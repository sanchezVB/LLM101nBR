# Capítulo 16 — Deployment

> **Objetivo de aprendizagem:** servir o modelo por HTTP e descobrir, medindo, as duas
> decisões que definem um serviço de LLM — **streaming** (o que o usuário sente) e
> **batching** (o que o servidor entrega). São coisas diferentes, e otimizar uma não
> otimiza a outra.

**Pré-requisitos:** Capítulos 12 (KV-cache, prefill vs decode) e 13 (quantização).

**Arquivos:**
- [`servidor.py`](servidor.py) — servidor HTTP com streaming, biblioteca padrão apenas
- [`carga.py`](carga.py) — mede TTFT, latência e vazão sob concorrência
- [`servidor_batch.py`](servidor_batch.py) — o mesmo servidor, agrupando requisições
- [`exercicios.md`](exercicios.md) — exercícios

> **Sem framework, de propósito.** Nada de FastAPI ou uvicorn — só `http.server`. O
> assunto do capítulo são as decisões de serviço, e um framework as esconde atrás de
> decoradores. Você vai escrever o `Transfer-Encoding: chunked` na mão.

---

## 1. Um modelo treinado não é um serviço

Até aqui o modelo viveu dentro de scripts. Servir é outra coisa, e as perguntas mudam:

| No treino você pergunta | Servindo você pergunta |
|---|---|
| a loss caiu? | quanto tempo até o usuário ver algo? |
| quanto tempo leva uma época? | quantos usuários simultâneos cabem? |
| cabe na memória? | o que acontece quando dez chegam juntos? |

Nenhuma dessas aparece numa curva de treino, e é por isso que este capítulo mede coisas
que nenhum capítulo anterior mediu.

---

## 2. Streaming: a otimização que não acelera nada

O servidor tem dois endpoints que produzem **o mesmo texto**:

```
POST /gerar           devolve a resposta inteira, de uma vez
POST /gerar/stream    devolve token por token, conforme são produzidos
```

O que muda é quando o primeiro byte chega:

| Endpoint | TTFT | Total |
|---|---|---|
| `/gerar` | 0,115 s | 0,115 s |
| `/gerar/stream` | **0,079 s** | 0,125 s |

O **total é o mesmo** — tem de ser, o servidor faz o mesmo trabalho. O que muda é o
**TTFT** (*time to first token*), e é ele que o usuário sente como "travou" ou
"respondeu".

### Por que isso é possível

Porque a geração é **autorregressiva**: os tokens existem um a um, em ordem. O `gerar` do
servidor é um **gerador Python** — ele não devolve o texto pronto, cede cada token no
instante em que existe:

```python
for t in gerar_tokens(ids, n):
    pedaco = decodificar([t]).encode("utf-8")
    self.wfile.write(f"{len(pedaco):X}\r\n".encode() + pedaco + b"\r\n")
    self.wfile.flush()          # sem isto, o SO junta tudo e não há streaming
```

Aquele `flush()` é o capítulo inteiro em uma linha. Sem ele o sistema operacional
acumula os bytes num buffer e entrega tudo junto — o código parece streaming e não é.

> Um modelo que produzisse a resposta inteira de uma vez não teria o que transmitir aos
> poucos. O streaming é grátis porque a arquitetura já é sequencial.

### ⚠️ E a magnitude aqui é pequena, por um motivo

A diferença medida é de 0,04 s — uns 30%. É pouco, e a causa é que **este modelo é rápido
demais para o streaming brilhar**: 40 tokens saem em 0,12 s, então o TTFT já era baixo e
boa parte dele é custo de HTTP, não de geração.

O ganho do streaming é proporcional ao **tempo total de geração**. Num modelo de verdade,
500 tokens a 30 tok/s levam ~17 segundos — e o TTFT cai de **17 s para ~0,3 s**. É a
diferença entre um sistema que parece travado e um que parece instantâneo.

Este capítulo mede o **mecanismo**. A magnitude vem da escala.

---

## 3. O que acontece quando vários clientes chegam juntos

O servidor usa `ThreadingHTTPServer`, então cada requisição roda numa thread. Parece que
vai escalar. Medindo:

| Clientes | Latência mediana | p95 | Vazão total |
|---|---|---|---|
| 1 | 0,15 s | 0,15 s | 268 tok/s |
| 2 | 0,20 s | 0,27 s | 301 tok/s |
| 4 | 0,32 s | 0,50 s | 320 tok/s |
| 8 | **0,54 s** | **0,96 s** | **334 tok/s** |

**Oito vezes mais clientes, 25% mais trabalho entregue.** E a latência cresce 3,6x, com o
p95 quase dobrando a mediana — há fila, e quem chega por último paga.

A causa está no `servidor.py`: existe uma **trava** em volta da geração, então as
requisições se **revezam** no modelo. Os 25% de ganho vêm das partes que rodam fora da
trava — HTTP, tokenização, JSON.

E vale ser preciso: **a trava não está errada.** Sem ela, várias threads chamariam o
modelo ao mesmo tempo e disputariam os mesmos núcleos, o que costuma sair pior. O problema
é outro — requisições concorrentes estão sendo tratadas como **independentes** quando
poderiam viajar **juntas**.

---

## 4. Batching: as requisições viajam juntas

A solução vem direto do [Capítulo 12](../12-inference-kv-cache/README.md): o decode é
limitado por **memória**. Ler os pesos do modelo custa o mesmo para 1 ou para 16
sequências — então processar 16 juntas custa quase o mesmo que processar 1.

Lá isso foi medido: gerar 16 sequências em paralelo custou **9,1x menos** que 16 gerações
separadas. Aqui é a mesma economia, aplicada a requisições de usuários diferentes.

O desenho inverte quem manda:

```
cliente A --\                              /--> tokens de A
cliente B ---+--> [fila] --> laço ---------+---> tokens de B
cliente C --/               (1 thread)      \--> tokens de C
```

Um **único thread** é dono do modelo. As requisições HTTP não geram nada — elas põem um
pedido numa fila e esperam os tokens voltarem. O thread-dono junta quem estiver esperando,
gera **um token para todos de uma vez**, distribui, e repete.

E quem termina **sai do lote na hora**, sem esperar os outros:

```python
for p in [p for p in ativos if p.restantes <= 0]:
    p.saida.put(None)
ativos = [p for p in ativos if p.restantes > 0]
```

É isso que torna o batching **contínuo** em vez de estático — e é o que vLLM e TGI fazem,
com muito mais cuidado.

---

## 5. O resultado

**Vazão total (tokens/s):**

| Clientes | Simples (trava) | Com batching | Ganho |
|---|---|---|---|
| 1 | 268 | 306 | 1,14x |
| 2 | 301 | 434 | 1,44x |
| 4 | 320 | 585 | 1,83x |
| 8 | 334 | **755** | **2,26x** |

**Latência (mediana / p95):**

| Clientes | Simples (trava) | Com batching |
|---|---|---|
| 1 | 0,15 s / 0,15 s | 0,13 s / 0,13 s |
| 2 | 0,20 s / 0,27 s | 0,18 s / 0,18 s |
| 4 | 0,32 s / 0,50 s | 0,27 s / 0,27 s |
| 8 | 0,54 s / **0,96 s** | **0,42 s / 0,42 s** |

**2,26x mais vazão com 8 clientes — e latência menor.** Não houve troca: o batching
ganhou nos dois eixos.

### O detalhe mais bonito da tabela

Olhe o **p95** das duas colunas.

Com a trava, ele é quase o dobro da mediana (0,96 contra 0,54). Isso é uma **fila**: a
maioria é atendida rápido e alguns esperam muito, porque chegaram por último.

Com batching, **p95 = mediana**. Todos avançam **no mesmo passo** — cada iteração produz
um token para todo mundo. Não existe "último da fila", então não existe cauda.

> Um p95 igual à mediana é a assinatura de um sistema onde ninguém espera pelos outros.
> Vale procurar por isso em qualquer serviço que você meça.

---

## 6. Um bug meu, e por que ele vale estar aqui

A primeira versão do `servidor_batch.py` era **mais lenta** que o servidor simples: 0,62 s
de latência contra 0,15 s, e 293 tok/s contra 334.

A causa era uma linha:

```python
ativos.append(FILA.get(timeout=None if not ativos else ESPERA_MS / 1000))
```

Ela esperava 8 ms por novas requisições **a cada passo de decode**. Com 40 tokens, isso
somava **320 ms de espera pura** por requisição. O ganho do agrupamento existia e estava
sendo inteiramente consumido pela espera.

A correção é uma regra que vale para qualquer laço de serviço:

> **Nunca bloqueie quando há trabalho a fazer.** Bloqueie só quando estiver ocioso; quando
> houver fila, drene sem esperar.

```python
if not ativos:
    ativos.append(FILA.get())            # ocioso: pode bloquear
while len(ativos) < LOTE_MAX:
    try:    ativos.append(FILA.get_nowait())   # ocupado: nunca espera
    except queue.Empty: break
```

Vale registrar o que quase aconteceu: eu tinha uma tabela mostrando batching **pior** que a
versão simples. Ela era verdadeira e teria ensinado a coisa errada, porque a causa não era
o batching — era a minha espera de 8 ms.

---

## 7. O que este capítulo não faz

Um servidor de produção tem várias coisas que aqui foram simplificadas de propósito, e
convém saber quais:

| Simplificação | O que a produção faz |
|---|---|
| todas as sequências recortadas no mesmo comprimento | *padding* com máscara, ou **paged attention** |
| KV-cache recalculado a cada passo do lote | cache persistente por sequência |
| sem limite de fila, sem timeout de admissão | *backpressure* — recusar quando cheio é melhor que atender mal |
| sem autenticação, sem limite de taxa | ambos, sempre |
| um processo, uma máquina | réplicas atrás de um balanceador |

A que mais dói é a primeira: recortar no menor comprimento é correto aqui porque todas as
requisições do teste começam parecidas, e seria errado com prompts de tamanhos diferentes.

> **Este capítulo mostra o mecanismo do batching, não uma implementação de produção.** A
> diferença entre os dois é o assunto de bibliotecas inteiras.

---

## 8. Resumo do capítulo

- **Streaming** não acelera nada e muda tudo: mesmo total, TTFT muito menor. O ganho é
  proporcional ao tempo de geração — pequeno aqui, enorme num modelo real.
- O `flush()` depois de cada token é o que separa streaming de streaming aparente.
- **Threads não escalam** um serviço de LLM: com a trava, 8 clientes rendem 25% mais vazão
  e 3,6x mais latência.
- **Batching escala**, porque o decode é limitado por memória — 2,26x de vazão *e* latência
  menor.
- **p95 = mediana** é a assinatura de um sistema sem fila. Meça o p95, não só a média.
- Num laço de serviço, **nunca bloqueie quando há trabalho** — foi o que fez o meu batching
  ficar mais lento que o servidor simples.

---

### Próximo capítulo

[**Capítulo 17 — Multimodal.**](../17-multimodal/) O último. Até aqui o modelo só viu
tokens de texto. E se a mesma arquitetura pudesse receber uma imagem?
