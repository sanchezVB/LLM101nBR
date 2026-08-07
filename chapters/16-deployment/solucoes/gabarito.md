# Gabarito — Capítulo 16

> Respostas discursivas. Os **números** vêm de [`carga.py`](../carga.py), rodado contra os
> dois servidores.
>
> ```bash
> python servidor.py          # terminal 1
> python carga.py             # terminal 2
> ```

---

## E1 — TTFT e latência

**1. O que o streaming melhora.**

O **tempo até o usuário ver alguma coisa**, e nada mais. O servidor faz o mesmo trabalho,
o texto é o mesmo, o total é o mesmo.

O que muda é a experiência: sem streaming, uma tela parada durante toda a geração e depois
tudo de uma vez. Com streaming, a primeira palavra quase imediatamente e a leitura
acompanha a produção.

> É uma otimização de **latência percebida**, não de latência. A distinção parece
> acadêmica até você usar os dois.

**2. Por que num LLM sim e num classificador não.**

Porque a geração é **autorregressiva**: a saída é uma sequência produzida em ordem, e cada
token existe antes do seguinte. Há o que transmitir aos poucos.

Um classificador produz **uma** saída, no fim de um único forward. Não existe "primeira
metade da etiqueta". Não há nada para transmitir antes de terminar.

O streaming não é uma técnica que se aplica a qualquer modelo — ele é uma consequência de
a saída ser sequencial.

**3. Sem o `flush()`.**

O sistema operacional acumula os bytes num buffer e envia quando ele enche ou quando a
conexão fecha. O resultado prático: o cliente recebe tudo de uma vez, no fim — exatamente
o comportamento do endpoint não-streaming.

E o código **continuaria parecendo streaming**: o laço escreve token por token, o
`Transfer-Encoding: chunked` está lá, tudo tem cara de certo. O defeito só aparece
medindo o TTFT.

> É o mesmo padrão do [Capítulo 12](../../12-inference-kv-cache/solucoes/gabarito.md): um
> bug que não levanta exceção e só é detectável por medição.

---

## E2 — Meça o seu próprio TTFT

| Tokens pedidos | TTFT (stream) | Total (stream) | **TTFT/total** | Total (sem stream) |
|---|---|---|---|---|
| 40 | 0,085 s | 0,131 s | **65,0%** | 0,117 s |
| 200 | 0,084 s | 0,872 s | **9,6%** | 0,762 s |

**1 e 2. O TTFT é praticamente constante; o total é que cresce.**

0,085 s com 40 tokens, 0,084 s com 200. Faz sentido: o primeiro token custa um prefill mais
um passo de decode, e isso não depende de quantos tokens virão depois.

Então a razão TTFT/total **despenca** conforme a resposta cresce: de 65% para 9,6%. O
streaming melhora pouco em respostas curtas e muito em respostas longas — e a tendência é
mecânica, não empírica.

**3. A extrapolação, e ela muda a leitura do capítulo.**

Um modelo real a 30 tok/s, 500 tokens:

| Medida | Sem streaming | Com streaming |
|---|---|---|
| tempo até ver algo | **~17 s** | **~0,3 s** |
| razão TTFT/total | 100% | ~1,8% |

Dezessete segundos de tela parada é o que separa "está quebrado" de "está pensando". A
magnitude de 30% medida no capítulo é pequena **porque o modelo é rápido demais para o
efeito aparecer** — não porque o efeito seja pequeno.

> Esta é a mesma armadilha do [Capítulo 12](../../12-inference-kv-cache/solucoes/gabarito.md),
> onde o KV-cache rendeu 1,93x em vez das dezenas que a teoria sugeria: **a escala do
> curso comprime os ganhos.** O mecanismo é o que se aprende; a magnitude vem da escala em
> que você o aplica.

---

## E3 — Por que threads não bastam

| Clientes | Latência mediana | p95 | Vazão total |
|---|---|---|---|
| 1 | 0,15 s | 0,15 s | 268 tok/s |
| 2 | 0,20 s | 0,27 s | 301 tok/s |
| 4 | 0,32 s | 0,50 s | 320 tok/s |
| 8 | 0,54 s | 0,96 s | **334 tok/s** |

**1. Não cresce proporcionalmente — cresce 25% para 8x mais clientes.** Se escalasse, os
334 seriam ~2.100.

**2. Tirar a trava não resolve, e provavelmente piora.**

Sem ela, oito threads chamam o modelo simultaneamente. Mas o PyTorch já usa **todos os
núcleos** internamente numa única matmul — então as threads passam a disputar os mesmos
núcleos, com troca de contexto e contenção de cache por cima.

O resultado típico é vazão igual ou pior, e latência mais irregular. **Meça antes de
opinar** — este exercício existe para isso.

**3. Por que agrupar resolve, e threads não.**

O [Capítulo 12](../../12-inference-kv-cache/solucoes/gabarito.md) estabeleceu que o decode
é limitado por **memória**: para produzir um token é preciso ler **todos** os pesos do
modelo. Lá isso foi medido — gerar 16 sequências em paralelo custou 9,1x menos que 16
gerações separadas.

A diferença é o que se compartilha:

| Estratégia | Leituras dos pesos, para 8 clientes |
|---|---|
| 8 threads independentes | **8** (cada uma lê tudo) |
| 1 lote de 8 | **1** (todos usam a mesma leitura) |

Threads dividem o *tempo* de acesso ao modelo. O batching divide o *custo* de cada acesso.
São coisas diferentes, e só a segunda ataca o gargalo real.

---

## E4 — O tamanho do lote

Oito clientes, 40 tokens cada:

| `--lote` | Latência mediana | Vazão | Ganho sobre lote 1 |
|---|---|---|---|
| 1 | 0,62 s | 288 tok/s | — |
| 4 | 0,47 s | 514 tok/s | 1,8x |
| 16 | 0,47 s | **673 tok/s** | **2,3x** |

**1.** A vazão mais que dobra de 1 para 16, e a latência **melhora** (0,62 → 0,47 s). Não
há troca aqui: mais lote é melhor nos dois eixos, até o ponto em que satura.

**2. `--lote 1` deveria reproduzir o servidor sem batching. Não reproduz exatamente** —
288 tok/s contra 334 do `servidor.py`.

O que mais mudou: no `servidor_batch.py` toda requisição passa por **duas filas**
(`queue.Queue`) e é servida por um thread que não é o do HTTP. Isso custa sincronização e
troca de contexto por token. Com lote 1 você paga esse custo sem receber nada em troca.

> Vale registrar como leitura de tabela: `--lote 1` **não** é o controle limpo que parece.
> Ele isola o efeito do agrupamento, mas carrega a sobrecarga da arquitetura de filas.

**3. O ganho satura, e a resposta é a mesma do E4 do Capítulo 8.**

De 4 para 16 o ganho é bem menor que de 1 para 4. O que limita é o mesmo: em algum ponto o
lote deixa de ser **limitado por memória** e passa a ser **limitado por cálculo** — a
matmul fica grande o bastante para ocupar os núcleos, e dali em diante dobrar o lote dobra
o tempo.

A diferença é que aqui há um segundo limite, que o Capítulo 8 não tinha: o **KV-cache
cresce com o lote** (Capítulo 12). Em escala real, é a memória que fecha a conta antes do
cálculo.

---

## E5 — A espera que custou caro

**1. A conta bate.**

Com a espera de 8 ms por passo, a latência foi de 0,15 s (servidor simples) para **0,62 s**.
A diferença é 0,47 s, contra os `40 × 8 ms = 0,32 s` previstos — a mesma ordem, com o resto
vindo da sobrecarga das filas medida no E4.

**2. Sim, existe um cenário em que ela ajuda: chegadas espaçadas.**

Se as requisições chegam de forma dispersa — uma a cada 5 ms, digamos — esperar um pouco
antes de começar junta várias num lote maior. Sem espera, a primeira requisição começa
sozinha e as seguintes perdem a carona.

É um compromisso real, e tem nome em sistemas de fila: **esperar aumenta a latência de
quem já chegou para aumentar a vazão de todos**. Serviços de verdade expõem isso como um
parâmetro (`max_waiting_time`, ou equivalente).

O erro da minha versão não foi ter espera — foi **esperar a cada passo de decode** em vez
de só na formação do lote. Esperar uma vez custa 8 ms; esperar 40 vezes custa 320.

**3. A regra:**

> **Nunca bloqueie quando há trabalho a fazer.** Bloqueie apenas quando estiver ocioso.

Ela aparece em qualquer laço de serviço. Um exemplo fora daqui: um consumidor de fila de
mensagens que faz `poll(timeout=100ms)` mesmo tendo mensagens no buffer local — ele
adiciona 100 ms a cada ciclo por nada. O padrão correto é drenar o que já está disponível
sem bloquear, e só então esperar.

---

## E6 — O p95 conta o que a média esconde

**1. Por que diferentes num caso e iguais no outro.**

| Servidor | Mediana | p95 |
|---|---|---|
| com trava | 0,54 s | **0,96 s** |
| com batching | 0,42 s | **0,42 s** |

Com a trava existe uma **fila**: as requisições são atendidas em sequência, então a
primeira espera pouco e a oitava espera sete gerações. A distribuição tem cauda porque as
posições na fila são diferentes.

Com batching **todos avançam no mesmo passo** — cada iteração produz um token para o lote
inteiro. Não há posição na fila, então não há cauda.

> **p95 = mediana é a assinatura de um sistema onde ninguém espera pelos outros.** Vale
> procurar por isso em qualquer serviço que você meça: quando os dois números se separam,
> há uma fila em algum lugar.

**2. Média 200 ms, p99 de 4 s.**

Uma em cada cem requisições demora **vinte vezes** a média. Para o usuário isso não é "às
vezes um pouco mais lento" — é o sistema travando de vez em quando, sem padrão aparente.

E numa página que faz 10 chamadas, a chance de pelo menos uma cair no p99 é de ~10%. A
cauda que parecia rara vira comum quando as requisições se acumulam.

Reportar só a média esconde precisamente o que o usuário lembra.

**3. Outra métrica de cauda para um serviço de LLM:**

O **p95 do TTFT**, separado do p95 da latência total. São coisas diferentes: um servidor
pode ter TTFT estável e latência total com cauda (respostas de tamanhos muito variados), ou
o contrário (fila na admissão). Só medindo os dois dá para saber onde está o problema.

Uma segunda: a **taxa de requisições recusadas ou expiradas**. Ela é zero até o sistema
saturar, e depois é a métrica mais importante que existe — e não aparece em nenhuma
estatística de latência, porque as requisições que falharam não têm latência.

---

## E7 — Backpressure (desafio)

**1 e 2.** Com fila ilimitada, a latência cresce sem limite: a 64 clientes cada um espera
o lote inteiro várias vezes, e as requisições mais antigas provavelmente já expiraram do
lado do cliente — o servidor gasta trabalho gerando respostas que ninguém vai ler.

O limite deve sair da sua medição: escolha o N tal que a latência do último da fila ainda
seja aceitável para o seu caso.

**3. Por que recusar pode ser melhor que atender devagar.**

Três razões, e a terceira é a que menos se pensa:

- **Um 429 é acionável.** O cliente sabe imediatamente que deve tentar depois, ou ir para
  outra réplica. Uma resposta que demora 40 s não informa nada — ele fica esperando.
- **Trabalho desperdiçado prejudica quem está sendo atendido.** Gerar tokens para um
  cliente que já desistiu consome exatamente os mesmos recursos de quem ainda está lá.
- **Sem limite, a degradação é global.** Com fila ilimitada, sobrecarga não faz alguns
  falharem — faz **todos** ficarem lentos. Recusar concentra o dano em quem chegou por
  último, em vez de espalhá-lo por todos.

> É a mesma escolha que o E6 revela na tabela: um sistema pode falhar de forma
> **concentrada e visível** ou **difusa e silenciosa**. A primeira é quase sempre melhor de
> operar — e é o mesmo argumento que o [Capítulo 12](../../12-inference-kv-cache/solucoes/gabarito.md)
> fez sobre o gloo travar sempre em vez de às vezes.
