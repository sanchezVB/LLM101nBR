# Gabarito — Capítulo 11

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> ```bash
> python solucoes/gabarito.py
> ```

## Sobre o orçamento de treino — leia antes dos números

O modelo da apostila leva **~18 minutos** por treino (3.000 passos). Este gabarito roda
vários experimentos, então usa **400 passos**. A arquitetura é **idêntica** à da apostila
(`n_embd=192`, 6 cabeças, 4 blocos, 2,2 M parâmetros) — só o número de passos muda.

Essa distinção importa. Reduzir passos é defensável para perguntas **comparativas**;
trocar a arquitetura não seria, porque mudaria o objeto em estudo.

Mas "defensável" não é "seguro". A aposta implícita é que a **ordem** entre as
configurações não depende do orçamento, mesmo que os valores absolutos dependam — e essa
aposta **já perdeu duas vezes neste curso**:

| Onde | Com orçamento curto | Com orçamento cheio |
|---|---|---|
| [Cap. 3, E4](../../03-ngram-model/solucoes/gabarito.md) | melhor `lr` = **1,0** | melhor `lr` = **0,1** |
| [Cap. 7, E5](../../07-optimization/solucoes/gabarito.md) | `3e-3` ganha | `1e-3` ganha |

Nos dois casos a pergunta era sobre **dinâmica de otimização** — e o orçamento é
justamente a variável que a dinâmica consome. As perguntas *deste* capítulo são
**estruturais** (quanto contexto, quantas posições, quantos dados), e deveriam ser
estáveis.

"Deveriam" não é medição. Rode [`_checagem_orcamento.py`](_checagem_orcamento.py), que
repete o E4 com o triplo dos passos e compara os **rankings**. É o mesmo teste que você
deve aplicar a qualquer experimento seu que rode com orçamento reduzido.

---

## E1 — Por que essa ordem?

**1. Por que o tokenizador é treinado só no texto de treino?**

Porque o BPE **aprende com os dados**: ele decide quais pares de bytes fundir contando
frequências. Se ele vir o texto de validação, essas contagens incluem material que o
modelo deveria não conhecer.

O que vazaria, concretamente: se `Memorial de Aires` entrasse no treino do tokenizador,
ele criaria tokens para nomes e expressões daquele livro. Na hora de avaliar, o texto de
validação ficaria **artificialmente mais barato de codificar** — menos tokens, cada um
mais previsível — e a loss cairia sem que o modelo tivesse ficado melhor.

É um vazamento sutil porque o tokenizador não é "o modelo". Mas ele faz parte do
**pipeline**, e tudo que toca o pipeline precisa respeitar a divisão.

**2. Por que dividir por obra e não sorteando parágrafos?**

Porque parágrafos da mesma obra **compartilham informação**. Um exemplo concreto: se um
parágrafo de *Dom Casmurro* que menciona "José Dias" cair no treino e o parágrafo seguinte
cair na validação, o modelo já viu o nome, o assento narrativo, o vocabulário da cena e
até a construção da frase anterior. Ele acerta a validação por ter visto quase a mesma
coisa — não por ter aprendido português.

Dividir por obra garante que a validação seja um texto que o modelo **nunca viu em
nenhuma forma**. É a diferença entre "consegue continuar este livro" e "consegue escrever
como Machado".

O [E2](e2_vazamento.py) mede exatamente esse efeito.

**3. Por que `uint16` e não `int64`?**

Porque o vocabulário tem 1.024 entradas e cabe folgado em 16 bits (que aguentam 65.536).
O `int64` guardaria os mesmos números em **8 bytes** em vez de 2:

| Tipo | Bytes/token | 621 mil tokens |
|---|---|---|
| `int64` | 8 | 4,97 MB |
| `uint16` | 2 | **1,24 MB** |

**Quatro vezes menos espaço**, e o ganho não é só de disco: é de **banda de memória**.
Como o treino lê os dados por `memmap` a cada batch, um arquivo 4x menor significa 4x
menos tráfego entre disco, cache e CPU. Em corpora de verdade — centenas de gigabytes —
essa escolha decide se o treino é limitado pelo cálculo ou pela leitura dos dados.

> ⚠️ O limite é real: com vocabulário acima de 65.536, o `uint16` **estoura silenciosamente**
> (os índices dão a volta). Tokenizadores modernos usam 100 mil a 200 mil tokens e
> precisam de `uint32`.

---

## E2 — A divisão errada

Solução em [`e2_vazamento.py`](e2_vazamento.py).

---

## E3 — O tokenizador é do domínio

**1. Os 20 tokens mais longos aprendidos:**

```
'José Dias ' | 'ima Justin' | 'que não ' | 'minha mã' | 'Capitú, '
'os olhos '  | 'da minha '  | 'inha mã'  | 'Capitú '  | 'José Di'
'a minha '   | 'seminari'   | 'ação, '   | 'almente ' | 'não me '
'que era '   | 'Capitú'     | 'ação '    | 'é que '   | 'primeir'
```

**6 dos 20** contêm nome de personagem (`José Dias`, `Capitú`, `Justin[a]`). Vários outros
são marcas do texto específico: `'minha mã[e]'`, `'seminari[o]'`, `'os olhos '` — o
narrador de *Dom Casmurro* falando da mãe, do seminário e dos olhos de ressaca.

Ninguém programou isso. O BPE só contou pares de bytes.

**2. Isso é bom ou ruim? Depende de onde você vai usá-lo.**

É **ótimo para este corpus e péssimo fora dele**. Um tokenizador que gasta um único token
em `'José Dias '` comprime Machado maravilhosamente — e não tem nenhuma fusão útil para
um artigo de medicina, onde cada `'hemoglobina'` seria fatiado em pedaços.

**3. O BPE treinado em nomes, aplicado a Machado.**

Mesmo trecho de prosa (109 bytes), dois BPEs de vocabulário 1.024:

| BPE treinado em | Tokens | Chars/token |
|---|---|---|
| Machado | **43** | 2,53 |
| nomes (Cap. 6) | **81** | 1,35 |

**1,88x mais tokens para dizer a mesma coisa.** O BPE de nomes aprendeu terminações como
`'ilson'` e `'erson'`, que quase não aparecem em texto corrido — e não aprendeu nada sobre
` que `, ` de ` ou `ção`.

O custo é direto: 1,88x mais tokens significa 1,88x mais posições de contexto gastas para
o mesmo texto, e um treino proporcionalmente mais caro para o mesmo conteúdo.

> **Nota de método.** A primeira versão deste gabarito comparava "81 tokens no trecho" (de
> um tokenizador) com "2,59 chars/token no corpus inteiro" (do outro). São **grandezas
> diferentes**, e a comparação parecia razoável sem ser. O número acima mede os dois no
> mesmo trecho.

---

## E4 — Tamanho do contexto

400 passos, modelo da apostila:

| `block_size` | Treino | Validação | ms/passo |
|---|---|---|---|
| 32 | 4,6154 | 4,8286 | 103,2 |
| 128 | 4,4455 | 4,6957 | 426,4 |
| 256 | **4,4013** | **4,6464** | 878,3 |

**1. Mais contexto melhora a loss** — e isso é o **contrário** do que acontecia no
[Capítulo 4](../../04-attention/solucoes/gabarito.md), onde `block_size=16` era pior que 8.

A diferença é o dado, não o modelo. Nomes têm ~7 letras: não há nada a lembrar de longe, e
contexto extra só adiciona ruído. Prosa tem dependências longas de verdade.

**2. Dobrar o contexto quadruplica o tempo? Não — e a resposta surpreende.**

| Mudança | Contexto | Tempo | Se fosse linear | Se fosse quadrático |
|---|---|---|---|---|
| 32 → 128 | 4x | **4,13x** | 4x | 16x |
| 128 → 256 | 2x | **2,06x** | 2x | 4x |

O crescimento é **linear**, apesar de a atenção ser O(T²).

O motivo: a atenção é **só uma parte** do custo. As camadas lineares — `qkv`, projeção, e
o feedforward que tem 4x a largura do modelo — crescem **linearmente** com T, e neste
tamanho elas dominam. O termo quadrático existe, mas só passa a mandar em contextos bem
maiores.

É por isso que "atenção é quadrática" é verdade e é enganoso ao mesmo tempo: verdade sobre
a fórmula, enganoso sobre onde o seu tempo está indo. Meça.

**3. Por que o contexto ajuda mais para prosa.**
Porque a dependência linguística é longa: concordância, referência a personagens, estrutura
de período, o assunto do parágrafo. Tudo isso liga tokens distantes. Um nome de 7 letras
não tem nada disso.

---

## E5 — Prever em todas as posições

| Modo | Treino | Validação | Previsões/batch | ms/passo |
|---|---|---|---|---|
| todas as posições | **4,4455** | **4,6957** | **4.096** | 365,1 |
| só a última | 6,0319 | 6,0564 | 32 | 360,4 |

**1.** Com `batch_size=32` e `block_size=128`: **4.096** previsões por batch contra **32**.
São **128x mais sinal de treino**.

**2. A diferença é enorme:** 4,70 contra 6,06 de loss de validação, com o mesmo orçamento
de passos. A versão que prevê só na última posição recebe 128x menos informação por passo.

**3. E o custo é praticamente o mesmo** — 365 ms contra 360 ms.

Esse é o ponto do exercício. As duas versões processam as 128 posições de qualquer jeito;
a única diferença é **quantas saídas da camada final entram na loss**. A versão "só a
última" faz o cálculo inteiro e **joga fora 99% dele**.

> **128x mais sinal, custo igual.** É o tipo de mudança que quase não existe em engenharia
> — normalmente se paga por ganho.

Os capítulos 3 a 7 usaram a versão ineficiente **de propósito**, porque ela mantinha a
métrica comparável entre capítulos. A partir daqui, não há motivo para isso.

---

## E6 — Dados sintéticos e *model collapse*

| Modelo | Treino | Validação **real** |
|---|---|---|
| modelo A (corpus de Machado) | 4,4455 | **4,6957** |
| modelo B (60 mil tokens gerados pelo A) | **4,3012** | **5,1453** |

**1 e 2. O modelo B fica pior na validação real** — degradação de **+0,4496**. O fenômeno
tem nome na literatura: ***model collapse***.

### O detalhe que torna isso perigoso

**Olhe a coluna de treino.** A loss de treino do modelo B é **menor** que a do A (4,30
contra 4,45). Se você olhasse só a curva de treino, concluiria que o B está aprendendo
melhor.

Não está. Texto gerado por um modelo é **mais previsível** que texto humano — menos
vocabulário, menos construções raras, menos surpresa. Ficar bom em prever texto fácil não
é ficar bom em prever Machado.

**O colapso chega disfarçado de progresso.** É por isso que a validação precisa vir de
dados reais e intocados — o mesmo princípio do E1 e do E2 deste capítulo.

### Por que acontece

A razão é informacional. O texto gerado por A é uma **amostra** da distribuição que A
aprendeu — e amostrar **perde a cauda**. Padrões raros aparecem pouco ou não aparecem, e o
B aprende uma versão ainda mais concentrada. Repetindo o ciclo, a diversidade colapsa.

> Nenhum modelo pode aprender dos próprios dados mais do que já sabe. A geração não cria
> informação nova sobre o mundo.

**3. Quando dados sintéticos ajudam de verdade:** quando existe uma **fonte de verdade
externa** para filtrar.

| Domínio | O verificador |
|---|---|
| matemática | dá para conferir a resposta |
| código | dá para rodar os testes |
| jogos | dá para saber quem ganhou |

Nesses casos o modelo gera muitas tentativas e um verificador seleciona as boas. A
informação que entra vem do **verificador**, não do modelo — e por isso o ciclo não
colapsa. É a base do que o [Capítulo 15](../../15-reinforcement-learning/) vai fazer.

---

## E7 — Escalando o corpus

### Versão do gabarito: frações do mesmo corpus

| Corpus | Tokens | Treino | Validação | **Gap** |
|---|---|---|---|---|
| 25% | 155.283 | 4,3627 | 4,7601 | 0,3975 |
| 50% | 310.567 | 4,4252 | 4,7330 | 0,3077 |
| 100% | 621.134 | 4,4455 | **4,6957** | **0,2501** |

**1 e 2.** Mais dados melhoram a validação, com **retorno decrescente**: dobrar de 25% para
50% rendeu 0,027 de loss; dobrar de novo rendeu 0,037. Ganhos na mesma casa para um esforço
que dobra.

**3. A lógica do Capítulo 3 vale em direção, não em magnitude.**

O **gap** entre treino e validação encolhe monotonicamente (0,40 → 0,31 → 0,25),
confirmando que parte do problema era memorização — exatamente como lá.

Mas compare a escala:

| Capítulo | Mudança nos dados | Efeito no gap |
|---|---|---|
| [Cap. 3](../../03-ngram-model/solucoes/gabarito.md) | 155 → 64.000 nomes | 5,7 → ~0 |
| Cap. 11 | 155 mil → 621 mil tokens | 0,40 → 0,25 |

Lá o modelo tinha ~10 exemplos por parâmetro; aqui já tem centenas de milhares de tokens.
**Melhorias ficam progressivamente mais caras** — é a forma de toda curva de escala, e é
por isso que os laboratórios discutem *ordens de grandeza* de dados, não porcentagens.

### Versão literal: acrescentando obras de verdade

O enunciado pede para **acrescentar obras**. A lista completa de Machado no Gutenberg tem
12 (confirmado na API, não chutado) — o `prepare_data.py` usa 5.
[`e7_mais_obras.py`](e7_mais_obras.py) baixa as outras 7 e mede.

**Duas decisões de método, que importam mais que o resultado:**

1. **O tokenizador fica fixo.** Usamos o `tokenizador.pkl` já treinado, sem retreinar. Se
   retreinássemos, mudaríamos **duas coisas ao mesmo tempo** (mais dados *e* outro
   tokenizador) e não saberíamos a qual atribuir a diferença. Custo honesto: o tokenizador
   fica levemente mal-adaptado às obras novas — é o preço de isolar a variável, e é o preço
   certo a pagar.
2. **A validação não muda.** *Memorial de Aires* continua fora do treino. Sem isso as
   losses não seriam comparáveis entre configurações.
