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
configurações não depende do orçamento, mesmo que os valores absolutos dependam. Essa
aposta **já tinha perdido duas vezes** neste curso:

| Onde | Com orçamento curto | Com orçamento cheio |
|---|---|---|
| [Cap. 3, E4](../../03-ngram-model/solucoes/gabarito.md) | melhor `lr` = **1,0** | melhor `lr` = **0,1** |
| [Cap. 7, E5](../../07-optimization/solucoes/gabarito.md) | `3e-3` ganha | `1e-3` ganha |

Nos dois casos a pergunta era sobre **dinâmica de otimização** — e o orçamento é
justamente a variável que a dinâmica consome. Já as perguntas *deste* capítulo são
**estruturais** (quanto contexto, quantas posições, quantos dados). Eu escrevi, aqui
mesmo, que por isso elas *deveriam* ser estáveis.

### E aí eu medi, e estava errado

[`checagem_orcamento.py`](checagem_orcamento.py) repete o E4 com o triplo dos passos e
compara os **rankings**:

| Orçamento | Ranking (melhor → pior) |
|---|---|
| 400 passos | **256** < 128 < 32 |
| 1.200 passos | **128** < 256 < 32 |

**O ranking virou.** Com 400 passos, `block_size=256` parecia o melhor; com 1.200 ele
perde para 128 (4,1270 contra 4,1649). O E4 não era estrutural o bastante para escapar.

Refeito com os **3.000 passos** da apostila, virou de novo — e o mesmo aconteceu com o E7:

| Exercício | Com 400 passos | Com 3.000 passos |
|---|---|---|
| **E4** (contexto) | 256 < 128 < 32 | **32 < 128 < 256** (inverteu por completo) |
| **E7** (mais obras) | 6 < 8 < 4 < 11 | **11 < 8 < 6 < 4** (inverteu por completo) |

Duas inversões totais, nos dois exercícios que testei. Por isso as respostas do E4 e do E7
vêm de [`e4_orcamento_cheio.py`](e4_orcamento_cheio.py) e
[`e7_orcamento_cheio.py`](e7_orcamento_cheio.py) — cerca de uma hora de treino cada. Vale a
hora: a alternativa era publicar respostas que a própria checagem do curso já desmentia.

> **E os outros exercícios deste capítulo?** E1, E3 e E5 não dependem de treino (são
> conceituais, de tokenização e de contagem de previsões), então o orçamento não os afeta.
> O **E6** (*model collapse*) foi reverificado em 3.000 passos
> ([`e6_orcamento_cheio.py`](e6_orcamento_cheio.py)) e a conclusão **se manteve, doze vezes
> mais forte**: a degradação passou de +0,45 para **+5,33**.
>
> Ou seja: dos três exercícios deste capítulo que dependem de treino, **dois inverteram** e
> **um se confirmou**. Não há como saber qual é qual sem medir — e essa é exatamente a
> razão de medir.

> **A regra que fica.** "Esta pergunta é estrutural, logo o orçamento não importa" é uma
> hipótese, não um argumento. Rode a sua configuração com o triplo dos passos e veja se a
> ordem se mantém. Se virar, o orçamento *é* a variável que decide — e o resultado curto
> não responde nada.

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

**Com os 3.000 passos da apostila** ([`e4_orcamento_cheio.py`](e4_orcamento_cheio.py)):

| `block_size` | Treino | Validação | **Gap** | minutos |
|---|---|---|---|---|
| 32 | 3,1947 | **3,8305** | **0,64** | 4,1 |
| 128 | 2,7816 | 3,9241 | 1,14 | 16,7 |
| 256 | 2,7419 | 4,0136 | 1,27 | 38,8 |

**1. Mais contexto PIORA a loss de validação neste corpus.** O melhor `block_size` é 32 —
o menor dos três.

Essa é a resposta oposta à que eu tinha escrito, e a história de como ela mudou é a parte
mais útil do exercício:

| Orçamento | Ranking (melhor → pior) | Vencedor |
|---|---|---|
| 400 passos | 256 < 128 < 32 | **256** |
| 1.200 passos | 128 < 256 < 32 | **128** |
| **3.000 passos** | **32 < 128 < 256** | **32** |

A ordem se **inverteu completamente** conforme o orçamento cresceu. Com 400 passos eu
concluí "mais contexto ajuda"; no orçamento de verdade, mais contexto atrapalha.

### Por que — olhe a coluna do gap

O contexto maior tem loss de **treino menor** (2,74 contra 3,19) e loss de **validação
maior**. Isso é overfitting, e o gap cresce monotonicamente com o contexto: 0,64 → 1,14 →
1,27.

A causa é um detalhe do desenho do experimento que vale internalizar. Com o **número de
passos fixo**, `block_size` maior significa **mais épocas sobre o mesmo corpus**:

| `block_size` | Tokens vistos em 3.000 passos | Épocas sobre 621 mil tokens |
|---|---|---|
| 32 | 3,1 M | ~5 |
| 128 | 12,3 M | ~20 |
| 256 | 24,6 M | ~40 |

Quarenta passagens por um corpus de 621 mil tokens, com um modelo de 2,2 M parâmetros, é
receita de memorização. O que a tabela mede não é só "quanto contexto ajuda" — é "quanto
contexto ajuda **a esta razão entre dados e capacidade**".

Com 400 passos ninguém tinha épocas suficientes para memorizar, e aí o contexto maior só
mostrava a sua vantagem. O overfitting só aparece quando você treina o bastante para ele
aparecer.

### E a comparação com o Capítulo 4 fica mais interessante

No [Capítulo 4](../../04-attention/solucoes/gabarito.md), `block_size=16` era pior que 8 —
e eu atribuí isso a nomes serem curtos. Aqui, com prosa, o contexto maior **também**
perde. A explicação "nomes são curtos" era verdadeira mas insuficiente: o fator comum aos
dois casos é **corpus pequeno demais para a capacidade do modelo**.

> **Contexto longo não é um bem em si.** Ele é uma aposta: você gasta capacidade e épocas
> para poder olhar mais longe. A aposta paga quando há dados suficientes para sustentá-la.
> Os modelos de contexto enorme que você vê por aí são treinados em trilhões de tokens —
> não em 1,6 MB de Machado.

> ⚠️ **Uma consequência para a apostila.** O capítulo treina com `block_size=128`, que por
> esta medida não é o ótimo para a loss de validação. A escolha continua defensável por
> outro motivo: um modelo com contexto 32 **não consegue** condicionar em mais de 32
> tokens na hora de gerar, por melhor que seja a sua loss média. Loss de validação mede
> previsão do próximo token, não coerência de texto longo. Mas é uma escolha que agora
> está medida, não presumida.

**2. Dobrar o contexto quadruplica o tempo? Não — e a resposta surpreende.**

| Mudança | Contexto | Tempo (3.000 passos) | Se fosse linear | Se fosse quadrático |
|---|---|---|---|---|
| 32 → 128 | 4x | **4,07x** | 4x | 16x |
| 128 → 256 | 2x | **2,32x** | 2x | 4x |

O crescimento é **quase linear**, apesar de a atenção ser O(T²). (A medição com 400
passos deu 4,13x e 2,06x — as duas concordam, e é o esperado: tempo *por passo* não
depende de quantos passos você dá.)

O motivo: a atenção é **só uma parte** do custo. As camadas lineares — `qkv`, projeção, e
o feedforward que tem 4x a largura do modelo — crescem **linearmente** com T, e neste
tamanho elas dominam. O termo quadrático existe, mas só passa a mandar em contextos bem
maiores.

É por isso que "atenção é quadrática" é verdade e é enganoso ao mesmo tempo: verdade sobre
a fórmula, enganoso sobre onde o seu tempo está indo. Meça.

**3. "Para prosa, o contexto maior ajuda mais do que ajudava para nomes. Por quê?"**

**A premissa da pergunta está errada, e o enunciado a colocou lá porque eu acreditava
nela.** Medido, o contexto maior não ajuda aqui — atrapalha.

O raciocínio que sustentava a premissa continua **correto e é insuficiente**: prosa
realmente tem dependências longas (concordância, referência a personagens, estrutura de
período), e nomes de 7 letras não têm nada disso. Existe, sim, informação útil a 200
tokens de distância em Machado.

O que falta ao raciocínio é a outra metade da conta. Poder usar contexto longo exige:

1. **informação útil lá atrás** — prosa tem, nomes não têm; e
2. **dados suficientes para aprender a usá-la sem memorizar o caminho**.

Este corpus atende ao item 1 e falha no item 2. O modelo tem contexto disponível e não
tem exemplos bastantes para aprender a explorá-lo — então usa a capacidade extra para
decorar o treino.

> É o tipo de erro que este curso repete de propósito: uma explicação **verdadeira** que
> não é a **dominante**. A mesma coisa aconteceu no [E2 do Capítulo 10](../../10-distributed/solucoes/gabarito.md),
> onde a propriedade do anel que eu citei era real mas não era o que mandava no tempo.

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

> ✅ **Reverificado no orçamento cheio.** Este era o último exercício do capítulo apoiado
> em 400 passos, num capítulo onde o E4 e o E7 tiveram a conclusão **invertida** ao passar
> para 3.000. Eu tinha publicado a conclusão com uma justificativa — não uma medição — de
> que ela deveria sobreviver. [`e6_orcamento_cheio.py`](e6_orcamento_cheio.py) mediu:
>
> | Modelo | Treino | Validação real |
> |---|---|---|
> | A (corpus de Machado) | 2,7816 | 3,9241 |
> | B (texto gerado pelo A) | **0,1149** | **9,2588** |
>
> **Degradação de +5,33**, contra +0,45 com 400 passos. Perplexidade de 50,6 para
> **10.496**. A conclusão não só se manteve como ficou **doze vezes mais forte** — e a
> justificativa que eu tinha dado (o argumento informacional não depende do orçamento)
> estava certa.
>
> Os números da seção abaixo continuam sendo os de 400 passos, porque é o que o
> `gabarito.py` roda em poucos minutos. As conclusões valem nos dois orçamentos.

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

E no orçamento cheio o contraste fica grotesco: **0,1149 de treino contra 9,2588 de
validação**. O modelo B decorou o texto sintético quase perfeitamente — loss de treino 24
vezes menor que a do A — enquanto ficava 2,4 vezes pior no texto real. É a demonstração
mais nítida deste curso de que **uma curva de treino excelente pode acompanhar um modelo
destruído**.

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

**1 e 2.** Mais dados melhoram a validação. Os ganhos são pequenos e da mesma ordem ao
dobrar o corpus (0,027 e depois 0,037).

> ⚠️ **Não leia isto como saturação.** Estes números são de **400 passos**, e a seção
> seguinte mostra que nesse orçamento o modelo mal completa uma passada pelo corpus — não
> há épocas suficientes para o overfitting aparecer, que é justamente o mecanismo pelo qual
> mais dados ajudam. Com os 3.000 passos da apostila, o ganho é **muito maior e não
> satura**. Esta tabela serve para ver o mecanismo depressa, não para concluir sobre escala.

**3. A lógica do Capítulo 3 vale em direção, não em magnitude.**

O **gap** entre treino e validação encolhe monotonicamente (0,40 → 0,31 → 0,25),
confirmando que parte do problema era memorização — exatamente como lá.

Mas compare a escala:

| Capítulo | Mudança nos dados | Efeito no gap |
|---|---|---|
| [Cap. 3](../../03-ngram-model/solucoes/gabarito.md) | 155 → 64.000 nomes | 5,7 → ~0 |
| Cap. 11 | 155 mil → 621 mil tokens | 0,40 → 0,25 |

Lá o modelo tinha ~10 exemplos por parâmetro; aqui já tem centenas de milhares de tokens.

> Eu ia escrever aqui que "melhorias ficam progressivamente mais caras, é a forma de toda
> curva de escala". A frase é verdadeira **em geral** e não descreve o que foi medido: no
> orçamento cheio o ganho não deu sinal de desacelerar. Ver a seção seguinte.

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

### Os resultados, com os 3.000 passos da apostila

[`e7_orcamento_cheio.py`](e7_orcamento_cheio.py):

| Obras | Tokens | Treino | Validação | **Gap** |
|---|---|---|---|---|
| 4 | 617.457 | 2,8302 | 3,9989 | 1,1687 |
| 6 | 812.973 | 2,9886 | 3,7763 | 0,7877 |
| 8 | 1.073.284 | 3,0770 | 3,7120 | 0,6350 |
| 11 | 1.427.186 | 3,1520 | **3,6712** | **0,5192** |

**Perfeitamente monotônico nas três colunas**, e cada uma conta um pedaço da mesma
história:

- a **validação** melhora sem parar — mais dados, melhor generalização;
- o **treino piora** sem parar — corpus mais diverso é mais difícil de decorar;
- o **gap** despenca de 1,17 para 0,52 — que é a definição de menos overfitting.

**1 e 2. Sim, acrescentar obras melhora, e não saturou** dentro do que foi testado. Indo de
617 mil para 1,43 milhão de tokens, a validação cai 0,33. Não há sinal de platô — o que
sugere que este modelo ainda está limitado por **dados**, não por capacidade.

### A resposta com 400 passos era o oposto

| Orçamento | Ranking (melhor → pior) |
|---|---|
| 400 passos | 6 < 8 < 4 < **11 (o pior)** |
| 3.000 passos | **11 (o melhor)** < 8 < 6 < 4 |

Com o orçamento curto, a conclusão era "melhora até 6 obras e depois piora". Com o
orçamento cheio, é "melhora sempre, monotonicamente". A inversão é **completa**.

E há uma lição a mais escondida aí. Eu tinha escrito uma hipótese específica para explicar
por que 11 obras seria pior: *Poesias Completas* é o único livro em **verso**, e misturar
poesia com prosa mudaria a distribuição. A hipótese era plausível, específica, e parecia
**confirmada** pelos dados de 400 passos.

No orçamento cheio, 11 obras é a **melhor** configuração. O efeito regularizador de mais
dados supera a diferença de gênero — e a minha explicação bonita estava explicando um
artefato de orçamento.

> **Por que os dois orçamentos discordam.** Com 400 passos, o modelo vê ~1,6 M de tokens:
> nem uma passada completa pelo corpus maior. Ninguém decora nada, e o único sinal que
> sobra é o de **distribuição** — daí a poesia atrapalhar. Com 3.000 passos ele vê ~12 M de
> tokens: 20 passadas pelo corpus pequeno contra 8,6 pelo grande. Aí o que domina é
> **overfitting**, e mais dados ganham com folga.
>
> Note que o confundimento aponta na direção **oposta** à do E4: lá, a configuração "maior"
> fazia *mais* épocas; aqui, faz *menos*. É o mesmo mecanismo produzindo conclusões
> invertidas — e é por isso que a pergunta "quantas épocas cada configuração faz?" deveria
> vir antes de qualquer interpretação.

**3. A lógica do Capítulo 3 vale, e agora com força.** Lá, aumentar os dados de 155 para 64
mil nomes levou o gap de 5,7 para ~0. Aqui, dobrar o corpus leva o gap de 1,17 para 0,52.
Direção idêntica, magnitude menor — e, ao contrário do que o experimento com frações do
corpus sugeria, **ainda longe de saturar**.
