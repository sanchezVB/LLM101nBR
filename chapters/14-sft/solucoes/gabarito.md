# Gabarito — Capítulo 14

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py) e de
> [`e2_quando_a_mascara_importa.py`](e2_quando_a_mascara_importa.py).
>
> ```bash
> python solucoes/gabarito.py
> ```

---

## E1 — Tokens especiais e vocabulário

**1. Por que um id novo, e não a string `###FIM###`.**

Porque o texto do usuário pode conter a string. Se alguém escrever `###FIM###` no meio de
uma pergunta, o seu parser corta ali — e o modelo, treinado a tratar aquilo como fronteira,
obedece.

Um token especial é uma garantia **estrutural**: não existe sequência de caracteres que o
tokenizador transforme no id 1026. A fronteira não pode ser falsificada pela entrada.

> É a base da defesa contra *prompt injection* por delimitador. Quando você lê que um
> modelo "confundiu a instrução do sistema com o texto do usuário", quase sempre há uma
> fronteira que era texto quando deveria ser estrutura.

**2. Quantos parâmetros novos, com `n_embd = 192`.**

| Matriz | Conta | Parâmetros |
|---|---|---|
| tabela de embeddings | 3 × 192 | 576 |
| camada de saída (pesos) | 3 × 192 | 576 |
| camada de saída (bias) | 3 | 3 |
| **total** | | **1.155** |

Confirmado pelo código: 2.196.352 → 2.197.507. Um aumento de **0,05%**.

**3. Zeros e valores aleatórios grandes.**

**Zeros na camada de saída não são neutros.** Um logit zero é um valor específico dentro de
uma distribuição já treinada, e para um modelo convergido a maioria dos logits reais é
*negativa* — então o token novo nasceria **mais provável** que muitos tokens legítimos. O
modelo passaria os primeiros passos consertando isso.

**Valores aleatórios grandes** são piores: o token novo domina a softmax, a loss explode nos
primeiros passos, e o gradiente que corrige isso perturba o resto da camada.

A **média** das linhas existentes evita os dois: o token novo nasce estatisticamente
indistinguível dos outros e se diferencia conforme aprende.

---

## E2 — Quando a máscara importa

### A medição

| Pedido/resposta | % de pedido | Com máscara | Sem máscara | Penalidade |
|---|---|---|---|---|
| 24 / ~34 | 42% | 3,9964 | 3,9898 | **−0,0066** |
| 64 / ~20 | 72% | 3,9450 | 3,9586 | +0,0136 |
| 104 / ~12 | 87% | 3,9588 | 3,9752 | **+0,0165** |

**1.** A explicação padrão é que treinar no pedido "gasta capacidade". O que ela **não**
diz, e deveria, é que a loss sem máscara é uma **média sobre as posições** — então o
tamanho relativo do pedido decide quanto ele dilui o sinal da resposta. Toda a dependência
está aí.

**2. A penalidade aparece a partir de ~70% de pedido**, e o sinal **inverte**: com pedido
curto, mascarar é marginalmente pior.

Repetindo a configuração de pedido longo com **seis sementes**:

| Semente | Com máscara | Sem máscara | Diferença |
|---|---|---|---|
| 1337 | 3,9588 | 3,9752 | +0,0165 |
| 42 | 3,9737 | 3,9796 | +0,0058 |
| 2024 | 3,9389 | 3,9690 | +0,0301 |
| 7 | 3,9537 | 3,9860 | +0,0322 |
| 99 | 3,9607 | 3,9867 | +0,0261 |
| 555 | 3,9536 | 3,9624 | +0,0088 |

**6 de 6 na mesma direção**, média **+0,0199 ± 0,0112**.

**3. Nulo e negativo pedem reações diferentes.**

No caso de pedido curto o resultado foi **negativo**, não nulo: mascarar saiu *pior*
(−0,0066). Um resultado nulo autorizaria "não detectei diferença"; um negativo consistente
autorizaria "a recomendação está invertida aqui". Este é pequeno demais para a segunda
afirmação — está dentro do que a semente move.

### O erro de análise que quase enterrou o resultado

Com as três primeiras sementes eu comparei a diferença entre variantes (0,0175) com a
variação da **mesma** variante entre sementes (0,0348) e concluí: *"o ruído supera o efeito"*.

**O critério estava errado para o desenho.** As duas variantes rodam com a **mesma
semente** — é um experimento **pareado**, e nele a variação que a semente causa afeta os
dois lados e **se cancela na diferença**.

| Critério | Números | Veredito |
|---|---|---|
| não pareado | ruído 0,0348 > efeito 0,0199 | "não estabelecido" |
| **pareado** | 6/6 na mesma direção, média > desvio | **estabelecido** |

Mesmos dados, conclusões opostas, decididas só pela estatística escolhida.

> É o segundo caso do curso em que a **medição estava certa e a leitura não**. O outro é o
> [Capítulo 13](../../13-quantization/solucoes/gabarito.md), onde uma métrica global marcava
> 1,3% de erro enquanto 48% das linhas estavam destruídas.

### E a magnitude

**+0,0199 sobre uma loss de 3,95 é 0,5%.** O efeito é real e pequeno. "Estatisticamente
detectável" e "importante na prática" não são a mesma coisa, e este número é o primeiro sem
ser o segundo — *nesta escala*. Em SFT real, com prompt de sistema longo e histórico de
conversa, a proporção é muito mais desfavorável que 87%.

---

## E3 — Dados de tamanho fixo

Com `TAM_RESPOSTA = 40` fixo, todos os 8.000 exemplos ficam com exatamente 66 tokens:

| Modelo | Taxa de parada | Comprimento mediano |
|---|---|---|
| SFT (com máscara) | 100% | **40** |
| SFT (sem máscara) | 100% | **40** |

**1 e 2. A taxa de parada é 100% e não significa nada.** A métrica que separa as duas
versões é o **comprimento mediano**: cravado em 40, sem variância.

O modelo aprendeu a **contar até 40**, não a concluir. Era a única regra consistente nos
dados — e ele a aprendeu perfeitamente.

Com a correção (resposta até o fim de frase), os comprimentos passam a variar de 39 a 82
tokens, 44 valores distintos, e a mediana medida vira 34 — próxima mas não igual à do
treino, que é o que se espera de um modelo que está lendo o texto.

**3. Regularidades acidentais num conjunto real:**

| Regularidade | O que o modelo aprende |
|---|---|
| toda resposta começa com "Claro!" | a dizer "Claro!" |
| respostas de um anotador são mais longas | o estilo daquele anotador |
| exemplos difíceis foram descartados na curadoria | a nunca dizer "não sei" |
| perguntas de código sempre vêm com bloco markdown | a formatar mesmo quando não cabe |

> O modelo aprende a regularidade que os seus dados **de fato contêm**, não a que você
> pretendia ensinar. Antes de treinar, procure o que é acidentalmente constante.

---

## E4 — A learning rate do finetuning

400 passos, 8.000 exemplos:

| `lr` | Loss na resposta | Taxa de parada | **Loss no texto puro** |
|---|---|---|---|
| `3e-5` | 4,2448 | **0%** | 3,9517 |
| **`3e-4`** | **3,9749** | 97% | 4,0372 |
| `1e-3` | 4,0181 | 100% | 4,1464 |
| *(modelo-base)* | — | 0% | **3,9104** |

**1. Não são as três.** Com `3e-5` a taxa de parada é **0%** — passos pequenos demais não
chegam a instalar um comportamento que não existia; só ajustam o que já estava lá.

E a melhor loss está no **meio**, não num extremo. Learning rate boa não é a maior nem a
menor.

**2. Olhe a última coluna.** Ela mede a loss no texto corrido do Capítulo 11, sem formato de
instrução — a capacidade original do modelo. O modelo-base marca 3,9104, e a coluna se
afasta **monotonicamente** conforme a learning rate cresce: 3,9517 → 4,0372 → 4,1464.

O modelo está trocando a competência antiga pela nova, e o câmbio piora com o tamanho do
passo.

**3. *Catastrophic forgetting*.** O finetuning vê um conjunto pequeno e homogêneo, e o
gradiente empurra **todos** os pesos na direção que serve àquele conjunto. Nada no objetivo
preserva o que foi aprendido antes.

A única proteção embutida é dar passos pequenos e poucos. É por isso que a learning rate de
finetuning é tipicamente 3 a 10 vezes menor que a do pré-treino — não é cautela, é o
mecanismo.

> Note que os dois eixos apontam para lados opostos, e o `3e-4` do capítulo cai onde eles se
> equilibram: melhor loss na tarefa nova, com metade do custo em esquecimento que o `1e-3`
> cobra.

---

## E5 — Quantos exemplos bastam?

| Exemplos | Loss na resposta | Taxa de parada |
|---|---|---|
| 500 | 4,8662 | **100%** |
| 2.000 | 4,1922 | **100%** |
| 8.000 | **3,9749** | 97% |

**1. As duas métricas se separam, e essa é a resposta.**

A **taxa de parada satura em 500** — já é 100%. O formato é barato de instalar: três tokens
especiais e uma convenção de onde cada um entra.

A **loss não satura**: 4,8662 → 4,1922 → 3,9749, quase **0,9** de melhora.

São coisas diferentes: uma mede se o modelo respeita o **formato**, a outra se ele escreve
uma resposta **boa**. Só a primeira é barata.

**2.** Por isso a comparação com o Capítulo 11 precisa ser dividida. Para o **formato**, a
curva satura cedo — ao contrário de lá. Para a **qualidade** da resposta, a curva se parece
com a de lá, porque a tarefa volta a ser modelar texto.

**3.** Resultados como o do **LIMA** ("mil exemplos bem escolhidos bastam") falam do
primeiro eixo. Se o que você quer é instalar **comportamento** — formato, tom, recusa, uso
de ferramenta — mil exemplos variados bastam mesmo, e a medição acima mostra por quê: é
pouca informação a transmitir.

O que mil exemplos **não** compram é capacidade de gerar conteúdo melhor. Essa vem do
pré-treino, e nenhum SFT a substitui.

---

## E6 — O *alignment tax*

| Modelo | Loss no texto corrido do Cap. 11 |
|---|---|
| modelo-base | 3,9104 |
| depois do SFT (`3e-4`) | 4,0372 (**+0,1268**) |

**1 e 2. O SFT piorou** a capacidade de modelar português comum em **+0,127** de loss. Não
é catastrófico, e não é zero.

**3.** É o ***alignment tax***: o custo, na competência original, de instalar um
comportamento novo. Ele **não é inevitável**, e há três formas conhecidas de reduzi-lo:

| Técnica | Como ataca |
|---|---|
| **misturar** dados de pré-treino no conjunto de SFT | o gradiente continua vendo texto comum |
| **LoRA** / congelar camadas | limita quanto os pesos originais podem se mover |
| lr menor, menos passos | o que o E4 mede |

Todas atacam a mesma causa: **nada no objetivo do SFT pede que o modelo lembre do que
sabia**.

---

## E7 — Conversa com vários turnos

**1. A máscara passa a ter várias faixas** — uma por resposta, em vez de uma só:

```
<|pedido|> A <|resposta|> B <|fim|> <|pedido|> C <|resposta|> D <|fim|>
\__ máscara __/\__ treina __/       \__ máscara __/\__ treina __/
```

Todos os turnos do usuário ficam mascarados, inclusive os do meio.

**2. O `<|fim|>` tem de vir no fim de CADA resposta**, e dá para decidir isso sem treinar.

Se ele aparecesse só no fim da conversa, a geração não teria onde parar ao terminar um turno
— o modelo continuaria e **inventaria a fala do usuário**. É exatamente o comportamento do
modelo-base que o capítulo inteiro combateu, só que agora dentro do formato.

**3.** É por isso que modelos reais usam um token de fim **por turno** (`<|im_end|>` e
equivalentes), e não um marcador único de conversa.

> E note que isso torna a conta do E2 **mais** relevante, não menos: numa conversa longa o
> histórico mascarado cresce e as respostas não. A fração de posições que seriam
> desperdiçadas sem máscara aumenta a cada turno — o regime desfavorável do E2 é o regime
> normal de um chat.
