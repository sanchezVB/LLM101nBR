# Gabarito — Capítulo 13

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> ```bash
> python solucoes/gabarito.py
> ```

---

## E1 — A aritmética

**1. Tensor em `[-3, 3]`, simétrica de 8 bits.**

```
escala = max|x| / 127 = 3/127 = 0,023622
q(1,5) = round(1,5 / 0,023622) = round(63,5) = 64      (ou 63, conforme o
                                                        desempate do round)
recuperado = 64 × 0,023622 = 1,51181
```

Erro de 0,0118 — cerca de **0,8%** do valor, e no máximo meia escala (0,0118) para
qualquer entrada. O erro máximo de quantização é sempre **metade do passo**.

**2. Por que `[-127, 127]` e não `[-128, 127]`.**

Para manter a **simetria**. Se usássemos −128, o intervalo representável seria assimétrico
e o mapeamento `q = x/escala` deixaria de ser simétrico em torno de zero: `+3` viraria 127
e `−3` viraria −128, valores com magnitudes diferentes.

Custa 1 nível em 256 — **0,4%** da resolução — e evita um caso especial em toda conta
subsequente. É uma troca boa, e é o que PyTorch faz por padrão.

**3. Tensor em `[10, 11]`.**

Com simétrica, `escala = 11/127 = 0,0866`. Mas o *intervalo útil* dos dados tem largura 1,
então cabem apenas `1/0,0866 ≈ 12 níveis` — de 256 disponíveis. Os outros 244 representam
valores entre −11 e +10, que nunca ocorrem.

Erro esperado: metade do passo sobre um valor típico de 10,5, ou seja
`0,0433/10,5 ≈ 0,4%`. Parece pouco, mas a *resolução dentro do intervalo real* é péssima.

A assimétrica usaria `escala = (11−10)/255 = 0,0039` — **22x mais fina** — e representaria
o mesmo intervalo com todos os 256 níveis.

> ⚠️ Com a ressalva do E2: como `[10, 11]` **não contém o zero**, é preciso estender o
> intervalo para `[0, 11]`, e aí a escala vira `11/255 = 0,0431`. Ainda **2x melhor** que a
> simétrica, mas não 22x. A extensão custa caro justamente nos casos mais deslocados.

---

## E2 — A armadilha do zero-point

Distribuição `randn(1000) + 5`, valores em `[1,24 ; 8,63]`:

| Variante | Erro |
|---|---|
| simétrica | 0,393% |
| assimétrica **sem** estender o zero | **1,007%** ← pior |
| assimétrica **com** estender o zero | **0,192%** |

**1. Com o bug, a simétrica ganha** — o oposto do que a teoria diz. Esse é justamente o
sintoma que denuncia o problema: quando um método que *deveria* ser melhor sai pior, há bug
antes de haver descoberta.

**2. O zero-point ideal é −170,7** e não cabe em `[-128, 127]`.

A razão é conceitual, não numérica: o zero-point é *o inteiro que representa o float 0,0*.
Se os dados vão de 1,24 a 8,63, o zero está **fora** do intervalo — e representá-lo exigiria
um inteiro que o tipo não tem.

**3. A cadeia causal completa**, e vale segui-la inteira:

```
dados em [1,24 ; 8,63], que não contêm o zero
  → zero-point ideal = qmin − xmin/escala  cai fora de [-128, 127]
  → o clamp o move para −128
  → o mapeamento inteiro se desloca em relação ao pretendido
  → q = x/escala + zero estoura o teto para os valores grandes
  → 12 de 1000 valores saturam em +127 e perdem a informação
  → o erro medido dispara
```

A correção — estender o intervalo para conter o zero — é o que PyTorch, TFLite e ONNX
fazem. Ela custa resolução e garante duas coisas: o zero-point cabe, e o float `0,0` tem um
inteiro exato (o que importa para *padding* e máscaras).

---

## E3 — A métrica que mente

Matriz 8×64 com uma linha 100x maior, quantização int8:

| Métrica | Per-tensor | Per-channel |
|---|---|---|
| erro relativo global (norma) | 1,48% | 0,52% |
| ... só nas 7 linhas normais | **56,15%** | 0,60% |
| **mediana do erro por elemento** | **100,00%** | 0,67% |

E o número que resolve a questão: **65,4%** dos pesos das linhas normais viraram
**zero exato**.

**1. Por que os dois primeiros números são tão diferentes.** O erro global é ~1,5%; o das
linhas normais é ~56%. A norma global é dominada pela linha grande, que quantiza bem — ela
tem valores grandes o bastante para usar toda a resolução disponível.

**2. A conta.** `‖w‖` soma **quadrados**. A linha 0 tem valores 100x maiores, logo contribui
**10.000x** mais que cada uma das outras para o denominador. O denominador de
`erro_relativo` é praticamente só ela; o numerador (o erro) está espalhado por toda a
matriz. Resultado: o erro das linhas pequenas acaba dividido por um número que não tem nada
a ver com elas.

**3. Uma métrica que denuncia sem desagregar: a mediana do erro relativo *por elemento*.**

```python
((rec - orig).abs() / orig.abs()).median()
```

Ela não tem denominador global, então nenhum elemento pode esconder os outros. Na tabela
acima ela marca **100%** — mais da metade dos pesos perdeu todo o seu valor.

Alternativa igualmente boa: **a fração de pesos que virou zero exato**. Ela transforma
"degradação" em "apagamento", que é o que de fato aconteceu.

> Isto vale muito além da quantização. Sempre que os dados forem heterogêneos em escala,
> uma média ponderada pela magnitude vai contar a história do subconjunto grande.

---

## E4 — Onde fica o precipício

| Bits | Erro no **peso** | Piora da **loss** | Perplexidade |
|---|---|---|---|
| `fp32` | — | — | 51,3 |
| 8 | 0,69% | **−0,0001** | 51,3 |
| 6 | 2,84% | +0,0021 | 51,4 |
| 4 | 12,55% | +0,0636 | 54,6 |
| 3 | 29,31% | +0,4657 | 81,7 |
| 2 | 80,84% | +3,5298 | **1.749,3** |

**1. Uma não prevê a outra**, e é o ponto do exercício.

O erro no peso cresce de forma **suave e regular** — dobra a cada bit. A piora da loss é
**desprezível até 4 bits e depois dispara**. São curvas de formatos completamente
diferentes.

A razão: a rede tem folga. Perturbações pequenas nos pesos são absorvidas — até deixarem de
ser pequenas.

> Corolário prático: **nunca escolha o número de bits olhando o erro de reconstrução.** Ele
> é fácil de medir e não responde à pergunta.

**2. O joelho está entre 4 e 3 bits.** A piora salta de +0,064 para +0,466 — **7x**. E de 3
para 2 o modelo é destruído: a perplexidade sai de 82 para **1.749**.

**3. Para servir este modelo eu escolheria int8**, e a justificativa precisa dos **dois**
eixos:

- **qualidade:** de graça (a loss não muda na quarta casa)
- **tamanho:** 4x menor, o que ataca o gargalo estabelecido no Capítulo 12

int4 também é defensável — custa 0,06 de loss e dá 8x. A escolha entre os dois depende de
quanta memória você precisa economizar. O que **não** é defensável é escolher por um eixo
só.

---

## E5 — Quantizando os embeddings também

| Configuração | Tamanho | Loss | Custo |
|---|---|---|---|
| int8, só `Linear` | 2,89 MB | 3,9371 | — |
| int8, **+ embeddings** | **2,20 MB** | 3,9371 | **+0,0000** |
| int4, só `Linear` | 1,90 MB | 4,0008 | — |
| int4, **+ embeddings** | **1,10 MB** | 4,0457 | +0,0449 |

**1.** Em **int8 é literalmente de graça**: 24% menor, loss idêntica até a quarta casa. Em
int4 o custo aparece — +0,045, o que quase dobra a penalidade total do int4 (0,064 → 0,109).

**2. Por que a resposta não é "no que tem a maior fração".**

Vale onde a fração é grande **o bastante para mover o total**. Num 7B, os embeddings são ~2%
dos parâmetros: quantizá-los muda o tamanho em 2%, o que não compensa nem o risco de
qualidade nem a complexidade. Aqui são 10%, e a economia de 24% é visível.

A pergunta é sempre *"quanto isso move o número que me importa"*, não *"qual é maior"*.

**3. A camada de saída merece cuidado extra**, por uma razão estrutural: ela produz os
**logits**, e a softmax é sensível a diferenças pequenas entre eles. Um erro que seria
inofensivo numa camada intermediária pode **reordenar** os tokens mais prováveis.

Muitos esquemas de produção deixam a camada de saída — e a de entrada — em precisão maior
justamente por isso.

---

## E6 — O que a quantização faz com a velocidade

| Configuração | Gerar 64 tokens |
|---|---|
| `float32` | 0,129 s |
| int8 | 0,184 s (**0,70x**) |

**1. O `float32` é mais rápido.** A quantização, desta forma, **custa** tempo.

**2. Onde o tempo vai.** No `forward` da `LinearQuantizada`:

```python
F.linear(x, desquantizar_simetrica(self.q, self.escala), self.bias)
```

A desquantização reconstrói a matriz inteira em `float32` — conversão de tipo mais
multiplicação pela escala, sobre **todos** os pesos, a **cada** chamada. Depois disso a
matmul é exatamente a mesma de antes. Trabalho extra, mesmo trabalho antigo.

**3. O que precisaria mudar.** Um kernel que multiplicasse `int8 × int8` acumulando em
`int32`, sem nunca materializar a matriz em float. Não se faz em PyTorch puro porque o
operador não existe — `F.linear` espera floats. É preciso descer para C++/CUDA, ou usar
llama.cpp, bitsandbytes, ONNX Runtime, que já fizeram isso.

> **"Menor" e "mais rápido" são independentes.** A primeira decorre da representação; a
> segunda exige que alguém escreva o kernel.

---

## E7 — Quantização por grupos

Matriz 256×256 com as 32 primeiras colunas 50x maiores, tudo em **int4**:

| Esquema | Erro global | **Erro nas colunas normais** | bits/peso reais |
|---|---|---|---|
| per-channel (1 escala/linha) | 11,23% | **100,00%** | 4,06 |
| grupos de 128 | 10,51% | 65,90% | 4,12 |
| grupos de 64 | 10,13% | 38,57% | 4,25 |
| grupos de 32 | 9,93% | **9,69%** | 4,50 |

### Eu caí na armadilha do meu próprio E3

Na primeira versão deste exercício eu medi **só o erro global** e conclui que os grupos
quase não valiam a pena: de 11,2% para 9,9%, contra 11% mais bits — praticamente empate.

Errado, e pelo motivo que o E3 acabou de ensinar. A norma global é dominada pelas 32 colunas
outlier, que quantizam bem em qualquer esquema.

Olhando a coluna que importa — as **colunas normais, que são 87,5% da matriz** — o quadro é
outro: o per-channel em int4 as **apaga por completo** (100% de erro), e grupos de 32
reduzem isso para 9,69%. **Dez vezes melhor, por 11% mais bits.**

**1 e 2.** Grupos menores reduzem o erro onde ele existe, porque cada escala cobre menos
variação. Com G=32 o bloco de outliers cai inteiro num grupo só e **para de contaminar os
demais** — que é exatamente o objetivo.

**3. A conta de bits por peso reais**, que costuma ser esquecida:

```
bits_reais = bits + (bits_da_escala / G)
```

Com int4 e G=64, guardando a escala em `fp16`: `4 + 16/64 = 4,25` bits por peso. Com G=32:
**4,5 bits**. Ou seja, "int4" com grupos pequenos não é 4 bits — é até **12% mais**.

É esse compromisso que os nomes dos formatos GGUF expõem: `Q4_0`, `Q4_K_M` e afins diferem
no tamanho do grupo e em quantos bits gastam com metadados. Escolher entre eles é escolher
um ponto nesta curva — e a escolha só faz sentido se você medir o erro **onde ele está**.
