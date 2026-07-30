# Gabarito — Capítulo 03

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> **Sobre o orçamento:** o gabarito treina **4.000 passos** (a apostila usa 20.000) para
> rodar em poucos minutos. Os valores absolutos ficam piores que os da apostila, mas as
> **comparações entre configurações** — que é o que os exercícios pedem — continuam
> válidas. Onde essa diferença muda a conclusão, está dito explicitamente (veja o E4).

---

## E1 — Leitura de código

**1. Por que `emb = C[X]` tem formato `(N, 3, 10)`?**
`X` tem formato `(N, 3)`: N exemplos, 3 índices de caractere cada. `C[X]` substitui cada
índice pela linha correspondente de `C`, que tem 10 números. Então:
`N` exemplos × `3` posições de contexto × `10` dimensões de embedding.

**2. Por que o `.view(N, -1)` antes da camada linear?**
A camada linear espera **um vetor** por exemplo, não uma matriz `3×10`. O `view`
reorganiza os mesmos números em `(N, 30)`, colando os três embeddings num vetor só. O `-1`
diz ao PyTorch "calcule esta dimensão". Note que `view` **não copia** dados — apenas
reinterpreta o mesmo bloco de memória.

**3. O que `F.cross_entropy(logits, Y)` faz?**
Exatamente o que fizemos à mão no Capítulo 1: **softmax** seguido de **negative
log-likelihood**. A diferença é que a versão do PyTorch é numericamente estável (evita
`exp()` de números grandes, que estouraria) e mais rápida.

---

## E2 — O efeito do contexto

| `block_size` | Parâmetros | Treino | Validação |
|--------------|-----------|--------|-----------|
| 1 | 7.897 | 2,3498 | 2,3401 |
| 3 | 11.897 | 2,0508 | 2,0452 |
| 5 | 15.897 | 2,0165 | 2,0114 |

**1.** Mais contexto **ajuda**: a validação cai de 2,34 para 2,01.

**2.** Com `block_size = 1` o MLP vira um **bigrama neural** — ele só vê o caractere
anterior, como no Capítulo 1. E de fato a loss (2,34) cai na faixa do bigrama daquele
capítulo (~2,4). É uma boa confirmação de que os dois modelos são a mesma coisa quando o
contexto é o mesmo.

**3.** Sim, os parâmetros crescem: a primeira camada tem `(n_embd × block_size) × n_hidden`
pesos. Cada caractere a mais de contexto acrescenta `n_embd × n_hidden = 2.000` pesos.

> É exatamente esse custo linear que motiva a **atenção** do Capítulo 4: lá, aumentar o
> contexto **não** aumenta o número de parâmetros.

---

## E3 — Tamanho da camada oculta

| `n_hidden` | Parâmetros | Treino | Validação |
|------------|-----------|--------|-----------|
| 50 | 3.197 | 2,1128 | 2,1023 |
| 200 | 11.897 | 2,0508 | 2,0452 |
| 500 | 29.297 | 2,0287 | 2,0271 |

**1.** Mais neurônios, mais parâmetros, loss menor.

**2.** **Retorno decrescente, e claro:** de 50 para 200 (parâmetros ×3,7) a validação
melhora 0,057. De 200 para 500 (parâmetros ×2,5) melhora só 0,018 — três vezes menos ganho.
Cada parâmetro novo contribui menos que o anterior.

---

## E4 — Learning rate

| `lr` | Validação | Situação |
|------|-----------|----------|
| 0,001 | 2,4547 | lenta demais |
| 0,01 | 2,2162 | |
| 0,1 | 2,0452 | ← o valor da apostila |
| **1,0** | **2,0019** | **melhor aqui** |
| 10,0 | — | **divergiu** |
| 50,0 | — | **divergiu** |

**1. Esta resposta provavelmente contraria o que você esperava.** A curva é em U, mas o
mínimo **não está no 0,1 que a apostila usa** — neste orçamento de 4.000 passos, `lr = 1,0`
vai melhor. A divergência só aparece em `10,0`.

Isso **não** significa que a apostila esteja errada. Ela treina 20.000 passos, e com
orçamento maior uma learning rate menor tem tempo de refinar e chega mais longe. A
conclusão correta:

> **A melhor learning rate depende do orçamento de passos.** Ela não é uma propriedade do
> modelo. O mesmo fenômeno reaparece no Capítulo 7 (exercício E5), onde `3e-3` ganha de
> `1e-3` justamente porque o experimento treina menos passos que a apostila.

**2.** Removendo o decaimento com a melhor `lr`: 2,0915 sem contra **2,0019** com — o
decaimento vale **0,09** de loss. Passos menores no fim permitem "assentar" no mínimo em
vez de ficar pulando em torno dele.

---

## E5 — Overfitting com dataset pequeno

Solução em [`e5_overfitting.py`](e5_overfitting.py):

```
=== DATASET PEQUENO (155 nomes) ===
loss treino    = 0.8001
loss validacao = 6.5086
```

**1.** Um abismo: treino 0,80 contra validação 6,51.

**2.** O mesmo modelo generaliza com 64 mil nomes e decora com 155 porque o número de
**parâmetros** (11.897) é maior que o número de **exemplos de treino** (880). Com essa
proporção, memorizar é mais fácil que aprender o padrão — e o modelo faz o que é mais
fácil. (É a mesma conta que reaparece no Capítulo 11: 0,28 token por parâmetro.)

**3.** Os nomes gerados se parecerem demais com nomes reais do dataset é **mau** sinal
porque significa que o modelo está **reproduzindo**, não **generalizando**. Um bom modelo
gera nomes que *poderiam* estar na lista mas não estão. Qualidade aparente na saída pode
esconder memorização — por isso se mede na validação, não no olho.

---

## E6 — Visualizando os embeddings

Com `n_embd = 2` (treino 2,2954 / validação 2,2873 — pior que com 10, como esperado):

| Distância média | Valor |
|-----------------|-------|
| vogal ↔ vogal | 0,732 |
| vogal ↔ consoante | 1,173 |
| **razão** | **1,60x** |

**1. As vogais ficam agrupadas.** Elas estão 1,6x mais próximas entre si do que das
consoantes. A rede descobriu sozinha que vogais têm papel parecido — **só por prever o
próximo caractere**. Ninguém ensinou fonética a ela.

Este é um dos resultados mais bonitos do curso: estrutura linguística emergindo de pura
estatística de sequência.

**2.** Reduzir para `n_embd = 2` piora a loss (menos capacidade por caractere), mas é o
único jeito de desenhar os embeddings num plano. É um trade-off comum em interpretabilidade:
sacrifica-se desempenho para poder **ver**. (Em modelos de verdade usa-se PCA ou t-SNE para
projetar embeddings de centenas de dimensões em 2, sem retreinar.)

---

## E7 — Contando os parâmetros

| Componente | Cálculo | Total |
|------------|---------|-------|
| `C` (embeddings) | 27 × 10 | 270 |
| `W1` | 30 × 200 | 6.000 |
| `b1` | 200 | 200 |
| `W2` | 200 × 27 | 5.400 |
| `b2` | 27 | 27 |
| **Total** | | **11.897** ✓ |

**1. Fórmula geral:**

```
vocab×n_embd + (n_embd×block_size)×n_hidden + n_hidden + n_hidden×vocab + vocab
```

**2.** Confere exatamente com o `11897` que o código informa.

**3. Quem domina:** `W1` (6.000) e `W2` (5.400) somam **96%** do total. As duas matrizes
grandes são o modelo; embeddings e vieses são detalhes.

Consequências:
- Dobrar `n_embd` quase dobra o total (ele aparece em `C` **e** em `W1`).
- Dobrar `n_hidden` também (aparece em `W1` **e** em `W2`).
- Dobrar `block_size` só afeta `W1` — cresce menos que os outros dois.
