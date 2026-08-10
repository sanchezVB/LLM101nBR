# Gabarito — Capítulo 17

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py), que roda com
> **600 passos** por variante (a apostila usa 1.500) — são vários treinos.
>
> ```bash
> python solucoes/gabarito.py
> ```

---

## E1 — Por que quantizar

**1. Compressão.**

| Medida | Pixels | Tokens | Razão |
|---|---|---|---|
| posições | 784 | 49 | **16x** |
| bits | 784 × 8 = 6.272 | 49 × 7 = 343 | **18x** |

Em posições e em bits a compressão é parecida — o que se ganha em quantidade de símbolos,
gasta-se um pouco em cada símbolo ser mais "caro" (7 bits contra 8). O ganho real é o
primeiro: **o que limita um Transformer é o comprimento da sequência**, não a largura do
vocabulário. Reduzir 784 para 49 é a diferença entre caber e não caber.

**2. Por que não usar os vetores contínuos direto.**

Porque a **saída** do modelo é uma distribuição sobre um conjunto finito. Um Transformer
autorregressivo termina numa softmax: ele escolhe entre K opções, amostra uma, e realimenta.

Com vetores contínuos não há o que amostrar — você teria de prever um vetor de 32 números
reais, e a loss viraria um erro quadrático em vez de uma cross-entropy. Isso é possível
(existem modelos assim) e muda o objetivo de "modelar uma distribuição" para "prever a
média", que produz saídas borradas.

> A quantização não é só compressão: é o que permite **amostrar**.

**3. A analogia com o BPE, e onde ela quebra.**

Onde funciona: ambos aprendem um alfabeto de pedaços recorrentes a partir dos dados, ambos
reduzem o comprimento da sequência, e ambos são específicos do domínio em que foram
treinados (o [E3 do Capítulo 11](../../11-datasets/solucoes/gabarito.md) mediu isso para
texto).

Onde quebra: o BPE é **exato** — decodificar desfaz o encode perfeitamente. O VQ-VAE
**perde informação**: a reconstrução é aproximada, e o erro de 0,0039 é justamente essa
perda. Um é compressão sem perdas de símbolos discretos; o outro é compressão com perdas
de um sinal contínuo.

---

## E2 — O straight-through estimator

| Variante | Erro de reconstrução | Códigos usados |
|---|---|---|
| **com** straight-through | **0,0069** | **67** |
| **sem** straight-through | 0,0756 | **1** |

**1. Sem o straight-through, tudo colapsa.** O erro fica 11x pior — e, mais revelador,
**um único código** é usado para o dataset inteiro.

O decoder ainda aprende: ele recebe gradiente normalmente. O que não aprende é o
**encoder** — o `argmin` corta a cadeia, nada diz a ele que vetores produzir, e ele fica
essencialmente na inicialização. Com o encoder parado, todos os vetores caem perto do mesmo
lugar, e a quantização vira uma constante.

**2. O que a linha faz.**

```python
z_q = z + (z_q - z).detach()
```

| Fase | Valor |
|---|---|
| **forward** | `z + z_q − z` = **`z_q`** — o vetor quantizado, exato |
| **backward** | `d/dz [z + constante]` = **1** — passa direto para o encoder |

O `.detach()` congela o *nó no grafo*, não o número. É contabilidade de derivadas, não
aritmética: o valor calculado é idêntico, e só o caminho do gradiente muda.

**3. Por que uma aproximação funciona.**

O encoder recebe o gradiente do decoder como se a quantização não existisse. Isso é falso —
mas é **quase** verdade, porque `z` e `z_q` são próximos por construção.

E são próximos justamente porque o *commitment loss* os mantém assim. As duas peças se
sustentam: a aproximação é boa exatamente na medida em que a outra perda faz o seu trabalho.

---

## E3 — As duas perdas

| `BETA_COMP` | Erro | Códigos usados |
|---|---|---|
| 0,00 | 0,0755 | **1** |
| 0,25 *(o do artigo)* | 0,0069 | 67 |
| **2,00** | **0,0046** | **105** |

**1. Com `BETA_COMP = 0` o resultado é catastrófico** — e é o mesmo colapso do E2, por
outra causa. Sem nada que o prenda, o encoder foge para uma região que o codebook não
alcança, e todo vetor cai no mesmo código.

**2. E aqui eu estava errado.**

Eu tinha escrito que `BETA_COMP` alto prenderia o encoder e pioraria a representação.
Medido, **2,0 é o melhor dos três nos dois eixos**: erro 0,0046 contra 0,0069, e 105
códigos usados contra 67.

O valor 0,25 é o do artigo original do VQ-VAE, e eu o tratei como se fosse um ótimo
demonstrado. Nesta configuração — MNIST, codebook de 128, 600 passos — não é. Um
*commitment* mais forte mantém o encoder perto do codebook, o que faz mais códigos serem
alcançáveis, o que melhora tudo.

> A lição não é "use 2,0". É que **um hiperparâmetro herdado de um artigo foi ajustado para
> outra configuração** — outro dataset, outro tamanho, outro orçamento. Vale medir na sua.
>
> É o mesmo padrão do [Capítulo 11](../../11-datasets/solucoes/gabarito.md), onde a melhor
> learning rate mudou conforme o orçamento de treino.

**3. O papel de cada perda:**

| Perda | O que ela move |
|---|---|
| `perda_codebook` | o **codebook**, em direção ao encoder |
| `perda_commit` | o **encoder**, em direção ao codebook |

São os dois lados do mesmo encontro, e cada uma move um lado.

Removendo a `perda_codebook`, os vetores do codebook só se moveriam pelo gradiente que
chega via decoder — muito mais fraco. Na prática ele ficaria quase congelado na
inicialização, e o encoder teria de se contorcer para caber num alfabeto aleatório.

---

## E4 — Códigos mortos

| Codebook | Usados | Fração | Erro |
|---|---|---|---|
| 32 | 32 | **100%** | 0,0070 |
| 64 | 55 | 86% | 0,0065 |
| 128 | 67 | 52% | 0,0069 |
| 256 | 57 | **22%** | 0,0066 |

**1. A fração usada cai, e o erro quase não melhora.**

De 32 para 256 códigos — oito vezes mais capacidade nominal — o erro vai de 0,0070 para
0,0066. Praticamente nada. **Um codebook maior não vira mais capacidade; vira mais códigos
mortos.**

O mecanismo é de realimentação: para ser escolhido, um código precisa estar perto de algum
vetor do encoder; para chegar perto, precisa ser escolhido e receber gradiente. Quem nasce
longe nunca entra no jogo.

> É o mesmo formato do colapso do E2 e do E3, e vale notar que os três aparecem por
> caminhos diferentes: sem gradiente no encoder, sem commitment, ou por inicialização
> azarada. O VQ-VAE tem várias maneiras de degenerar para "poucos códigos fazem tudo".

**2 e 3. A correção, e o que ela realmente melhora.**

Reinicializar códigos mortos sobre vetores do batch quase sempre **aumenta muito a fração
usada** e **melhora pouco o erro de reconstrução**.

Isso não a torna inútil — torna-a mal avaliada pela estatística de uso. O ganho real
aparece depois: um codebook bem aproveitado dá ao Transformer um vocabulário mais
informativo, e é lá que se deve medir.

> Cuidado com a métrica que melhora sem que o sistema tenha melhorado no que importa. É o
> mesmo tipo de armadilha do [Capítulo 15](../../15-rl/solucoes/gabarito.md), onde β = 0,5
> tinha o menor custo em português — porque o modelo não estava aprendendo nada.

---

## E5 — O mesmo Transformer

**1. Zero linhas do modelo mudaram.**

```python
from modelo import GPT
```

É a mesma classe, byte por byte, que escreveu prosa nos capítulos 11 a 15. O que muda é a
config:

| Config | `vocab_size` | `block_size` | `n_embd` |
|---|---|---|---|
| Capítulo 11 (texto) | 1.024 | 128 | 192 |
| Capítulo 17 (imagem) | 128 | 49 | 128 |

**2. A comparação de perplexidade não é justa**, e entender por que vale mais que o número.

| Modelo | Perplexidade | Vocabulário | Normalizada |
|---|---|---|---|
| Capítulo 11 | 51,7 | 1.024 | 5,0% |
| Capítulo 17 | 3,9 | 128 | 3,0% |

Perplexidade é "entre quantas opções o modelo está efetivamente escolhendo". Comparar 3,9
com 51,7 ignora que os vocabulários diferem 8x.

E mesmo normalizada a comparação é fraca, porque as **tarefas** são diferentes: prever o
próximo pedaço de um dígito manuscrito é bem mais restrito que prever a próxima palavra de
Machado. Perplexidades só se comparam dentro da mesma tarefa e do mesmo tokenizador.

**3. A ordem de varredura é arbitrária para imagem, e cria um problema real.**

O modelo gera da esquerda para a direita, de cima para baixo — porque é assim que se
achatou o mapa 7×7. Mas imagem não tem ordem natural, e a consequência é concreta: **o
modelo desenha a parte de cima sem saber o que virá embaixo, e não pode voltar atrás.** Um
traço mal começado no topo condena o dígito inteiro.

Texto tem ordem natural (o tempo); imagem não tem. Alternativas:

| Abordagem | Ideia |
|---|---|
| ordem aleatória | o modelo recebe a posição junto com o token e pode gerar em qualquer ordem |
| múltiplas escalas | gera uma versão grosseira primeiro, refina depois |
| **difusão** | abandona a autorregressão: refina a imagem **inteira** de uma vez, várias vezes |

A terceira é a que dominou geração de imagem, e a razão é exatamente esta: ela não precisa
escolher uma ordem.

---

## E6 — Condicionar pelo dígito

O caminho é o mesmo do [Capítulo 14](../../14-sft/solucoes/gabarito.md): reserve ids novos
acima do vocabulário existente (aqui, 128 a 137 para os dez dígitos), expanda a tabela de
embeddings e a camada de saída, e ponha o rótulo como primeiro token da sequência.

**3. O que mudaria para condicionar por uma legenda de verdade.**

Estruturalmente, **quase nada** — e é esse o ponto. Em vez de um token de condição, você
põe a legenda tokenizada pelo BPE do Capítulo 6:

```
<|texto|> um dígito sete inclinado <|imagem|> t₁ t₂ ... t₄₉
```

O modelo aprende a prever os tokens de imagem condicionados aos de texto, com a mesma loss
de sempre.

O que muda de verdade é o **dado**: você precisa de pares imagem-legenda, e é por isso que
este capítulo usa o rótulo do MNIST em vez de legendas. Dez rótulos existem; um corpus
pareado útil começa em dezenas de gigabytes.

> A arquitetura de um "texto → imagem" não é o obstáculo. O obstáculo são os dados
> pareados — a mesma conclusão que o Capítulo 11 tirou sobre texto, em escala maior.

---

## E7 — Um modelo, duas modalidades (desafio)

**1 e 2.** Juntar os vocabulários funciona mecanicamente: ids 0–1023 para texto, 1024–1151
para imagem, e um GPT treinado em sequências dos dois tipos. Espere alguma interferência —
a capacidade é dividida entre duas distribuições sem relação.

**3. A pergunta que importa: o modelo aprende alguma relação entre as modalidades? Não.**

E a razão está no objetivo de treino. O modelo é recompensado por prever o próximo token
**dentro de cada sequência**. Se nenhuma sequência contém texto *e* imagem, nada no
gradiente jamais liga um token de texto a um token de imagem — as duas metades do
vocabulário nunca coocorrem.

O resultado é **um modelo com duas personalidades**, não um modelo multimodal. Ele modela
duas distribuições que por acaso compartilham pesos.

Para que aprendesse, seria preciso o que este capítulo não tem: **sequências que misturem
as duas modalidades**. É por isso que dados pareados são o recurso caro em multimodalidade
— e por que "juntar os tokenizadores" resolve a parte fácil do problema.

> Vale terminar o curso com isto: a arquitetura quase nunca é o obstáculo. Do Capítulo 1 ao
> 17, o que mudou a qualidade foi sempre **dados, escala e medição honesta** — nunca uma
> ideia arquitetural esperta.
