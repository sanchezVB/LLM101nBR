# Gabarito — Capítulo 10

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py), medidos numa
> máquina de 12 núcleos.
>
> ```
> python solucoes/gabarito.py
> ```
>
> O E3 roda em **subprocesso com timeout**, porque o objetivo literal daquele exercício é
> fazer o programa travar — e um gabarito que trava para sempre não serve para ninguém.

---

## E1 — Entendendo o coletivo

**1. Quatro processos, o rank `i` tem o valor `i`.**

| Operação | Resultado em **cada** rank |
|---|---|
| `all_reduce` com `SUM` | `0+1+2+3 = 6` |
| `all_reduce` com `MAX` | `3` |

O ponto do "all": **todos** terminam com o mesmo valor. Não é o rank 0 que fica com a
soma — é todo mundo.

**2. As quatro operações.**

| Operação | Quem fica com o resultado |
|---|---|
| `reduce` | **um** rank (o destino) |
| `all_reduce` | **todos** |
| `gather` | **um** rank recebe a lista completa |
| `all_gather` | **todos** recebem a lista completa |

`reduce`/`all_reduce` **combinam** os valores (soma, máximo). `gather`/`all_gather`
**empilham** sem combinar: cada rank fica com os N tensores separados.

**3. Por que o `broadcast` não serve para sincronizar gradientes.**

Porque o `broadcast` copia o valor de **um** rank para os demais — e nenhum rank tem o
valor certo para transmitir. Cada um calculou o gradiente do **seu** pedaço do batch; o
gradiente correto é a **média dos quatro**, que não existe em lugar nenhum antes de
alguém somá-los.

Usar `broadcast` do rank 0 equivaleria a jogar fora os dados dos outros três processos —
todo o trabalho deles seria descartado. É exatamente o que a soma do `all_reduce` evita.

---

## E2 — Escalando o número de processos

Tempo de um `all_reduce`, melhor de 3 rodadas, **duas passadas independentes** (12 núcleos):

| Processos | 10 mil | 1 milhão | 10 milhões |
|---|---|---|---|
| 2 | 0,54 / 0,56 ms | 21,0 / 15,3 ms | 264,9 / 280,2 ms |
| 4 | 2,35 / 1,83 ms | **9,8 / 8,7 ms** | **93,6 / 105,8 ms** |
| 8 | 5,81 / 5,70 ms | 13,3 / 12,7 ms | 174,2 / 177,7 ms |

Reprodutibilidade entre passadas: **1,02x a 1,38x**. O efeito discutido abaixo é de
**2,7x** — bem acima do ruído.

**1. Não há uma resposta só — ela depende do tamanho do tensor, e é essa a descoberta.**

- **10 mil elementos:** o tempo **cresce ~10x** de 2 para 8 processos.
- **10 milhões:** o tempo **cai quase 3x** de 2 para 4 processos.

Sim, *cai*. Com mais processos, o all-reduce do tensor grande fica **mais rápido**.

> Eu tinha escrito que o tempo sempre cresce com N, apoiado na propriedade do anel (cada
> processo troca ~2× o tamanho do tensor, independentemente de N). Essa propriedade é
> verdadeira — e não é a que domina.

A explicação é a mesma dicotomia do [Capítulo 8](../../08-device/solucoes/gabarito.md),
**latência vs vazão**:

| Tamanho do tensor | O que domina | Efeito de mais processos |
|---|---|---|
| Tensor pequeno | **latência** dos N−1 saltos do anel | mais saltos → **pior** |
| Tensor grande | **vazão**: transferir e somar 40 MB | trabalho dividido → **melhor** |

No caso grande, o anel divide o trabalho em pedaços de `tamanho/N`. Com N=2 cada processo
lida com 20 MB por etapa; com N=4, com 10 MB. Pedaços menores cabem melhor na cache, e a
**soma** — que é trabalho real, não só transferência — se espalha por mais núcleos.

**2. No esquema ingênuo** (todos → rank 0 → todos), o rank 0 receberia `(N−1) × tamanho`:
o tempo cresceria **linearmente** com N, e ele viraria um gargalo cada vez pior. E o ganho
observado em N=4 seria **impossível** — o rank 0 é um só, e não paraleliza.

**3. Onde os processos passam a competir por CPU.** Nesta máquina de 12 núcleos, N=8 perde
para N=4 nas colunas grandes: o tempo piora por **contenção**, não por comunicação. Numa
máquina só, "mais processos" tem um teto físico, e ele fica perto do número de núcleos.

> ⚠️ **Cuidado ao generalizar.** Tudo isto foi medido em *uma* máquina, com gloo em CPU.
> Num cluster de verdade, com GPUs e rede dedicada, o equilíbrio entre latência e vazão é
> outro — mas a **pergunta a fazer** é a mesma.

---

## E3 — Provoque um deadlock

Este exercício me pegou. Vale ler a previsão antes do resultado.

### A previsão (errada)

A explicação clássica do deadlock em `send`/`recv` é o **buffer de socket**: o sistema
operacional aceita mensagens pequenas e devolve o controle imediatamente, sem esperar
ninguém receber. Sob essa teoria, tensores pequenos passariam por acidente e só os
grandes travariam — o que faria deste um bug clássico de "passa no teste, quebra em
produção".

Eu escrevi essa resposta antes de medir. O próprio enunciado original do exercício dava
essa dica.

### A medição

| Tensor | Alternância par/ímpar | Resultado |
|---|---|---|
| 1 elemento | sim | terminou |
| 1 elemento | **não** | **TRAVOU** |
| 100 elementos | sim | terminou |
| 100 elementos | **não** | **TRAVOU** |
| 5 milhões | sim | terminou |
| 5 milhões | **não** | **TRAVOU** |

**Trava sempre.** Até com **um único elemento**. O `dist.send` do gloo só retorna quando o
`recv` correspondente foi postado do outro lado — não existe tamanho pequeno o bastante
para escapar.

A explicação do buffer de socket vale para **sockets crus e para MPI**. Não vale para o
gloo, que fez outra escolha de projeto.

### O que aprender com isso

**Sobre o gloo: a escolha dele é boa.** Um deadlock que acontece *sempre* é muito melhor
que um que só aparece em produção com tensores grandes. O gloo trocou "às vezes funciona"
por "nunca funciona", e **falhar de forma determinística é um recurso, não um defeito** —
o bug aparece no primeiro teste que você rodar, na sua máquina, com dados de brinquedo.

**Sobre você (e sobre mim).** Eu tinha um modelo mental correto e o apliquei a uma
implementação que não o segue. O modelo estava certo *em geral* e errado *naquele caso* —
e nenhuma quantidade de raciocínio teria revelado isso. Só medir revela.

É o mesmo padrão do [E2 do Capítulo 4](../../04-attention/solucoes/gabarito.md) e do
[E3 do Capítulo 5](../../05-transformer/solucoes/gabarito.md).

**E note o modo de falha:** não há exceção, não há mensagem. O programa simplesmente
para — exatamente como aconteceu no desenvolvimento deste capítulo, com o hostname que
resolvia para um IP público (Seção 8 da apostila).

---

## E4 — A loss que engana

Solução em [`e4_loss_correta.py`](e4_loss_correta.py).

**1. Por que os ranks imprimem losses diferentes se os pesos são idênticos?**
Porque cada rank calcula a loss no **seu** micro-batch, e os micro-batches são diferentes.
Os *pesos* são idênticos (o DDP garante isso sincronizando os gradientes); os *dados* não.

**2.** Com `all_reduce` da loss dividido por `world_size`, os valores passam a bater entre
os ranks — porque agora todos estão reportando a mesma quantidade: a loss média sobre o
batch global.

**3. O que aconteceria com a curva se você registrasse só o rank 0.**
Ela ficaria **mais ruidosa** do que o treino realmente é. A loss do rank 0 é uma amostra
de batch 64; a loss global é uma amostra de batch 256. Quatro vezes mais dados, metade do
ruído.

O perigo prático: você olha a curva serrilhada, conclui que o treino está instável, e sai
mexendo na learning rate para consertar um problema que **existe só na medição**.

---

## E5 — Batch efetivo e learning rate

300 passos, batch **local** fixo em 64:

| Processos | Batch efetivo | lr | Loss global |
|---|---|---|---|
| 1 | 64 | `1e-3` | 2,0095 |
| 2 | 128 | `1e-3` | 1,9417 |
| 4 | 256 | `1e-3` | 1,8494 |
| 2 | 128 | `1,4e-3` (×√2) | **1,9155** |
| 4 | 256 | `2,0e-3` (×√4) | **1,7846** |

**1.** Com o batch local fixo, mais processos = batch efetivo maior. Como o número de
**passos** é o mesmo, o modelo vê mais dados — mas dá o mesmo número de **atualizações**.
A loss melhora, porque cada gradiente é menos ruidoso.

**2. Sim, ajustar a lr melhora ainda mais.** Escalar por `√world_size` recuperou parte da
diferença nos dois casos (1,9417 → 1,9155 e 1,8494 → 1,7846). Gradiente menos ruidoso
tolera — e pede — passo maior.

**3. Por que "mais GPUs" pode deixar o treino mais lento em número de passos.**
Porque você processa mais exemplos por passo, mas continua andando com o **mesmo tamanho
de passo**. Mais dados por atualização, mesmo número de atualizações: é preciso mais
passos para chegar ao mesmo lugar. O ganho está em tempo de parede por passo, não em
progresso por passo — e só se materializa se você ajustar a learning rate junto.

É a mesma armadilha do [E4 do Capítulo 8](../../08-device/solucoes/gabarito.md) (batch
maior é mais eficiente por exemplo, mas não é grátis do ponto de vista da otimização).

---

## E6 — Meça o ZeRO

Estado do otimizador, por processo:

| Otimizador | Processos | Normal | ZeRO-1 | Redução |
|---|---|---|---|---|
| AdamW | 2 | 8,54 MB | 4,33 MB | 2,0x |
| AdamW | 4 | 8,54 MB | 2,23 MB | 3,8x |
| **SGD sem momentum** | 2 | **0,00 MB** | **0,00 MB** | — |
| **SGD sem momentum** | 4 | **0,00 MB** | **0,00 MB** | — |

**1. Sim, o estado por processo cai para ~1/N** — como a apostila mede com 4 processos.

**2. E aqui está o ponto do exercício: com SGD sem momentum, o ZeRO-1 não economiza
nada.** Zero contra zero.

O motivo é direto: o SGD puro **não guarda estado nenhum** — ele usa o gradiente e
descarta. Se o otimizador não guarda nada, não há o que fatiar. O ZeRO-1 economiza
exatamente aquilo que o otimizador guarda.

(Os estágios 2 e 3, que fatiam **gradientes** e **pesos**, continuariam ajudando — eles
não dependem do otimizador.)

**3. Com bf16 nos pesos** (Capítulo 9), a conta muda de forma interessante. Pesos e
gradientes passam de 4 para 2 bytes por parâmetro, mas o estado do AdamW costuma ficar em
fp32 por precisão. Com 8 processos:

| Estratégia | Pesos | Gradientes | Estado | Total/GPU |
|---|---|---|---|---|
| DDP | 2 GB | 2 GB | 8 GB | 12 GB |
| ZeRO-1 | 2 GB | 2 GB | 1 GB | **5 GB** |

O peso **relativo** do estado do otimizador **aumenta** quando os pesos encolhem — ou
seja, precisão reduzida torna o ZeRO ainda mais valioso. As duas técnicas se reforçam.

---

## E7 — Quando distribuir *não* compensa

Juntando os números medidos:

- tempo de cálculo de um passo (modelo pequeno, CPU, Capítulo 8): **~12,8 ms**
- all-reduce de ~150 mil parâmetros (0,6 MB): a coluna "10 mil" do E2, escalada — fica na
  casa de **alguns milissegundos**

**1 e 2.** Para o modelo pequeno deste curso, comunicação e cálculo ficam na **mesma ordem
de grandeza**. Distribuir em 4 processos dividiria o cálculo por 4 mas acrescentaria
comunicação a cada passo — ganho líquido pequeno ou negativo.

**3. A regra geral:**

> Distribuir compensa quando
> **tempo de cálculo por passo ≫ tempo de comunicação por passo**

O cálculo cresce com o **tamanho do modelo** e com o **batch**. A comunicação cresce só
com o tamanho do **modelo** — o all-reduce troca gradientes, e há um gradiente por
parâmetro, não por exemplo.

Logo: **aumentar o batch melhora a razão de graça.** É por isso que treino distribuído de
verdade usa batches enormes — não é só por velocidade, é para que a comunicação valha a
pena.

E é a mesma conclusão contra-intuitiva do [E7 do Capítulo 8](../../08-device/solucoes/gabarito.md):
para o modelo deste curso, a resposta certa é **não distribuir**.
