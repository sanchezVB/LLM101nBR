# Gabarito — Capítulo 08

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> Para exercitar a GPU, rode com o Python que tem o `torch-directml`:
> `C:\dml312\Scripts\python.exe solucoes/gabarito.py`
>
> ⚠️ **Seus números não vão bater com os daqui, e está tudo bem.** Veja a nota sobre
> variabilidade no E2.

---

## E1 — Latência vs vazão

**1. Por que a GPU perde numa matmul de 128×128?**
Porque há um **custo fixo** por operação — lançar o kernel, coordenar o dispositivo,
sincronizar — que não depende do tamanho do trabalho. Numa matmul minúscula, esse custo
fixo é maior que o cálculo em si. A CPU faz a conta e devolve; a GPU passa a maior parte
do tempo se organizando.

**2. Por que a CPU fica em ~340 GFLOP/s e a GPU vai de 167 a 5.500?**
A CPU **satura cedo**: ela tem poucos núcleos e já os usa quase todos mesmo em matrizes
pequenas. A GPU tem **milhares** de núcleos, e só os ocupa quando há trabalho para todos.
Em matrizes pequenas, a maioria fica ociosa.

**3. Uma tarefa com muitos `if` e desvios roda melhor em quê?**
Em **CPU**. Núcleos de GPU executam em grupos que compartilham a mesma instrução; quando
o código diverge (uns entram no `if`, outros não), o hardware precisa executar os dois
caminhos em sequência e descartar o que não vale — perdendo boa parte do paralelismo. A
CPU tem previsão de desvio e execução fora de ordem justamente para esse tipo de código.

---

## E2 — Encontre o seu ponto de virada

Uma execução (AMD Radeon RX 7600 via DirectML):

| Tamanho | CPU (ms) | GPU (ms) | Speedup | GFLOP/s GPU |
|---------|----------|----------|---------|-------------|
| 128 | 0,05 | 0,03 | 1,52x | 136 |
| 256 | 0,18 | 0,04 | 4,51x | 857 |
| 512 | 1,36 | 0,10 | 13,89x | 2.745 |
| 1024 | 7,47 | 0,40 | **18,55x** | 5.332 |
| 2048 | 62,94 | 3,72 | 16,92x | 4.619 |

**1 e 2.** A GPU ganha a partir de algumas centenas, e o speedup cresce até estabilizar
em torno de 15–18x.

### ⚠️ Sobre a variabilidade — e um erro que eu cometi

**Estes números não batem com os da apostila** (que mediu 0,75x em 128 e 15,26x em 1024). E
não porque um dos dois esteja errado.

Ao preparar este gabarito eu rodei a mesma medição três vezes e obtive **1,08x, 1,52x e
2,09x** para 128×128. A variação vinha de **outros processos disputando a CPU** — o gabarito
de outro capítulo estava treinando ao fundo enquanto eu media.

Medição de tempo é sensível a:
- carga da máquina (o meu erro)
- estado térmico e frequência do processador
- até se você multiplica `a @ b` ou `a @ a` — a segunda tem melhor localidade de cache e
  deixa a CPU mais rápida

**O que é estável, e o que você deve verificar:**
- a **forma** da curva: speedup baixo nos tamanhos pequenos, crescendo até estabilizar
- a **ordem de grandeza** do ganho máximo (uma a duas dezenas de vezes)
- o fato de que a **CPU satura** e a **GPU escala**

Se quiser números comparáveis, feche tudo e rode com a máquina ociosa. É irônico e
apropriado que o capítulo sobre medir corretamente tenha me pegado justamente aqui.

**3.** O pico de GFLOP/s medido fica bem abaixo da especificação da placa. Isso é normal: a
especificação supõe uso perfeito das unidades, precisão reduzida e nenhum gargalo de
memória. Os capítulos 9 e 10 atacam exatamente essa diferença.

---

## E3 — A armadilha do `.item()`

| Medição | ms/passo |
|---|---|
| sem ler a loss | 1,18 |
| lendo `.item()` a cada passo | 1,57 (**+33%**) |

**1 e 2.** Um terço a mais de tempo, e a causa são **duas coisas** que o `.item()` faz:

- **transfere** um valor da GPU para a CPU
- para poder fazê-lo, **espera** a GPU terminar tudo o que estava na fila

A segunda é a cara. Ela destrói o paralelismo entre CPU e GPU: normalmente a CPU vai
enfileirando o próximo passo enquanto a GPU trabalha no atual; com o `.item()`, ela para e
espera.

**3. Como registrar sem pagar o preço:**
- registre a cada N passos (100, por exemplo) — você precisa da **curva**, não de todos os
  pontos
- ou acumule os valores num tensor **na GPU** e leia só no fim

---

## E4 — Tamanho do batch

| Batch | ms/passo | ms por exemplo |
|-------|----------|----------------|
| 32 | 0,149 | 0,00466 |
| 128 | 0,174 | 0,00136 |
| 512 | 0,183 | 0,00036 |
| 2048 | 0,477 | **0,00023** |

**1 e 2.** O tempo por passo cresce **muito menos** que proporcionalmente: batch 64x maior
custa só 3x mais tempo. Por isso o tempo **por exemplo** cai 20x. É a mesma lição da
Seção 4 da apostila.

**3.** O limite é a memória da placa — você verá um erro de alocação.

**4. Cuidado com a conclusão, e este é o ponto do exercício.** Batch maior processa mais
exemplos por segundo, mas **cada passo continua sendo um passo de gradiente**. Se você
dobrar o batch sem ajustar a learning rate, o treino faz metade do progresso por exemplo
visto. Releia o E5 do Capítulo 7 e a Seção 6 do Capítulo 10.

---

## E5 — Meça errado de propósito

| Medição | Tempo | "Speedup" |
|---------|-------|-----------|
| GPU **sem** drenar | 0,054 ms | **230,6x** |
| GPU **com** drenar | 0,460 ms | 26,9x |

**A medição errada exagera em ~9x** (e numa das execuções, em 18x).

**1.** Sem drenar, a GPU parece absurdamente rápida — porque estamos medindo o tempo de
**enfileirar** a operação, não o de executá-la. O Python devolve o controle imediatamente.

**2. Por que a CPU não é afetada?** Porque nela a execução é **síncrona**: quando a linha
termina, a conta terminou. Não existe fila.

**3. Como detectar que a medição está contaminada:**
- speedups que **não crescem de forma monotônica** com o tamanho
- valores fisicamente implausíveis (centenas de vezes)
- tempos que não mudam quando o trabalho aumenta

---

## E6 — Operações que caem de volta na CPU

Solução em [`e6_fallback.py`](e6_fallback.py):

| Otimizador | GPU (ms/passo) | Speedup | Fallback |
|------------|----------------|---------|----------|
| AdamW | 75,2 | 2,65x | `aten::lerp.Scalar_out` |
| SGD + momentum | 32,7 | **5,94x** | nenhum |
| SGD puro | 30,9 | **6,23x** | nenhum |

Trocar o otimizador **mais que dobra** o ganho. Aquela única operação custava ~42 ms por
passo — mais da metade do tempo total. E não aparece como erro: só como lentidão.

---

## E7 — Quando vale a pena mudar para GPU?

Juntando os números da Seção 7 da apostila:

| Modelo | Parâmetros | Speedup |
|--------|-----------|---------|
| pequeno (o do curso) | 153 mil | **0,30x** — a GPU perde |
| médio | 3,2 M | 2,65x |
| grande | 18,9 M | 6,82x |

**1.** O ponto de equilíbrio fica entre 153 mil e 3,2 milhões de parâmetros — mais perto do
limite inferior, já que 3,2 M já dá 2,65x.

**2. O custo não é só de execução.** Instalar um ambiente separado, descobrir operações não
suportadas (o `lerp` do E6) e depurar travamentos consomem **horas**. Para um treino de 6
minutos, não compensa; para um de 6 horas, compensa muito. A conta é de tempo total, não de
tempo de GPU.

**3. Para o modelo pequeno deste curso, a recomendação é CPU** — e essa é a resposta
contra-intuitiva do capítulo. "Usar GPU" não é automaticamente certo: é uma decisão que
depende da escala e que **se mede**.
