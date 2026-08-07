# Gabarito — Capítulo 15

> Respostas discursivas. Os **números** vêm de [`experimento.py`](../experimento.py) e de
> [`e4_dial_do_kl.py`](e4_dial_do_kl.py).
>
> ```bash
> python experimento.py                        # ~2h30 (3 configurações)
> python solucoes/e4_dial_do_kl.py 0.1 0.5     # ~50 min por beta
> ```

---

## E1 — SFT e RL, lado a lado

**1. De onde vem o sinal.**

| Método | Fonte do sinal |
|---|---|
| **SFT** | um **alvo**: existe um texto de referência, e a loss mede a distância até ele |
| **RL** | um **juiz**: não há texto de referência, e a loss mede se o que saiu pontuou bem |

A diferença é de *fonte*, não de algoritmo. Os dois fazem descida de gradiente sobre os
mesmos pesos.

**2. Por que o RL precisa gerar.**

Porque a recompensa é uma função da **saída**, e a saída não existe antes de o modelo
produzi-la. No SFT o alvo já está no disco; aqui é preciso rodar o modelo até o fim para
ter o que avaliar.

O custo é grande e é o motivo de este capítulo ser o mais lento do curso: cada passo de
treino contém **48 passos de geração** — e com dois modelos, política e referência. Um
passo de SFT é um forward e um backward; um passo de RL é 96 forwards mais um backward.

> É por isso que RLHF é caro de um jeito que finetuning não é, e por que técnicas como o
> DPO — que eliminam a geração — ganharam tração tão rápido.

**3. Por que o baseline não muda o gradiente esperado.**

Porque `E[R − média(R)] = 0`. Subtrair uma constante que não depende da ação não altera a
esperança do gradiente:

```
E[(R − b) · ∇log P]  =  E[R · ∇log P] − b · E[∇log P]
                                          \_____________/
                                                = 0
```

O último termo é zero porque `E[∇log P] = ∇E[1] = 0` — a soma das probabilidades é sempre
1, então o gradiente dela é nulo.

O que muda é a **variância**. Sem baseline, uma recompensa de 0,9 e outra de 0,8 são as
duas "boas" e as duas são reforçadas com força parecida. Com baseline, a de 0,9 é reforçada
e a de 0,8 é **penalizada** — o sinal passa a ser sobre qual é melhor, não sobre qual é
positiva.

---

## E2 — Escreva uma recompensa ruim

A resposta é sua, mas há um padrão que quase sempre aparece e vale nomear.

Toda recompensa da forma "meça **X** como proxy de **Y**" tem uma região onde X e Y andam
juntos — normalmente a região onde estão os dados naturais — e uma região onde eles se
descolam. O RL vai direto para a segunda, porque é lá que X é maior.

| Recompensa | X (medido) | Y (pretendido) | Onde descola |
|---|---|---|---|
| `pontos` | fração de tokens de pontuação | frases bem pontuadas | `'. '` |
| comprimento mínimo | número de tokens | resposta completa | repetição, enrolação |
| "contém a palavra da pergunta" | sobreposição léxica | responde ao que foi perguntado | eco do enunciado |
| "poucas repetições" | tokens distintos | fluência | palavras raras aleatórias |

**3.** O exercício pede para identificar o X e o Y da sua própria recompensa. Se você não
conseguir separá-los, provavelmente escreveu Y como se fosse mensurável — e é justamente
esse o momento em que uma recompensa ruim nasce parecendo razoável.

> Note que **nenhuma** das quatro acima está "com bug". Todas medem exatamente o que
> dizem. O erro está em confundir a medida com a intenção.

---

## E3 — O baseline importa mesmo?

**1 e 2.** Sem baseline, o aprendizado continua acontecendo — só que mais devagar e com a
curva de recompensa muito mais ruidosa.

O motivo é o do E1: o gradiente esperado é o mesmo, então a direção média está certa. O que
piora é a **variância** de cada estimativa individual. Com recompensas todas positivas,
cada passo reforça tudo que foi amostrado, e o sinal útil — "isto foi melhor que aquilo" —
fica submerso.

**3. Recompensa sempre negativa é o caso que expõe o problema.**

Se `R ∈ [−1, 0]`, então sem baseline **toda** ação é penalizada, sempre. O modelo empurra
para baixo a probabilidade de tudo que gera, inclusive do que gerou a recompensa menos
ruim. Ele aprende — porque as coisas piores são penalizadas mais — mas por um caminho
absurdo: reduzindo a probabilidade de tudo, e diferenciando só pelas taxas.

Isso é o argumento para **normalizar recompensas** (subtrair a média, dividir pelo desvio).
A escala e o deslocamento da sua recompensa não deveriam mudar o algoritmo, e sem
normalização eles mudam.

---

## E4 — Onde fica a faixa útil do β

Recompensa `pontos` (mal especificada), 400 passos, partindo de loss 4,0784:

| β | Recompensa | KL | Loss real | **Custo** | Regime |
|---|---|---|---|---|---|
| **0,00** | 0,058 → **0,996** | 14,87 | 4,8211 | **+0,7427** | hackeia por completo — `'. '` |
| 0,02 | 0,058 → 0,958 | 11,63 | 4,3963 | +0,3179 | hackeia quase igual — `'.'` |
| **0,10** | 0,055 → **0,218** | 1,41 | 4,0832 | **+0,0048** | **aprende sem custo** |
| 0,50 | 0,053 → 0,064 | 0,31 | 4,0810 | +0,0026 | mal se move |
| 2,00 | 0,053 → 0,055 | 0,23 | 4,0800 | +0,0016 | congelado |

**1. A faixa útil existe, é β ≈ 0,1 — e é um ponto só.**

Em β = 0,10 a recompensa quadruplica (0,055 → 0,218), a KL cai de 14,87 para 1,41, e o
custo em português é **+0,005**, indistinguível de zero. A resposta volta a ser texto.

De um lado dele, em β = 0,02, o modelo hackeia quase completo. Do outro, em β = 0,5, ele
paralisa: a recompensa sobe 0,011 em 400 passos. **Um fator de cinco para cada lado**, e a
janela fecha.

Isso é desconfortável, e é o resultado. O β não é um parâmetro que você ajusta "mais ou
menos" — é uma escolha entre três regimes qualitativamente diferentes, e só um deles serve.

**2. A transição entre 0,02 e 0,10 é abrupta, e o motivo é estrutural.**

A recompensa efetiva é `R − β·KL`. O hacking total rende `R ≈ 1` e custa `KL ≈ 15`. Então:

| β | Ganho do hack | Custo do hack | Compensa? |
|---|---|---|---|
| 0,02 | +1,0 | −0,30 | **sim, muito** |
| 0,10 | +1,0 | −1,50 | **não** |

O β não modula suavemente *quanto* o modelo hackeia — ele decide se hackear **vale a
pena**. É uma mudança de sinal numa desigualdade, e por isso a transição parece um degrau
em vez de uma rampa.

> É a mesma forma do precipício entre 4 e 3 bits no [Capítulo 13](../../13-quantization/solucoes/gabarito.md):
> um parâmetro contínuo produzindo um comportamento com limiar.

**3. Custo zero não é sucesso — e as duas últimas linhas provam.**

Em β = 0,5 o custo é **+0,0026**, quase metade do custo em β = 0,10. Pela coluna do custo,
seria a melhor configuração da tabela.

Olhe a recompensa: **0,053 → 0,064**. Em 400 passos, o modelo aprendeu 0,011. Ele não foi
protegido; foi **impedido de aprender**. Em β = 2,0 isso fica explícito — KL de 0,23, e a
política praticamente não se moveu do ponto de partida.

> É o mesmo formato de erro que o [Capítulo 13](../../13-quantization/solucoes/gabarito.md)
> encontrou na quantização: uma métrica que parece boa porque o sistema **não está fazendo
> nada**. Lá era o erro global escondendo linhas zeradas; aqui é o custo baixo escondendo
> um modelo que não treinou.

Por isso a tabela precisa das **duas** colunas. Otimizar só o custo leva ao β infinito, que
é exatamente o mesmo que não treinar — e tem custo zero.

---

## E5 — A recompensa não é o objetivo

**1. As ordens são opostas.**

| Por recompensa (melhor → pior) | Por loss em Machado (melhor → pior) |
|---|---|
| `pontos` sem freio (0,996) | *(partida, sem RL)* — 4,0784 |
| `pontos` com freio (0,958) | `comprimento` com freio — 4,1701 |
| `comprimento` com freio (0,743) | `pontos` com freio — 4,3963 |
| *(partida)* — sem recompensa | `pontos` sem freio — **4,8211** |

O **campeão** de recompensa é o **pior** modelo. Não é coincidência nem azar: ele é o pior
justamente **porque** foi o melhor em otimizar a coisa errada.

**2. O que um relatório só com a recompensa diria.**

Diria: "o RL funcionou — a recompensa subiu 17 vezes". Cada palavra verdadeira, e a
conclusão que o leitor tira é falsa.

O engano é específico: a recompensa **sempre sobe**. Ela mede se o otimizador está
funcionando, não se o modelo melhorou. Reportá-la sozinha é reportar que o gradiente
descendo — informação sobre o algoritmo, disfarçada de informação sobre o resultado.

**3. Duas métricas que um projeto real precisa acompanhar:**

| Métrica | O que detecta que a recompensa não detecta |
|---|---|
| **KL da referência** | que a política está se afastando — o mecanismo, antes do dano |
| **Desempenho numa tarefa que o RL não vê** (aqui, loss em Machado) | que a capacidade original está sendo consumida |

Uma terceira, quando há gente disponível: **avaliação humana em amostra**, justamente
porque ela não pode ser otimizada diretamente pelo modelo. É por isso que laboratórios
mantêm avaliações humanas caras mesmo tendo métricas automáticas baratas — a métrica barata
é o que o modelo está otimizando, então ela é a que menos serve para julgá-lo.

---

## E6 — Conserte a recompensa

**1 e 2.** Há consertos que ajudam: penalizar pontuação repetida, exigir um mínimo de
tokens não-pontuação entre os pontos, medir a fração de *frases bem formadas* em vez de
*caracteres de pontuação*.

Cada um fecha o buraco que você viu. E o modelo procura o próximo.

**3. Não existe recompensa que resista a qualquer otimização, e isso não é pessimismo — é
a estrutura do problema.**

Se a recompensa é uma função computável da saída, e você otimiza contra ela com força
suficiente, o modelo encontra o máximo dessa função. A pergunta nunca é "esta recompensa
tem buracos?" — tem — e sim **"o máximo dela é aceitável?"**.

Isso reorganiza o que fazer:

| Estratégia | Por que funciona (ou não) |
|---|---|
| escrever recompensa melhor | **corrida sem fim** — fecha um buraco, abre outro |
| limitar a otimização (KL, poucos passos) | **compra tempo**, não resolve — é o E4 |
| **recompensa verificável** | muda o jogo: se o máximo é "a resposta está certa", hackear = acertar |

A terceira é a razão de a área ter migrado para **matemática, código e tarefas com
verificador**. Não é que esses domínios sejam mais importantes — é que neles a recompensa
não é um proxy. Quando o teste passa, o teste passou.

> E é a mesma conclusão do [E6 do Capítulo 11](../../11-datasets/solucoes/gabarito.md),
> chegando por outro caminho: dados gerados pelo próprio modelo ajudam **quando existe um
> verificador externo**. Lá era sobre dados, aqui é sobre recompensa, e o critério é o
> mesmo.

---

## E7 — DPO (desafio)

**1 e 2.** A implementação é direta e vale pelo contraste: o DPO não gera durante o treino
— ele consome pares prontos, com uma loss que se parece com uma cross-entropy. Um passo de
DPO custa o mesmo que um passo de SFT, contra os 96 forwards de um passo de REINFORCE.

**3. O DPO não evita reward hacking, e a razão está no enunciado do exercício.**

Você rotulou os pares com a **mesma recompensa ruim**. O DPO vai otimizar a preferência que
recebeu, com a mesma obediência literal — e a preferência recebida diz que respostas com
mais pontuação são melhores.

Trocar o algoritmo não conserta a especificação. O DPO elimina o modelo de recompensa
explícito e o custo da geração; **não** elimina a distância entre o que você mediu e o que
você queria.

> Se a sua preferência vier de humanos, o problema muda de lugar mas não desaparece:
> humanos preferem respostas longas, confiantes e bem formatadas — e um modelo otimizado
> contra essa preferência aprende a ser longo, confiante e bem formatado. Inclusive quando
> está errado.
