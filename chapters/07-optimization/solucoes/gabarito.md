# Gabarito — Capítulo 07

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> **Orçamento:** 2.500 passos por configuração (a apostila usa 15.000). Onde essa redução
> muda a conclusão, está dito.

> **Aviso geral deste gabarito:** quatro das oito respostas contrariam a expectativa —
> inclusive a minha, quando escrevi os exercícios. Todas foram corrigidas pela medição.

---

## E1 — Leitura de código

**1. O que a correção de viés faz?**
`m` e `v` começam em **zero**. No primeiro passo, `m = (1−β₁)·g = 0,1·g` — dez vezes menor
que o gradiente real. As médias móveis **subestimam** no início. Dividir por `(1 − β₁ᵗ)`
compensa exatamente isso: no passo 1 o divisor é `0,1`, multiplicando `m` por 10.

Ela importa mais no início porque é lá que o viés é grande. Conforme `t` cresce, `β₁ᵗ → 0`,
o divisor tende a 1 e a correção **desaparece sozinha**.

**2. Weight decay somado ao gradiente vs aplicado ao parâmetro:**
No Adam original, o termo de decay entrava no gradiente e depois passava pela **divisão por
√v**. Isso distorce a regularização: parâmetros com gradiente grande recebiam *menos*
decay, sem nenhuma razão. O AdamW aplica o decay direto no parâmetro, fora do caminho
adaptativo — daí o "W" de *decoupled*.

**3. Por que `√fan_in` e não `fan_in`?**
Um neurônio soma `fan_in` produtos independentes. A **variância** de uma soma de `n` termos
independentes cresce com `n`, logo o **desvio padrão** cresce com `√n`. Como queremos
controlar o desvio padrão, dividimos por `√fan_in`.

---

## E2 — O warmup é necessário?

| Warmup | Validação | Norma média (100 primeiros passos) |
|--------|-----------|-----------------------------------|
| **0** | **1,9198** | 1,223 |
| 50 | 1,9211 | 1,351 |
| 200 | 1,9219 | 1,498 |
| 500 | 1,9226 | 1,645 |

**1. Neste modelo o warmup não ajuda** — e piora de forma monotônica. Gastar 500 dos 2.500
passos com learning rate reduzida é desperdiçar 20% do orçamento num modelo que não tem
problema de estabilidade.

**2. Cuidado com a métrica — e eu escolhi a errada.**

A norma dos gradientes nos primeiros passos **aumenta** com o warmup (1,223 → 1,645). Seria
tentador ler isso como "o warmup deixa o treino mais instável". Não é.

Com warmup a learning rate começa minúscula, então o modelo demora mais para sair da região
de inicialização — onde a loss é alta e os gradientes são **naturalmente maiores**. A norma
alta mede **lentidão**, não instabilidade. A métrica não testava o que eu queria testar.

**3. Então por que o warmup existe?** Porque nos primeiros passos `m` e `v` do Adam quase
não têm informação. Em modelos **grandes**, com batches grandes, um passo ruim no início
pode levar a loss a um regime do qual ela não se recupera.

> **O warmup é um seguro contra um risco que cresce com a escala.** Nesta escala, ele só
> custa. Isso não o torna errado — torna este modelo insuficiente para julgá-lo.

---

## E3 — Calibrando o gradient clipping

Norma do gradiente sem clipping: **média 1,324, máxima 2,427**.

| Clip | Validação | % de passos cortados |
|------|-----------|---------------------|
| 0,1 | 1,9206 | **100,0%** |
| 1,0 | 1,9205 | 99,6% |
| 3,0 | 1,9219 | **0,0%** |
| 100,0 | 1,9219 | 0,0% |

**1.** Com `clip = 3,0` (acima da norma típica de 1,32) **nada** é cortado — o comportamento
correto de uma rede de segurança que não é acionada.

**2. Com `clip = 0,1`, 100% dos passos são cortados**: isso deixa de ser clipping e vira
**normalização** do gradiente — o erro documentado na Seção 5 da apostila.

Mas seja honesto com o número: **aqui isso não prejudicou** (1,9206 contra 1,9219 sem
clipping — até marginalmente melhor). Normalizar todo gradiente é um algoritmo
**diferente**, não necessariamente pior. Ele só não é o que você pensa que está rodando.

> O problema de configurar errado nem sempre é perder desempenho. Às vezes é **não saber
> qual algoritmo você está usando** — e então não conseguir explicar o resultado.

**3.** Com `clip = 100` nada é cortado e a loss é idêntica à de não ter clipping.
Conclusão honesta: **neste modelo o clipping é dispensável**. Não há picos. Ele existe para
o caso patológico.

**4.** O clipping altera só o **tamanho** do gradiente, não a **direção**: todos os
componentes são multiplicados pelo mesmo fator. A informação de "para onde ir" é preservada.

---

## E4 — O ganho certo para a GELU

Desvio padrão das ativações na 8ª camada:

| Ganho | tanh | ReLU | GELU |
|-------|------|------|------|
| 1,000 | 0,2483 | 0,0493 | 0,0068 |
| 1,414 (√2) | 0,5515 | **0,7879** | 0,5393 |
| **1,500** | 0,5892 | 1,2635 | **1,0508** |
| 1,667 (5/3) | **0,6472** | 2,9400 | 2,7308 |
| 2,000 | 0,7253 | 12,6212 | 12,2773 |

**1. O ganho que mantém o desvio ~1 para GELU é ≈ 1,49.**

**2.** Fica **entre** o da ReLU (1,414) e o da tanh (1,667), mais perto da ReLU. Faz sentido:
a GELU é uma versão suave da ReLU e deixa passar um pouco menos de sinal, precisando de um
ganho ligeiramente maior.

**3. O PyTorch não tem entrada para GELU:**

```
torch.nn.init.calculate_gain('relu') = 1.4142
torch.nn.init.calculate_gain('gelu') -> ValueError
```

Na prática quase ninguém ajusta esse ganho para GELU — porque modelos modernos usam
**LayerNorm**, que re-normaliza a cada bloco e torna o valor exato bem menos crítico (a
apostila mede isso na Seção 2).

---

## E5 — A curva da learning rate

Solução em [`e5_curva_lr.py`](e5_curva_lr.py). Curva em U, com mínimo em **3e-3** naquele
orçamento — maior que o `1e-3` da apostila, porque **a melhor learning rate depende do
orçamento de passos**.

---

## E6 — AdamW vs SGD no modelo de verdade

| Otimizador | `lr` | Validação |
|------------|------|-----------|
| **AdamW** | 1e-3 | **1,9219** |
| SGD | 1e-3 | 2,6488 |
| SGD | 1e-2 | 2,2738 |
| SGD | 1e-1 | 2,0090 |
| SGD | 5e-1 | 1,9294 |
| SGD + momentum | 1e-2 | 2,0105 |
| **SGD + momentum** | 1e-1 | **1,9213** |

**1.** Com a **mesma** learning rate (1e-3), o SGD fica muito atrás: 2,65 contra 1,92. Um
passo bom para o AdamW é minúsculo para o SGD.

**2. Mas ajustando a learning rate, o SGD+momentum EMPATA com o AdamW** (1,9213 contra
1,9219). O SGD puro fica logo atrás (1,9294).

> Ou seja: neste modelo o AdamW **não é magicamente superior** — ele é superior **com a
> learning rate padrão**. A grande vantagem prática do Adam não é chegar mais longe, é
> chegar lá **sem que você precise caçar a learning rate certa**. Isso vale muito quando
> cada tentativa custa horas de treino.

**3.** O modelo tem parâmetros de escalas muito diferentes (embeddings, pesos de atenção,
ganhos de LayerNorm, vieses). Uma única learning rate não serve para todos — é o experimento
do `optimizers.py`. O Adam normaliza o passo por parâmetro e escapa desse aperto.

---

## E7 — Por que o weight decay atrapalhou

| Weight decay | Treino | Validação | Gap |
|--------------|--------|-----------|-----|
| 0,00 | 1,9243 | **1,9207** | −0,0036 |
| 0,01 | 1,9256 | 1,9219 | −0,0037 |
| 0,10 | 1,9382 | 1,9334 | −0,0048 |
| 0,50 | 2,0015 | 1,9938 | −0,0077 |

**1 e 2. Olhe as três colunas juntas.** Aumentar o weight decay piora **treino e validação
juntos**, e o **gap entre eles quase não muda**. Esse padrão é a assinatura de que a
regularização **não está resolvendo overfitting** — ela está apenas removendo capacidade.

Se houvesse overfitting, o esperado seria: treino piora, validação **melhora**, gap
diminui. Não é o que acontece.

**3.** Com gap praticamente nulo no baseline (−0,004), **não há overfitting para combater**.
Weight decay é remédio para uma doença que este modelo não tem.

**4.** Ele ajudaria com **poucos dados** — releia o E5 do Capítulo 3: com 155 nomes, treino
0,80 contra validação 6,51. Lá o gap é enorme, e a regularização teria o que fazer.

---

## E8 — Outros agendamentos

| Agendamento | Validação |
|-------------|-----------|
| constante | 1,9449 |
| cosseno | 1,9219 |
| **linear** | **1,9206** |
| degraus | 1,9404 |

Diferença entre o melhor e o pior: **0,0243**.

**1 e 2. Decair ganha de não decair — mas os agendamentos não são equivalentes entre si.**
O linear e o cosseno ficam praticamente empatados e claramente à frente; os **degraus**
ficam quase tão ruins quanto o constante.

Faz sentido: com apenas 2.500 passos, o primeiro degrau (em 50%) demora demais a chegar.
Agendamentos por degrau foram feitos para treinos longos, em que cada patamar tem tempo de
render.

E o cosseno ser o padrão da área é mais **convenção** do que superioridade medida — pelo
menos nesta escala, o linear empata ou ganha.

**3. O problema de depender de `max_steps`:** o cosseno precisa saber o total de passos
**de antemão**. Se você quiser continuar um treino já terminado, a curva já chegou ao fim e
não há como esticá-la sem estragar o formato — a learning rate ficaria presa no mínimo.

É um incômodo real, e existem alternativas: o **WSD** (*warmup-stable-decay*) mantém a
learning rate constante por tempo indeterminado e só decai no fim, permitindo decidir
depois quando parar.
