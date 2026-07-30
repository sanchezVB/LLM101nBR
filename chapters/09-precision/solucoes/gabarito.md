# Gabarito — Capítulo 09

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py), que roda na
> CPU em segundos. O **E6** depende de GPU — a resposta está abaixo, com os números que
> a apostila mediu.

---

## E1 — Contas com bits

**1. Qual formato representa `100000`? E `1e-9`?**

- `100000` → só o **bf16**. O fp16 estoura em 65.504.
- `1e-9` → só o **bf16**. O fp16 zera abaixo de ~6e-8 (contando com subnormais).

Os dois casos têm a mesma causa: os 8 bits de expoente do bf16 contra os 5 do fp16.

**2. Quantos valores distintos entre 1 e 2?**

Com `m` bits de mantissa, exatamente **2^m** valores. Então:
- bf16 (7 bits) → **128** valores entre 1 e 2
- fp16 (10 bits) → **1.024** valores
- fp32 (23 bits) → ~8,4 milhões

**3. Por que dobrar os bits de mantissa não dobra a precisão?**
Porque a relação é **exponencial**, não linear: cada bit a mais **dobra** o número de
valores representáveis. Ir de 7 para 14 bits multiplica a resolução por `2^7 = 128`, não
por 2.

---

## E2 — Achando os limites na mão

```
fp16: estoura em 2^15  = 3,277e+04   (finfo.max  = 6,550e+04)
      zera    em 2^-25 = 2,980e-08   (finfo.tiny = 6,104e-05)
bf16: estoura em 2^127 = 1,701e+38   (finfo.max  = 3,390e+38)
      zera    em 2^-134= 4,592e-41   (finfo.tiny = 1,175e-38)
```

**1.** O expoente máximo bate com o `finfo.max` — a potência de 2 exata fica um pouco
abaixo do máximo, que usa a mantissa cheia (`1,111…₂ × 2^15`).

**2. Por que dá para ir muito abaixo do `.tiny` antes de zerar?** Por causa dos números
**subnormais**. Abaixo do menor valor normal, o formato abre mão de precisão para continuar
representando valores cada vez menores — usa a mantissa como se o expoente estivesse
travado no mínimo. É uma "rampa de saída" gradual em vez de um corte seco. Repare na
distância: o fp16 tem `.tiny = 6,1e-05` mas só zera em `3,0e-08`, três ordens de grandeza
abaixo.

**3.** O bf16 estoura e zera em expoentes muito maiores porque tem 8 bits de expoente — o
mesmo alcance do fp32.

---

## E3 — O epsilon na prática

| `x` | `1+x` em fp32 | `1+x` em fp16 | `1+x` em bf16 |
|-----|---------------|---------------|---------------|
| 1e-1 | 1,100000 | 1,099609 | 1,101562 |
| 1e-2 | 1,010000 | 1,009766 | 1,007812 |
| 1e-3 | 1,001000 | 1,000977 | **1,000000** |
| 1e-4 | 1,000100 | **1,000000** | **1,000000** |

(Os valores em negrito são somas que **não mudaram nada** — o `x` foi arredondado para
fora.)

**1.** Em fp16 a soma deixa de ter efeito por volta de `x = 1e-4` (epsilon = 9,77e-04).

**2.** Em bf16 o limite é **maior** (epsilon = 7,81e-03): ele já falha em `1e-3`. Tem 3
bits menos de mantissa, então distingue menos casas decimais.

**3. É exatamente por isso que os pesos mestres ficam em fp32.** Uma atualização típica de
treino tem ordem de `1e-4` a `1e-6`. Somada a um peso de ordem 1 em 16 bits, ela é
arredondada para fora e o peso **não muda** — como a apostila mede (o peso continua em
`1,000000` depois de 100 atualizações).

---

## E4 — A janela do loss scaling

| Pesos | Escala | % zerados | infs | Situação |
|-------|--------|-----------|------|----------|
| 0,02 | 1 | **50,3%** | 0 | perde gradiente |
| 0,02 | 1.024 | 0,1% | 0 | ok |
| 0,02 | 2²⁰ | 0,0% | 2 | overflow |
| **0,50** | 1 | **0,1%** | 0 | **ok** |
| 0,50 | 1.024 | 0,0% | 0 | ok |
| 0,50 | 2²⁰ | 0,0% | 1 | overflow |

**1 e 2. Leia a tabela com cuidado — o efeito não é o que parece.**

Com pesos maiores (0,50), o underflow praticamente **desaparece**: com escala 1, os zerados
caem de 50,3% para 0,1%. O limite **inferior** da janela desce muito.

Mas o limite **superior** quase não se mexe: os dois casos começam a estourar por volta de
2²⁰. Então a janela **não se desloca — ela alarga**.

E faz sentido: o *underflow* depende da magnitude dos gradientes (que muda com os pesos),
enquanto o *overflow* depende do teto do formato (65.504), que é **fixo**. Só o piso se
move.

**3. O scaler dinâmico continua se justificando**, mas por um motivo mais preciso: a
magnitude dos gradientes **cai** conforme o modelo converge. Uma escala fixa escolhida no
início pode ficar pequena demais depois, reintroduzindo underflow justamente na fase de
ajuste fino.

---

## E5 — Melhorando o GradScaler

| Fator | Escala final | Passos descartados |
|-------|--------------|--------------------|
| 1,01 | 70.966 | **0** / 40 |
| 2,00 | 524.288 | 4 / 40 |
| 16,00 | 65.536 | **6** / 40 |

**1.** Fator **grande** (16) reage rápido mas passa do ponto: sobe demais, estoura, desce
demais — e descarta 50% mais passos. Fator **pequeno** (1,01) é estável (zero descartes)
mas demora a se ajustar quando a magnitude dos gradientes muda.

**2. Por que um piso é útil?** Ele evita que uma sequência de overflows derrube a escala a
um valor tão baixo que **reintroduza underflow**. Sem piso, o scaler pode se auto-sabotar
ao reagir de forma exagerada a um pico isolado.

**3. Por que `growth_interval = 2000`?** Porque errar para cima custa **um passo
descartado**, mas errar com **frequência** faz o treino descartar passos o tempo todo. É
melhor subir raramente e ficar estável do que perseguir o ótimo e viver descartando.

---

## E6 — Seu hardware acelera 16 bits? *(precisa de GPU)*

Medido na máquina desta apostila (AMD Radeon RX 7600 via DirectML), matmul 1024×1024:

| | fp32 | fp16 | Ganho |
|---|------|------|-------|
| CPU | 5,35 ms | 1.951 ms | ~0,003x |
| Radeon | 0,40 ms | 0,41 ms | **0,98x** |

**1.** Nesta máquina, **nenhum ganho**.

**2. Isso significa que precisão reduzida é inútil? Não** — e o exercício existe para essa
distinção. São **dois benefícios independentes**:

- **memória**: garantido pela aritmética (metade dos bytes), vale em qualquer hardware
- **velocidade**: depende de o hardware **e** o backend terem unidades de 16 bits

Aqui o segundo não aparece. Numa NVIDIA com tensor cores, aparece — tipicamente 2x a 8x.

**3.** Se você tem NVIDIA, compare o ganho medido com a especificação da placa. Ele
raramente chega ao pico teórico, porque nem toda operação do modelo é matmul.

---

## E7 — Treino em precisão mista

Na versão CPU do `gabarito.py`, um problema de brinquedo treinado 400 passos:

| Modo | Loss final |
|------|-----------|
| fp32 | 0,0463 |
| bf16 | 0,0463 |
| fp16 (sem scaler) | 0,0463 |

**1 e 3.** O autocast com **bf16** chega à mesma loss do fp32 — não estraga o treino.
Esperado: o bf16 tem o alcance do fp32.

**2. Aqui o fp16 sem scaler também deu idêntico — e isso exige honestidade sobre o que o
teste não mostra.**

Não significa que loss scaling seja dispensável. Significa que **este problema não provoca
underflow**. Compare com o `loss_scaling.py` da apostila: lá os pesos são deliberadamente
pequenos (`p.mul_(0.02)`), os gradientes ficam na casa de `1e-8`, e **50,3% deles zeram**.
Aqui os pesos são os da inicialização padrão, os gradientes são saudáveis, e nada zera.

> **A lição metodológica:** um teste que passa não prova que a proteção é desnecessária —
> prova que o teste não exercita o caso que ela protege. É a mesma armadilha do E2 do
> Capítulo 4, com a máscara causal.

Modelos profundos de verdade têm gradientes muito menores nas camadas iniciais, e é lá que
o fp16 sem scaler quebra.

**4. Por que a economia de memória não chega aos 50%?** Porque a cópia **mestra** dos pesos
e o estado do otimizador (`m` e `v` do AdamW) continuam em fp32. Só as ativações e as
matmuls usam 16 bits.
