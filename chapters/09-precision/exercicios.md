# Exercícios — Capítulo 09 (Precision)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Os exercícios E1 a E5 rodam **na CPU** e não precisam de GPU. O E6 e o E7 precisam de
> GPU para a parte de velocidade.

---

### E1 — Contas com bits (aquecimento)
Sem rodar código:
1. O bf16 tem 8 bits de expoente e 7 de mantissa; o fp16 tem 5 e 10. Qual dos dois
   representa o número `100000`? E qual representa `0,000000001` (1e-9)?
2. Quantos valores distintos um formato com 7 bits de mantissa consegue representar
   **entre 1 e 2**? E com 10 bits? (Dica: é `2^bits`.)
3. Por que dobrar os bits de mantissa **não** dobra a precisão, mas a multiplica por
   muito mais?

---

### E2 — Ache o limite na mão
Escreva um laço que, começando em 1,0, multiplique por 2 até que o valor vire `inf` em
fp16.
1. Em que potência de 2 isso acontece? Confere com `torch.finfo(torch.float16).max`?
2. Faça o mesmo dividindo por 2 até chegar a zero. Em que potência? Compare com
   `.tiny` — por que dá para ir **abaixo** do `.tiny` antes de zerar? (Pesquise
   "números subnormais".)
3. Repita para bf16 e explique a diferença.

---

### E3 — O epsilon na prática
1. Em fp16, calcule `1.0 + x` para `x` = 1e-2, 1e-3, 1e-4. A partir de qual valor a soma
   deixa de mudar o resultado?
2. Repita em bf16. O limite é maior ou menor? Por quê?
3. Relacione com a Seção 4 da apostila: por que os **pesos mestres** precisam ficar em
   fp32?

---

### E4 — A janela do loss scaling
No `loss_scaling.py`, mexa no `modelinho` para que os gradientes fiquem **maiores**
(aumente o `p.mul_(0.02)` para `p.mul_(0.5)`).
1. A porcentagem de gradientes zerados em fp16 muda?
2. A janela de escalas seguras se move? Para cima ou para baixo?
3. Isso justifica o scaler **dinâmico**? Explique com os seus números.

---

### E5 — Melhore o GradScaler
O `GradScalerSimples` da apostila reduz a escala pela metade a cada overflow e a dobra a
cada N passos bons.
1. Qual o problema de um `fator` muito grande (ex.: 16)? E muito pequeno (ex.: 1,01)?
2. Implemente uma variante que, ao detectar overflow, **não** reduza abaixo de um piso
   configurável. Por que um piso é útil?
3. O `torch.amp.GradScaler` real usa `growth_interval=2000` por padrão, não 100. Por que
   um intervalo tão longo?

---

### E6 — Seu hardware acelera 16 bits? (precisa de GPU)
Rode `precision_bench.py` na sua máquina.
1. O fp16 é mais rápido que o fp32 na sua GPU? Quanto?
2. **Cuidado com a conclusão:** na máquina desta apostila o ganho foi **nulo** (0,99x).
   Se o seu também for, isso significa que precisão reduzida é inútil? (Dica: releia a
   Seção 6 — há dois benefícios distintos.)
3. Se você tem NVIDIA: procure "tensor cores" na especificação da sua placa. O ganho que
   você mediu é compatível com o prometido?

---

### E7 — Treine de verdade em precisão mista (desafio)
Pegue o `transformer.py` do Capítulo 5 e adapte para usar `torch.autocast`.
1. Com `bfloat16` (sem scaler): a loss final fica igual à do fp32? Compare com os
   números do Capítulo 5 (validação 1,811).
2. Com `float16` **sem** scaler: o treino degrada? Meça.
3. Com `float16` **com** `torch.amp.GradScaler`: recupera a qualidade?
4. Meça o **pico de memória** nos três casos (use `torch.cuda.max_memory_allocated()` se
   tiver NVIDIA). A economia chega aos 50% teóricos? Por que não?
