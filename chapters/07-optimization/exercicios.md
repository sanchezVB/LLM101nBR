# Exercícios — Capítulo 07 (Optimization)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

> **Aviso de tempo:** os exercícios que treinam o Transformer levam ~6 minutos cada na
> CPU. Reduza `max_steps` (ex.: 4000) para experimentar mais rápido — as *comparações
> relativas* seguem válidas, só não compare com os números absolutos da apostila.

---

### E1 — Leitura de código (aquecimento)
Sem rodar:
1. No nosso `AdamW`, o que a **correção de viés** (`m / (1 - beta1**t)`) faz? Por que ela
   importa mais nos primeiros passos e vira irrelevante depois?
2. Qual a diferença entre somar o *weight decay* ao gradiente (Adam original) e
   aplicá-lo direto no parâmetro (AdamW)? Por que a segunda forma é melhor?
3. Por que `initialization.py` divide por `sqrt(fan_in)` e não por `fan_in`?

---

### E2 — O warmup é necessário?
No `train_tuned.py`, coloque `WARMUP_STEPS = 0` e treine.
1. A loss dos primeiros passos fica pior? O treino chega ao mesmo lugar no fim?
2. Olhe a **norma do gradiente** nos primeiros passos (o script reporta média e máxima).
   Por que o começo do treino é o momento mais instável?
3. Explique o warmup em uma frase: por que começar com passos pequenos ajuda um
   otimizador adaptativo como o Adam? (Dica: nos primeiros passos, as médias móveis `m`
   e `v` têm pouquíssima informação.)

---

### E3 — Calibrando o gradient clipping
O script informa a **norma média** do gradiente e **quantos passos foram clipados**. Use
essas duas informações.
1. Com o `GRAD_CLIP = 3.0` do código, que percentual dos passos é cortado? Isso é
   coerente com "cortar apenas as exceções"?
2. Reduza para `1.0` e depois `0.1`. Em que ponto o clipping deixa de ser uma rede de
   segurança e passa a ser uma **normalização** de todo passo? (A apostila conta que a
   primeira versão deste capítulo cortava 99% dos passos — o erro está documentado na
   Seção 5.)
3. Aumente para `100` (nunca corta). A loss piora? Se não piorar, o que isso diz sobre a
   necessidade de clipping **neste** modelo?
4. O clipping altera a **direção** do gradiente ou apenas o seu **tamanho**? Por que essa
   distinção importa?

---

### E4 — O ganho certo para a GELU
O `initialization.py` mostra o ganho correto para `tanh` (5/3) e `ReLU` (√2). Nosso
Transformer usa **GELU**.
1. Adapte a função `rodar` para aceitar GELU e descubra empiricamente qual ganho mantém
   o desvio padrão estável ao longo das 8 camadas.
2. O valor ficou mais perto do da ReLU ou do da tanh? Isso faz sentido, dado que a GELU
   é uma versão suave da ReLU?
3. Compare com `torch.nn.init.calculate_gain` — ela tem entrada para GELU? Se não, o que
   isso sugere na prática?

---

### E5 — A curva da learning rate
Treine com `base_lr` = 1e-4, 3e-4, 1e-3, 3e-3 e 1e-2 (reduza `max_steps` para 3000).
1. Monte a tabela de loss de validação. O formato é de um "U"? Onde está o mínimo?
2. O que acontece com `1e-2` — o treino diverge ou só fica ruim?
3. **Você provavelmente vai achar um mínimo em `3e-3`, maior que o `1e-3` da apostila.**
   Isso contradiz o capítulo? (Dica: a apostila treina 15.000 passos, e este exercício
   3.000. Com pouco orçamento, compensa andar mais rápido.)
4. Por que a melhor learning rate depende do **tamanho do batch**? (Pense: com batch
   maior, o gradiente é menos ruidoso.)
> Solução de referência: [`solucoes/e5_curva_lr.py`](solucoes/e5_curva_lr.py).

---

### E6 — AdamW vs SGD no modelo de verdade
O `optimizers.py` compara os otimizadores num problema de brinquedo. Faça a comparação
no Transformer: troque o `torch.optim.AdamW` por `torch.optim.SGD` (com e sem momentum).
1. Com a **mesma** learning rate, o SGD chega perto? 
2. Ajuste a learning rate do SGD para o melhor valor que encontrar. Ele empata com o
   AdamW?
3. Por que redes com muitos parâmetros de escalas diferentes (embeddings, pesos de
   atenção, ganhos de LayerNorm) favorecem otimizadores adaptativos?

---

### E7 — Por que o weight decay atrapalhou? (importante)
A ablação da apostila mostra que `weight_decay = 0.1` **piorou** o modelo em `−0,0532` — o
maior efeito da tabela, e negativo.
1. Varie o weight decay: `0.0`, `0.01`, `0.1`, `0.5`. Monte a tabela de loss de validação.
   Existe um valor melhor que o default `0.01`?
2. Compare a loss de **treino** com a de **validação** em cada caso. Quando o weight decay
   aumenta, as duas sobem juntas ou a de treino sobe mais? O que isso indica?
3. **A pergunta que importa:** weight decay combate *overfitting*. Olhe os números do
   baseline (treino 1,791 vs validação 1,811). Existe overfitting para combater? Se não
   existe, o que a regularização forte está fazendo com a capacidade do modelo?
4. Em que cenário você **esperaria** que `weight_decay = 0.1` ajudasse? (Dica: releia a
   solução do E5 do Capítulo 3, com 155 nomes.)

---

### E8 — Outro agendamento (desafio)
Implemente e compare três agendamentos, todos com o mesmo warmup:
- **constante** (o baseline)
- **cosseno** (o nosso)
- **linear** (decai em linha reta até `MIN_LR_FRAC`)
- **degraus** (divide por 10 em 50% e 80% do treino)

1. Qual vence no nosso problema? A diferença é grande?
2. O cosseno é o padrão em treinos de LLM. Pelos seus números, isso se justifica ou é
   convenção?
3. Um detalhe importante: o agendamento depende de saber `max_steps` de antemão. Que
   problema isso cria se você quiser **continuar** um treino já terminado?
