# Exercícios — Capítulo 01 (Bigram Language Model)

Faça na ordem; os primeiros são de fixação, os últimos são mais desafiadores.
Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

---

### E1 — Leitura de código (aquecimento)
Sem rodar, responda:
1. Por que `vocab_size` é 27, e não 26?
2. No `bigram.py`, o que aconteceria se trocássemos `(N + 1)` por `N` (sem
   suavização) ao calcular a loss? Por quê?
3. Para que serve o `manual_seed`?

---

### E2 — Mexa nos dados
Adicione 5 nomes ao final de [`names.txt`](names.txt) (pode ser de amigos, ou nomes
inventados). Rode `bigram.py` de novo. A loss subiu ou desceu? Os nomes gerados
mudaram? Explique em 1–2 frases por que adicionar dados muda o modelo.

---

### E3 — O efeito da suavização (*smoothing*)
No `bigram.py`, troque o `+ 1` por `+ 0`, `+ 10` e `+ 100` (três execuções).
1. Como a **loss** muda em cada caso?
2. Como os **nomes gerados** mudam (mais "criativos" ou mais "genéricos")?
3. Explique: por que `+ 0` pode dar problema?

---

### E4 — Temperatura no sampling
No `bigram.py`, antes de amostrar, eleve a distribuição a uma potência e
re-normalize: `p = P[ix] ** T; p = p / p.sum()`.
Teste `T = 0.3`, `T = 1.0` e `T = 3.0`.
1. Com `T` baixo, os nomes ficam mais "óbvios" ou mais "aleatórios"? E com `T` alto?
2. Relacione isso com o conceito de **temperatura** que você já ouviu falar em LLMs.

---

### E5 — Faça contagem e rede convergirem (a equivalência)
Rodando os dois arquivos como vêm, você acha a loss por contagem (**~2,4**) maior que
a da rede (**~2,2**). Elas *deveriam* ser a mesma coisa — vamos provar igualando a
"quantidade de suavização" dos dois lados:
1. No `bigram.py`, troque o `+ 1` por `+ 0.01`. Qual a nova loss por contagem?
   (Deve cair para ~2,15 — perto do MLE, o melhor possível.)
2. No `bigram_nn.py`, zere a regularização (troque `0.01 *` por `0.0 *`) e aumente os
   passos de 200 para 1000. Qual a loss pura da rede agora?
3. Os dois valores se encontram em torno de **~2,1**? Explique, com suas palavras, por
   que `+1` (smoothing) e o termo `(W**2)` (regularização L2) são "o mesmo botão".

---

### E6 — Avaliando um nome específico (desafio)
Escreva uma função `nll_de(nome)` que calcule a negative log-likelihood **de um único
nome** segundo o modelo de contagem `P`. Use-a para responder:
1. Qual tem menor NLL (é mais "provável" segundo o modelo): `ana` ou `xwq`?
2. Teste com o seu próprio nome. Ele é "provável" para o modelo?

---

### E7 — Rumo ao trigrama (desafio, antecipa o Cap. 3)
Modifique o modelo de contagem para um **trigrama**: prever o próximo caractere a
partir dos **dois** caracteres anteriores.
1. Em vez de uma matriz 27×27, qual o formato da tabela de contagens agora?
2. A loss melhora em relação ao bigrama? Por quê?
3. Que novo problema de "dados esparsos" (muitos zeros) começa a aparecer? Como a
   suavização ajuda — e por que ela não resolve tudo? (Essa limitação é exatamente o
   que motiva o MLP do Capítulo 3.)
