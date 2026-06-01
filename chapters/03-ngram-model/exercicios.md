# Exercícios — Capítulo 03 (N-gram model / MLP)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

---

### E1 — Leitura de código (aquecimento)
Sem rodar, responda:
1. Por que `emb = C[X]` tem formato `(N, 3, 10)`? O que significa cada uma das três
   dimensões?
2. Por que precisamos do `.view(N, -1)` antes da primeira camada linear?
3. O que `F.cross_entropy(logits, Y)` faz, em termos do que fizemos "à mão" no
   Capítulo 1?

---

### E2 — O efeito do contexto (`block_size`)
Rode `mlp.py` com `block_size` = 1, 3 e 5 (três execuções).
1. Anote a loss de validação em cada caso. Mais contexto ajuda?
2. Com `block_size = 1`, o MLP vira essencialmente um bigrama "neural". A loss fica
   parecida com a do Capítulo 1 (~2,4)? Por quê?
3. O número de parâmetros muda com o `block_size`? Onde, no código, isso aparece?

---

### E3 — Tamanho da camada oculta
Varie `n_hidden` em 50, 200 e 500.
1. Como muda o número de parâmetros e a loss de validação?
2. Camada maior sempre dá loss menor? Onde parece haver um retorno decrescente?

---

### E4 — Learning rate
1. Troque a learning rate inicial (`0.1`) por `1.0` e por `0.001`. Descreva o que
   acontece com a curva de loss em cada caso (instável? lenta?).
2. Remova o decaimento (deixe `lr = 0.1` o tempo todo). A loss final piora? Por que o
   decaimento ajuda no fim do treino?

---

### E5 — Overfitting com dataset pequeno (o mais importante)
Copie os **155 nomes** do Capítulo 1 para um arquivo `names_pequeno.txt` e rode o MLP
com ele (mude o nome do arquivo no código, ou use a solução pronta).
1. O que acontece com a loss de **treino** versus a de **validação**?
2. Por que o **mesmo modelo** generaliza com 64 mil nomes mas decora com 155?
3. Os nomes "gerados" passam a se parecer demais com nomes reais do dataset. Por quê
   isso é um **mau** sinal, e não um bom?
> Solução de referência: [`solucoes/e5_overfitting.py`](solucoes/e5_overfitting.py).

---

### E6 — Visualizando os embeddings (desafio)
Após treinar, a tabela `C` tem um vetor por caractere. Para "ver" o que a rede
aprendeu, treine com `n_embd = 2` (assim cada caractere é um ponto no plano) e
plote os 27 caracteres com `matplotlib`.
1. As **vogais** (a, e, i, o, u) ficam agrupadas? E outros grupos fazem sentido?
2. Por que reduzir para `n_embd = 2` piora a loss, mas ainda assim é útil para
   visualizar?

---

### E7 — Conte os parâmetros na mão (desafio)
O código imprime `parametros: 11897`. Mostre, somando as partes (`C`, `W1`, `b1`,
`W2`, `b2`), de onde vem esse número — em função de `vocab_size`, `block_size`,
`n_embd` e `n_hidden`. Confira que sua fórmula reproduz o 11897.
