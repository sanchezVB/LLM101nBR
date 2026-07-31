# Exercícios — Capítulo 13 (Quantização)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Estes exercícios usam o checkpoint do Capítulo 11 (`../11-datasets/modelo.pt`).

---

### E1 — A aritmética (aquecimento)
Sem rodar:
1. Um tensor tem valores em `[-3, 3]`. Qual a escala simétrica para int8? Qual inteiro
   representa o float `1,5`? E que float se recupera dele?
2. Por que a quantização simétrica usa `[-127, 127]` em vez de `[-128, 127]`, se o int8
   comporta o −128?
3. Um tensor tem todos os valores em `[10, 11]`. Qual o erro relativo esperado com
   quantização **simétrica** de 8 bits? E por que a assimétrica ajudaria tanto aqui?

---

### E2 — Reproduza a armadilha do zero-point (importante)
No `quantizacao.py`, remova as duas linhas que estendem o intervalo para incluir o zero
(`xmin = min(xmin, 0)` e `xmax = max(xmax, 0)`).
1. Meça o erro da assimétrica para uma distribuição `randn(1000) + 5`. Compare com a
   simétrica. Qual ganha?
2. Imprima o *zero-point ideal* (`qmin - xmin/escala`) antes do `clamp`. Ele cabe em int8?
3. Quantos valores saturam no teto (`q == 127`)? Explique a cadeia causal completa: do
   zero-point fora do intervalo até o erro medido.

---

### E3 — A métrica que mente (importante)
Reproduza a matriz 8×64 com uma linha 100x maior.
1. Meça o erro **global** e o erro **só nas 7 linhas normais**, com per-tensor. Por que os
   dois números são tão diferentes?
2. A norma usada em `erro_relativo` é dominada por quê? Escreva a conta.
3. Proponha **outra** métrica que teria denunciado o problema sem precisar desagregar por
   linha. (Dica: pense em erro relativo *por elemento*.)

---

### E4 — Onde fica o precipício
Rode o `quantizar_modelo.py`.
1. Faça a tabela de duas colunas: *erro de reconstrução do peso* (do `quantizacao.py`) e
   *piora da loss* (do `quantizar_modelo.py`), para 8, 6, 4, 3 e 2 bits. Uma prevê a outra?
2. Entre quais bits está o joelho? O que acontece com a perplexidade ali?
3. Se você tivesse que escolher um número de bits para servir este modelo, qual escolheria?
   Justifique com os dois eixos (qualidade e tamanho), não com um só.

---

### E5 — Quantize os embeddings também
O `quantizar_modelo.py` tem o argumento `incluir_embeddings`.
1. Ative-o e meça: quanto o modelo encolhe a mais? Quanto a loss piora?
2. Neste modelo os embeddings são 10% dos parâmetros. Num modelo de 7B são ~2%. Em qual dos
   dois vale mais a pena quantizá-los, e por quê a resposta **não** é "no que tem a maior
   fração"?
3. A camada de saída (`lm`) mapeia `n_embd → vocab_size`. Ela é um `nn.Linear`, então já
   está sendo quantizada. Há algum motivo para tratá-la com mais cuidado que as outras?

---

### E6 — Meça o que a memória realmente faz
1. Meça o tempo de gerar 64 tokens com `float32` e com `int8`. Qual é mais rápido?
2. Explique, a partir do `forward` da `LinearQuantizada`, exatamente onde o tempo extra é
   gasto.
3. Escreva o que precisaria mudar para que a quantização de fato acelerasse. Por que isso
   não se faz em PyTorch puro?

---

### E7 — Quantização por grupos (desafio)
Entre o per-tensor (1 escala) e o per-channel (uma por linha) existe o meio-termo usado por
GGUF e GPTQ: uma escala a cada **G** valores consecutivos da linha (tipicamente G = 64
ou 128).
1. Implemente. Para uma matriz `(saida, entrada)`, isso significa `entrada/G` escalas por
   linha.
2. Meça o erro de reconstrução com G = 256 (equivalente ao per-channel de uma linha de 256),
   128, 64 e 32. Quanto se ganha?
3. Faça a conta do **custo real em bits por peso**: com int4 e G = 64, quantos bits cada
   peso ocupa de fato, contando as escalas em `float16`? Compare com os 4 bits nominais.
