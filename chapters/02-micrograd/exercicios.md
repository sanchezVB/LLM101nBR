# Exercícios — Capítulo 02 (Micrograd)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

---

### E1 — Derivada na unha (aquecimento)
Sem código, só caneta:
1. Para `f = a * b` com `a = 3`, `b = 4`, quanto vale `df/da`? E `df/db`?
2. Para `d = a*b + c`, quanto vale `dd/dc`? Por que a soma "não altera" o gradiente?
3. Explique, com suas palavras, por que `tanh` saturada (entrada grande) produz
   gradiente quase zero.

---

### E2 — Conferindo com diferença finita
A derivada é `[f(x+h) - f(x)] / h` para `h` pequeno. Escolha uma expressão simples
com `Value` (ex.: `L = a*b + c`), calcule `a.grad` via `backward()`, e depois
compare com a estimativa numérica: recalcule `L` com `a` somado de `h = 1e-6` e use a
fórmula. Os dois valores batem? (Essa é a forma clássica de **testar** um autograd.)

---

### E3 — Adicione uma operação nova
Implemente o método `log()` na classe `Value` (logaritmo natural).
1. Qual a derivada local de `log(x)`? (Dica: `d(ln x)/dx = 1/x`.)
2. Escreva o `_backward` correspondente, seguindo o padrão de `exp()`.
3. Teste comparando com `math.log` e a diferença finita do E2.

---

### E4 — Por que zerar o gradiente?
No `nn.py`, **comente** a linha `model.zero_grad()` e rode de novo.
1. O que acontece com a loss ao longo dos passos?
2. Explique usando o fato de que os gradientes são acumulados com `+=`.
3. Relacione com o `W.grad = None` do Capítulo 1.

---

### E5 — Bate com o PyTorch? (desafio)
Escreva um script que monte **a mesma expressão** com `Value` e com `torch.tensor(...,
requires_grad=True)`, rode `backward()` nos dois e compare `data` e todos os `grad`.
1. Eles batem (com tolerância de `1e-6`)?
2. Troque uma operação (ex.: use `relu` em vez de `tanh`) e confirme que ainda batem.
> A solução de referência está em [`solucoes/e5_check_vs_torch.py`](solucoes/e5_check_vs_torch.py).

---

### E6 — Mexa na arquitetura e na learning rate
No `nn.py`:
1. Troque `MLP(3, [4, 4, 1])` por `MLP(3, [8, 1])` e depois `MLP(3, [16, 16, 1])`.
   Quantos parâmetros cada um tem? A loss final muda?
2. Varie a learning rate (`0.05`) para `0.5` e `0.001`. O que acontece com `0.5`
   (instável?) e com `0.001` (lento?)?
3. Troque a ativação `tanh` por `relu` nos neurônios. Treina melhor ou pior aqui?

---

### E7 — Visualize o grafo (desafio, opcional)
Escreva uma função que percorra `_prev` recursivamente a partir de um `Value` e
imprima o grafo (cada nó: seu `_op`, `data` e `grad`). Use-a numa expressão pequena
para "ver" a estrutura que a `backward()` percorre. (Se conhecer a lib `graphviz`,
desenhe de verdade — é opcional.)
