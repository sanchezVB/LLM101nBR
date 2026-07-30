# Exercícios — Capítulo 05 (Transformer)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

> **Aviso de tempo:** os exercícios que treinam modelos levam alguns minutos cada na
> CPU. Para experimentar mais rápido, reduza `max_steps` (ex.: 3000) — as *comparações
> relativas* continuam válidas, só não compare com os números absolutos da apostila.

---

### E1 — Leitura de código (aquecimento)
Sem rodar, responda:
1. No `Block`, por que escrevemos `x = x + self.sa(self.ln1(x))` e não
   `x = self.sa(self.ln1(x))`? O que a soma acrescenta?
2. Por que `head_size = n_embd // n_head`? O que aconteceria com o custo se cada
   cabeça tivesse `head_size = n_embd`?
3. Qual a diferença entre o que a **atenção** faz e o que o **feedforward** faz dentro
   de um bloco?

---

### E2 — Sem as conexões residuais
No `Block`, troque as duas linhas por `x = self.sa(self.ln1(x))` e
`x = self.ff(self.ln2(x))` (sem o `x +`).
1. A loss final piora? Quanto?
2. Aumente `n_layer` para 6 **sem** os residuais e depois **com**. A diferença cresce
   com a profundidade?
3. Explique o papel do residual como "caminho livre" para o gradiente.
> Solução de referência: [`solucoes/e2_ablacoes.py`](solucoes/e2_ablacoes.py).

---

### E3 — Sem LayerNorm
Remova as LayerNorms do bloco (aplique as sub-camadas direto em `x`).
1. O treino fica instável? A loss oscila mais entre passos?
2. Tente também aumentar a learning rate para `3e-3` sem LayerNorm. O que acontece?
3. Por que normalizar as ativações ajuda a estabilizar o treino?

---

### E4 — Profundidade e número de cabeças
Treine com:
- `n_layer` = 1, 3 e 6 (mantendo `n_head = 4`)
- `n_head` = 1, 4 e 8 (mantendo `n_layer = 3`)

1. Anote parâmetros e loss de validação em cada caso.
2. Mais profundidade sempre ajuda? Onde começa o retorno decrescente?
3. Com o **mesmo** `n_embd`, mudar `n_head` altera o número de parâmetros? Por quê?
   (Dica: olhe `head_size = n_embd // n_head`.)

---

### E5 — Uma cabeça grande vs várias pequenas
Compare `n_head = 1` (uma cabeça de 64) com `n_head = 8` (oito de 8), ambos com
`n_embd = 64` — ou seja, praticamente o mesmo número de parâmetros.
1. Qual vai melhor?
2. Por que várias cabeças pequenas podem ganhar de uma grande? (Pense: cada cabeça
   pode se especializar numa relação diferente.)
3. Existe um limite? O que acontece com `n_head = 64` (cabeças de dimensão 1)?

---

### E6 — Dropout (regularização, desafio)
O GPT-2 usa **dropout**: durante o treino, zera aleatoriamente uma fração das
ativações, o que dificulta a memorização. Adicione `nn.Dropout(0.1)` na saída da
projeção da atenção e no fim do feedforward.
1. Com nosso dataset (64 mil nomes), treino e validação já andam juntos. O dropout
   ajuda, atrapalha ou é indiferente aqui?
2. Em que situação o dropout seria claramente útil? (Releia a lição do Cap. 3 sobre
   *overfitting*.)
3. Por que o dropout deve ser desligado na avaliação (`model.eval()`)?

---

### E7 — Conte os parâmetros (desafio)
O código informa **153.499** parâmetros. Deduza esse número somando as partes:
embeddings (token e posição), por bloco (atenção: `k`,`q`,`v`,`proj`; feedforward: duas
camadas; duas LayerNorms), a LayerNorm final e a `lm_head`.
1. Escreva a fórmula em função de `vocab_size`, `block_size`, `n_embd`, `n_head` e
   `n_layer`.
2. Confira que ela reproduz 153.499.
3. Qual parte domina o total? Se você dobrar `n_embd`, o número de parâmetros dobra ou
   quadruplica?
