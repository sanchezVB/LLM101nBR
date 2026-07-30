# Exercícios — Capítulo 04 (Attention)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

---

### E1 — Leitura de código (aquecimento)
Sem rodar, responda:
1. O que a matriz `tril` (triangular inferior) impede? O que aconteceria com o
   **treino** se o modelo pudesse ver o futuro?
2. Por que dividimos as pontuações por `sqrt(head_size)`?
3. Qual a diferença de papel entre **query**, **key** e **value**?

---

### E2 — Sem a máscara causal (vazamento de dados)
No `model.py`, comente a linha do `masked_fill` (a máscara causal) e treine de novo.
1. A loss de treino cai mais rápido? Ela fica **artificialmente** boa?
2. Gere nomes com esse modelo. A qualidade acompanha a loss?
3. Explique por que remover a máscara é **trapaça**: o modelo passa a usar a resposta
   para prever a resposta.

---

### E3 — Sem o embedding posicional
Remova o `+ pos` do `forward` (use apenas `tok`) e treine de novo.
1. A loss piora? Quanto?
2. **Por quê?** Sem posição, a atenção vê o contexto como um *conjunto* e não como uma
   *sequência* — ou seja, `ana` e `naa` ficariam indistinguíveis. Explique com suas
   palavras por que isso atrapalha na hora de prever o próximo caractere.

---

### E4 — Sem a escala `sqrt(head_size)`
Remova o fator `* k.shape[-1]**-0.5`.
1. O que acontece com a loss no começo do treino?
2. Imprima os pesos de atenção (`wei`) de um batch. Eles ficam mais "concentrados"
   (perto de 0 e 1) do que com a escala?
3. Relacione com o experimento do fim do `attention.py`: softmax de valores grandes
   satura, e gradiente saturado quase não aprende.

---

### E5 — Tamanho do contexto (`block_size`)
Treine com `block_size` = 3, 8 e 16.
1. Anote a loss de validação. Mais contexto sempre ajuda?
2. Para nomes (palavras curtas, ~7 letras), faz sentido um contexto de 16? O que o
   embedding posicional aprende nas posições que quase nunca são usadas?
3. Como o custo de computação da atenção cresce com o `block_size`? (Dica: a matriz de
   afinidades é `T × T`.)

---

### E6 — Inspecionando o que o modelo olha (desafio)
Faça o modelo devolver também os pesos `wei` e, para um nome específico (ex.: o
contexto `. . . . . a n a`), imprima a linha da última posição.
1. Em quais posições o modelo mais presta atenção ao prever o próximo caractere?
2. Você provavelmente vai achar **muito peso nos tokens de preenchimento `.`**, e não
   nas letras. Antes de chamar isso de bug, pense: o softmax **obriga** os pesos a
   somarem 1. Se o modelo não precisa trazer informação de nenhuma posição específica,
   onde ele "estaciona" esse peso? (Esse fenômeno tem nome em LLMs grandes: *attention
   sink*.)
3. A distribuição é claramente **não uniforme** — compare com o *bag of words* da
   Seção 3, onde todos os pesos eram iguais. O que isso mostra sobre o mecanismo?
> Solução de referência: [`solucoes/e6_inspecionar_atencao.py`](solucoes/e6_inspecionar_atencao.py).

---

### E7 — Multi-head attention (desafio, antecipa o Cap. 5)
Uma cabeça só aprende **um** tipo de relação. Implemente **multi-head attention**:
crie `n_heads` cabeças com `head_size = n_embd // n_heads`, rode todas em paralelo e
**concatene** as saídas.
1. Com o mesmo total de parâmetros, 4 cabeças pequenas vão melhor que 1 grande?
2. Por que ter várias cabeças pode ajudar? (Pense: uma cabeça olha a vogal anterior,
   outra o começo da palavra...)
3. Essa é exatamente a peça que abre o Capítulo 5 — o Transformer.
