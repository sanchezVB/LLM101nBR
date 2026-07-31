# Exercícios — Capítulo 14 (SFT)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Rode `python preparar_sft.py` e `python sft.py` antes dos exercícios. Juntos levam menos
> de seis minutos.

---

### E1 — Tokens especiais e vocabulário (aquecimento)
Sem rodar:
1. Por que `<|fim|>` precisa ser um **id novo** em vez de uma string como `###FIM###`?
   Descreva um caso concreto em que a string falharia.
2. Ao aumentar o vocabulário de 1.024 para 1.027, quantos parâmetros novos aparecem? Faça a
   conta com `n_embd = 192`, lembrando das **duas** matrizes e do bias.
3. As linhas novas são inicializadas com a **média** das existentes. O que aconteceria com
   zeros na camada de saída? E com valores aleatórios grandes?

---

### E2 — Quando a máscara importa (importante)
O capítulo mediu que mascarar o pedido **não faz diferença** com pedido de 24 tokens e
resposta de ~34.
1. Antes de rodar: escreva por que você espera que a máscara importe, e o que na sua
   explicação depende do **tamanho relativo** do pedido.
2. Rode [`solucoes/e2_quando_a_mascara_importa.py`](solucoes/e2_quando_a_mascara_importa.py),
   que varia a proporção. A partir de qual fração de pedido a penalidade aparece?
3. Um resultado nulo ("não faz diferença") e um resultado negativo ("faz diferença ao
   contrário") pedem reações diferentes. Qual dos dois você obteve no caso de pedido curto,
   e o que cada um autorizaria você a concluir?

---

### E3 — Quebre os dados de propósito (importante)
Volte o `preparar_sft.py` para resposta de **tamanho fixo** (era `TAM_RESPOSTA = 40`).
1. Treine e meça a taxa de parada e o comprimento **mediano**. O que você vê?
2. A taxa de parada é 100% nos dois casos. Isso significa que a versão de tamanho fixo é
   igualmente boa? Que métrica separa as duas?
3. Generalize: se todas as respostas do seu conjunto de SFT começassem com "Claro!", o que o
   modelo aprenderia? Dê mais dois exemplos de regularidade acidental que apareceriam num
   conjunto real.

---

### E4 — A learning rate do finetuning
O `sft.py` usa `lr = 3e-4`, três vezes menor que a do pré-treino.
1. Rode com `1e-3` (a do pré-treino) e com `3e-5`. Meça a loss na resposta e a taxa de
   parada nas três.
2. Com learning rate alta, o que acontece com a capacidade do modelo de escrever português
   comum? (Sugestão: gere texto **sem** o formato de instrução e compare com o modelo-base.)
3. O fenômeno tem nome — *catastrophic forgetting*. Explique por que finetuning com poucos
   dados e learning rate alta o provoca.

---

### E5 — Quantos exemplos bastam?
1. Treine com 500, 2.000 e 8.000 exemplos, mantendo os passos fixos. Como muda a taxa de
   parada? E a loss?
2. Compare com o Capítulo 11, onde mais dados sempre ajudaram. Por que a curva aqui é
   diferente?
3. Na literatura de SFT há resultados mostrando que **mil exemplos bem escolhidos** batem
   dezenas de milhares aleatórios. O que a sua medição sugere sobre o porquê?

---

### E6 — O modelo esqueceu de escrever?
1. Meça a loss do modelo **depois do SFT** no conjunto de validação **original** do
   Capítulo 11 (texto corrido, sem formato de instrução). Compare com o modelo-base.
2. O SFT melhorou, piorou ou não mexeu na capacidade de modelar português comum?
3. Este é o *alignment tax*: o custo, na capacidade original, de ensinar um comportamento
   novo. Ele é inevitável? O que reduziria o custo?

---

### E7 — Um formato de conversa de verdade (desafio)
O formato deste capítulo tem um turno só. Modelos de chat têm vários.
1. Estenda o formato para `<|pedido|> A <|resposta|> B <|pedido|> C <|resposta|> D <|fim|>`.
   O que muda na máscara?
2. Onde entra o `<|fim|>` — no fim de cada resposta ou só no fim da conversa? Qual das duas
   escolhas permite a geração parar no turno certo?
3. Modelos reais usam um token de fim **por turno** (`<|im_end|>` e afins) e param nele.
   Implemente assim e verifique que a geração para no fim do primeiro turno, não da
   conversa inteira.
