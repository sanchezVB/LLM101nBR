# Exercícios — Capítulo 17 (Multimodal)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Rode `python preparar_dados.py` e `python vqvae.py` antes. Juntos levam ~3 minutos.

---

### E1 — Por que quantizar (aquecimento)
Sem rodar:
1. Uma imagem 28×28 vira 49 tokens. Qual a taxa de compressão em **posições**? E se você
   contasse em **bits**, considerando que cada pixel é um byte e cada token cabe em 7 bits?
2. Por que não alimentar o Transformer com os vetores contínuos do encoder, pulando a
   quantização? O que se perderia? (Dica: pense no que a camada de saída do modelo produz.)
3. O capítulo chama o VQ-VAE de "BPE das imagens". Em que a analogia é boa, e em que ela
   quebra?

---

### E2 — O straight-through estimator (importante)
1. Remova a linha `z_q = z + (z_q - z).detach()` e passe `z_q` direto para o decoder.
   Treine 300 passos. O que acontece com a perda de reconstrução, e por quê?
2. Escreva o que o `.detach()` faz no forward e no backward, separadamente. Por que a
   expressão vale `z_q` num e `z` no outro?
3. O estimador é **aproximado** — o gradiente que o encoder recebe não é o gradiente
   verdadeiro. Por que isso não impede o treino de funcionar?

---

### E3 — As duas perdas (importante)
O quantizador tem `perda_codebook` e `perda_commit`.
1. Treine com `BETA_COMP = 0` (sem commitment loss). Meça o erro de reconstrução e o
   número de códigos usados. O que piora?
2. Treine com `BETA_COMP = 2.0`. O que acontece agora, e por quê?
3. Explique o papel de cada perda em uma frase, e diga o que aconteceria removendo a
   **outra** (a `perda_codebook`).

---

### E4 — Códigos mortos
O capítulo mede 93 códigos usados de 128.
1. Varie o tamanho do codebook (`--codebook 32`, `64`, `256`, `512`) e monte a tabela de
   códigos usados e erro de reconstrução. A fração usada cresce ou cai com o tamanho?
2. Implemente a correção mais simples: a cada N passos, reinicialize os códigos não usados
   sobre vetores aleatórios do batch atual. Quantos códigos passam a ser usados?
3. A correção melhorou o **erro de reconstrução**, ou só a estatística de uso? Se só a
   estatística, isso ainda vale a pena? Justifique.

---

### E5 — O mesmo Transformer
1. No `gerar_imagens.py`, a config do GPT é `{vocab_size: 128, block_size: 49, ...}`.
   Compare com a do Capítulo 11. Quantas linhas de código do modelo precisaram mudar para
   ele aceitar imagens?
2. A perplexidade medida é 3,9 entre 128 tokens. Compare com a do Capítulo 11 (51,7 entre
   1.024). O modelo de imagens é "melhor"? O que torna a comparação difícil?
3. O modelo gera as 49 posições em ordem de varredura (esquerda para a direita, de cima
   para baixo). Isso é natural para texto e arbitrário para imagem. Que problema essa ordem
   cria, e o que você tentaria no lugar?

---

### E6 — Condicionar pelo dígito
O MNIST tem rótulos, e o `preparar_dados.py` já os salva.
1. Acrescente o rótulo como **primeiro token** da sequência (reserve 10 ids novos, como o
   Capítulo 14 fez com os tokens especiais). Retreine.
2. Gere dez imagens, uma por dígito, começando cada sequência pelo rótulo correspondente.
   O modelo obedece?
3. Você acabou de construir um gerador **condicional** — o mesmo mecanismo de um "texto →
   imagem", com um vocabulário de condição de tamanho 10 em vez de uma frase. O que
   mudaria para condicionar por uma legenda de verdade?

---

### E7 — Um modelo, duas modalidades (desafio)
1. Junte os vocabulários: tokens de texto do Capítulo 11 (ids 0–1023) e tokens de imagem
   deste capítulo (ids 1024–1151). Treine **um** GPT em sequências dos dois tipos.
2. Meça a loss em cada modalidade separadamente e compare com os modelos especializados.
   Houve interferência? Em qual direção?
3. **A pergunta que importa:** um modelo treinado nas duas modalidades separadamente, sem
   nenhum exemplo *pareado*, aprende alguma relação entre elas? Justifique pelo que o
   objetivo de treino recompensa — e diga o que seria preciso para que aprendesse.
