# Exercícios — Capítulo 06 (Tokenization)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de
olhar**.

---

### E1 — Unicode e UTF-8 (aquecimento)
Sem rodar código:
1. Quantos **bytes** ocupa `"pão"` em UTF-8? E `"pao"`? Por quê?
2. `ord("ç")` vale 231. Isso significa que `"ç"` cabe em 1 byte? Explique.
3. Por que um tokenizador que começa pelos **bytes** nunca encontra um caractere
   "desconhecido"?

---

### E2 — O tamanho do vocabulário
Rode `bpe.py` com `VOCAB_SIZE` = 300, 512 e 1024.
1. Como muda a **taxa de compressão** em cada caso?
2. O ganho é linear? Onde começa o retorno decrescente?
3. Qual o custo de um vocabulário maior? (Pense na camada de saída do modelo: a
   `lm_head` do Capítulo 5 tem `n_embd × vocab_size` parâmetros.)

---

### E3 — As primeiras fusões
Olhe a lista de fusões que o `bpe.py` imprime (as 10 primeiras).
1. Por que `('a','n') -> 'an'` é a primeira? O que isso diz sobre nomes brasileiros?
2. Várias fusões iniciais terminam em `\n` (fim de linha), como `'a\n'` e `'on\n'`.
   Por quê? O que isso revela sobre o formato do nosso arquivo de treino?
3. Entre os tokens mais longos aparecem `'ilson'`, `'erson'`, `'ilton'`. O tokenizador
   descobriu **sufixos de nomes** sem ninguém ensinar. Como?

---

### E4 — A ordem das fusões importa
No método `encode`, trocamos a escolha do par por `min(stats, key=...)` usando a ordem
de aprendizado. Experimente trocar por uma escolha arbitrária (por exemplo, o primeiro
par encontrado que esteja em `self.merges`).
1. O `decode(encode(x)) == x` continua valendo?
2. A compressão fica pior? Por quê a ordem de aprendizado é a ordem correta?
3. Dica: uma fusão tardia pode usar um token criado por uma fusão anterior. O que
   acontece se você aplicar a tardia primeiro?

---

### E5 — O imposto do português (importante)
O `bpe.py` mostra que `"A informação sobre a ação..."` é fatiada com 16 tokens de
fragmentos de byte (`<0xNN>`), porque o tokenizador foi treinado em nomes **sem**
acento.
1. Treine um BPE **em texto português com acentos** e verifique se ele aprende `'ção'`
   (ou `'ão'`) como um token só. Use como corpus os próprios `README.md` do curso.
2. Compare a contagem de tokens da mesma frase nos dois tokenizadores.
3. Isso tem consequência prática real: APIs de LLM cobram **por token**. Se o
   tokenizador foi treinado majoritariamente em inglês, escrever em português custa
   mais. Estime esse sobrecusto a partir dos seus números.
> Solução de referência: [`solucoes/e5_bpe_portugues.py`](solucoes/e5_bpe_portugues.py).

---

### E6 — Tokens fora do domínio
Note que o caso com japonês e emoji (veja `bpe.py`) teve compressão `1.00x` — nenhuma.
1. Por que o tokenizador não comprime nada nesse caso?
2. Ele ainda consegue **representar** o texto? (Rode e confirme.) Por que isso é uma
   garantia importante?
3. O que aconteceria com um tokenizador baseado em **palavras** ao receber uma palavra
   em japonês?

---

### E7 — Integrando com o modelo (desafio, conecta ao Cap. 5)
Nosso Transformer do Capítulo 5 usa 27 tokens (um por letra). Adapte-o para usar o
tokenizador BPE deste capítulo.
1. O que muda em `vocab_size` e, por consequência, no número de parâmetros?
2. Com tokens maiores, cada posição carrega mais informação. Um `block_size` de 8
   passa a cobrir mais texto — quanto mais, aproximadamente, dada a taxa de compressão?
3. A loss fica comparável à do Capítulo 5? **Cuidado:** ela **não** é diretamente
   comparável. Explique por que uma loss por *token* com vocabulário diferente não pode
   ser comparada diretamente. (Dica: prever entre 27 opções e prever entre 512 é uma
   tarefa de dificuldade diferente.)
