# Exercícios — Capítulo 10 (Distributed)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> **Todos os exercícios rodam na CPU**, com múltiplos processos. Não é preciso ter GPU nem
> mais de uma máquina.
>
> **Se algo travar:** rode `python dist_utils.py` para diagnosticar a rede, e sempre use
> timeout ao experimentar (`timeout 60 python ...` no Linux/Git Bash). Travamento
> silencioso é o modo de falha normal aqui — veja a Seção 8 da apostila.

---

### E1 — Entendendo o coletivo (aquecimento)
Sem rodar:
1. Com 4 processos onde o rank `i` tem o valor `i`, qual o resultado do `all_reduce` com
   `SUM` em cada rank? E com `MAX`?
2. Qual a diferença entre `all_reduce` e `reduce`? E entre `all_gather` e `gather`?
3. Por que o `broadcast` **não** é suficiente para sincronizar gradientes? (Pense: quem
   teria o valor correto para transmitir?)

---

### E2 — Escalando o número de processos
Rode `python allreduce.py N` com N = 2, 4 e 8.
1. O tempo do all-reduce de 10 milhões de elementos cresce com N? Quanto?
2. Compare com o que aconteceria no esquema ingênuo (todos → rank 0 → todos): esse tempo
   cresceria **como** em função de N?
3. Sua máquina tem quantos núcleos? A partir de qual N os processos passam a competir por
   CPU em vez de somar? (Rode `python dist_utils.py` para ver.)

---

### E3 — Provoque um deadlock (importante)
No `ring_allreduce_manual`, remova a alternância par/ímpar — faça **todos** os processos
chamarem `dist.send` antes de `dist.recv`.
1. O programa trava? Sempre, ou só com tensores grandes?
2. Explique: por que com tensores **pequenos** pode funcionar por acidente? (Dica: buffers
   do sistema operacional.)
3. Esse é um bug que passa em teste pequeno e quebra em produção. Que lição isso dá sobre
   testar código distribuído?

---

### E4 — A loss que engana
No `ddp_train.py`, os ranks imprimem losses diferentes (2,18 / 2,36 / 2,17 / 2,31).
1. Por quê, se os pesos são idênticos?
2. Implemente o registro **correto** da loss: faça `all_reduce` dela e divida por
   `world_size`. Os valores passam a bater entre os ranks?
3. Se você registrasse só a loss do rank 0 num gráfico, o que aconteceria com a aparência
   da curva?
> Solução de referência: [`solucoes/e4_loss_correta.py`](solucoes/e4_loss_correta.py).

---

### E5 — Batch efetivo e learning rate
Treine o `ddp_train.py` com 1, 2 e 4 processos, mantendo o batch **por processo** fixo.
1. O batch efetivo muda. A loss após 50 passos melhora, piora ou fica igual?
2. Agora ajuste a learning rate por `√world_size` e repita. Muda a conclusão?
3. Por que "mais GPUs" pode deixar o treino **mais lento em número de passos** se você não
   mexer na learning rate?

---

### E6 — Meça o ZeRO nos três estágios
O `zero_memory.py` mede o ZeRO-1 de verdade e calcula os estágios 2 e 3 por aritmética.
1. Confirme a conta do ZeRO-1: com N processos, o estado medido é mesmo ~1/N?
2. A tabela assume AdamW em fp32. Refaça a conta para **SGD sem momentum** (que não guarda
   estado). O ZeRO ainda ajuda? Quanto?
3. E se combinássemos com o Capítulo 9 (bf16 nos pesos)? Recalcule a linha de 8 processos.

---

### E7 — Quando distribuir *não* compensa (desafio)
Junte o que você mediu neste capítulo e no Capítulo 8.
1. Estime: para o modelo pequeno do curso (153 mil parâmetros), quanto tempo levaria o
   all-reduce dos gradientes por passo? Compare com o tempo de cálculo de um passo (12,8 ms
   na CPU, do Capítulo 8).
2. Distribuir esse modelo em 4 processos aceleraria ou atrasaria o treino?
3. Formule a regra geral: qual a relação entre **tempo de cálculo** e **tempo de
   comunicação** que decide se vale a pena distribuir?
