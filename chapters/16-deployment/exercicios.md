# Exercícios — Capítulo 16 (Deployment)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Estes exercícios precisam de **dois terminais**: um com o servidor de pé, outro rodando
> o cliente. Suba o servidor com `python servidor.py` e meça com `python carga.py`.

---

### E1 — TTFT e latência (aquecimento)
Sem rodar:
1. O `/gerar` e o `/gerar/stream` produzem o mesmo texto e levam o mesmo tempo total. Então
   o que exatamente o streaming melhora? Descreva em termos do que o usuário observa.
2. Por que o streaming é possível num LLM e não seria, digamos, num classificador de
   imagens que devolve uma etiqueta?
3. O servidor chama `self.wfile.flush()` depois de cada token. O que aconteceria sem essa
   linha, e por que o código continuaria "parecendo" streaming?

---

### E2 — Meça o seu próprio TTFT (importante)
1. Rode o `carga.py` e anote TTFT e total nos dois endpoints. Qual a razão entre eles?
2. Agora peça **200 tokens** em vez de 40 (`TOKENS = 200` no `carga.py`). O que acontece
   com a razão TTFT/total nos dois casos? Explique a tendência.
3. Extrapole: para um modelo que gera 30 tokens/s e uma resposta de 500 tokens, quanto
   seria o TTFT com e sem streaming? Isso muda a sua opinião sobre a magnitude medida no
   capítulo?

---

### E3 — Por que threads não bastam (importante)
1. Rode `carga.py` contra o `servidor.py` e monte a tabela de vazão por número de clientes.
   A vazão cresce proporcionalmente? Quanto?
2. Remova a trava (`TRAVA`) do `servidor.py` e meça de novo. A vazão melhorou? (Cuidado: o
   resultado pode surpreender — meça antes de opinar.)
3. Explique, usando o que o Capítulo 12 estabeleceu sobre o decode, por que threads
   independentes não resolvem — e por que agrupar resolve.

---

### E4 — O tamanho do lote
No `servidor_batch.py`, o `LOTE_MAX` é 16.
1. Meça a vazão com `--lote 1`, `--lote 4` e `--lote 16`, sempre com 8 clientes. Como ela
   varia?
2. `--lote 1` deveria reproduzir o comportamento do servidor sem batching. Reproduz? Se
   não, o que mais mudou entre os dois servidores?
3. A partir de que tamanho de lote o ganho satura? O que limita — e a resposta é a mesma
   do E4 do Capítulo 8?

---

### E5 — A espera que custou caro
O capítulo relata um bug: esperar 8 ms por novos pedidos **a cada passo** de decode.
1. Reintroduza a espera e meça. Quanto ela custa por requisição de 40 tokens? A conta
   bate com `40 × 8 ms`?
2. Existe algum cenário em que essa espera **ajudaria**? (Dica: pense em requisições que
   chegam espaçadas em vez de todas juntas.)
3. Formule a regra geral que a correção segue, e dê um exemplo de outro sistema onde ela
   se aplica.

---

### E6 — O p95 conta o que a média esconde
1. Compare mediana e p95 nos dois servidores, com 8 clientes. Por que eles são diferentes
   num caso e iguais no outro?
2. Um serviço tem latência média de 200 ms e p99 de 4 s. Descreva o que o usuário
   experimenta — e por que reportar só a média seria enganoso.
3. Que outra métrica de cauda você acompanharia num serviço de LLM, além do p95 de
   latência? (Pense no que mais pode ter cauda longa.)

---

### E7 — Backpressure (desafio)
O servidor deste capítulo aceita requisições sem limite: a fila cresce indefinidamente.
1. Meça o que acontece com 64 clientes simultâneos. A latência ainda é utilizável?
2. Implemente um limite: se a fila passar de N pedidos, responda **HTTP 429** em vez de
   enfileirar. Escolha o N a partir da sua medição do item 1.
3. Argumente: por que recusar uma requisição pode ser **melhor** que atendê-la devagar?
   Pense do ponto de vista do cliente que já está sendo atendido.
