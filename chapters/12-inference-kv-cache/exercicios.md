# Exercícios — Capítulo 12 (KV-cache)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Estes exercícios usam o checkpoint do Capítulo 11 (`../11-datasets/modelo.pt`). Se ele
> não existir, rode antes `cd ../11-datasets && python train_text.py`.

---

### E1 — O que dá e o que não dá para cachear (aquecimento)
Sem rodar:
1. Por que `k` e `v` de uma posição não mudam quando um token novo chega, mas `q` precisa
   ser recalculado a cada passo?
2. Se o modelo **não** tivesse máscara causal (atenção bidirecional, como no BERT), o
   KV-cache continuaria válido? Justifique.
3. Gerar 128 tokens com contexto 128 processa quantas posições no caminho ingênuo? E no
   caminho com cache? Qual a fração de trabalho descartado no primeiro?

---

### E2 — Quebre a posição de propósito (importante)
No `forward` do `modelo.py`, troque o cálculo da posição por `torch.arange(T)` — o bug
descrito na Seção 4 da apostila.
1. O programa levanta alguma exceção? O texto continua saindo?
2. Compare com a saída de referência (`gerar_ingenuo`). A partir de qual token elas
   divergem, e por quê **daquele** token em diante?
3. Leia os dois textos. Se você não tivesse a referência para comparar, conseguiria dizer
   qual está errado? O que isso diz sobre depurar otimizações de inferência?

---

### E3 — Quebre a máscara de propósito (importante)
Agora aplique a máscara também no modo decode: troque a condição por
`w = w.masked_fill(self.tril[:T, :T_total] == 0, float("-inf"))`, sem a guarda
`if T == T_total`.
1. O que acontece com o texto? Descreva o padrão que aparece.
2. Explique a partir da matriz triangular: qual linha `tril[:1, :T_total]` seleciona, e o
   que ela permite o token enxergar?
3. Este bug é mais fácil ou mais difícil de notar que o do E2? Por quê?

---

### E4 — Meça o ganho
Rode o `benchmark_cache.py` com a máquina ociosa.
1. Como o **ms por token** evolui com o comprimento, nos dois caminhos? Faça o gráfico
   mentalmente antes de olhar a tabela.
2. O speedup cresce, satura ou cai conforme o texto fica mais longo? Por quê?
3. Para **um único** token gerado, o cache ajuda? Existe comprimento em que ele atrapalha?

---

### E5 — A conta de memória
1. Escreva a fórmula do tamanho do cache em função de `n_layer`, `n_head`, `head_size`,
   `T`, `batch` e bytes por número. Confira contra `bytes_do_cache()`.
2. Para um modelo de 7B (32 camadas, 32 cabeças, `head_size` 128) em bf16, com contexto
   8.192: quanto ocupa o cache por usuário? E quantos usuários simultâneos cabem numa
   placa de 80 GB, descontando os ~14 GB dos pesos?
3. O cache cresce com `n_layer × n_head × head_size`. Note que `n_head × head_size =
   n_embd`. Reescreva a fórmula em função de `n_embd` — o que isso diz sobre qual decisão
   de arquitetura controla o custo de servir o modelo?

---

### E6 — Cache com batch
O `gerar_com_cache` gera uma sequência por vez. Faça-o gerar `B` sequências em paralelo.
1. O que muda na forma do cache? E no código?
2. Meça: gerar 32 sequências de 64 tokens em paralelo custa quanto, comparado a 32
   gerações separadas?
3. Onde está o problema quando as sequências têm **comprimentos diferentes** (umas
   terminam antes)? Descreva pelo menos uma estratégia para lidar com isso.

---

### E7 — Multi-query attention (desafio)
A MQA usa **uma só** cabeça de K/V, compartilhada por todas as cabeças de Q. O cache cai
por um fator de `n_head`.
1. Implemente. Dica: `k` e `v` passam a ter forma `(B, 1, T, hs)`, e o *broadcast* faz o
   resto na multiplicação com `q`.
2. Quanto o cache encolhe no modelo deste curso? E num 7B?
3. **Cuidado com a conclusão.** Você pode implementar a MQA sobre os pesos já treinados,
   mas o modelo foi treinado com 6 cabeças de K/V independentes. Medindo a loss antes e
   depois, o que você espera? O que isso diz sobre quando a MQA precisa entrar no
   projeto?
