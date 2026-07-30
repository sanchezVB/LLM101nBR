# Gabarito — Capítulo 05

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> **Orçamento:** 2.500 passos por configuração (a apostila usa 15.000).

---

## E1 — Leitura de código

**1. Por que `x = x + self.sa(self.ln1(x))` e não `x = self.sa(self.ln1(x))`?**
A soma é a **conexão residual**. Ela muda duas coisas:

- **na ida**: a sub-camada não precisa produzir a representação inteira — só uma
  **correção** ao que já estava lá. Aprender um ajuste é mais fácil que reconstruir tudo.
- **na volta**: a derivada de uma soma **distribui** o gradiente igualmente. O `x +` cria
  um caminho pelo qual o gradiente flui até as primeiras camadas **sem ser multiplicado
  por nada**.

**2. Por que `head_size = n_embd // n_head`?**
Para que a concatenação das cabeças volte a ter `n_embd`. Com `n_embd = 64` e
`n_head = 4`, são 4 cabeças de 16 que concatenam de volta em 64 — **quatro cabeças
pequenas custam o mesmo que uma grande**. Se cada cabeça tivesse `head_size = n_embd`, o
custo seria `n_head` vezes maior.

**3. Atenção vs feedforward:**
A **atenção comunica** — move informação entre posições. O **feedforward computa** —
processa cada posição isoladamente, sem trocar nada. Um bloco faz as duas coisas: as
posições conversam, depois cada uma pensa sozinha sobre o que ouviu.

---

## E2 — Sem as conexões residuais

Solução em [`e2_ablacoes.py`](e2_ablacoes.py):

| Configuração | Loss validação |
|--------------|----------------|
| 3 blocos, residual **ON** | 1,9029 |
| 3 blocos, residual **OFF** | 2,1971 (**+0,294**) |
| 6 blocos, residual **ON** | 1,8865 |
| 6 blocos, residual **OFF** | 2,7728 (**+0,886**) |

A penalidade **triplica** ao dobrar a profundidade. E o mais revelador: **sem** residuais,
ir de 3 para 6 blocos **piora** o modelo (2,197 → 2,773); **com** residuais, melhora
(1,903 → 1,887).

> Sem conexões residuais, profundidade atrapalha. Com elas, ajuda.

---

## E3 — Sem LayerNorm

| Blocos | `lr` | LayerNorm | Validação |
|--------|------|-----------|-----------|
| 3 | 1e-3 | ON | 1,9365 |
| 3 | 1e-3 | **OFF** | **1,9318** |
| 3 | 3e-3 | **ON** | **1,9504** |
| 3 | 3e-3 | OFF | 2,0104 |
| 8 | 1e-3 | ON | 1,9228 |
| 8 | 1e-3 | **OFF** | **1,9181** |
| 8 | 3e-3 | **ON** | **1,9709** |
| 8 | 3e-3 | OFF | **DIVERGIU** |

**Penalidade de remover a LayerNorm:**

| | `lr` = 1e-3 | `lr` = 3e-3 |
|---|---|---|
| 3 blocos | **−0,0047** | +0,0600 |
| 8 blocos | **−0,0047** | **divergiu** |

**1 e 2. A resposta não é "LayerNorm sempre ajuda".** Leia a tabela de penalidades:

- Com `lr = 1e-3`, remover a LayerNorm dá **−0,0047 nas duas profundidades** — ela é
  dispensável, e até marginalmente prejudicial, tanto com 3 quanto com 8 blocos.
- Com `lr = 3e-3`, remover custa +0,06 com 3 blocos e **destrói o treino** com 8.

**Não é a profundidade que cria a necessidade — é a learning rate.** A profundidade
**amplifica a falha**: o mesmo `lr` que apenas degrada um modelo de 3 blocos aniquila um de
8.

**3.** Faz sentido pela teoria da Seção 4 da apostila: a LayerNorm impede que o erro de
escala se acumule. Com passos pequenos as ativações não saem de escala, e não há o que
corrigir. Com passos grandes elas saem — e aí o número de camadas decide se o estrago é
recuperável.

> **A lição, que se repete no curso:** uma peça pode ser **correta e necessária em geral**,
> mas dispensável na configuração específica que você está testando. Isso não a torna
> inútil — torna o seu teste insuficiente para julgá-la.

---

## E4 — Profundidade e número de cabeças

| `n_layer` | `n_head` | Parâmetros | Validação |
|-----------|----------|-----------|-----------|
| 1 | 4 | 53.915 | 1,9818 |
| 3 | 4 | 153.499 | 1,9365 |
| 6 | 4 | 302.875 | *(ver saída)* |

| `n_layer` | `n_head` | Parâmetros | Validação |
|-----------|----------|-----------|-----------|
| 3 | 1 | **153.499** | 1,9618 |
| 3 | 4 | **153.499** | 1,9365 |
| 3 | 8 | **153.499** | 1,9227 |

**1 e 2.** Mais profundidade ajuda, com retorno decrescente.

**3. Mudar `n_head` não muda o número de parâmetros** — repare que as três linhas da
segunda tabela têm exatamente **153.499**. Como `head_size = n_embd // n_head`, o total de
pesos das cabeças é constante. O que muda é **como esses pesos são organizados**: quantos
softmaxes independentes existem.

---

## E5 — Uma cabeça grande vs várias pequenas

| `n_head` | `head_size` | Parâmetros | Validação |
|----------|-------------|-----------|-----------|
| 1 | 64 | 153.499 | 1,9618 |
| 8 | 8 | 153.499 | 1,9227 |
| **64** | **1** | 153.499 | **1,9137** |

**1 e 2.** Várias cabeças pequenas ganham, e com **exatamente os mesmos parâmetros**. A
melhora vem só da organização: cada cabeça se especializa numa relação diferente, e a
projeção combina o que todas trouxeram.

**3. Esta resposta não saiu como eu previa.**

Eu esperava um limite: com `n_head = 64` cada cabeça tem dimensão 1, e um produto escalar
de vetores de dimensão 1 é só a multiplicação de dois números — pareceria pobre demais para
expressar afinidade.

**A medição diz o contrário: 64 cabeças foi o melhor dos três.**

Por que o limite não aparece aqui? Porque a tarefa é simples (nomes, contexto 8). Sessenta
e quatro padrões de atenção escalares, combinados pela projeção, dão conta. O custo de cada
cabeça ser pobre é compensado pela quantidade.

Na prática os modelos usam `head_size` entre 32 e 128 — mas por razões que este experimento
**não exercita**: tarefas complexas onde cada cabeça precisa comparar representações ricas,
e eficiência de hardware (cabeças muito pequenas geram matmuls pequenas, que a GPU odeia —
ver Capítulo 8).

> **Vale saber a regra prática E saber por que ela existe.** Uma regra da literatura pode
> não se reproduzir na sua escala.

---

## E6 — Dropout

| Dropout | Treino | Validação | Gap |
|---------|--------|-----------|-----|
| **0,0** | 1,9438 | **1,9365** | −0,0073 |
| 0,1 | 1,9517 | 1,9448 | −0,0069 |
| 0,3 | 1,9774 | 1,9681 | −0,0093 |

**1. O dropout atrapalha aqui**, e de forma monotônica. Com 64 mil nomes, treino e
validação já andam juntos — o gap é praticamente nulo. Sem overfitting para combater, o
dropout só remove capacidade.

É exatamente a mesma lição do weight decay no Capítulo 7, e o padrão é o mesmo: **treino e
validação pioram juntos, e o gap não muda**. Regularização que não está resolvendo nada.

**2.** Ele ajudaria com **poucos dados** — releia o E5 do Capítulo 3 (155 nomes: treino
0,80 contra validação 6,51).

**3. Por que desligar na avaliação?** Porque com dropout ativo você estaria medindo um
modelo **aleatoriamente mutilado** — a loss viria pior e ruidosa, sem significar nada. O
`model.eval()` desliga o dropout (e ajusta a BatchNorm, quando existe).

---

## E7 — Contando os parâmetros

Com `n_embd = 64`, `n_head = 4`, `n_layer = 3`, `block_size = 8`, `vocab = 27`:

| Componente | Total |
|------------|-------|
| `token_emb` (27 × 64) | 1.728 |
| `pos_emb` (8 × 64) | 512 |
| **por bloco** (qkv + proj + ff + 2 LayerNorm) | 49.792 |
| × 3 blocos | 149.376 |
| LayerNorm final | 128 |
| `lm_head` (64 × 27 + 27) | 1.755 |
| **Total** | **153.499** ✓ |

**1 e 2.** A fórmula reproduz exatamente o número que o código informa.

**3. Quem domina:** os blocos somam 149.376 de 153.499 — **97%**. E dentro de cada bloco, o
**feedforward** é o maior pedaço, por causa da expansão 4×.

Consequências para escalar:
- Dobrar `n_embd` **quadruplica** o custo dos blocos (todas as matrizes são `ne × ne` ou
  `ne × 4ne`).
- Dobrar `n_layer` apenas **dobra**.

É por isso que modelos grandes crescem mais em profundidade do que em largura — e por que
`n_embd` é a decisão de arquitetura mais cara.
