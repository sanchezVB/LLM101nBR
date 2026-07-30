# Gabarito — Capítulo 04

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> **Orçamento:** 4.000 passos por configuração (a apostila usa 20.000). Valores absolutos
> ficam piores; as comparações continuam válidas.

---

## E1 — Leitura de código

**1. O que a `tril` impede, e o que aconteceria sem ela no treino?**
Ela impede que uma posição veja o **futuro**. Sem ela, ao prever o token seguinte a
posição poderia consultar justamente a resposta — a loss de treino despencaria e o modelo
não aprenderia nada útil, porque na hora de gerar texto o futuro não existe. É *data
leakage*.

(O exercício E2 mostra que isso tem uma sutileza importante — veja abaixo.)

**2. Por que dividir por `√head_size`?**
O produto escalar `q · k` é a soma de `head_size` produtos, então a sua magnitude cresce
com a dimensão. Pontuações grandes fazem o softmax **saturar**: a distribuição vira quase
um one-hot e o gradiente morre. Dividir por `√head_size` mantém a variância em torno de 1
independentemente do tamanho da cabeça.

**3. Query, key e value:**
- **query**: "o que eu estou procurando?"
- **key**: "o que eu tenho a oferecer?" (o endereço pelo qual sou encontrado)
- **value**: "o que eu de fato entrego, se você me escolher" (o conteúdo)

Separar *key* de *value* dá liberdade ao modelo: como sou encontrado e o que eu entrego
são coisas diferentes.

---

## E2 — Sem a máscara causal

**Este exercício tem uma armadilha que vale mais que a pergunta.**

| Configuração | Treino | Validação |
|--------------|--------|-----------|
| 1 camada, máscara **ON** | 2,1589 | 2,1517 |
| 1 camada, máscara **OFF** | 2,1589 | **2,1517** |
| 2 camadas, máscara **ON** | 2,2238 | 2,2122 |
| 2 camadas, máscara **OFF** | 2,1614 | **2,1513** |

**Com uma camada, remover a máscara não muda absolutamente nada.** Os números são
idênticos até a quarta casa. Não é erro de medição.

O motivo é estrutural: este modelo usa **apenas a última posição** para prever, e a última
linha da matriz triangular **já é toda de uns**. Para a posição que importa, a máscara
nunca fez diferença — ela só afetava posições intermediárias, cujas saídas são
**descartadas**.

**Com duas camadas a história muda.** A saída das posições intermediárias da camada 1
alimenta a camada 2, e a última posição da camada 2 as lê. Aí o futuro realmente vaza, e a
loss sem máscara fica artificialmente melhor (2,1513 contra 2,2122).

**E onde a máscara é absolutamente essencial:** quando o modelo prevê em **todas** as
posições (Capítulo 11). Ali a posição `t` veria literalmente o token `t+1` que deve prever
— vazamento direto e devastador.

> **A lição:** uma proteção pode estar **correta** e ainda assim ser **inócua** no desenho
> atual. Removê-la e não ver diferença **não prova que é inútil** — prova que o seu teste
> não exercita o caminho que ela protege. É exatamente assim que proteções de segurança
> são removidas "porque não faziam nada".

---

## E3 — Sem o embedding posicional

| | Treino | Validação |
|---|--------|-----------|
| **com** posicional | 2,1589 | 2,1517 |
| **sem** posicional | 2,2075 | 2,1959 |
| | | **+0,0442** |

**1 e 2.** A loss piora. A atenção é **invariante a permutações**: ela calcula afinidades
entre pares de posições, mas nada no mecanismo diz **qual veio antes**. Sem o embedding
posicional, `ana` e `naa` viram o mesmo conjunto de tokens, e o modelo perde a capacidade
de usar a **ordem** — que em linguagem é quase tudo.

A penalidade de 0,044 pode parecer modesta, mas note: para *nomes*, boa parte do sinal está
em *quais* letras aparecem (estatística de caracteres), não só na ordem. Em prosa, o mesmo
experimento seria muito mais destrutivo.

---

## E4 — Sem a escala `1/√head_size`

| | Treino | Validação | Peso máximo médio de atenção |
|---|--------|-----------|------------------------------|
| **com** escala | 2,1589 | 2,1517 | **0,406** |
| **sem** escala | 2,3508 | 2,3428 | **0,985** |

**1 e 2.** Sem a escala a loss piora bastante (+0,19), e a causa aparece na terceira
coluna: o peso máximo por linha sobe de 0,41 para **0,985**. A distribuição de atenção
virou praticamente um **one-hot** — o modelo escolhe uma posição e ignora todas as outras.

**3.** Softmax saturado tem gradiente quase nulo. É exatamente o mesmo efeito da `tanh`
saturada do Capítulo 2: o modelo trava numa escolha e perde a capacidade de ajustá-la. A
escala não é um detalhe de normalização — é o que mantém o mecanismo **treinável**.

---

## E5 — Tamanho do contexto

| `block_size` | Parâmetros | Treino | Validação |
|--------------|-----------|--------|-----------|
| 3 | 11.103 | 2,2194 | 2,2110 |
| **8** | 11.363 | 2,1589 | **2,1517** |
| 16 | 11.779 | 2,2122 | 2,1988 |

**1. A resposta não é "mais contexto é sempre melhor".** De 3 para 8 melhora; de 8 para 16
**piora**. Não é retorno decrescente — é retorno **negativo**.

Por quê? Nomes têm ~7 letras. Com `block_size = 16`, a imensa maioria das posições do
contexto é **preenchimento** (`.`), e o modelo gasta capacidade processando padding em vez
de sinal.

**2.** Somando a isso, os embeddings posicionais das posições distantes quase nunca são
exercitados: recebem pouco gradiente, permanecem próximos da inicialização aleatória e
injetam ruído.

> **A lição:** o contexto deve ser dimensionado pelo **tamanho real do dado**. Contexto
> grande demais não é neutro — custa computação **e** qualidade. (No Capítulo 11, com
> prosa, `block_size = 128` faz sentido justamente porque o texto de verdade é longo.)

**3. Dois custos que crescem de formas diferentes:**
- **computação**: cresce com **T²** (a matriz de afinidades é `T × T`)
- **parâmetros**: cresce só com o embedding posicional — de 11.103 para 11.779, quase nada

É o **oposto** do MLP do Capítulo 3, onde os parâmetros cresciam linearmente com o
contexto. A atenção troca custo de parâmetros por custo de computação.

---

## E6 — Inspecionando os pesos

Solução em [`e6_inspecionar_atencao.py`](e6_inspecionar_atencao.py).

O achado principal: boa parte do peso cai nos tokens de **preenchimento** (`.`), não nas
letras. Isso **não** é bug — o softmax obriga os pesos a somarem 1, então quando o modelo
não precisa trazer informação de nenhuma posição específica, ele "estaciona" o peso numa
posição inofensiva. O fenômeno tem nome na literatura de LLMs: ***attention sink***.

---

## E7 — Multi-head attention

| Cabeças | `head_size` | Parâmetros | Treino | Validação |
|---------|-------------|-----------|--------|-----------|
| 1 | 52 | 11.363 | 2,1589 | 2,1517 |
| 2 | 26 | 14.119 | 2,0752 | 2,0643 |
| **4** | **13** | 14.119 | 2,0463 | **2,0367** |

**1. Várias cabeças pequenas ganham.** De 1 para 4 cabeças a validação melhora 0,115 — um
ganho grande para este capítulo.

Note que 2 e 4 cabeças têm **exatamente os mesmos 14.119 parâmetros**: `head_size =
n_embd // n_head`, então o total de pesos das cabeças é constante. A melhoria de 2 para 4
vem **só da organização**, não de capacidade extra.

> A versão de 1 cabeça tem menos parâmetros (11.363) porque não tem a projeção de saída.
> A comparação limpa é entre 2 e 4 cabeças.

**2. Por que várias ajudam?** Uma cabeça aprende **um** critério de "onde olhar". Várias
aprendem vários — uma pode seguir a vogal anterior, outra o começo da palavra, outra a
última consoante. Como as saídas são concatenadas e projetadas, o modelo combina esses
critérios.

**3.** É exatamente a peça que abre o Capítulo 5.
