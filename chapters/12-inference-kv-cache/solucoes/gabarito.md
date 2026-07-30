# Gabarito — Capítulo 12

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).
>
> ```bash
> python solucoes/gabarito.py
> ```
>
> Os exercícios E2 e E3 introduzem os bugs **de propósito**, em subclasses, para que você
> veja o modo de falha sem editar o modelo bom.

---

## E1 — O que dá e o que não dá para cachear

**1. Por que `k` e `v` são cacheáveis e `q` não é.**

`k` e `v` da posição *t* dependem só do token em *t* e da sua posição — e nada disso muda
quando um token novo chega em *t+1*. São propriedades **do passado**, e o passado está
fechado.

`q` é diferente por natureza: a cada passo há um token novo fazendo uma pergunta nova ao
histórico. A query é a única coisa que muda de verdade a cada iteração. Cachear query não
faria sentido — ela é usada uma vez e descartada.

**2. Sem máscara causal, o cache continuaria válido? Não.**

Com atenção bidirecional, a representação da posição 3 depende do token 7. Quando o token
8 chega, **todas** as representações anteriores mudam — o `k` e o `v` guardados ficam
obsoletos.

É uma observação que vale guardar: **o KV-cache só existe porque o modelo é causal.**
Modelos tipo BERT não têm KV-cache, e não é falta de otimização — é que a otimização não
existe para eles. A restrição que parecia limitação (só olhar para trás) é o que torna a
geração eficiente.

**3. As contas.**

| Caminho | Posições processadas |
|---|---|
| Ingênuo | 1 + 2 + … + 128 = **8.256** |
| Com cache | **128** (uma por token) |

Descartado no caminho ingênuo: **1 − 128/8.256 = 98,4%**.

---

## E2 — Quebrando a posição de propósito

Trocando a posição absoluta por `torch.arange(T)`:

| Verificação | Resultado |
|---|---|
| Levantou exceção? | **não** |
| Gerou texto? | **sim**, 81 tokens |
| Primeira divergência | token **4** |

```
referência : 'de vocês e vinham a vida.\n\n--Papae não ira a Deus que a convidasse...'
com o bug  : 'de vocês\nde lerguiamos justa e uma dez do mariamente o a um homem...'
```

**1.** Nenhuma exceção, nenhum `NaN`, nenhum aviso. O texto sai com a mesma aparência de
sempre. É o pior tipo de bug: ele não avisa.

**2. Por que a divergência começa no token 4, e não no 1.**

Porque o **prefill** processa o prompt inteiro de uma vez, e ali `T == T_total` — então
`arange(T)` por acaso dá as posições certas. O bug só se manifesta na fase de **decode**,
quando passamos um token só e ele acha que está na posição 0.

Isso é instrutivo por si: o bug se esconde exatamente na fase que o teste rápido (gerar um
tokenzinho e ver se sai algo) não exercita.

**3. Conseguiria dizer qual está errado sem a referência? Não.**

Os dois textos são igualmente ruins a olho nu — porque um modelo de 2,2 M parâmetros já
escreve mal quando está **certo**. A degradação se esconde no ruído de base.

> Numa otimização de inferência, **"parece bom" não é evidência de nada.** É por isso que
> o critério do capítulo é *saída idêntica*.

---

## E3 — Quebrando a máscara de propósito

Aplicando a máscara também no decode:

| Verificação | Resultado |
|---|---|
| Primeira divergência | token **3** |
| Tokens distintos na saída | **54 de 80** |

**1. Eu previ repetição. Errado.**

Ao escrever este gabarito eu afirmei que o texto "degenera em repetição, porque todo token
passa a ser gerado a partir do mesmo contexto". A contagem desmente: 54 tokens distintos
em 80. A amostragem continua estocástica — o que se perdeu foi o **contexto**, não a
diversidade.

### O que o modelo virou — medido, não suposto

O teste certo não é olhar o texto. É perguntar: *de que a previsão ainda depende?* Duas
histórias com o **mesmo primeiro token** e 19 tokens do meio completamente diferentes:

| Modelo | Logits idênticos? | Diferença máxima |
|---|---|---|
| com o bug | **sim** | **0,00** |
| correto | não | 6,22 |

Bit a bit idênticos. O modelo passou a prever a partir de `(primeiro token, token atual,
posição)` e **mais nada**.

> Ele virou, literalmente, o modelo do [Capítulo 1](../../01-bigram-language-model/): um
> **bigrama**. Toda a maquinaria de atenção continua rodando, consumindo tempo, e não
> transporta informação nenhuma.

**2. Por que só o primeiro token sobrevive.**

No decode `T == 1`, então `tril[:1, :T_total]` seleciona a **primeira** linha da matriz
triangular: `[1, 0, 0, …, 0]`. Ela permite olhar apenas a posição 0 — daí o primeiro token
continuar influenciando enquanto todo o resto desaparece.

**3. Eu também previ que este bug seria mais fácil de notar. Errado de novo.**

O raciocínio era que ele "falha de forma escandalosa". Compare os dois textos: são
igualmente ruins.

> **A lição dos dois exercícios juntos.** Em modelo pequeno, nenhum dos dois bugs se
> denuncia pela leitura. Um deles rebaixa um Transformer a um bigrama e **você não percebe
> olhando**. O teste de equivalência não é burocracia — é o único instrumento que
> funciona.

E note que em modelo grande a situação é pior, não melhor: um GPT de verdade com esses
bugs ainda escreveria português fluente. A fluência vem da camada de saída; a coerência é
que vem do contexto.

---

## E4 — Meça o ganho

Rode [`benchmark_cache.py`](../benchmark_cache.py) com a máquina ociosa.

**1.** O **ms por token** do caminho ingênuo **cresce** com o comprimento — cada token novo
reprocessa um contexto maior. O do caminho com cache fica praticamente **constante**. É o
ponto todo do capítulo.

**2.** O speedup **cresce** com o comprimento, porque o custo do caminho ingênuo cresce e o
do cache não. Ele para de crescer quando o contexto satura em `block_size` (128 aqui):
daí em diante o ingênuo também passa a ter custo constante, só que alto.

**3. Para um único token, o cache não ajuda** — é exatamente um prefill nos dois casos. O
cache começa a pagar a partir do segundo token gerado. Não há comprimento em que ele
*atrapalhe*: o custo extra é a alocação da memória do cache, que não aparece no relógio
neste tamanho.

---

## E5 — A conta de memória

**1. A fórmula**, conferida contra `bytes_do_cache()`:

```
2 (K e V) × n_layer × batch × n_head × T × head_size × bytes
```

Para o modelo do curso com `T=128`: **786.432 bytes** (768 KB) — manual e método batem.

**2. Um 7B em bf16, contexto 8.192:**

| Grandeza | Valor |
|---|---|
| Cache por usuário | **4,3 GB** |
| Placa de 80 GB − 14 GB de pesos | 66 GB livres |
| **Usuários simultâneos** | **15** |

Quinze. Uma placa de topo de linha, um modelo que cabe folgado nela, e quinze conversas ao
mesmo tempo.

**3. A reescrita que muda a leitura.**

Como `n_head × head_size = n_embd`:

```
2 × n_layer × batch × T × n_embd × bytes
```

**O cache não depende de quantas cabeças você usa, nem do tamanho delas** — depende da
**largura** e da **profundidade** do modelo. Redistribuir `n_embd` entre mais ou menos
cabeças não muda nada no custo de servir.

O que muda é **quebrar a simetria**: se as cabeças de Q forem muitas e as de K/V forem
poucas, o produto `n_head_kv × head_size` fica menor que `n_embd`. É precisamente isso que
MQA e GQA fazem — e é por isso que elas são a intervenção óbvia nesta fórmula.

---

## E6 — Cache com batch

| Modo | Tempo |
|---|---|
| 16 sequências de 64 tokens **em paralelo** | **0,53 s** |
| 16 sequências de 64 tokens **separadas** | 4,78 s |
| **ganho** | **9,1x** |

**1. Nada muda no código.** A dimensão de batch já está na forma do cache
`(B, n_head, T, hs)` e todas as operações são em lote — bastou passar um prompt com B
linhas. Que a mudança seja nula é um bom sinal sobre a implementação.

**2.** O ganho é grande pela razão do [Capítulo 8](../../08-device/solucoes/gabarito.md): o
decode é limitado por **memória**, não por cálculo. Ler os pesos custa o mesmo para 1 ou
para 16 sequências, então o custo se dilui.

> É por isso que serviço de inferência agrupa requisições de **usuários diferentes** no
> mesmo batch. Não é para economizar — é que a alternativa desperdiça quase tudo.

**3. Comprimentos diferentes são o problema de verdade.** Se uma sequência termina no token
10 e outra no 200, o batch inteiro roda até a mais longa acabar: as posições já terminadas
gastam cálculo à toa e o cache delas continua ocupando memória.

| Estratégia | Como funciona |
|---|---|
| Preenchimento com máscara | simples e desperdiçador |
| **Continuous batching** | retira a sequência pronta e põe outra da fila no lugar |
| **Paged attention** (vLLM) | aloca o cache em páginas, sem reservar o pior caso por sequência |

---

## E7 — Multi-query attention (desafio)

MQA improvisada sobre os pesos treinados (média das 6 cabeças de K/V):

| Configuração | Loss de validação | Cache (T=128) |
|---|---|---|
| Multi-head original | **3,9357** | 768 KB |
| MQA improvisada | **5,2697** (**+1,3339**) | **128 KB** (6x menor) |

**1 e 2.** O cache cai por um fator de `n_head` — 6x aqui, **32x** num 7B. É a maior
economia disponível nesta fórmula.

**3. E a loss desaba.** +1,33 de loss é um modelo destruído, não degradado.

Não há surpresa: o modelo foi **treinado** com 6 cabeças de K/V independentes, e cada
cabeça de Q aprendeu a perguntar para a **sua**. Substituir as seis por uma média destrói
essa correspondência.

> **MQA não é uma otimização que se liga na hora de servir. É uma decisão de
> arquitetura.** O modelo precisa nascer assim (ou passar por *uptraining* com dados).
> Llama 2 70B e Mistral usam GQA porque foram projetados com ela.

A lição geral vale além deste exercício, e fecha o capítulo:

> "Economiza memória" e "dá para aplicar depois" são afirmações **independentes**. O
> KV-cache é gratuito porque **não muda a conta** — é a mesma matemática, reorganizada. A
> MQA não é, porque muda. Toda vez que alguém oferecer uma otimização, a primeira
> pergunta é: *isso altera o resultado?*
