# Gabarito — Capítulo 01

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py), que roda os
> experimentos; rode-o e compare com os seus resultados.

---

## E1 — Leitura de código

**1. Por que `vocab_size` é 27, e não 26?**
São as 26 letras `a–z` **mais** o token especial de fronteira `.`, que marca início e fim
de nome. Sem ele o modelo não saberia com que letra começar nem quando parar.

**2. O que aconteceria trocando `(N + 1)` por `N` no cálculo da loss?**
Bigramas nunca vistos teriam contagem 0, logo probabilidade 0, e `log(0) = -infinito` — a
loss viraria infinita.

Um detalhe que o `gabarito.py` mostra e que vale entender: com este dataset, avaliando
**nos mesmos dados do treino**, a loss com `+0` sai finita (2,1473). Isso porque todo
bigrama avaliado foi, por definição, visto no treino. O infinito aparece assim que você
avalia um nome **novo** que contenha um par inédito. É por isso que se suaviza mesmo
quando "parece funcionar".

**3. Para que serve o `manual_seed`?**
Fixa o gerador de números aleatórios, tornando a execução **reprodutível** — você e eu
vemos os mesmos nomes gerados. Útil para estudar e depurar; em produção você deixaria
aleatório.

---

## E2 — Mexa nos dados

Medido (`gabarito.py`):

| Dataset | Loss |
|---------|------|
| 155 nomes originais | 2,3832 |
| + 5 nomes **estranhos** (`zwqk`, `xyzabc`…) | 2,3982 (**+0,0150**) |
| + 5 nomes **típicos** (`ana`, `maria`…) | 2,3750 (**−0,0081**) |

**A resposta correta é "depende", e é isso que o exercício quer ensinar.** Não existe
"mais dados = loss menor". O que existe é: dados que **seguem o padrão** que o modelo já
capturou reduzem a loss; dados com padrões **raros** a aumentam, porque o modelo precisa
gastar probabilidade em bigramas incomuns.

Se você acrescentou nomes de amigos (portugueses, típicos), provavelmente viu a loss cair
um pouco. Se acrescentou nomes inventados esquisitos, viu subir.

---

## E3 — O efeito da suavização

| Suavização | Loss |
|-----------|------|
| `+0` | 2,1473 |
| `+1` | 2,3832 |
| `+10` | 2,8674 |
| `+100` | 3,2114 |

**1.** A loss **cresce** monotonicamente com a suavização. Suavizar afasta o modelo dos
dados observados.

**2.** Os nomes gerados ficam mais **genéricos e aleatórios** conforme a suavização
aumenta — a distribuição tende ao uniforme, e o modelo "esquece" o que aprendeu. Com `+0`
eles grudam mais nos padrões do dataset.

**3.** Ver E1.2 acima: `+0` produz probabilidade zero para pares não vistos, e o `log(0)`
quebra a conta assim que você sai dos dados de treino.

---

## E4 — Temperatura no sampling

Medido, com a mesma semente:

| T | Nomes gerados |
|---|---------------|
| 0,3 | `celia`, `lo`, `la`, `ria`, `ca` |
| 1,0 | `cexzma`, `ogosuraro`, `zktahwelo`, `imjtta` |
| 3,0 | `cexzm`, `loglkurkicczktyh`, `mvlzimjttainrlkf` |

**1.** T **baixo** → nomes mais "óbvios", curtos e repetitivos: a temperatura concentra a
massa de probabilidade nas opções mais prováveis. T **alto** → mais aleatórios e
impronunciáveis: ela achata a distribuição.

**2.** É exatamente o parâmetro `temperature` das APIs de LLM. Note a forma usada:
`p**(1/T)`. Com `T < 1` o expoente é maior que 1, o que **acentua** as diferenças entre as
probabilidades; com `T > 1` ele as **achata**. Nos modelos modernos a mesma operação é
aplicada aos *logits* antes do softmax (dividir os logits por T), o que é matematicamente
equivalente.

---

## E5 — Igualando contagem e rede neural

| Configuração | Loss |
|--------------|------|
| Contagem `+1` | 2,3832 |
| Contagem `+0,01` | **2,1518** |
| Rede, `reg=0.01`, 200 passos | 2,1774 |
| Rede, `reg=0.0`, 1000 passos | **2,1600** |

**1 e 2.** Com a suavização **igualada** dos dois lados, os valores convergem para a mesma
faixa (~2,15–2,18).

**3.** É a confirmação da tese do capítulo: contar e otimizar chegam ao **mesmo modelo**. A
diferença que se via na apostila (2,38 contra 2,21) vinha só do fato de que o `+1` é uma
suavização **muito mais forte** que o `reg=0.01`. São o mesmo botão, em escalas diferentes.

---

## E6 — NLL de um nome específico

Solução em [`e6_nll_de_nome.py`](e6_nll_de_nome.py). Resultados:

```
ana          -> NLL = 1.7871
maria        -> NLL = 1.7626
vinicius     -> NLL = 2.4613
xwq          -> NLL = 3.6696
kffkz        -> NLL = 3.4179
```

**1.** `ana` tem NLL bem menor que `xwq` — o modelo considera `ana` muito mais provável.

**2.** Nomes com bigramas comuns em português (`ma`, `ri`, `an`) pontuam bem. Nomes com
sequências raras (`xw`, `kz`) pontuam mal. `vinicius` fica no meio: tem `vi` e `ni`
comuns, mas `ci` seguido de `us` é menos frequente no dataset.

---

## E7 — Rumo ao trigrama

| Modelo | Suavização | Loss |
|--------|-----------|------|
| Bigrama | `+0,01` | 2,1518 |
| **Trigrama** | `+1,0` | 2,3449 |
| **Trigrama** | `+0,01` | **1,1523** |

**1.** A tabela passa de `27×27 = 729` para `27×27×27 = **19.683**` células.

**2.** Com suavização comparável (`+0,01`), a loss **melhora muito**: 2,15 → 1,15. Mais
contexto ajuda de verdade.

> **Atenção ao efeito da suavização aqui:** com `+1`, o trigrama fica *pior* que o bigrama
> (2,34 contra 2,15). Isso não é o trigrama sendo ruim — é que somar 1 a **19.683** células
> injeta muito mais "massa uniforme" do que somar 1 a 729. Quanto maior a tabela, mais
> destrutiva é a mesma suavização. Comparar modelos com suavização fixa é comparar coisas
> diferentes.

**3.** O problema novo é a **esparsidade**: a maioria das células nunca é preenchida. A
suavização evita o `log(0)`, mas **não inventa informação** — para combinações nunca vistas
o modelo apenas chuta uniformemente. E isso piora exponencialmente: com `k` caracteres de
contexto são `27^(k+1)` células.

É exatamente essa parede que motiva o **Capítulo 3**: em vez de contar células, o MLP
aprende uma representação **densa e compartilhada**, onde contextos parecidos se apoiam
mutuamente.
