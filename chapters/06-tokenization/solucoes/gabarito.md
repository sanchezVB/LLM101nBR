# Gabarito — Capítulo 06

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py) (roda em ~2
> minutos, sem treinar rede nenhuma).

---

## E1 — Unicode e UTF-8

**1. Quantos bytes ocupa `"pão"`?**

```
'pao' : 3 caracteres, 3 bytes -> [112, 97, 111]
'pão' : 3 caracteres, 4 bytes -> [112, 195, 163, 111]
```

Mesmo número de **caracteres**, um byte a mais. O `ã` ocupa **2 bytes** em UTF-8
(`195, 163`), enquanto `a` ocupa 1.

**2. `ord("ç")` vale 231 — isso significa que cabe em 1 byte?**

**Não, e este é o erro clássico do capítulo.** 231 é o **code point** (a identidade
Unicode do caractere), não a codificação. Em UTF-8, todo code point acima de 127 usa 2
bytes ou mais — `ç` vira `[195, 167]`.

Confundir code point com byte leva a bugs sutis: você conta 3 caracteres e aloca 3 bytes.

**3. Por que começar pelos bytes elimina o "desconhecido"?**

Existem exatamente **256** valores de byte, e todos estão no vocabulário base. Qualquer
texto — de qualquer idioma, mais emoji, mais código-fonte — é uma sequência de bytes, logo
é sempre representável. Não há como aparecer algo "fora do vocabulário".

---

## E2 — Tamanho do vocabulário

| Vocabulário | Fusões | Tokens (treino) | Compressão | Ganho |
|-------------|--------|-----------------|-----------|-------|
| 300 | 44 | 95.043 | 1,578x | — |
| 512 | 256 | 66.936 | 2,241x | +0,663 |
| 1024 | 768 | 52.965 | 2,832x | +0,591 |

**1 e 2.** A compressão melhora, com **retorno decrescente**: as primeiras 256 fusões
valem +0,663 de compressão; as 512 seguintes valem só +0,591. As fusões mais frequentes
são aprendidas primeiro, e as seguintes cobrem casos cada vez mais raros.

**3. O custo de um vocabulário maior não está no tokenizador — está no modelo.** Duas
matrizes crescem com `vocab_size`:

- a tabela de embeddings: `vocab_size × n_embd`
- a camada de saída (`lm_head`): `n_embd × vocab_size`

Dobrar o vocabulário dobra as duas. É um trade-off direto: **sequências mais curtas
(bom) contra modelo maior (custo)**. Modelos reais usam entre 32 mil e 200 mil tokens,
e essa escolha é uma decisão de arquitetura, não um detalhe.

---

## E3 — As primeiras fusões

```
256: 'an'      261: 'il'
257: 'a\n'     262: 'on\n'
258: 'e\n'     263: 'ar'
259: 'el'      264: 'o\n'
260: 'on'      265: 'er'
```

**1. Por que `('a','n')` é a primeira?** Porque `an` é a sequência de duas letras mais
frequente em nomes brasileiros: **an**a, aless**an**dra, fern**an**da, lu**an**a,
**an**tonio…

**2. Por que tantas fusões terminam em `\n`?** Quatro das dez primeiras contêm quebra de
linha. O motivo é o **formato do arquivo**: um nome por linha. A quebra de linha faz parte
do padrão de **terminação** dos nomes, e o tokenizador está literalmente aprendendo "como
um nome acaba".

> Isso revela algo importante: o tokenizador aprende o **formato do arquivo**, não só a
> língua. Se você mudasse para nomes separados por vírgula, ele aprenderia `,` no lugar.
> Formatação é dado.

**3. Os tokens mais longos:**

```
'ilson\n', 'erson\n', 'isson\n', 'ilene\n', 'ilton\n', 'ilane\n',
'laine\n', 'iane\n', 'ison\n', 'iele\n', 'iana\n', 'aldo\n', ...
```

São **sufixos produtivos de nomes brasileiros**. O algoritmo não tem nenhuma noção de
morfologia — ele só conta pares. Estatística pura virando estrutura linguística.

---

## E4 — A ordem das fusões importa

Aplicando as fusões numa ordem arbitrária em vez da ordem de aprendizado:

| Texto | Ordem correta | Ordem errada | Round-trip |
|-------|---------------|--------------|------------|
| `maria eduarda` | 7 | 8 | ✅ |
| `joao vinicius` | 9 | 9 | ✅ |
| `ana beatriz de souza` | 15 | 15 | ✅ |
| **20.000 caracteres** | **9.137** | **10.041** | **+9,89%** |

**1. O round-trip continua valendo.** O `decode` desfaz qualquer sequência de fusões
válidas, então o texto volta igual. A ordem errada **não quebra** o tokenizador — só o
deixa pior.

**2. A compressão piora ~10%** em texto de tamanho razoável. Significativo, ainda que não
catastrófico.

> **Repare no contraste entre as duas partes da tabela** — e é a lição metodológica do
> exercício. Nos três textos curtos, dois deram **exatamente o mesmo** número de tokens.
> Quem medisse só neles concluiria que "a ordem quase não importa". Só com volume o efeito
> de ~10% aparece de forma confiável. **Amostra pequena esconde efeito real.**

**3. Por que a regra é obrigatória mesmo assim?** Dois motivos:

- Uma fusão tardia pode usar um token criado por uma anterior; aplicá-la antes desmonta a
  cadeia.
- Sem ordem fixa, o mesmo texto poderia produzir tokens diferentes em execuções
  diferentes. Um tokenizador precisa ser **determinístico** — senão o modelo recebe
  entradas inconsistentes.

---

## E5 — O imposto do português

Solução em [`e5_bpe_portugues.py`](e5_bpe_portugues.py). Resumo: um BPE do **mesmo
tamanho** treinado em português com acentos aprende `'ação '`, `'ção '`, `'não '`, `'são '`
como tokens únicos, e gasta **53% menos tokens** (64 contra 135 em três frases).

---

## E6 — Tokens fora do domínio

| Texto | Bytes | Tokens | Compressão | Round-trip |
|-------|-------|--------|-----------|------------|
| nomes (mesmo domínio) | 2.000 | 900 | **2,22x** | ✅ |
| frase em português | 60 | 50 | 1,20x | ✅ |
| japonês + emoji | 17 | 17 | **1,00x** | ✅ |
| código Python | 33 | 29 | 1,14x | ✅ |

**1.** Fora do domínio a compressão desaba para perto de 1,00x: nenhuma das fusões
aprendidas se aplica, então cada byte vira um token.

**2. Mas o round-trip continua `True` em todos os casos** — e é essa a garantia que
importa. O tokenizador **não comprime** o que não conhece, mas **nunca falha**. Japonês,
emoji e código-fonte são todos representáveis, porque o vocabulário base são os 256 bytes.

**3. Um tokenizador de palavras** receberia uma palavra japonesa e não teria entrada
nenhuma para ela — precisaria emitir `<UNK>`, **perdendo a informação de forma
irreversível**. É a diferença entre "comprime mal" e "não representa".

---

## E7 — Integrando com o modelo

Este exercício é mais de raciocínio que de medição.

**1. O que muda em `vocab_size` e nos parâmetros?** De 27 para 512 (ou 1024). As duas
matrizes ligadas ao vocabulário crescem na mesma proporção: embeddings
(`vocab_size × n_embd`) e `lm_head` (`n_embd × vocab_size`). Com o modelo do Capítulo 5
(`n_embd = 64`), isso vai de ~3,5 mil para ~65 mil parâmetros só nessas duas.

**2. Quanto texto cabe no mesmo `block_size`?** Com compressão de 2,24x, uma janela de 8
tokens passa a cobrir ~18 caracteres em vez de 8. **O contexto efetivo mais que dobra sem
custo nenhum** — é o principal ganho prático da tokenização.

**3. A loss é comparável? Não, e entender por quê é o ponto do exercício.**

A cross-entropy mede a incerteza **por token**. Prever entre 27 opções e prever entre 512
são tarefas de dificuldade diferente: o piso de um chute uniforme passa de `ln(27) = 3,30`
para `ln(512) = 6,24`. Uma loss de 2,0 significa coisas muito diferentes nos dois casos.

Para comparar de forma justa, converte-se para **bits por caractere** (BPC):

```
BPC = loss_por_token / ln(2) / caracteres_por_token
```

Essa métrica é independente do tokenizador — e é por isso que é usada na literatura para
comparar modelos com vocabulários diferentes. É exatamente a armadilha que o Capítulo 11
aponta ao aposentar o benchmark de 1,776.
