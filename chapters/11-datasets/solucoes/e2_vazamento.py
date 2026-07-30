"""
Solucao do Exercicio E2 — medindo o vazamento da divisao errada.

Em vez de treinar dois modelos (caro), medimos o vazamento DIRETAMENTE: quanto do
texto de validacao ja' aparece, literalmente, no texto de treino.

A metrica: pegamos todas as sequencias de N tokens seguidos ("n-gramas") do
conjunto de validacao e perguntamos quantas delas tambem existem no treino. Se a
divisao for honesta, esse numero e' baixo -- sao apenas expressoes comuns da
lingua. Se houver vazamento, ele dispara: o modelo vai ser avaliado em trechos
que ja' viu.

Comparamos as duas divisoes:
  CORRETA : por obra (4 livros no treino, 1 na validacao)
  ERRADA  : todos os livros juntos, pedacos sorteados

Run (a partir da pasta do capitulo):
    python solucoes/e2_vazamento.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AQUI = Path(__file__).resolve().parent.parent


def n_gramas(tokens, n):
    """Conjunto de todas as sequencias de n tokens seguidos."""
    return {tuple(tokens[i:i + n]) for i in range(0, len(tokens) - n, 1)}


def medir_vazamento(tokens_treino, tokens_val, n):
    tr = n_gramas(tokens_treino, n)
    va = n_gramas(tokens_val, n)
    comuns = len(va & tr)
    return comuns / len(va) if va else 0.0


def main():
    treino = np.fromfile(AQUI / "treino.bin", dtype=np.uint16).astype(np.int32)
    val = np.fromfile(AQUI / "val.bin", dtype=np.uint16).astype(np.int32)

    # amostras: medir n-gramas do corpus inteiro custa muita memoria
    AMOSTRA = 200_000
    tr_c = treino[:AMOSTRA]
    va_c = val[:60_000]

    print("=== divisao CORRETA (por obra) ===")
    print(f"  treino: {len(treino):,} tokens (4 obras)")
    print(f"  val   : {len(val):,} tokens ('Memorial de Aires', obra separada)\n")

    print(f"  {'n-grama':>8s} {'% do val que ja aparece no treino':>36s}")
    correta = {}
    for n in (3, 5, 8, 12):
        p = medir_vazamento(tr_c, va_c, n)
        correta[n] = p
        print(f"  {n:>8d} {p:>35.2%}")

    # -----------------------------------------------------------------------
    # A divisao ERRADA: junta tudo e sorteia pedacos. Simulamos no nivel dos
    # tokens (equivale a embaralhar paragrafos antes de dividir).
    # -----------------------------------------------------------------------
    print("\n=== divisao ERRADA (tudo junto, pedacos sorteados) ===")
    tudo = np.concatenate([treino, val])
    rng = np.random.default_rng(1337)

    PEDACO = 512
    n_pedacos = len(tudo) // PEDACO
    indices = rng.permutation(n_pedacos)
    corte = int(0.85 * n_pedacos)

    pedacos = tudo[:n_pedacos * PEDACO].reshape(n_pedacos, PEDACO)
    tr_err = pedacos[indices[:corte]].reshape(-1)
    va_err = pedacos[indices[corte:]].reshape(-1)

    print(f"  treino: {len(tr_err):,} tokens (pedacos de todas as 5 obras)")
    print(f"  val   : {len(va_err):,} tokens (pedacos das MESMAS obras)\n")

    print(f"  {'n-grama':>8s} {'% do val que ja aparece no treino':>36s}")
    errada = {}
    for n in (3, 5, 8, 12):
        p = medir_vazamento(tr_err[:AMOSTRA], va_err[:60_000], n)
        errada[n] = p
        print(f"  {n:>8d} {p:>35.2%}")

    # -----------------------------------------------------------------------
    print("\n=== comparacao ===")
    print(f"  {'n-grama':>8s} {'correta':>10s} {'errada':>10s} {'quantas vezes pior':>20s}")
    for n in (3, 5, 8, 12):
        razao = errada[n] / correta[n] if correta[n] else float("inf")
        print(f"  {n:>8d} {correta[n]:>9.2%} {errada[n]:>9.2%} {razao:>19.1f}x")

    print("""
Como ler:

  n-gramas CURTOS (3 tokens) aparecem muito nos DOIS casos -- e' so' a lingua
  portuguesa: "de que a", "nao se ", etc. Isso nao e' vazamento, e' o idioma.

  n-gramas LONGOS (8, 12 tokens) sao a assinatura do vazamento. Uma sequencia
  especifica de 8 tokens aparecer nos dois lados nao acontece por acaso.

  A divisao errada mostra MAIS sobreposicao em todos os tamanhos, e a diferenca
  e' mais visivel justamente nos 8-gramas. O efeito e' consistente -- mas seja
  honesto sobre a MAGNITUDE: aqui ele e' moderado, nao dramatico. Duas razoes:

    1. O nosso "embaralhamento" e' de blocos de 512 tokens. Trechos diferentes do
       mesmo livro continuam sendo texto diferente. Se o embaralhamento fosse por
       FRASE, ou se houvesse documentos duplicados, o vazamento seria bem maior.

    2. Este corpus e' de UM autor so'. Machado escreve parecido em todos os
       livros, entao ate' a divisao honesta tem sobreposicao real de estilo e
       vocabulario. Isso e' informacao legitima, nao vazamento.

  Em corpus da web -- onde o MESMO artigo aparece copiado em dezenas de sites --
  o efeito e' muito maior, e por isso a deduplicacao e' um passo obrigatorio nos
  pipelines de verdade.

Como isso passa despercebido:

  A loss de validacao MELHORA. Quem so' olha o numero conclui que o modelo ficou
  melhor, publica o resultado, e so' descobre o problema quando o modelo encontra
  dados de verdade. Por isso a revisao de um pipeline de dados deve perguntar
  sempre: "de onde vem o conjunto de validacao, e ele poderia conter algo que o
  treino tambem contem?".""")


if __name__ == "__main__":
    main()
