r"""
preparar_sft.py — monta o dataset de instrucao a partir do corpus do Capitulo 11.

O modelo-base sabe CONTINUAR texto. O SFT ensina outra coisa: responder dentro de
um FORMATO e, sobretudo, PARAR. Aqui construimos os exemplos que ensinam isso.

Formato de cada exemplo:

    <|pedido|> {trecho de contexto} <|resposta|> {continuacao} <|fim|>
    \_________________ mascarado ______________/\____ treinado ____/

A mascara e' o ponto tecnico do capitulo: a loss e' calculada SO' na resposta.
O modelo nao precisa aprender a gerar o pedido -- o pedido vem do usuario.

Run (a partir da pasta do capitulo):
    python preparar_sft.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
CAP11 = AQUI.parent / "11-datasets"
sys.path.insert(0, str(CAP11))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Tokens especiais: ids NOVOS, acima do vocabulario de 1024 do Capitulo 11.
#
# Eles NAO sao texto. Nenhuma sequencia de caracteres do corpus produz o id
# 1024 -- e' isso que os torna delimitadores confiaveis. Se usassemos uma
# string como "###RESPOSTA###", o proprio texto poderia conte-la.
# ---------------------------------------------------------------------------
PEDIDO, RESPOSTA, FIM = 1024, 1025, 1026
NOMES_ESPECIAIS = {PEDIDO: "<|pedido|>", RESPOSTA: "<|resposta|>", FIM: "<|fim|>"}
VOCAB_SFT = 1027

IGNORAR = -100          # o valor que a cross_entropy do PyTorch descarta

TAM_PEDIDO = 24         # tokens de contexto
RESP_MIN, RESP_MAX = 12, 56    # a resposta termina em FIM DE FRASE, dentro disto
BLOCO = 128             # o contexto do modelo

# A PRIMEIRA VERSAO DESTE ARQUIVO usava resposta de tamanho FIXO (40 tokens), e o
# resultado foi um modelo que aprendeu a CONTAR ate' 40, nao a concluir: a mediana
# de parada saiu exatamente em 40, e a taxa foi 100% com e sem mascara.
#
# Com todos os exemplos do mesmo tamanho, a unica regra consistente nos dados e'
# posicional. O modelo aprendeu a regra que estava la' -- so' nao era a que eu
# queria ensinar. Agora a resposta termina em fim de frase, e o comprimento
# varia: para parar na hora certa o modelo precisa olhar o TEXTO.


def tokens_de_fim_de_frase(vocab):
    """Ids cujo texto contem um sinal de fim de frase."""
    fins = set()
    for tid, b in vocab.items():
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if any(c in s for c in ".!?"):
            fins.add(tid)
    return fins


def montar(dados, n_exemplos, fins, semente=1337):
    """Recorta trechos do corpus e os embrulha no formato de instrucao.

    A resposta vai ate' o primeiro FIM DE FRASE apos RESP_MIN tokens -- entao o
    comprimento varia de exemplo para exemplo, e parar exige ler o texto.

    Devolve dois arrays (n_exemplos, BLOCO):
        X    -- os tokens de entrada
        Y    -- os alvos, com IGNORAR em tudo que nao for resposta
    """
    rng = np.random.default_rng(semente)
    X = np.full((n_exemplos, BLOCO), FIM, dtype=np.int64)
    Y = np.full((n_exemplos, BLOCO), IGNORAR, dtype=np.int64)

    maximo = TAM_PEDIDO + RESP_MAX
    for i in range(n_exemplos):
        ini = rng.integers(0, len(dados) - maximo - 1)
        trecho = dados[ini:ini + maximo].astype(np.int64)
        resposta = trecho[TAM_PEDIDO:]
        # corta no primeiro fim de frase depois do minimo
        corte = len(resposta)
        for k in range(RESP_MIN, len(resposta)):
            if int(resposta[k]) in fins:
                corte = k + 1
                break
        seq = np.concatenate([
            [PEDIDO], trecho[:TAM_PEDIDO],
            [RESPOSTA], resposta[:corte],
            [FIM],
        ])
        n = len(seq)
        X[i, :n] = seq

        # A MASCARA. Y[t] e' o que o modelo deve prever DEPOIS de ver X[t].
        # Queremos treinar a partir da posicao do <|resposta|> (que deve prever
        # o primeiro token da resposta) ate' a ultima posicao da resposta (que
        # deve prever o <|fim|>).
        ini_resp = 1 + TAM_PEDIDO           # indice do token <|resposta|>
        Y[i, ini_resp:n - 1] = seq[ini_resp + 1:n]

    return X, Y


def descrever(X, Y, vocab, i=0):
    """Mostra um exemplo, marcando o que entra na loss e o que nao entra."""
    def txt(t):
        if t in NOMES_ESPECIAIS:
            return NOMES_ESPECIAIS[t]
        return vocab[t].decode("utf-8", errors="replace")

    n = int(np.argmax(X[i] == FIM)) + 1
    print("  exemplo completo:")
    print("    " + "".join(txt(int(t)) for t in X[i, :n]).replace("\n", "\\n"))
    treinados = int((Y[i] != IGNORAR).sum())
    print(f"\n  posicoes: {n} no total, {treinados} entram na loss "
          f"({treinados/n:.0%}), {n-treinados} mascaradas")


if __name__ == "__main__":
    from dataset import carregar, carregar_tokenizador

    _, vocab = carregar_tokenizador()
    treino, val = carregar("treino"), carregar("val")

    print("=" * 74)
    print("Montando o dataset de instrucao")
    print("=" * 74)
    fins = tokens_de_fim_de_frase(vocab)
    print(f"  tokens que terminam frase: {len(fins)} de {len(vocab)}")
    Xtr, Ytr = montar(treino, 8000, fins, semente=1337)
    Xva, Yva = montar(val, 500, fins, semente=99)
    print(f"  treino: {Xtr.shape[0]:,} exemplos | validacao: {Xva.shape[0]:,}\n")
    descrever(Xtr, Ytr, vocab)

    np.savez_compressed(AQUI / "sft_dados.npz",
                        Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
    tam = (AQUI / "sft_dados.npz").stat().st_size
    print(f"\n  gravado em sft_dados.npz ({tam/1024:.0f} KB)")
    print(f"""
  Repare em duas coisas:

  1. O PEDIDO NAO ENTRA NA LOSS. O modelo nao precisa aprender a escrever a
     pergunta -- ela vem de fora. Treinar nela gasta capacidade com uma tarefa
     que ninguem pediu, e e' um erro comum. O E2 mede o estrago.

  2. O <|fim|> ENTRA. Ele e' o unico alvo que o modelo-base nunca viu, e
     aprende-lo e' a diferenca entre 'continua texto para sempre' e 'responde e
     para'. E' o resultado central deste capitulo, e da' para medir com um
     numero binario: a taxa de parada.""")
