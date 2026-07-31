"""
avaliar.py — o SFT funcionou? Com um numero, nao com uma impressao.

A tentacao aqui e' gerar duas amostras, ler, e declarar que 'melhorou'. Num
modelo de 2,2 M parametros isso nao significa nada: ele escreve mal antes e
depois. O Capitulo 12 ja' mostrou que leitura nao detecta nem um bug que rebaixa
o modelo a um bigrama.

Entao medimos o que e' binario e objetivo:

    TAXA DE PARADA -- dado um pedido, o modelo emite <|fim|> dentro do orcamento?

O modelo-base nao consegue, por construcao: o token nao existe no vocabulario
dele. E' esse abismo que o SFT atravessa.

Run (a partir da pasta do capitulo):
    python avaliar.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import GPT, carregar, carregar_tokenizador
from preparar_sft import (PEDIDO, RESPOSTA, FIM, NOMES_ESPECIAIS,
                          TAM_PEDIDO, RESP_MIN, RESP_MAX)

ORCAMENTO = 120        # quantos tokens deixamos gerar antes de desistir


def carregar_sft(nome):
    caminho = AQUI / nome
    if not caminho.exists():
        return None
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m = GPT(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def gerar_resposta(m, pedido, orcamento=ORCAMENTO, temperatura=0.8, top_k=40,
                   semente=0):
    """Gera ate' emitir <|fim|> ou estourar o orcamento.

    Devolve (tokens da resposta, parou_sozinho).
    """
    g = torch.Generator().manual_seed(semente)
    idx = torch.tensor([[PEDIDO] + list(pedido) + [RESPOSTA]], dtype=torch.long)
    saida = []
    for _ in range(orcamento):
        logits, _ = m(idx[:, -m.block_size:])
        logits = logits[:, -1, :] / temperatura
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        prox = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g)
        t = int(prox.item())
        if t == FIM:
            return saida, True
        saida.append(t)
        idx = torch.cat((idx, prox), dim=1)
    return saida, False


def medir(m, pedidos, rotulo):
    """Taxa de parada e comprimento medio das respostas."""
    paradas, comprimentos = 0, []
    for i, p in enumerate(pedidos):
        resp, parou = gerar_resposta(m, p, semente=i)
        paradas += parou
        comprimentos.append(len(resp))
    n = len(pedidos)
    return {
        "rotulo": rotulo,
        "taxa": paradas / n,
        "comp_medio": float(np.mean(comprimentos)),
        "comp_mediano": float(np.median(comprimentos)),
    }


# ===========================================================================
if __name__ == "__main__":
    _, vocab = carregar_tokenizador()
    dados = np.load(AQUI / "sft_dados.npz")
    Xva = dados["Xva"]

    # os pedidos da validacao, sem os delimitadores
    pedidos = [Xva[i, 1:1 + TAM_PEDIDO].tolist() for i in range(40)]

    base, _ = carregar()
    sft = carregar_sft("modelo_sft.pt")
    sft_sem = carregar_sft("modelo_sft_sem_mascara.pt")

    if sft is None:
        raise SystemExit("modelo_sft.pt nao encontrado. Rode antes: python sft.py")

    print("=" * 74)
    print("1. O modelo aprendeu a PARAR?")
    print("=" * 74)
    print(f"  {len(pedidos)} pedidos da validacao, orcamento de {ORCAMENTO} tokens\n")
    print(f"  {'modelo':>26s} {'taxa de parada':>16s} {'comp. medio':>13s} "
          f"{'mediano':>9s}")

    linhas = [medir(sft, pedidos, "SFT (com mascara)")]
    if sft_sem is not None:
        linhas.append(medir(sft_sem, pedidos, "SFT (sem mascara)"))
    for r in linhas:
        print(f"  {r['rotulo']:>26s} {r['taxa']:>15.0%} {r['comp_medio']:>13.1f} "
              f"{r['comp_mediano']:>9.0f}")
    print(f"  {'modelo-base':>26s} {'0%':>15s} {'(nunca para)':>13s} {'--':>9s}")
    print(f"""
  O modelo-base marca 0% por CONSTRUCAO: o token <|fim|> nao existe no
  vocabulario dele. Nao e' que ele erre -- ele nao tem como acertar. Essa e' a
  distancia que o SFT atravessa.""")

    # -----------------------------------------------------------------------
    print("=" * 74)
    print("2. Como a resposta se parece")
    print("=" * 74)

    def txt(ts):
        return "".join(NOMES_ESPECIAIS.get(int(t),
                       vocab[int(t)].decode("utf-8", errors="replace") if int(t) < 1024
                       else "?") for t in ts).replace("\n", "\\n")

    for i in (0, 1):
        resp, parou = gerar_resposta(sft, pedidos[i], semente=i)
        print(f"\n  pedido  : {txt(pedidos[i])!r}")
        print(f"  resposta: {txt(resp)!r}")
        print(f"  parou sozinho: {parou} ({len(resp)} tokens)")

    print("""
  Leia as respostas e nao se impressione: e' um modelo de 2,2 M parametros
  treinado em 1,6 MB de texto. Ele nao ficou inteligente. O que mudou -- e o que
  da' para medir -- e' o COMPORTAMENTO: ele agora respeita um formato e termina.

  Essa distincao vale para modelos grandes tambem. O SFT nao acrescenta
  conhecimento; ele muda o que o modelo FAZ com o conhecimento que ja' tem.""")
