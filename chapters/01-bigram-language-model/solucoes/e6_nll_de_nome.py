"""
Solucao do Exercicio E6 — negative log-likelihood de um unico nome.

Reaproveita a tabela de probabilidades P construida por contagem e mede o quao
"provavel" um nome e' segundo o modelo. NLL menor = nome mais plausivel.

Run:
    python e6_nll_de_nome.py
"""

import torch

# --- rebuild P exactly like in bigram.py (count-based, with +1 smoothing) ---
# caminho relativo ao ARQUIVO, nao ao diretorio de onde se roda -- assim o
# script funciona tanto de dentro de solucoes/ quanto da pasta do capitulo
from pathlib import Path
words = (Path(__file__).resolve().parent.parent / "names.txt").read_text(
    encoding="utf-8").splitlines()
words = [w.strip() for w in words if w.strip()]

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(itos)

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        N[stoi[a], stoi[b]] += 1

P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)


def nll_de(nome):
    """Average negative log-likelihood of a single name under the bigram model."""
    chs = ["."] + list(nome.lower()) + ["."]
    log_likelihood = 0.0
    n = 0
    for a, b in zip(chs, chs[1:]):
        if a not in stoi or b not in stoi:   # char outside a-z: skip safely
            continue
        log_likelihood += torch.log(P[stoi[a], stoi[b]])
        n += 1
    return (-log_likelihood / n).item() if n else float("inf")


for nome in ["ana", "xwq", "maria", "vinicius", "kffkz"]:
    print(f"{nome:12s} -> NLL = {nll_de(nome):.4f}")

# Esperado: 'ana' e 'maria' tem NLL baixa (parecem nomes reais);
# 'xwq' e 'kffkz' tem NLL alta (combinacoes raras no dataset).
