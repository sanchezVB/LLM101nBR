"""
Solucao do Exercicio E2 — ablacao das conexoes residuais.

Treina o MESMO Transformer com e sem residuais, em duas profundidades, para
mostrar que (a) residuais ajudam e (b) a vantagem CRESCE com a profundidade --
que e' exatamente o motivo de eles existirem.

Usamos menos passos (4000) do que a apostila para o experimento ser rapido; os
numeros absolutos ficam piores que os do transformer.py, mas a COMPARACAO entre
as quatro configuracoes e' o que interessa.

Tempo estimado: ~6 minutos na CPU.

Run (a partir da pasta do capitulo):
    python solucoes/e2_ablacoes.py
"""
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCK_SIZE = 8
N_EMBD = 64
N_HEAD = 4
STEPS = 4000
BATCH = 64

words = open(Path(__file__).resolve().parent.parent / "names.txt", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
V = len(stoi)


def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build(words[:n1])
Xdev, Ydev = build(words[n1:n2])


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - m) / torch.sqrt(v + self.eps) + self.beta


class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.key = nn.Linear(N_EMBD, hs, bias=False)
        self.query = nn.Linear(N_EMBD, hs, bias=False)
        self.value = nn.Linear(N_EMBD, hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(w, dim=-1) @ v


class Block(nn.Module):
    def __init__(self, residual: bool):
        super().__init__()
        hs = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(hs) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.ff = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD)
        )
        self.ln1, self.ln2 = LayerNorm(N_EMBD), LayerNorm(N_EMBD)
        self.residual = residual

    def forward(self, x):
        sa = self.proj(torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1))
        x = x + sa if self.residual else sa
        ff = self.ff(self.ln2(x))
        x = x + ff if self.residual else ff
        return x


class Model(nn.Module):
    def __init__(self, n_layer, residual):
        super().__init__()
        self.te = nn.Embedding(V, N_EMBD)
        self.pe = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(residual) for _ in range(n_layer)])
        self.ln_f = LayerNorm(N_EMBD)
        self.lm = nn.Linear(N_EMBD, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T))
        x = self.ln_f(self.blocks(x))
        return self.lm(x[:, -1, :])


def treinar(n_layer, residual):
    torch.manual_seed(1337)
    m = Model(n_layer, residual)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(STEPS):
        ix = torch.randint(0, Xtr.shape[0], (BATCH,))
        loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def sl(X, Y, chunk=4096):
        m.eval()
        tot = n = 0
        for i in range(0, X.shape[0], chunk):
            tot += F.cross_entropy(m(X[i : i + chunk]), Y[i : i + chunk], reduction="sum").item()
            n += Y[i : i + chunk].numel()
        m.train()
        return tot / n

    return sl(Xdev, Ydev)


print(f"Treinando 4 configuracoes com {STEPS} passos cada (aguarde alguns minutos)...\n")
print(f"{'config':32s} {'val loss':>9s}")
resultados = {}
for n_layer in (3, 6):
    for residual in (True, False):
        val = treinar(n_layer, residual)
        resultados[(n_layer, residual)] = val
        tag = f"{n_layer} blocos, residual={'ON ' if residual else 'OFF'}"
        print(f"{tag:32s} {val:9.4f}", flush=True)

print("\n--- analise ---")
for n_layer in (3, 6):
    com = resultados[(n_layer, True)]
    sem = resultados[(n_layer, False)]
    print(f"{n_layer} blocos: sem residual e' {sem - com:+.4f} pior que com residual")
print()
print("A penalidade de remover os residuais tende a CRESCER com a profundidade:")
print("sem o caminho direto do residual, o gradiente precisa atravessar todas as")
print("sub-camadas para chegar ao inicio da rede, e vai se degradando no percurso.")
print("Foi essa ideia (ResNet, 2015) que tornou praticavel treinar redes profundas.")
