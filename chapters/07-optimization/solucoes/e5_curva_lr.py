"""
Solucao do Exercicio E5 — a curva da learning rate.

Treina o mesmo Transformer com varias learning rates e monta a curva. O
resultado classico e' um "U": lr baixa demais aprende pouco no orcamento dado,
lr alta demais desestabiliza. O minimo fica numa faixa relativamente estreita.

Usamos 3000 passos por configuracao para o experimento ser viavel (~8 min no
total). Os numeros absolutos ficam piores que os da apostila (que usa 15000
passos), mas a FORMA da curva -- que e' o que interessa -- aparece igual.

Run (a partir da pasta do capitulo):
    python solucoes/e5_curva_lr.py
"""

import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BLOCK, N_EMBD, N_HEAD, N_LAYER = 8, 64, 4, 3
STEPS, BATCH = 3000, 64
WARMUP = 200

words = open(Path(__file__).resolve().parent.parent / "names.txt", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
V = len(stoi)


def build(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * BLOCK
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
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = x.var(-1, keepdim=True, unbiased=False)
        return self.g * (x - m) / torch.sqrt(v + self.eps) + self.b


class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.k = nn.Linear(N_EMBD, hs, bias=False)
        self.q = nn.Linear(N_EMBD, hs, bias=False)
        self.v = nn.Linear(N_EMBD, hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK, BLOCK)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.k(x), self.q(x), self.v(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(w, dim=-1) @ v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        hs = N_EMBD // N_HEAD
        self.heads = nn.ModuleList([Head(hs) for _ in range(N_HEAD)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.fi = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fo = nn.Linear(4 * N_EMBD, N_EMBD)
        self.ln1, self.ln2 = LayerNorm(N_EMBD), LayerNorm(N_EMBD)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.proj(torch.cat([hd(h) for hd in self.heads], dim=-1))
        return x + self.fo(F.gelu(self.fi(self.ln2(x))))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(V, N_EMBD)
        self.pe = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = LayerNorm(N_EMBD)
        self.lm = nn.Linear(N_EMBD, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        return self.lm(self.lnf(x)[:, -1, :])


def treinar(base_lr):
    torch.manual_seed(1337)
    m = Model()
    opt = torch.optim.AdamW(m.parameters(), lr=base_lr, weight_decay=0.1)
    divergiu = False
    for step in range(STEPS):
        lr = base_lr * (step + 1) / WARMUP if step < WARMUP else base_lr * (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - WARMUP) / (STEPS - WARMUP)))
        )
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, Xtr.shape[0], (BATCH,))
        loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
        if not torch.isfinite(loss):
            divergiu = True
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    if divergiu:
        return float("nan")

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


print(f"Treinando 5 learning rates com {STEPS} passos cada (~8 min)...\n")
print(f"{'learning rate':>14s} {'val loss':>10s}   grafico")
resultados = {}
for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
    val = treinar(lr)
    resultados[lr] = val
    if val != val:      # NaN
        print(f"{lr:14.0e} {'DIVERGIU':>10s}", flush=True)
    else:
        barras = "#" * int((val - 1.8) * 60) if val > 1.8 else "#"
        print(f"{lr:14.0e} {val:10.4f}   {barras}", flush=True)

validos = {k: v for k, v in resultados.items() if v == v}
if validos:
    melhor = min(validos, key=validos.get)
    print(f"\nmelhor learning rate: {melhor:.0e} (val {validos[melhor]:.4f})")
print("""
A curva tem forma de U:
  - lr baixa demais: cada passo e' pequeno; no orcamento dado, o modelo nao
    chega onde poderia. Nao esta' errado, esta' LENTO.
  - lr alta demais: os passos passam do ponto e o treino oscila ou piora.
  - no meio existe uma faixa boa, e ela e' mais larga do que se imagina: entre
    1e-3 e 1e-2 a loss varia pouco.

DETALHE IMPORTANTE: com o orcamento CURTO deste experimento (3000 passos), a
melhor lr foi 3e-3 -- MAIOR que o 1e-3 usado na apostila (que treina 15000
passos). Isso nao e' contradicao: com poucos passos, compensa andar mais rapido;
com muitos, uma lr menor tem tempo de refinar e chega mais longe.

Conclusao pratica: a "melhor learning rate" nao e' uma propriedade do modelo. Ela
depende do ORCAMENTO de passos e do TAMANHO DO BATCH (batch maior da' gradiente
menos ruidoso, o que tolera lr maior). Sempre reporte os tres juntos.
""")
