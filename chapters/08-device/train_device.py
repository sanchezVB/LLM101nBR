"""
train_device.py — treinar na CPU vs na GPU, em tamanhos crescentes de modelo.

O benchmark.py mostrou que a GPU so' ganha acima de um certo tamanho de matriz.
Aqui a pergunta e' pratica: **o nosso Transformer se beneficia?**

Testamos o mesmo modelo em tres tamanhos, nos dois dispositivos, medindo o tempo
por passo de treino. E verificamos algo igualmente importante: a GPU produz o
MESMO resultado que a CPU?

Run:
    python train_device.py
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from device import pegar_device, sincronizar

dev, rotulo = pegar_device()
cpu = torch.device("cpu")

BLOCK_SIZE = 8
PASSOS = 100          # poucos passos: aqui medimos VELOCIDADE, nao qualidade

# ---------------------------------------------------------------------------
# Dados (os mesmos nomes dos capitulos anteriores).
# ---------------------------------------------------------------------------
words = open("names.txt", encoding="utf-8").read().splitlines()
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
Xtr, Ytr = build(words[: int(0.8 * len(words))])
print(f"dados: {tuple(Xtr.shape)}")


# ---------------------------------------------------------------------------
# O modelo (mesma arquitetura do Cap. 5, com dimensao configuravel).
# ---------------------------------------------------------------------------
class Bloco(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.fi = nn.Linear(n_embd, 4 * n_embd)
        self.fo = nn.Linear(4 * n_embd, n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        # q, k, v de uma vez: uma matmul grande em vez de tres pequenas
        q, k, v = self.qkv(h).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, hs): cada cabeca em paralelo
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x))))


class Modelo(nn.Module):
    def __init__(self, n_embd, n_head, n_layer):
        super().__init__()
        self.te = nn.Embedding(V, n_embd)
        self.pe = nn.Embedding(BLOCK_SIZE, n_embd)
        self.blocos = nn.ModuleList([Bloco(n_embd, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T, device=idx.device))
        for b in self.blocos:
            x = b(x)
        return self.lm(self.lnf(x)[:, -1, :])


def medir(device, n_embd, n_head, n_layer, batch):
    """Treina PASSOS passos e devolve o tempo medio por passo (ms) e a loss."""
    torch.manual_seed(1337)
    m = Modelo(n_embd, n_head, n_layer).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    nparam = sum(p.nelement() for p in m.parameters())

    # Move os dados UMA VEZ (a licao do benchmark.py: nao transferir a cada passo)
    Xd, Yd = Xtr.to(device), Ytr.to(device)
    g = torch.Generator().manual_seed(1337)

    def um_passo():
        ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g).to(device)
        loss = F.cross_entropy(m(Xd[ix]), Yd[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(5):        # aquecimento
        um_passo()
    sincronizar(device)

    t0 = time.perf_counter()
    for _ in range(PASSOS):
        loss = um_passo()
    sincronizar(device)
    dt = (time.perf_counter() - t0) / PASSOS

    return dt * 1000, loss.item(), nparam


# ---------------------------------------------------------------------------
CONFIGS = [
    # rotulo,          n_embd, n_head, n_layer, batch
    ("pequeno (cap. 5)",    64,      4,       3,     64),
    ("medio",              256,      8,       4,    256),
    ("grande",             512,      8,       6,    512),
]

def rodar_comparacao():
    print(f"\n=== tempo por passo de treino: CPU vs {rotulo} ===")
    print(f"{'modelo':<18s} {'params':>9s} {'batch':>6s} {'CPU (ms)':>10s} "
          f"{'GPU (ms)':>10s} {'speedup':>9s}")

    resultados = []
    for nome, ne, nh, nl, bs in CONFIGS:
        t_cpu, loss_cpu, npar = medir(cpu, ne, nh, nl, bs)
        t_gpu, loss_gpu, _ = medir(dev, ne, nh, nl, bs)
        sp = t_cpu / t_gpu
        resultados.append((nome, npar, bs, t_cpu, t_gpu, sp, loss_cpu, loss_gpu))
        print(f"{nome:<18s} {npar:9d} {bs:6d} {t_cpu:10.1f} {t_gpu:10.1f} {sp:8.2f}x",
              flush=True)

    # -----------------------------------------------------------------------
    # A GPU da' o MESMO resultado que a CPU?
    # -----------------------------------------------------------------------
    print("\n=== a GPU calcula o mesmo que a CPU? ===")
    print(f"{'modelo':<18s} {'loss CPU':>10s} {'loss GPU':>10s} {'diferenca':>11s}")
    for nome, _, _, _, _, _, lc, lg in resultados:
        print(f"{nome:<18s} {lc:10.4f} {lg:10.4f} {abs(lc-lg):11.2e}")

    print("""
  As losses ficam proximas, mas NAO identicas -- e isso e' esperado. A GPU usa
  kernels diferentes e soma os numeros em outra ORDEM, e em ponto flutuante a
  ordem da soma altera o ultimo digito (o mesmo efeito que vimos no Cap. 4, na
  comparacao das versoes da atencao). O que importa e' que a diferenca seja da
  ordem do arredondamento, e nao de um erro de logica.""")

    melhor = max(resultados, key=lambda r: r[5])
    print(f"\nmaior ganho: {melhor[0]} com {melhor[5]:.2f}x")
    if resultados[0][5] < 1:
        print(f"Note que o modelo PEQUENO (o nosso, dos capitulos anteriores) e'"
              f" {1/resultados[0][5]:.1f}x MAIS LENTO na GPU.")
        print("Ele e' pequeno demais: as matmuls dele nao enchem a GPU, e o custo fixo")
        print("por operacao domina. GPU nao acelera trabalho pequeno -- acelera trabalho")
        print("GRANDE e paralelo.")


# So' roda quando este arquivo e' EXECUTADO, nao quando e' importado. Sem essa
# protecao, `from train_device import Modelo` (que a solucao do E6 faz) dispararia
# o benchmark inteiro.
if __name__ == "__main__":
    rodar_comparacao()
