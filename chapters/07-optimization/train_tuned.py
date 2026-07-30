"""
Treino afinado — ABLACAO das tecnicas de otimizacao no Transformer do Cap. 5.

Comparacao controlada: MESMA arquitetura, MESMOS dados, MESMA semente, MESMO
numero de passos. So' as tecnicas de otimizacao mudam -- e mudam UMA POR VEZ,
para sabermos a contribuicao de cada uma:

  1. baseline          lr constante, sem clipping (a configuracao do Cap. 5)
  2. + agendamento     warmup + cosine decay
  3. + clipping        gradient clipping bem calibrado
  4. tudo junto        agendamento + clipping + weight decay + init escalada

SOBRE O LIMITE DO CLIPPING: a primeira versao deste arquivo usava GRAD_CLIP=1.0
e cortava 99% dos passos -- ou seja, nao estava capturando picos, estava
normalizando TODO gradiente. Isso descaracteriza a tecnica. O script agora MEDE
a norma tipica antes (ver a coluna "norma media") e usa um limite acima dela,
que e' como a escolha deve ser feita na pratica.

Tempo: ~25 minutos na CPU (quatro treinos de ~6 min).

Run:
    python train_tuned.py
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Hiperparametros (identicos ao Capitulo 5).
# ---------------------------------------------------------------------------
block_size, n_embd, n_head, n_layer = 8, 64, 4, 3
max_steps, batch_size = 15000, 64
base_lr = 1e-3

# Especificos do treino afinado
WARMUP_STEPS = 500        # passos de aquecimento da learning rate
MIN_LR_FRAC = 0.1         # a lr termina em 10% do valor de pico
GRAD_CLIP = 3.0           # norma maxima; ACIMA da norma tipica (~1.2), para
                          # cortar so' os picos e nao todo passo
WEIGHT_DECAY = 0.1

# ---------------------------------------------------------------------------
# Dados.
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
vocab_size = len(stoi) + 0
itos = {i: c for c, i in stoi.items()}


def build_dataset(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f"treino: {tuple(Xtr.shape)} | val: {tuple(Xdev.shape)} | teste: {tuple(Xte.shape)}")


# ---------------------------------------------------------------------------
# Modelo (o mesmo do Capitulo 5).
# ---------------------------------------------------------------------------
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
        self.key = nn.Linear(n_embd, hs, bias=False)
        self.query = nn.Linear(n_embd, hs, bias=False)
        self.value = nn.Linear(n_embd, hs, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        return F.softmax(w, dim=-1) @ v


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.ff_in = nn.Linear(n_embd, 4 * n_embd)
        self.ff_out = nn.Linear(4 * n_embd, n_embd)
        self.ln1, self.ln2 = LayerNorm(n_embd), LayerNorm(n_embd)

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.proj(torch.cat([hd(h) for hd in self.heads], dim=-1))
        x = x + self.ff_out(F.gelu(self.ff_in(self.ln2(x))))
        return x


class Transformer(nn.Module):
    def __init__(self, init_escalada=False):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        if init_escalada:
            self._init_escalada()

    def _init_escalada(self):
        """Truque do GPT-2: encolher a inicializacao das camadas que ESCREVEM no
        caminho residual, por 1/sqrt(2 * n_layer).

        Motivo: cada bloco SOMA a sua contribuicao ao residual. Com N blocos, as
        variancias se acumulam e o sinal cresce com a profundidade. Encolher a
        escrita compensa esse acumulo. Sao 2 escritas por bloco (atencao e
        feedforward), dai o 2 * n_layer.
        """
        escala = (2 * n_layer) ** -0.5
        for b in self.blocks:
            for camada in (b.proj, b.ff_out):
                with torch.no_grad():
                    camada.weight.mul_(escala)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.ln_f(x)[:, -1, :])


# ---------------------------------------------------------------------------
# Agendamento da learning rate: warmup linear + decaimento cosseno.
# ---------------------------------------------------------------------------
def lr_agendada(step):
    """warmup: sobe de 0 ao pico. depois: desce em cosseno ate MIN_LR_FRAC."""
    if step < WARMUP_STEPS:
        return base_lr * (step + 1) / WARMUP_STEPS
    progresso = (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS)
    cos = 0.5 * (1 + math.cos(math.pi * progresso))          # vai de 1 a 0
    return base_lr * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * cos)


# ---------------------------------------------------------------------------
# Treino.
# ---------------------------------------------------------------------------
def treinar(agendamento=False, clipping=False, wd_alto=False, init_escalada=False):
    torch.manual_seed(1337)
    modelo = Transformer(init_escalada=init_escalada)
    wd = WEIGHT_DECAY if wd_alto else 0.01
    opt = torch.optim.AdamW(modelo.parameters(), lr=base_lr, weight_decay=wd)

    t0 = time.time()
    normas = []
    clipados = 0
    for step in range(max_steps):
        if agendamento:
            lr = lr_agendada(step)
            for grupo in opt.param_groups:
                grupo["lr"] = lr

        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        loss = F.cross_entropy(modelo(Xtr[ix]), Ytr[ix])

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Medimos a norma SEMPRE (para poder calibrar o limite); cortamos so'
        # quando o clipping esta' ligado. clip_grad_norm_ devolve a norma que
        # havia ANTES do corte, e reescala sem mudar a direcao.
        limite = GRAD_CLIP if clipping else float("inf")
        norma = torch.nn.utils.clip_grad_norm_(modelo.parameters(), limite).item()
        normas.append(norma)
        if clipping and norma > GRAD_CLIP:
            clipados += 1

        opt.step()

        if step % 5000 == 0:
            print(f"    step {step:5d} | loss {loss.item():.4f} | {time.time()-t0:.0f}s", flush=True)

    @torch.no_grad()
    def split_loss(X, Y, chunk=4096):
        modelo.eval()
        tot = n = 0
        for i in range(0, X.shape[0], chunk):
            tot += F.cross_entropy(modelo(X[i : i + chunk]), Y[i : i + chunk], reduction="sum").item()
            n += Y[i : i + chunk].numel()
        modelo.train()
        return tot / n

    return {
        "modelo": modelo,
        "treino": split_loss(Xtr, Ytr),
        "val": split_loss(Xdev, Ydev),
        "teste": split_loss(Xte, Yte),
        "norma_media": sum(normas) / len(normas),
        "norma_max": max(normas),
        "clipados": clipados,
        "segundos": time.time() - t0,
    }


# Cada tecnica e' testada ISOLADAMENTE contra o baseline, e no fim todas juntas.
# So' assim se sabe quem contribui, quem e' neutro e quem atrapalha.
CONFIGS = [
    ("1. baseline (cap. 5)", {}),
    ("2. so' agendamento", dict(agendamento=True)),
    ("3. so' clipping", dict(clipping=True)),
    ("4. so' weight decay 0.1", dict(wd_alto=True)),
    ("5. so' init escalada", dict(init_escalada=True)),
    ("6. tudo junto", dict(agendamento=True, clipping=True, wd_alto=True, init_escalada=True)),
]

resultados = {}
for rotulo, kwargs in CONFIGS:
    print(f"\n=== {rotulo} ===")
    r = treinar(**kwargs)
    resultados[rotulo] = r
    extra = ""
    if kwargs.get("clipping"):
        extra = f" | clipados: {r['clipados']}/{max_steps} ({r['clipados']/max_steps:.0%})"
    print(f"  val = {r['val']:.4f} | norma do grad: media {r['norma_media']:.3f}, "
          f"maxima {r['norma_max']:.3f}{extra}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print(f"{'configuracao':24s} {'treino':>9s} {'val':>9s} {'teste':>9s} {'vs baseline':>13s}")
base_val = resultados["1. baseline (cap. 5)"]["val"]
for rotulo, _ in CONFIGS:
    r = resultados[rotulo]
    delta = base_val - r["val"]
    marca = f"{delta:+.4f}" if rotulo != "1. baseline (cap. 5)" else "—"
    print(f"{rotulo:24s} {r['treino']:9.4f} {r['val']:9.4f} {r['teste']:9.4f} {marca:>13s}")

print("\nMesma arquitetura, mesmos dados, mesma semente -- so' otimizacao.")
print(f"(norma tipica do gradiente ~{resultados['1. baseline (cap. 5)']['norma_media']:.2f}; "
      f"limite de clipping = {GRAD_CLIP})")

# A melhor configuracao e' escolhida pelos DADOS, nao pela nossa expectativa.
melhor_rotulo = min(resultados, key=lambda r: resultados[r]["val"])
melhor = resultados[melhor_rotulo]
print(f"\nMELHOR configuracao: {melhor_rotulo} (val {melhor['val']:.4f})")

# ---------------------------------------------------------------------------
@torch.no_grad()
def amostrar(modelo):
    ctx = [0] * block_size
    out = []
    while True:
        probs = F.softmax(modelo(torch.tensor([ctx])), dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0 or len(out) > 20:
            break
        out.append(itos[ix])
        ctx = ctx[1:] + [ix]
    return "".join(out)


print(f"\nNomes gerados pela melhor configuracao ({melhor_rotulo}):")
for _ in range(8):
    print("  ", amostrar(melhor["modelo"]))
