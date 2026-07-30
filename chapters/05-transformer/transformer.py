"""
Transformer — a arquitetura completa, no estilo GPT-2.

Juntamos tudo o que construimos: os embeddings do Cap. 3, a self-attention do
Cap. 4 e a LayerNorm do layernorm.py. As pecas NOVAS deste capitulo sao:

  1. MULTI-HEAD attention: varias cabecas em paralelo, cada uma livre para
     aprender um tipo diferente de relacao. As saidas sao concatenadas.
  2. CONEXOES RESIDUAIS (x + f(x)): criam um "caminho livre" para o gradiente,
     o que torna possivel empilhar muitas camadas.
  3. BLOCO: comunicacao (attention) seguida de computacao (feedforward), cada
     uma com LayerNorm e residual.
  4. PROFUNDIDADE: varios blocos empilhados.

Mesma tarefa e mesma metrica dos capitulos 3 e 4, para a loss ser comparavel:
MLP = 1.967 | atencao sozinha = 2.099 | atencao + feedforward = 1.913

Run:
    python transformer.py
"""

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# Hiperparametros.
# ---------------------------------------------------------------------------
block_size = 8       # tamanho do contexto
n_embd = 64          # dimensao dos embeddings
n_head = 4           # numero de cabecas de atencao (head_size = n_embd // n_head)
n_layer = 3          # numero de blocos empilhados
max_steps = 15000
batch_size = 64
learning_rate = 1e-3

# ---------------------------------------------------------------------------
# 1. Dados.
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(itos)


def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


random.seed(42)
random.shuffle(words)
n1, n2 = int(0.8 * len(words)), int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f"treino: {tuple(Xtr.shape)} | val: {tuple(Xdev.shape)} | teste: {tuple(Xte.shape)}")


# ---------------------------------------------------------------------------
# 2. As pecas.
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Normaliza cada posicao ao longo das features (ver layernorm.py)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class Head(nn.Module):
    """Uma cabeca de self-attention causal (identica a' do Capitulo 4)."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Varias cabecas em paralelo, concatenadas e projetadas de volta.

    Cada cabeca tem dimensao n_embd // n_head, de modo que a concatenacao
    volta a ter n_embd -- o custo total e' o mesmo de uma cabeca grande, mas
    o modelo ganha a liberdade de aprender VARIAS relacoes diferentes.
    """

    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        # projecao: mistura o que as cabecas trouxeram, cada uma no seu "canto"
        self.proj = nn.Linear(head_size * n_head, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.proj(out)


class FeedForward(nn.Module):
    """Processamento por posicao (nao troca informacao entre posicoes)."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),     # expande (o 4x e' a convencao do GPT)
            nn.GELU(),
            nn.Linear(4 * dim, dim),     # volta ao tamanho original
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Um bloco do Transformer: comunicacao + computacao.

    Usamos PRE-NORM (a LayerNorm vem ANTES da sub-camada, e nao depois): e' o
    arranjo do GPT-2 e treina de forma mais estavel que o do artigo original.

    Note o padrao `x = x + sublayer(norm(x))`: a soma e' a CONEXAO RESIDUAL. Ela
    da' ao gradiente um caminho direto de volta -- sem ela, empilhar varios
    blocos torna o treino muito mais dificil.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # comunicacao entre posicoes
        x = x + self.ff(self.ln2(x))    # computacao dentro de cada posicao
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)                 # norm final, antes dos logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        x = x[:, -1, :]          # ultima posicao (mesma metrica dos caps. 3 e 4)
        return self.lm_head(x)


model = Transformer()
n_params = sum(p.nelement() for p in model.parameters())
print(f"parametros: {n_params}  ({n_layer} blocos, {n_head} cabecas, n_embd={n_embd})")

# ---------------------------------------------------------------------------
# 3. Treino.
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
t0 = time.time()

for step in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1500 == 0:
        print(f"step {step:5d}/{max_steps} | loss {loss.item():.4f} | {time.time()-t0:.0f}s")

print(f"treino concluido em {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# 4. Avaliacao.
# ---------------------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y, chunk=4096):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        xb, yb = X[i : i + chunk], Y[i : i + chunk]
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
        n += yb.numel()
    model.train()
    return total / n


print(f"\nloss treino     = {split_loss(Xtr, Ytr):.4f}")
print(f"loss validacao  = {split_loss(Xdev, Ydev):.4f}")
print(f"loss teste      = {split_loss(Xte, Yte):.4f}")
print("referencias: MLP (cap.3) = 1.967 | atencao (cap.4) = 2.099 | atencao+ff = 1.913")


# ---------------------------------------------------------------------------
# 5. Geracao.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_name():
    context = [0] * block_size
    out = []
    while True:
        probs = F.softmax(model(torch.tensor([context])), dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
        if len(out) > 20:
            break
    return "".join(out)


print("\nNomes gerados pelo Transformer:")
for _ in range(10):
    print("  ", sample_name())
