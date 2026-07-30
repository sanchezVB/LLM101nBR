"""
Um modelo de linguagem com self-attention (uma cabeca), treinado nos nomes.

Mesma tarefa do Capitulo 3 (dado um contexto, prever o proximo caractere), para
que a loss seja DIRETAMENTE comparavel: o MLP chegou a ~1.97 com ~11.9k
parametros. Aqui usamos ~11.4k -- comparacao justa.

A diferenca em relacao ao MLP: (a) o contexto e' maior (8 caracteres em vez de 3)
e (b) o modelo DECIDE onde olhar dentro desse contexto, em vez de concatenar tudo.

EXPERIMENTO DO CAPITULO: troque USE_FEEDFORWARD para True e rode de novo. Essa
unica mudanca revela por que o Transformer precisa de MAIS do que atencao --
veja a discussao na secao 9 do README.

Run:
    python model.py
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

# >>> O experimento central: rode com False, depois com True. <<<
USE_FEEDFORWARD = False

# ---------------------------------------------------------------------------
# 1. Dados e vocabulario.
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(itos)

block_size = 8      # contexto de 8 caracteres (o MLP do cap. 3 usava 3)
n_embd = 52         # dimensao dos embeddings
head_size = 52      # dimensao da cabeca de atencao


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
# 2. Uma cabeca de self-attention (o que construimos em attention.py, agora
#    empacotado como um modulo do PyTorch).
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # buffer = tensor que acompanha o modulo mas NAO e' parametro treinavel
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # afinidades escaladas + mascara causal + softmax
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v       # (B, T, head_size)


class FeedForward(nn.Module):
    """Processamento por posicao: expande, aplica GELU, volta.

    Note que isto NAO troca informacao entre posicoes -- cada posicao e'
    processada isoladamente. E' o complemento da atencao (que so' comunica,
    sem "pensar"). Juntos formam o bloco do Transformer (Capitulo 5).
    """

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class AttentionLM(nn.Module):
    """Embeddings (token + posicao) -> self-attention -> [feedforward] -> logits."""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # POSICIONAL: a atencao por si so' nao sabe a ORDEM dos tokens (ela ve'
        # um conjunto, nao uma sequencia). O embedding de posicao injeta "onde"
        # cada token esta'.
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd, head_size, block_size)
        self.ff = FeedForward(head_size) if USE_FEEDFORWARD else None
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.token_emb(idx)                                    # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))       # (T, n_embd)
        x = tok + pos                                                # conteudo + posicao
        x = self.head(x)                                             # (B, T, head_size)
        if self.ff is not None:
            x = x + self.ff(x)       # conexao residual (o "x +"), detalhada no Cap. 5
        x = x[:, -1, :]      # usamos a ultima posicao para prever o proximo char
        return self.lm_head(x)                                       # (B, vocab_size)


model = AttentionLM()
n_params = sum(p.nelement() for p in model.parameters())
print(f"parametros: {n_params}  (feedforward: {'ON' if USE_FEEDFORWARD else 'OFF'})")

# ---------------------------------------------------------------------------
# 3. Treino. Agora usamos um otimizador do PyTorch (AdamW) em vez do gradient
#    descent na mao -- ele adapta o passo por parametro. Detalhes no Cap. 7.
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
max_steps = 20000
batch_size = 64

for step in range(max_steps):
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    logits = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 2000 == 0:
        print(f"step {step:5d}/{max_steps} | loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 4. Avaliacao nos tres splits.
# ---------------------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y, chunk=8192):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):        # em pedacos, para nao pesar a memoria
        xb, yb = X[i : i + chunk], Y[i : i + chunk]
        total += F.cross_entropy(model(xb), yb, reduction="sum").item()
        n += yb.numel()
    model.train()
    return total / n


print(f"\nloss treino     = {split_loss(Xtr, Ytr):.4f}")
print(f"loss validacao  = {split_loss(Xdev, Ydev):.4f}")
print(f"loss teste      = {split_loss(Xte, Yte):.4f}")
print("(referencia: o MLP do Capitulo 3, mesma tarefa, ficou em ~1.97)")


# ---------------------------------------------------------------------------
# 5. Geracao de nomes novos.
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_name():
    context = [0] * block_size
    out = []
    while True:
        logits = model(torch.tensor([context]))
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
        if len(out) > 20:      # trava de seguranca
            break
    return "".join(out)


print("\nNomes gerados:")
for _ in range(10):
    print("  ", sample_name())
