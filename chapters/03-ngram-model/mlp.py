"""
N-gram model — um MLP (multi-layer perceptron) char-level, em PyTorch.

No Capitulo 1 o bigrama olhava SO o caractere anterior. Aqui olhamos os ultimos
`block_size` caracteres (o "contexto") e alimentamos um MLP que aprende a prever
o proximo. E' a arquitetura classica de Bengio et al. (2003), a ponte do modelo
de brinquedo para o modelo de verdade.

Fluxo: contexto -> embeddings -> camada oculta (GELU) -> logits -> softmax.
Treinamos com cross-entropy e avaliamos em splits de treino/validacao/teste.

Run:
    python mlp.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(2147483647)

# ---------------------------------------------------------------------------
# 1. Dados e vocabulario (igual aos capitulos anteriores).
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(itos)

# ---------------------------------------------------------------------------
# 2. Hiperparametros do modelo.
# ---------------------------------------------------------------------------
block_size = 3      # quantos caracteres de contexto usamos para prever o proximo
n_embd = 10         # dimensao do embedding de cada caractere
n_hidden = 200      # neuronios na camada oculta


def build_dataset(words):
    """Transforma uma lista de nomes em tensores X (contextos) e Y (alvos)."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size            # comeca com '...' (tudo fronteira)
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]      # desliza a janela: dropa o 1o, anexa o atual
    return torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------------------
# 3. Splits treino/validacao/teste (80/10/10). Avaliar fora do treino e' o
#    que distingue "decorar" de "generalizar".
# ---------------------------------------------------------------------------
import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])
print(f"treino: {tuple(Xtr.shape)} | val: {tuple(Xdev.shape)} | teste: {tuple(Xte.shape)}")

# ---------------------------------------------------------------------------
# 4. Parametros do modelo.
#    C  : tabela de embeddings (vocab_size x n_embd)
#    W1,b1 : camada oculta ; W2,b2 : camada de saida (logits)
#    O fator de escala em W1/W2 deixa as ativacoes em boa faixa no inicio
#    (motivado no Capitulo 7 - aqui so usamos um valor razoavel).
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / (n_embd * block_size) ** 0.5
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
b2 = torch.randn(vocab_size, generator=g) * 0.0
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True
print(f"parametros: {sum(p.nelement() for p in parameters)}")


def forward(X):
    """Calcula os logits do modelo para um batch de contextos X."""
    emb = C[X]                             # (N, block_size, n_embd): olha os embeddings
    x = emb.view(emb.shape[0], -1)         # (N, block_size*n_embd): concatena o contexto
    h = F.gelu(x @ W1 + b1)                # (N, n_hidden): camada oculta com GELU
    logits = h @ W2 + b2                   # (N, vocab_size)
    return logits


# ---------------------------------------------------------------------------
# 5. Treino: mini-batches + cross-entropy + decaimento da learning rate.
# ---------------------------------------------------------------------------
max_steps = 20000
batch_size = 32

for step in range(max_steps):
    # mini-batch: um subconjunto aleatorio dos dados a cada passo (mais rapido)
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    logits = forward(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])   # = softmax + negative log-likelihood

    for p in parameters:
        p.grad = None
    loss.backward()

    lr = 0.1 if step < 15000 else 0.01        # decai a lr no fim para refinar
    for p in parameters:
        p.data += -lr * p.grad

    if step % 2000 == 0:
        print(f"step {step:5d}/{max_steps} | loss {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 6. Avaliacao: loss em treino, validacao e teste (sem calcular gradiente).
# ---------------------------------------------------------------------------
@torch.no_grad()
def split_loss(X, Y):
    logits = forward(X)
    return F.cross_entropy(logits, Y).item()


print(f"\nloss treino     = {split_loss(Xtr, Ytr):.4f}")
print(f"loss validacao  = {split_loss(Xdev, Ydev):.4f}")
print(f"loss teste      = {split_loss(Xte, Yte):.4f}")

# ---------------------------------------------------------------------------
# 7. Geracao: amostra nomes novos do modelo treinado.
# ---------------------------------------------------------------------------
print("\nNomes gerados pelo MLP:")
for _ in range(10):
    out = []
    context = [0] * block_size
    while True:
        logits = forward(torch.tensor([context]))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
        context = context[1:] + [ix]
    print("  ", "".join(out))
