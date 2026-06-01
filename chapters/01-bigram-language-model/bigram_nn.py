"""
Bigram Language Model — versao rede neural (neural network).

Mesma tarefa do bigram.py, mas agora as probabilidades vem de uma unica
camada linear: uma matriz de pesos W de tamanho 27x27, treinada por gradient
descent. No fim, os pesos aprendidos reproduzem a MESMA tabela de
probabilidades que obtivemos contando -- so que via otimizacao. Esse e o
ponto central do capitulo: contar e um caso particular de aprender.

Run:
    python bigram_nn.py
"""

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. Data + vocabulary (same as the count-based version).
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]

chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
vocab_size = len(itos)

# ---------------------------------------------------------------------------
# 2. Build the training set of bigrams: xs (input char), ys (target char).
# ---------------------------------------------------------------------------
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        xs.append(stoi[ch1])
        ys.append(stoi[ch2])
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print(f"{num} exemplos de treino (bigramas)")

# ---------------------------------------------------------------------------
# 3. The "network": a single 27x27 weight matrix. With one-hot inputs, the
#    matmul (xenc @ W) just selects a row of W, so each row plays the role of
#    the log-counts ("logits") for "what character comes next".
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

# one-hot encode the inputs once (they don't change between steps)
xenc = F.one_hot(xs, num_classes=vocab_size).float()  # (num, 27)

# ---------------------------------------------------------------------------
# 4. Training loop: forward (softmax) -> loss (NLL) -> backward -> update.
# ---------------------------------------------------------------------------
for step in range(200):
    # --- forward pass ---
    logits = xenc @ W                                  # (num, 27) log-counts
    counts = logits.exp()                              # equivalent to N
    probs = counts / counts.sum(dim=1, keepdim=True)   # softmax -> probabilities
    # loss = mean negative log-likelihood + L2 regularization.
    # The regularization term pushes W toward 0, which is the NN equivalent of
    # the "+1" Laplace smoothing in bigram.py.
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01 * (W ** 2).mean()

    # --- backward pass ---
    W.grad = None      # reset gradients (otherwise PyTorch accumulates them)
    loss.backward()    # autograd fills W.grad

    # --- update ---
    W.data += -50 * W.grad   # gradient descent step (learning rate = 50)

    if step % 20 == 0:
        print(f"step {step:3d} | loss {loss.item():.4f}")

print(f"loss final = {loss.item():.4f}")

# ---------------------------------------------------------------------------
# 5. Sample from the trained network (same procedure as before, but the
#    probabilities now come from W instead of a counts table).
# ---------------------------------------------------------------------------
print("\nNomes gerados pela rede:")
for _ in range(10):
    out = []
    ix = 0
    while True:
        x = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
        logits = x @ W
        probs = logits.exp()
        probs = probs / probs.sum(dim=1, keepdim=True)
        ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            break
        out.append(itos[ix])
    print("  ", "".join(out))
