"""
Bigram Language Model — versao por contagem (count-based).

Conta com que frequencia cada caractere segue outro nos nomes de treino,
transforma essas contagens em probabilidades e usa a tabela para (a) gerar
nomes novos e (b) medir a qualidade do modelo via negative log-likelihood.

Run:
    python bigram.py
"""

import torch

# ---------------------------------------------------------------------------
# 1. Load the data: one name per line, lowercase, a-z only.
# ---------------------------------------------------------------------------
words = open("names.txt", "r", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]
print(f"{len(words)} nomes carregados. Exemplos: {words[:8]}")

# ---------------------------------------------------------------------------
# 2. Build the vocabulary. We add a special boundary token '.' that marks
#    both the start and the end of a name.
# ---------------------------------------------------------------------------
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}  # string -> index
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}          # index -> string
vocab_size = len(itos)
print(f"vocab_size = {vocab_size}")  # 27 = 26 letters + '.'

# ---------------------------------------------------------------------------
# 3. Count bigrams. N[i, j] = how many times char j follows char i.
# ---------------------------------------------------------------------------
N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        N[stoi[ch1], stoi[ch2]] += 1

# ---------------------------------------------------------------------------
# 4. Turn counts into probabilities (row-normalized). The "+1" is Laplace
#    smoothing: it removes zero probabilities so the model never assigns
#    probability 0 to a bigram it simply never saw in training.
# ---------------------------------------------------------------------------
P = (N + 1).float()
P = P / P.sum(dim=1, keepdim=True)

# ---------------------------------------------------------------------------
# 5. Sample new names from the model.
# ---------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647)  # reproducible randomness


def sample_name():
    out = []
    ix = 0  # always start at the boundary token '.'
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:  # sampled the boundary token -> end of name
            break
        out.append(itos[ix])
    return "".join(out)


print("\nNomes gerados pelo modelo:")
for _ in range(10):
    print("  ", sample_name())

# ---------------------------------------------------------------------------
# 6. Evaluate the model: average negative log-likelihood over every bigram.
#    Lower is better. This is exactly the "loss" we minimize in bigram_nn.py.
# ---------------------------------------------------------------------------
log_likelihood = 0.0
n = 0
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]
        log_likelihood += torch.log(prob)
        n += 1

nll = -log_likelihood / n
print(f"\nnegative log-likelihood (loss) = {nll.item():.4f}")
