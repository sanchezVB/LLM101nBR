"""
Solucao do Exercicio E5 — overfitting com dataset pequeno.

Roda EXATAMENTE o mesmo MLP, mas com poucos nomes, e mostra a loss de treino
despencando enquanto a de validacao dispara. E' o retrato classico do
overfitting: o modelo decora em vez de generalizar.

Compare o resultado deste script (dataset pequeno) com o mlp.py (dataset grande):
mesma arquitetura, conclusao oposta -- a diferenca e' so a quantidade de dados.

Run (a partir da pasta do capitulo):
    python solucoes/e5_overfitting.py
"""
import torch
import torch.nn.functional as F
import random

# 155 nomes (os mesmos do Capitulo 1)
NOMES_PEQUENO = """ana maria joao jose pedro paulo lucas marcos mateus tiago andre bruno
carlos daniel eduardo felipe gabriel gustavo henrique igor joaquim leonardo miguel
nicolas otavio rafael ricardo rodrigo samuel thiago victor vinicius william arthur
bernardo caio diego enzo fabio fernando francisco guilherme heitor ian ivan julio
leandro luan luiz marcelo mauricio murilo nathan otto pablo renan sergio vitor wesley
yuri alice amanda beatriz bianca camila carla carolina clara debora eduarda elaine
fernanda gabriela helena isabela isadora jessica juliana larissa laura leticia livia
luana manuela mariana marina melissa natalia patricia paula priscila rafaela raquel
rebeca renata sabrina sofia sophia tatiane vanessa vitoria yasmin adriana alessandra
bruna cristina daniela denise flavia giovanna ingrid joana kelly monica nicole regina
simone viviane agatha bento cecilia davi elisa emanuel esther francisca gael ines lara
lorenzo lucca maite noah olivia pietra ravi theo valentina anthony benicio catarina
emilly isaac liz murielle otelo pamela quesia rogerio silvana ubirajara vladimir
washington ximena zelia""".split()

block_size = 3
n_embd, n_hidden = 10, 200


def build(words):
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    X, Y = [], []
    for w in words:
        ctx = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y), len(stoi)


random.seed(42)
random.shuffle(NOMES_PEQUENO)
n1 = int(0.8 * len(NOMES_PEQUENO))
Xtr, Ytr, V = build(NOMES_PEQUENO[:n1])
Xdev, Ydev, _ = build(NOMES_PEQUENO[n1:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((V, n_embd), generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5 / 3) / (n_embd * block_size) ** 0.5
b1 = torch.randn(n_hidden, generator=g) * 0.01
W2 = torch.randn((n_hidden, V), generator=g) * 0.01
b2 = torch.zeros(V)
ps = [C, W1, b1, W2, b2]
for p in ps:
    p.requires_grad = True


def fwd(X):
    h = F.gelu(C[X].view(X.shape[0], -1) @ W1 + b1)
    return h @ W2 + b2


for step in range(20000):
    ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
    loss = F.cross_entropy(fwd(Xtr[ix]), Ytr[ix])
    for p in ps:
        p.grad = None
    loss.backward()
    lr = 0.1 if step < 15000 else 0.01
    for p in ps:
        p.data += -lr * p.grad


@torch.no_grad()
def sl(X, Y):
    return F.cross_entropy(fwd(X), Y).item()


print("=== DATASET PEQUENO (155 nomes) ===")
print(f"loss treino    = {sl(Xtr, Ytr):.4f}")
print(f"loss validacao = {sl(Xdev, Ydev):.4f}")
print("\nObserve o ABISMO entre treino e validacao: o modelo DECOROU o treino")
print("(loss baixa) mas NAO generaliza (loss de validacao alta). Isso e' overfitting.")
print("Compare com o mlp.py (64 mil nomes), onde treino ~ validacao ~ 1.97.")
