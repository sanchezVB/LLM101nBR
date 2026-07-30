"""
Solucao do Exercicio E6 — inspecionando onde o modelo presta atencao.

Treina uma versao rapida do modelo (5000 passos, so' para ter pesos razoaveis) e
imprime a distribuicao de atencao da ULTIMA posicao para alguns contextos. Assim
"abrimos" o modelo e vemos em quais caracteres anteriores ele se apoia na hora de
prever o proximo.

Run (a partir da pasta do capitulo):
    python solucoes/e6_inspecionar_atencao.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)

words = open(Path(__file__).resolve().parent.parent / "names.txt", encoding="utf-8").read().splitlines()
words = [w.strip() for w in words if w.strip()]
chars = sorted(set("".join(words)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

block_size, n_embd, head_size = 8, 24, 24


def build(ws):
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
Xtr, Ytr = build(words[: int(0.8 * len(words))])


class Head(nn.Module):
    """Igual a' do model.py, mas devolve TAMBEM os pesos de atencao."""

    def __init__(self):
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
        return wei @ v, wei          # <- devolve os pesos para inspecao


class AttentionLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(V, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = Head()
        self.lm_head = nn.Linear(head_size, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T))
        x, wei = self.head(x)
        return self.lm_head(x[:, -1, :]), wei


model = AttentionLM()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("treinando (5000 passos, versao rapida)...")
for step in range(5000):
    ix = torch.randint(0, Xtr.shape[0], (64,))
    logits, _ = model(Xtr[ix])
    loss = F.cross_entropy(logits, Ytr[ix])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
print(f"loss final do treino rapido: {loss.item():.4f}\n")


@torch.no_grad()
def inspecionar(prefixo):
    """Mostra a atencao da ultima posicao para o contexto dado."""
    ctx = [0] * block_size
    for ch in prefixo:
        ctx = ctx[1:] + [stoi[ch]]
    logits, wei = model(torch.tensor([ctx]))
    ultima = wei[0, -1]                      # atencao da ultima posicao
    chars_ctx = [itos[i] for i in ctx]

    print(f"contexto: {' '.join(chars_ctx)}   (prevendo o proximo apos '{prefixo}')")
    for c, p in zip(chars_ctx, ultima.tolist()):
        barra = "#" * int(p * 40)
        print(f"    '{c}'  {p:.3f}  {barra}")
    probs = F.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, 3)
    sugestoes = ", ".join(f"'{itos[i.item()]}' ({p.item():.2f})" for p, i in zip(top.values, top.indices))
    print(f"    -> proximos mais provaveis: {sugestoes}\n")


for prefixo in ["ana", "mari", "jo", "vinici"]:
    inspecionar(prefixo)

print("Duas coisas para observar:")
print()
print("1. A atencao NAO e' uniforme. Compare com a media do 'bag of words', onde")
print("   todas as posicoes tinham exatamente o mesmo peso. Aqui o modelo distribui")
print("   de forma desigual -- ele esta' escolhendo.")
print()
print("2. Boa parte do peso costuma cair nos tokens de preenchimento '.', e nao nas")
print("   letras! Isso NAO e' um erro do modelo. Quando ele nao precisa trazer")
print("   informacao de nenhuma posicao especifica, ele precisa colocar o peso em")
print("   ALGUM lugar (o softmax obriga a soma a ser 1), e escolhe uma posicao")
print("   inofensiva -- um 'estacionamento' para a atencao. Esse fenomeno e' real e")
print("   bem documentado em LLMs grandes, onde e' chamado de 'attention sink'.")
