"""
Gabarito executavel do Capitulo 03 — MLP.

Roda E2, E3, E4, E6 e E7. As respostas discursivas estao em gabarito.md.

NOTA SOBRE O ORCAMENTO: a apostila treina 20.000 passos; aqui usamos 4.000 para
que o gabarito inteiro rode em poucos minutos. Os numeros ABSOLUTOS ficam piores
que os da apostila, mas as COMPARACOES entre configuracoes -- que e' o que os
exercicios pedem -- continuam validas.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent

PASSOS = 4000

palavras = (CAP / "names.txt").read_text(encoding="utf-8").split()
chars = sorted(set("".join(palavras)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

random.seed(42)
random.shuffle(palavras)
n1, n2 = int(0.8 * len(palavras)), int(0.9 * len(palavras))


def construir(ws, block_size):
    X, Y = [], []
    for w in ws:
        ctx = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def treinar(block_size=3, n_embd=10, n_hidden=200, lr=0.1, decair=True, passos=PASSOS):
    """Devolve (loss_treino, loss_val, n_parametros, C) -- C serve para o E6."""
    Xtr, Ytr = construir(palavras[:n1], block_size)
    Xdev, Ydev = construir(palavras[n1:n2], block_size)

    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((V, n_embd), generator=g)
    fan_in = n_embd * block_size
    W1 = torch.randn((fan_in, n_hidden), generator=g) * (5 / 3) / fan_in ** 0.5
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    W2 = torch.randn((n_hidden, V), generator=g) * 0.01
    b2 = torch.zeros(V)
    ps = [C, W1, b1, W2, b2]
    for p in ps:
        p.requires_grad = True

    for passo in range(passos):
        ix = torch.randint(0, Xtr.shape[0], (32,), generator=g)
        emb = C[Xtr[ix]].view(32, -1)
        loss = F.cross_entropy(F.gelu(emb @ W1 + b1) @ W2 + b2, Ytr[ix])
        for p in ps:
            p.grad = None
        loss.backward()
        taxa = lr if (not decair or passo < passos * 0.75) else lr * 0.1
        for p in ps:
            p.data += -taxa * p.grad
        if not torch.isfinite(loss):
            return float("nan"), float("nan"), sum(p.nelement() for p in ps), C

    @torch.no_grad()
    def perda(X, Y):
        emb = C[X].view(X.shape[0], -1)
        return F.cross_entropy(F.gelu(emb @ W1 + b1) @ W2 + b2, Y).item()

    return perda(Xtr, Ytr), perda(Xdev, Ydev), sum(p.nelement() for p in ps), C


# ===========================================================================
print("=" * 72)
print(f"E2 — o efeito do contexto (block_size), {PASSOS} passos")
print("=" * 72)
print(f"  {'block_size':>11s} {'params':>8s} {'treino':>8s} {'val':>8s}")
for bs in (1, 3, 5):
    tr, va, npar, _ = treinar(block_size=bs)
    print(f"  {bs:>11d} {npar:>8d} {tr:>8.4f} {va:>8.4f}")
print("""
  Respostas:
  1. Mais contexto AJUDA: a loss de validacao cai de block_size 1 para 3 e 5.
  2. Com block_size=1 o MLP vira um "bigrama neural" -- ele so' ve' o caractere
     anterior, exatamente como o Capitulo 1. A loss fica na faixa do bigrama.
  3. O numero de parametros CRESCE com o block_size, porque a primeira camada
     tem (n_embd * block_size) x n_hidden pesos. Dobrar o contexto dobra essa
     matriz -- e' o custo que motiva a atencao no Capitulo 4.""")

# ===========================================================================
print("=" * 72)
print(f"E3 — tamanho da camada oculta, {PASSOS} passos")
print("=" * 72)
print(f"  {'n_hidden':>9s} {'params':>8s} {'treino':>8s} {'val':>8s}")
for nh in (50, 200, 500):
    tr, va, npar, _ = treinar(n_hidden=nh)
    print(f"  {nh:>9d} {npar:>8d} {tr:>8.4f} {va:>8.4f}")
print("""
  Respostas:
  1. Mais neuronios = mais parametros e, em geral, loss menor.
  2. O ganho tem RETORNO DECRESCENTE: de 50 para 200 melhora bem mais do que de
     200 para 500, apesar de o numero de parametros crescer muito mais.""")

# ===========================================================================
print("=" * 72)
print(f"E4 — learning rate, {PASSOS} passos")
print("=" * 72)
print(f"  {'lr':>8s} {'val':>10s}   situacao")
resultados_lr = {}
for lr in (0.001, 0.01, 0.1, 1.0, 10.0, 50.0):
    tr, va, _, _ = treinar(lr=lr)
    resultados_lr[lr] = va
    if va != va:
        sit = "DIVERGIU (nao-numero)"
    elif va > 2.5:
        sit = "lenta demais / instavel"
    else:
        sit = ""
    print(f"  {lr:>8} {va:>10.4f}   {sit}")

melhor_lr = min((v, k) for k, v in resultados_lr.items() if v == v)[1]
print(f"\n  melhor deste experimento: lr={melhor_lr}")

tr_sem, va_sem, _, _ = treinar(lr=melhor_lr, decair=False)
tr_com, va_com, _, _ = treinar(lr=melhor_lr, decair=True)
print(f"\n  com lr={melhor_lr}:  sem decaimento {va_sem:.4f} | "
      f"com decaimento {va_com:.4f}  ({va_com - va_sem:+.4f})")
print("""
  Respostas (e a primeira provavelmente contraria o que voce esperava):

  1. A curva e' em U, mas o MINIMO nao esta' no 0.1 que a apostila usa -- neste
     orcamento de 4.000 passos, learning rates bem MAIORES vao melhor. So' bem
     acima disso o treino desestabiliza.

     Isso nao significa que a apostila esteja errada: ela treina 20.000 passos,
     e com orcamento maior uma lr menor tem tempo de refinar e alcanca um
     resultado melhor. A conclusao correta e' que **a melhor learning rate
     depende do orcamento de passos** -- exatamente o que o Capitulo 7 mede de
     forma sistematica (e onde o mesmo fenomeno reaparece: 3e-3 ganha de 1e-3
     quando se treina pouco).

  2. O decaimento ajuda: passos menores no fim permitem "assentar" no minimo em
     vez de ficar pulando em volta dele.""")

# ===========================================================================
print("=" * 72)
print("E6 — visualizando os embeddings (n_embd=2)")
print("=" * 72)
tr, va, _, C2 = treinar(n_embd=2)
print(f"  treino {tr:.4f} | val {va:.4f}  (pior que com n_embd=10, como esperado)\n")

vogais = set("aeiou")
emb = C2.detach()
# distancia media dentro do grupo das vogais vs entre vogais e consoantes
idx_vog = [stoi[c] for c in vogais if c in stoi]
idx_cons = [i for c, i in stoi.items() if c not in vogais and c != "."]


def dist_media(ids_a, ids_b):
    total, n = 0.0, 0
    for i in ids_a:
        for j in ids_b:
            if i != j:
                total += (emb[i] - emb[j]).norm().item()
                n += 1
    return total / n


d_vv = dist_media(idx_vog, idx_vog)
d_vc = dist_media(idx_vog, idx_cons)
print(f"  distancia media VOGAL-VOGAL      : {d_vv:.3f}")
print(f"  distancia media VOGAL-CONSOANTE  : {d_vc:.3f}")
print(f"  razao: {d_vc/d_vv:.2f}x  ({'vogais agrupadas' if d_vc > d_vv else 'SEM agrupamento'})")
print("""
  Resposta: as vogais ficam MAIS PROXIMAS entre si do que das consoantes -- a
  rede descobriu sozinha que elas tem papel parecido, so' por prever o proximo
  caractere. Ninguem ensinou fonetica a ela.

  n_embd=2 piora a loss (menos capacidade por caractere), mas e' o unico jeito de
  desenhar os embeddings num plano. E' um trade-off comum: reduzir dimensao para
  poder VER, sabendo que o modelo fica pior.""")

# ===========================================================================
print("=" * 72)
print("E7 — contando os parametros na mao")
print("=" * 72)
bs, ne, nh = 3, 10, 200
formula = V * ne + (ne * bs) * nh + nh + nh * V + V
_, _, real, _ = treinar(passos=1)
print(f"  C  (embeddings) : {V} x {ne}        = {V*ne:>6d}")
print(f"  W1              : {ne*bs} x {nh}      = {ne*bs*nh:>6d}")
print(f"  b1              : {nh}             = {nh:>6d}")
print(f"  W2              : {nh} x {V}       = {nh*V:>6d}")
print(f"  b2              : {V}              = {V:>6d}")
print(f"  {'-'*46}")
print(f"  formula         : {formula:>6d}")
print(f"  medido no codigo: {real:>6d}   {'OK' if formula == real else 'DIVERGE'}")
print(f"""
  Formula geral:
    vocab*n_embd + (n_embd*block_size)*n_hidden + n_hidden + n_hidden*vocab + vocab

  Quem domina: W1 tem {ne*bs*nh} pesos e W2 tem {nh*V} -- juntos sao
  {100*(ne*bs*nh + nh*V)/formula:.0f}% do total. Dobrar n_embd quase dobra o total
  (ele aparece em C e em W1); dobrar n_hidden tambem, pois aparece em W1 e W2.""")
