"""
Gabarito executavel do Capitulo 01 — roda os exercicios que pedem medicao.

Cada secao corresponde a um exercicio. As respostas discursivas estao em
gabarito.md; aqui ficam os NUMEROS, para voce comparar com os seus.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
from pathlib import Path

import torch

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))


def carregar(arquivo="names.txt"):
    palavras = (CAP / arquivo).read_text(encoding="utf-8").split()
    chars = sorted(set("".join(palavras)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return palavras, stoi, itos


def contagens(palavras, stoi, V):
    N = torch.zeros((V, V), dtype=torch.float32)
    for w in palavras:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            N[stoi[a], stoi[b]] += 1
    return N


def loss_contagem(palavras, stoi, V, suavizacao):
    N = contagens(palavras, stoi, V)
    P = N + suavizacao
    P = P / P.sum(1, keepdim=True)
    total, n = 0.0, 0
    for w in palavras:
        chs = ["."] + list(w) + ["."]
        for a, b in zip(chs, chs[1:]):
            total += torch.log(P[stoi[a], stoi[b]])
            n += 1
    return (-total / n).item()


# ===========================================================================
print("=" * 70)
print("E2 — mexer nos dados")
print("=" * 70)
palavras, stoi, itos = carregar()
V = len(stoi)
base = loss_contagem(palavras, stoi, V, 1)
print(f"  {len(palavras)} nomes originais      -> loss {base:.4f}")

extras = ["zwqk", "xyzabc", "kkkk", "wwww", "qqzz"]   # nomes propositalmente estranhos
p2 = palavras + extras
_, stoi2, _ = carregar()
base2 = loss_contagem(p2, stoi2, len(stoi2), 1)
print(f"  + 5 nomes ESTRANHOS      -> loss {base2:.4f}  ({base2-base:+.4f})")

comuns = ["ana", "maria", "joao", "jose", "pedro"]     # nomes ja' tipicos
p3 = palavras + comuns
base3 = loss_contagem(p3, stoi2, len(stoi2), 1)
print(f"  + 5 nomes TIPICOS        -> loss {base3:.4f}  ({base3-base:+.4f})")
print("""
  Resposta: depende do que voce acrescenta. Nomes com padroes raros AUMENTAM a
  loss (o modelo tem de gastar probabilidade em bigramas incomuns); nomes que
  seguem o padrao ja' dominante a DIMINUEM. Nao e' "mais dados = melhor loss" --
  e' "dados mais previsiveis = melhor loss".""")

# ===========================================================================
print("=" * 70)
print("E3 — o efeito da suavizacao")
print("=" * 70)
print(f"  {'suavizacao':>12s} {'loss':>10s}   observacao")
for s in (0, 1, 10, 100):
    L = loss_contagem(palavras, stoi, V, s)
    if s == 0:
        obs = "menor loss, mas P=0 em bigramas nao vistos"
    elif s >= 100:
        obs = "quase uniforme: o modelo 'esquece' os dados"
    else:
        obs = ""
    print(f"  {s:>12d} {L:>10.4f}   {obs}")
print("""
  Respostas:
  1. A loss CRESCE com a suavizacao. Suavizar afasta o modelo dos dados.
  2. Com suavizacao alta os nomes gerados ficam mais aleatorios (a distribuicao
     tende ao uniforme); com +0 ficam mais "grudados" no dataset.
  3. Com +0, um bigrama nunca visto tem P=0 -> log(0) = -inf. Neste dataset a
     loss ainda sai finita porque avaliamos NOS MESMOS dados do treino (todo
     bigrama avaliado foi visto). Avalie num nome novo com um par inedito e a
     loss vira infinito. E' exatamente por isso que se suaviza.""")

# ===========================================================================
print("=" * 70)
print("E4 — temperatura no sampling")
print("=" * 70)
N = contagens(palavras, stoi, V)
P = (N + 1)
P = P / P.sum(1, keepdim=True)


def amostrar(P, T, semente=2147483647, n=6):
    g = torch.Generator().manual_seed(semente)
    saidas = []
    for _ in range(n):
        out, ix = [], 0
        while True:
            p = P[ix] ** (1.0 / T)          # T<1 concentra, T>1 achata
            p = p / p.sum()
            ix = torch.multinomial(p, 1, generator=g).item()
            if ix == 0 or len(out) > 15:
                break
            out.append(itos[ix])
        saidas.append("".join(out))
    return saidas


for T in (0.3, 1.0, 3.0):
    print(f"  T={T:<4} -> {', '.join(amostrar(P, T))}")
print("""
  Respostas:
  1. T BAIXO deixa os nomes mais "obvios" e repetitivos (concentra a massa nas
     opcoes mais provaveis). T ALTO deixa mais aleatorios e impronunciaveis.
  2. E' exatamente o parametro `temperature` das APIs de LLM. Note que usamos
     p**(1/T): T<1 eleva a um expoente >1, acentuando as diferencas; T>1 as
     achata. E' a mesma operacao aplicada aos logits nos modelos modernos.""")

# ===========================================================================
print("=" * 70)
print("E5 — igualando contagem e rede neural")
print("=" * 70)
import torch.nn.functional as F

xs, ys = [], []
for w in palavras:
    chs = ["."] + list(w) + ["."]
    for a, b in zip(chs, chs[1:]):
        xs.append(stoi[a])
        ys.append(stoi[b])
xs, ys = torch.tensor(xs), torch.tensor(ys)


def treinar_nn(reg, passos, lr=50):
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((V, V), generator=g, requires_grad=True)
    xenc = F.one_hot(xs, num_classes=V).float()
    for _ in range(passos):
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = -probs[torch.arange(len(ys)), ys].log().mean() + reg * (W ** 2).mean()
        W.grad = None
        loss.backward()
        W.data += -lr * W.grad
    logits = xenc @ W
    probs = logits.exp()
    probs = probs / probs.sum(1, keepdim=True)
    return -probs[torch.arange(len(ys)), ys].log().mean().item()   # loss PURA


print(f"  contagem com +1     : {loss_contagem(palavras, stoi, V, 1):.4f}")
print(f"  contagem com +0.01  : {loss_contagem(palavras, stoi, V, 0.01):.4f}")
print(f"  rede, reg=0.01, 200 : {treinar_nn(0.01, 200):.4f}")
print(f"  rede, reg=0.0, 1000 : {treinar_nn(0.0, 1000):.4f}")
print("""
  Resposta: com a suavizacao IGUALADA dos dois lados, os numeros convergem para
  a mesma faixa (~2,1). Isso confirma que contar e otimizar chegam ao MESMO
  modelo -- a diferenca que se via antes vinha so' do "+1" ser uma suavizacao
  muito mais forte que o "reg=0.01".""")

# ===========================================================================
print("=" * 70)
print("E7 — trigrama")
print("=" * 70)
T3 = torch.zeros((V, V, V), dtype=torch.float32)
for w in palavras:
    chs = ["."] + list(w) + ["."]
    for a, b, c in zip(chs, chs[1:], chs[2:]):
        T3[stoi[a], stoi[b], stoi[c]] += 1

for s in (1.0, 0.01):
    P3 = T3 + s
    P3 = P3 / P3.sum(2, keepdim=True)
    total, n = 0.0, 0
    for w in palavras:
        chs = ["."] + list(w) + ["."]
        for a, b, c in zip(chs, chs[1:], chs[2:]):
            total += torch.log(P3[stoi[a], stoi[b], stoi[c]])
            n += 1
    print(f"  trigrama, suavizacao +{s:<5}: loss {(-total/n).item():.4f}")

zeros = (T3 == 0).float().mean().item()
print(f"\n  celulas da tabela: {V}x{V}x{V} = {V**3}")
print(f"  celulas VAZIAS: {zeros:.1%}")
print("""
  Respostas:
  1. A tabela vira 27x27x27 = 19.683 celulas (contra 729 do bigrama).
  2. A loss MELHORA em relacao ao bigrama -- mais contexto ajuda mesmo.
  3. Mas olhe a fracao de celulas vazias: a maioria dos trigramas NUNCA aparece.
     Isso e' o problema dos dados esparsos, e piora exponencialmente com o
     tamanho do contexto (27^(k+1) celulas). A suavizacao evita o log(0), mas
     nao inventa informacao: para combinacoes nunca vistas o modelo so' chuta.
     E' exatamente essa parede que motiva o MLP do Capitulo 3, que APRENDE uma
     representacao densa em vez de contar celulas.""")
