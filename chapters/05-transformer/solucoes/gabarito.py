"""
Gabarito executavel do Capitulo 05 — Transformer.

Roda E3, E4, E5, E6 e E7 (o E2 ja' tem solucao propria em e2_ablacoes.py).

ORCAMENTO: 2.500 passos por configuracao (a apostila usa 15.000). Os valores
absolutos ficam piores; as COMPARACOES continuam validas.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
PASSOS = 2500
BLOCK = 8

palavras = (CAP / "names.txt").read_text(encoding="utf-8").split()
chars = sorted(set("".join(palavras)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
V = len(stoi)
random.seed(42)
random.shuffle(palavras)
n1, n2 = int(0.8 * len(palavras)), int(0.9 * len(palavras))


def construir(ws):
    X, Y = [], []
    for w in ws:
        ctx = [0] * BLOCK
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


Xtr, Ytr = construir(palavras[:n1])
Xdev, Ydev = construir(palavras[n1:n2])


class Bloco(nn.Module):
    def __init__(self, ne, nh, usar_ln=True, dropout=0.0):
        super().__init__()
        self.hs = ne // nh
        self.nh = nh
        self.qkv = nn.Linear(ne, 3 * ne, bias=False)
        self.proj = nn.Linear(ne, ne)
        self.fi = nn.Linear(ne, 4 * ne)
        self.fo = nn.Linear(4 * ne, ne)
        self.usar_ln = usar_ln
        self.ln1 = nn.LayerNorm(ne) if usar_ln else nn.Identity()
        self.ln2 = nn.LayerNorm(ne) if usar_ln else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK, BLOCK)))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.nh, self.hs).transpose(1, 2)
        k = k.view(B, T, self.nh, self.hs).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hs).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.fo(F.gelu(self.fi(self.ln2(x)))))


class Modelo(nn.Module):
    def __init__(self, ne=64, nh=4, nl=3, usar_ln=True, dropout=0.0):
        super().__init__()
        self.te = nn.Embedding(V, ne)
        self.pe = nn.Embedding(BLOCK, ne)
        self.blocos = nn.ModuleList([Bloco(ne, nh, usar_ln, dropout) for _ in range(nl)])
        self.lnf = nn.LayerNorm(ne) if usar_ln else nn.Identity()
        self.lm = nn.Linear(ne, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T))
        for b in self.blocos:
            x = b(x)
        return self.lm(self.lnf(x)[:, -1, :])


def treinar(ne=64, nh=4, nl=3, usar_ln=True, dropout=0.0, lr=1e-3, passos=PASSOS):
    torch.manual_seed(1337)
    m = Modelo(ne, nh, nl, usar_ln, dropout)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1337)
    oscilacao = []
    for passo in range(passos):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
        loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if passo >= passos - 200:
            oscilacao.append(loss.item())
        if not torch.isfinite(loss):
            return float("nan"), float("nan"), sum(p.nelement() for p in m.parameters()), 0.0

    @torch.no_grad()
    def perda(X, Y, chunk=8192):
        m.eval()
        tot = n = 0
        for i in range(0, X.shape[0], chunk):
            tot += F.cross_entropy(m(X[i:i+chunk]), Y[i:i+chunk], reduction="sum").item()
            n += Y[i:i+chunk].numel()
        m.train()
        return tot / n

    desvio = torch.tensor(oscilacao).std().item() if oscilacao else 0.0
    return perda(Xtr, Ytr), perda(Xdev, Ydev), sum(p.nelement() for p in m.parameters()), desvio


# ===========================================================================
print("=" * 74)
print(f"E3 — sem LayerNorm, {PASSOS} passos")
print("=" * 74)
print(f"  {'configuracao':>28s} {'treino':>9s} {'val':>9s} {'oscilacao':>10s}")
for usar_ln in (True, False):
    for lr in (1e-3, 3e-3):
        tr, va, _, osc = treinar(usar_ln=usar_ln, lr=lr)
        rot = f"{'COM' if usar_ln else 'SEM'} LayerNorm, lr={lr:g}"
        va_s = f"{va:9.4f}" if va == va else "  DIVERGIU"
        print(f"  {rot:>28s} {tr:>9.4f} {va_s} {osc:>10.4f}")
print("""
  Respostas:
  1 e 2. Sem LayerNorm o treino fica pior e MAIS INSTAVEL -- olhe a coluna
  'oscilacao' (desvio padrao da loss nos ultimos 200 passos). Com learning rate
  maior a diferenca aumenta, e sem LayerNorm o treino pode ate' divergir.
  3. Normalizar mantem as ativacoes numa faixa previsivel a cada bloco, entao o
     erro de escala nao se acumula com a profundidade -- exatamente o que a
     Secao 4 da apostila mede com os ganhos de inicializacao.""")

# ===========================================================================
print("=" * 74)
print(f"E4 — profundidade e numero de cabecas, {PASSOS} passos")
print("=" * 74)
print(f"  {'n_layer':>8s} {'n_head':>7s} {'params':>8s} {'treino':>9s} {'val':>9s}")
for nl in (1, 3, 6):
    tr, va, npar, _ = treinar(nl=nl, nh=4)
    print(f"  {nl:>8d} {4:>7d} {npar:>8d} {tr:>9.4f} {va:>9.4f}")
print()
for nh in (1, 4, 8):
    tr, va, npar, _ = treinar(nl=3, nh=nh)
    print(f"  {3:>8d} {nh:>7d} {npar:>8d} {tr:>9.4f} {va:>9.4f}")
print("""
  Respostas:
  1 e 2. Mais profundidade ajuda, com retorno decrescente (e, em modelo pequeno
     com dados limitados, pode ate' piorar por overfitting).
  3. Mudar n_head NAO muda o numero de parametros: head_size = n_embd // n_head,
     entao 4 cabecas de 16 tem exatamente os mesmos pesos que 1 cabeca de 64.
     O que muda e' COMO esses pesos sao organizados -- quantos softmaxes
     independentes existem.""")

# ===========================================================================
print("=" * 74)
print(f"E5 — uma cabeca grande vs varias pequenas, {PASSOS} passos")
print("=" * 74)
print(f"  {'n_head':>7s} {'head_size':>10s} {'params':>8s} {'val':>9s}")
for nh in (1, 8, 64):
    tr, va, npar, _ = treinar(nl=3, nh=nh)
    print(f"  {nh:>7d} {64//nh:>10d} {npar:>8d} {va:>9.4f}")
print("""
  Respostas:
  1 e 2. Varias cabecas pequenas costumam ganhar: cada uma pode se especializar
     numa relacao diferente, e o modelo combina o que todas trouxeram.
  3. Existe LIMITE: com n_head=64 cada cabeca tem dimensao 1, e um produto
     escalar de vetores de dimensao 1 quase nao consegue expressar afinidade --
     a atencao vira quase uma comparacao de dois numeros. Na pratica usa-se
     head_size entre 32 e 128.""")

# ===========================================================================
print("=" * 74)
print(f"E6 — dropout, {PASSOS} passos")
print("=" * 74)
print(f"  {'dropout':>8s} {'treino':>9s} {'val':>9s} {'gap':>8s}")
for d in (0.0, 0.1, 0.3):
    tr, va, _, _ = treinar(dropout=d)
    print(f"  {d:>8.1f} {tr:>9.4f} {va:>9.4f} {va-tr:>8.4f}")
print("""
  Respostas:
  1. Com este dataset (64 mil nomes) treino e validacao ja' andam juntos -- o
     'gap' e' pequeno. Sem overfitting para combater, o dropout tende a
     ATRAPALHAR: ele so' remove capacidade. E' a mesma licao do weight decay no
     Capitulo 7.
  2. Dropout ajudaria com POUCOS dados (releia o E5 do Capitulo 3: 155 nomes,
     treino 0.80 vs val 6.51).
  3. Na avaliacao o dropout DEVE ser desligado (model.eval()), senao voce estaria
     medindo um modelo aleatoriamente mutilado -- e a loss de validacao viria
     pior e ruidosa, sem significar nada.""")

# ===========================================================================
print("=" * 74)
print("E7 — contando os parametros")
print("=" * 74)
ne, nh, nl = 64, 4, 3
te = V * ne
pe = BLOCK * ne
por_bloco = (3 * ne * ne) + (ne * ne + ne) + (ne * 4 * ne + 4 * ne) + (4 * ne * ne + ne) + 4 * ne
lnf = 2 * ne
lm = ne * V + V
formula = te + pe + nl * por_bloco + lnf + lm
_, _, real, _ = treinar(passos=1)
print(f"  token_emb   : {V} x {ne}                 = {te:>7d}")
print(f"  pos_emb     : {BLOCK} x {ne}                  = {pe:>7d}")
print(f"  por bloco   : qkv + proj + ff + 2 LayerNorm = {por_bloco:>7d}")
print(f"  x {nl} blocos                               = {nl*por_bloco:>7d}")
print(f"  LayerNorm final                            = {lnf:>7d}")
print(f"  lm_head     : {ne} x {V} + {V}             = {lm:>7d}")
print(f"  {'-'*52}")
print(f"  formula                                    = {formula:>7d}")
print(f"  medido                                     = {real:>7d}   "
      f"{'OK' if formula == real else 'DIVERGE'}")
print(f"""
  Quem domina: os blocos somam {nl*por_bloco} de {formula} ({100*nl*por_bloco/formula:.0f}%),
  e dentro de cada bloco o FEEDFORWARD e' o maior pedaco (a expansao 4x).
  Dobrar n_embd QUADRUPLICA o custo dos blocos (todas as matrizes sao ne x ne
  ou ne x 4ne), enquanto dobrar n_layer apenas dobra.""")
