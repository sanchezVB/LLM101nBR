"""
Gabarito executavel do Capitulo 04 — attention.

Roda E2, E3, E4, E5 e E7. As respostas discursivas estao em gabarito.md.

ORCAMENTO: 4.000 passos por configuracao (a apostila usa 20.000), para o
gabarito inteiro rodar em ~10 min. Os valores absolutos ficam piores que os da
apostila; as COMPARACOES entre configuracoes continuam validas.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
PASSOS = 4000

palavras = (CAP / "names.txt").read_text(encoding="utf-8").split()
chars = sorted(set("".join(palavras)))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
V = len(stoi)
random.seed(42)
random.shuffle(palavras)
n1, n2 = int(0.8 * len(palavras)), int(0.9 * len(palavras))


def construir(ws, bs):
    X, Y = [], []
    for w in ws:
        ctx = [0] * bs
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


class Cabeca(nn.Module):
    def __init__(self, ne, hs, bs, mascarar=True, escalar=True):
        super().__init__()
        self.k = nn.Linear(ne, hs, bias=False)
        self.q = nn.Linear(ne, hs, bias=False)
        self.v = nn.Linear(ne, hs, bias=False)
        self.mascarar, self.escalar = mascarar, escalar
        self.register_buffer("tril", torch.tril(torch.ones(bs, bs)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.k(x), self.q(x), self.v(x)
        w = q @ k.transpose(-2, -1)
        if self.escalar:
            w = w * k.shape[-1] ** -0.5
        if self.mascarar:
            w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        self.ultimo_w = w.detach()
        return w @ v


class Modelo(nn.Module):
    def __init__(self, bs=8, ne=52, n_cabecas=1, usar_pos=True,
                 mascarar=True, escalar=True, n_camadas=1):
        super().__init__()
        hs = ne // n_cabecas
        self.te = nn.Embedding(V, ne)
        self.pe = nn.Embedding(bs, ne) if usar_pos else None
        # cada "camada" tem o seu conjunto de cabecas
        self.camadas = nn.ModuleList([
            nn.ModuleList([Cabeca(ne, hs, bs, mascarar, escalar) for _ in range(n_cabecas)])
            for _ in range(n_camadas)
        ])
        self.proj = nn.Linear(hs * n_cabecas, ne) if (n_cabecas > 1 or n_camadas > 1) else None
        self.lm = nn.Linear(ne if self.proj is not None else hs * n_cabecas, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx)
        if self.pe is not None:
            x = x + self.pe(torch.arange(T))
        for cabecas in self.camadas:
            y = torch.cat([c(x) for c in cabecas], dim=-1)
            x = self.proj(y) if self.proj is not None else y
        return self.lm(x[:, -1, :])


def treinar(bs=8, ne=52, n_cabecas=1, usar_pos=True, mascarar=True,
            escalar=True, n_camadas=1, passos=PASSOS):
    torch.manual_seed(1337)
    Xtr, Ytr = construir(palavras[:n1], bs)
    Xdev, Ydev = construir(palavras[n1:n2], bs)
    m = Modelo(bs, ne, n_cabecas, usar_pos, mascarar, escalar, n_camadas)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1337)
    for _ in range(passos):
        ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
        loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def perda(X, Y, chunk=8192):
        m.eval()
        tot = n = 0
        for i in range(0, X.shape[0], chunk):
            tot += F.cross_entropy(m(X[i:i+chunk]), Y[i:i+chunk], reduction="sum").item()
            n += Y[i:i+chunk].numel()
        m.train()
        return tot / n

    npar = sum(p.nelement() for p in m.parameters())
    return perda(Xtr, Ytr), perda(Xdev, Ydev), npar, m


# ===========================================================================
print("=" * 72)
print(f"E2 — sem a mascara causal (vazamento), {PASSOS} passos")
print("=" * 72)
print(f"  {'configuracao':>34s} {'treino':>9s} {'val':>9s}")
for n_cam in (1, 2):
    for masc in (True, False):
        tr, va, _, _ = treinar(mascarar=masc, n_camadas=n_cam)
        rot = f"{n_cam} camada(s), mascara {'ON ' if masc else 'OFF'}"
        print(f"  {rot:>34s} {tr:>9.4f} {va:>9.4f}")
print("""
  Resposta (e este exercicio tem uma armadilha que vale mais que a pergunta):

  Com UMA camada, remover a mascara nao muda NADA -- os numeros sao identicos.
  Nao e' erro de medicao. O motivo e' estrutural: este modelo usa apenas a
  ULTIMA posicao para prever, e a ultima linha da matriz triangular ja' e' toda
  de uns. Ou seja, para a posicao que importa a mascara nunca fez diferenca; ela
  so' afetava posicoes intermediarias cujas saidas sao DESCARTADAS.

  Com DUAS camadas a coisa muda: a saida das posicoes intermediarias da camada 1
  alimenta a camada 2, e a ultima posicao da camada 2 as le'. Ai' o futuro
  realmente vaza, e a loss sem mascara fica artificialmente melhor.

  Onde a mascara e' absolutamente essencial: quando o modelo preve em TODAS as
  posicoes (Capitulo 11). La', a posicao t veria literalmente o token t+1 que
  deve prever -- vazamento direto e devastador.

  A licao: uma protecao pode estar CORRETA e ainda assim ser inocua no seu
  desenho atual. Testar removendo-a e nao ver diferenca nao prova que ela e'
  inutil -- prova que o seu teste nao exercita o caminho que ela protege.""")

# ===========================================================================
print("=" * 72)
print(f"E3 — sem o embedding posicional, {PASSOS} passos")
print("=" * 72)
tr_p, va_p, _, _ = treinar(usar_pos=True)
tr_n, va_n, _, _ = treinar(usar_pos=False)
print(f"  {'COM posicional':>22s} {tr_p:>9.4f} {va_p:>9.4f}")
print(f"  {'SEM posicional':>22s} {tr_n:>9.4f} {va_n:>9.4f}")
print(f"  penalidade: {va_n - va_p:+.4f}")
print("""
  Respostas:
  1 e 2. Sem posicional a loss PIORA. A atencao e' invariante a permutacao: ela
  calcula afinidades entre pares, mas nada no mecanismo diz QUAL veio antes.
  Sem essa informacao, 'ana' e 'naa' viram o mesmo conjunto de tokens, e o
  modelo perde a capacidade de usar a ORDEM -- que em linguagem e' quase tudo.""")

# ===========================================================================
print("=" * 72)
print(f"E4 — sem a escala 1/sqrt(head_size), {PASSOS} passos")
print("=" * 72)
tr_e, va_e, _, m_e = treinar(escalar=True)
tr_x, va_x, _, m_x = treinar(escalar=False)
print(f"  {'COM escala':>22s} {tr_e:>9.4f} {va_e:>9.4f}")
print(f"  {'SEM escala':>22s} {tr_x:>9.4f} {va_x:>9.4f}")

# quao concentrados ficam os pesos de atencao?
Xd, _ = construir(palavras[n1:n2], 8)
with torch.no_grad():
    for nome, mod in (("com escala", m_e), ("sem escala", m_x)):
        mod(Xd[:256])
        w = mod.camadas[0][0].ultimo_w[:, -1, :]   # atencao da ultima posicao
        print(f"  {nome:>22s}: peso MAXIMO medio por linha = {w.max(dim=-1).values.mean():.3f}")
print("""
  Respostas:
  1 e 2. Sem a escala, os pesos de atencao ficam MUITO mais concentrados (o peso
  maximo por linha sobe): o softmax satura e vira quase um one-hot.
  3. Softmax saturado tem gradiente quase nulo -- e' o mesmo efeito da tanh
     saturada do Capitulo 2. O modelo trava numa escolha e para de aprender a
     ajusta-la.""")

# ===========================================================================
print("=" * 72)
print(f"E5 — tamanho do contexto, {PASSOS} passos")
print("=" * 72)
print(f"  {'block_size':>11s} {'params':>8s} {'treino':>9s} {'val':>9s}")
for bs in (3, 8, 16):
    tr, va, npar, _ = treinar(bs=bs)
    print(f"  {bs:>11d} {npar:>8d} {tr:>9.4f} {va:>9.4f}")
print("""
  Respostas (a numero 1 nao e' "mais contexto e' sempre melhor"):

  1. De 3 para 8 melhora. De 8 para 16 PIORA. Nao e' so' retorno decrescente --
     e' retorno NEGATIVO.

     Por que? Nomes tem ~7 letras. Com block_size=16, a imensa maioria das
     posicoes do contexto e' PREENCHIMENTO ('.'), e o modelo gasta capacidade
     processando padding em vez de sinal. Alem disso, os embeddings posicionais
     das posicoes distantes quase nunca sao exercitados: recebem pouco gradiente
     e continuam proximos da inicializacao, injetando ruido.

     A licao: o contexto deve ser dimensionado pelo TAMANHO REAL do dado. Um
     contexto grande demais nao e' neutro -- ele custa computacao E qualidade.
     (No Capitulo 11, com prosa, contexto 128 faz sentido porque o texto de
     verdade e' longo.)

  2. Sim: as posicoes distantes recebem pouco gradiente e ficam mal treinadas.

  3. O custo de COMPUTACAO da atencao cresce com T^2 (a matriz de afinidades e'
     TxT), mas o numero de PARAMETROS cresce so' com o embedding posicional --
     de 11.103 para 11.779, quase nada. Sao dois custos diferentes que crescem
     de formas diferentes, e e' o oposto do MLP do Capitulo 3, onde os
     PARAMETROS cresciam linearmente com o contexto.""")

# ===========================================================================
print("=" * 72)
print(f"E7 — multi-head attention, {PASSOS} passos")
print("=" * 72)
print(f"  {'cabecas':>8s} {'head_size':>10s} {'params':>8s} {'treino':>9s} {'val':>9s}")
for nc in (1, 2, 4):
    tr, va, npar, _ = treinar(n_cabecas=nc)
    print(f"  {nc:>8d} {52//nc:>10d} {npar:>8d} {tr:>9.4f} {va:>9.4f}")
print("""
  Respostas:
  1. Varias cabecas pequenas costumam ir melhor que uma grande, com custo
     parecido -- cada uma pode se especializar numa relacao diferente.
  2. Uma cabeca aprende UM criterio de "onde olhar". Varias aprendem varios: uma
     pode seguir a vogal anterior, outra o inicio da palavra. Como as saidas sao
     concatenadas e projetadas, o modelo combina esses criterios.
  3. E' exatamente a peca que abre o Capitulo 5.

  OBS: aqui as cabecas tem projecao de saida (proj), que a versao de 1 cabeca
  nao tem -- por isso a contagem de parametros nao e' identica. A comparacao
  honesta e' entre 2 e 4 cabecas, que tem a mesma estrutura.""")
