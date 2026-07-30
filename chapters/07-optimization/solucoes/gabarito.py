"""
Gabarito executavel do Capitulo 07 — otimizacao.

Roda E2, E3, E4, E6, E7 e E8 (o E5 ja' tem solucao propria).

ORCAMENTO: 2.500 passos por configuracao (a apostila usa 15.000), para o
gabarito rodar em ~20 min em vez de horas. Os valores absolutos ficam piores;
as COMPARACOES continuam validas.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
PASSOS = 2500
BLOCK, N_EMBD, N_HEAD, N_LAYER = 8, 64, 4, 3

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
    def __init__(self):
        super().__init__()
        self.hs = N_EMBD // N_HEAD
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.fi = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fo = nn.Linear(4 * N_EMBD, N_EMBD)
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK, BLOCK)))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x))))


class Modelo(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(V, N_EMBD)
        self.pe = nn.Embedding(BLOCK, N_EMBD)
        self.blocos = nn.ModuleList([Bloco() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(N_EMBD)
        self.lm = nn.Linear(N_EMBD, V)

    def forward(self, idx):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T))
        for b in self.blocos:
            x = b(x)
        return self.lm(self.lnf(x)[:, -1, :])


def treinar(lr=1e-3, warmup=200, agendar=True, clip=None, wd=0.01,
            otimizador="adamw", passos=PASSOS):
    torch.manual_seed(1337)
    m = Modelo()
    if otimizador == "adamw":
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    elif otimizador == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)

    g = torch.Generator().manual_seed(1337)
    normas, clipados, primeiras_losses = [], 0, []
    for passo in range(passos):
        if agendar:
            if passo < warmup:
                taxa = lr * (passo + 1) / max(1, warmup)
            else:
                prog = (passo - warmup) / max(1, passos - warmup)
                taxa = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
            for grupo in opt.param_groups:
                grupo["lr"] = taxa

        ix = torch.randint(0, Xtr.shape[0], (64,), generator=g)
        loss = F.cross_entropy(m(Xtr[ix]), Ytr[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norma = torch.nn.utils.clip_grad_norm_(
            m.parameters(), clip if clip else float("inf")).item()
        normas.append(norma)
        if clip and norma > clip:
            clipados += 1
        opt.step()
        if passo < 100:
            primeiras_losses.append(loss.item())
        if not torch.isfinite(loss):
            return dict(val=float("nan"), treino=float("nan"), normas=normas,
                        clipados=clipados, inicio=primeiras_losses)

    @torch.no_grad()
    def perda(X, Y, chunk=8192):
        m.eval()
        tot = n = 0
        for i in range(0, X.shape[0], chunk):
            tot += F.cross_entropy(m(X[i:i+chunk]), Y[i:i+chunk], reduction="sum").item()
            n += Y[i:i+chunk].numel()
        m.train()
        return tot / n

    return dict(treino=perda(Xtr, Ytr), val=perda(Xdev, Ydev), normas=normas,
                clipados=clipados, inicio=primeiras_losses)


def fmt(v):
    return f"{v:8.4f}" if v == v else "DIVERGIU"


# ===========================================================================
print("=" * 74)
print(f"E2 — o warmup e' necessario? ({PASSOS} passos)")
print("=" * 74)
print(f"  {'warmup':>8s} {'val':>10s} {'norma media (100 primeiros passos)':>36s}")
for w in (0, 50, 200, 500):
    r = treinar(warmup=w)
    n100 = sum(r["normas"][:100]) / 100
    print(f"  {w:>8d} {fmt(r['val']):>10s} {n100:>36.3f}")
print("""
  Respostas (e o resultado contraria a expectativa -- inclusive a minha):

  1. Neste modelo o warmup NAO ajuda: warmup=0 da' a melhor loss (1.9198) e
     aumentar o warmup piora de forma monotonica (1.9226 com 500 passos). O
     motivo e' simples: gastar 500 dos 2.500 passos com learning rate reduzida
     e' desperdicar 20% do orcamento num modelo que nao tem problema de
     estabilidade.

  2. CUIDADO COM A METRICA. A norma media dos gradientes nos 100 primeiros
     passos AUMENTA com o warmup (1.223 -> 1.645), e seria tentador ler isso
     como 'o warmup deixa o treino mais instavel'. Nao e' isso.

     Com warmup a learning rate comeca minuscula, entao o modelo demora mais
     para sair da regiao de inicializacao -- onde a loss e' alta e os gradientes
     sao naturalmente maiores. A norma alta mede LENTIDAO, nao instabilidade.
     Escolhi uma metrica que nao testava o que eu queria testar.

  3. Entao por que o warmup existe? Porque nos primeiros passos as medias moveis
     m e v do Adam quase nao tem informacao -- sao estimativas ruins baseadas em
     pouquissimos gradientes. Em modelos GRANDES, com batches grandes, um passo
     ruim no inicio pode levar a loss a um regime do qual ela nao se recupera.
     Este modelo e' pequeno e robusto demais para exibir esse problema.

     Licao: o warmup e' um seguro contra um risco que cresce com a escala.
     Nesta escala, ele so' custa.""")

# ===========================================================================
print("=" * 74)
print(f"E3 — calibrando o gradient clipping ({PASSOS} passos)")
print("=" * 74)
base = treinar(clip=None)
norma_media = sum(base["normas"]) / len(base["normas"])
norma_max = max(base["normas"])
print(f"  norma do gradiente sem clipping: media {norma_media:.3f}, maxima {norma_max:.3f}\n")
print(f"  {'clip':>8s} {'val':>10s} {'% de passos cortados':>22s}")
for c in (0.1, 1.0, 3.0, 100.0):
    r = treinar(clip=c)
    print(f"  {c:>8.1f} {fmt(r['val']):>10s} {100*r['clipados']/PASSOS:>21.1f}%")
print(f"""
  Respostas:
  1. Com clip=3.0 (acima da norma tipica de {norma_media:.2f}) quase nada e'
     cortado -- e' o comportamento correto de uma rede de seguranca.
  2. Com clip=0.1 TODO passo e' cortado (100%): isso deixa de ser clipping e
     vira normalizacao do gradiente -- o erro que a apostila documenta na Secao 5.

     Mas seja honesto com o numero: aqui isso NAO prejudicou (1.9206 contra
     1.9219 sem clipping -- ate' marginalmente melhor). Normalizar todo gradiente
     e' um algoritmo DIFERENTE, nao necessariamente pior; ele so' nao e' o que
     voce pensa que esta' rodando. O problema de configurar errado nem sempre e'
     perder desempenho: as vezes e' nao saber qual algoritmo voce esta' usando.
  3. Com clip=100 nada e' cortado, e a loss e' praticamente a mesma de nao ter
     clipping. Conclusao honesta: NESTE modelo o clipping nao e' necessario --
     nao ha' picos. Ele existe para o caso patologico, nao para o caso comum.
  4. O clipping altera so' o TAMANHO do gradiente, nao a DIRECAO: todos os
     componentes sao multiplicados pelo mesmo fator. A informacao de 'para onde
     ir' e' preservada.""")

# ===========================================================================
print("=" * 74)
print("E4 — o ganho certo para a GELU")
print("=" * 74)


def testar_ganho(ganho, ativacao, n_camadas=8, dim=256):
    torch.manual_seed(1337)
    h = torch.randn(1024, dim)
    for _ in range(n_camadas):
        W = torch.randn(dim, dim) * ganho / dim ** 0.5
        h = ativacao(h @ W)
    return h.std().item()


print(f"  {'ganho':>8s} {'tanh':>10s} {'relu':>10s} {'gelu':>10s}")
for ganho in (1.0, 1.414, 1.5, 1.667, 2.0):
    linha = [testar_ganho(ganho, f) for f in (torch.tanh, F.relu, F.gelu)]
    print(f"  {ganho:>8.3f} " + " ".join(f"{v:>10.4f}" for v in linha))

# procura o ganho que mantem std ~1 para GELU
melhor, melhor_d = None, 1e9
for i in range(80, 260):
    ganho = i / 100
    s = testar_ganho(ganho, F.gelu)
    if abs(s - 1.0) < melhor_d:
        melhor, melhor_d = ganho, abs(s - 1.0)
print(f"\n  ganho que mantem std~1 na 8a camada com GELU: ~{melhor:.2f}")
try:
    print(f"  torch.nn.init.calculate_gain('relu') = "
          f"{torch.nn.init.calculate_gain('relu'):.4f} (= sqrt(2))")
    torch.nn.init.calculate_gain("gelu")
except Exception as e:
    print(f"  calculate_gain('gelu') -> {type(e).__name__}: nao existe entrada para GELU")
print("""
  Respostas:
  1 e 2. O ganho da GELU fica proximo do da ReLU (sqrt(2) = 1.414), um pouco
     acima -- faz sentido, ja' que a GELU e' uma versao suave da ReLU e deixa
     passar um pouco menos de sinal.
  3. O PyTorch NAO tem entrada para GELU em calculate_gain. Na pratica quase
     ninguem ajusta: modelos modernos usam LayerNorm, que re-normaliza a cada
     bloco e torna o ganho exato bem menos critico (a apostila mede isso).""")

# ===========================================================================
print("=" * 74)
print(f"E6 — AdamW vs SGD no modelo de verdade ({PASSOS} passos)")
print("=" * 74)
print(f"  {'otimizador':>16s} {'lr':>8s} {'val':>10s}")
r = treinar(otimizador="adamw", lr=1e-3)
print(f"  {'AdamW':>16s} {1e-3:>8.0e} {fmt(r['val']):>10s}")
for lr in (1e-3, 1e-2, 1e-1, 5e-1):
    r = treinar(otimizador="sgd", lr=lr)
    print(f"  {'SGD':>16s} {lr:>8.0e} {fmt(r['val']):>10s}")
for lr in (1e-2, 1e-1):
    r = treinar(otimizador="sgdm", lr=lr)
    print(f"  {'SGD + momentum':>16s} {lr:>8.0e} {fmt(r['val']):>10s}")
print("""
  Respostas:
  1. Com a MESMA learning rate (1e-3), o SGD fica MUITO atras: 2.65 contra 1.92.
     Um passo que e' bom para o AdamW e' minusculo para o SGD.

  2. Mas ajustando a learning rate o quadro muda: o SGD+momentum com lr=1e-1
     chega a 1.9213 -- EMPATA com o AdamW (1.9219). O SGD puro com lr=5e-1 fica
     logo atras (1.9294).

     Ou seja: neste modelo o AdamW nao e' magicamente superior -- ele e'
     superior COM A LEARNING RATE PADRAO. A grande vantagem pratica do Adam nao
     e' chegar mais longe, e' chegar la' sem que voce precise cacar a lr certa.
     Isso vale muito quando cada tentativa custa horas de treino.
  3. O modelo tem parametros de escalas MUITO diferentes: embeddings, pesos de
     atencao, ganhos de LayerNorm, vieses. Uma unica learning rate nao serve para
     todos (e' o experimento do optimizers.py). O Adam normaliza o passo por
     parametro e escapa desse aperto.""")

# ===========================================================================
print("=" * 74)
print(f"E7 — por que o weight decay atrapalhou ({PASSOS} passos)")
print("=" * 74)
print(f"  {'weight decay':>13s} {'treino':>9s} {'val':>9s} {'gap (val-treino)':>18s}")
for wd in (0.0, 0.01, 0.1, 0.5):
    r = treinar(wd=wd)
    gap = r["val"] - r["treino"]
    print(f"  {wd:>13.2f} {fmt(r['treino']):>9s} {fmt(r['val']):>9s} {gap:>18.4f}")
print("""
  Respostas:
  1 e 2. Aumentar o weight decay piora TREINO e VALIDACAO juntos, e o 'gap'
     entre eles quase nao muda. Esse padrao e' a assinatura de que a
     regularizacao NAO esta' resolvendo overfitting -- ela so' esta' removendo
     capacidade.
  3. Com gap praticamente nulo no baseline, nao ha' overfitting para combater.
     Weight decay e' remedio para uma doenca que este modelo nao tem.
  4. Ele ajudaria com POUCOS dados -- releia o E5 do Capitulo 3 (155 nomes:
     treino 0.80 vs val 6.51). La' o gap e' enorme, e a regularizacao teria o
     que fazer.""")

# ===========================================================================
print("=" * 74)
print(f"E8 — outros agendamentos ({PASSOS} passos)")
print("=" * 74)


def treinar_agendamento(tipo, lr=1e-3, warmup=200, passos=PASSOS):
    torch.manual_seed(1337)
    m = Modelo()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(1337)
    for passo in range(passos):
        if passo < warmup:
            taxa = lr * (passo + 1) / warmup
        else:
            prog = (passo - warmup) / max(1, passos - warmup)
            if tipo == "constante":
                taxa = lr
            elif tipo == "cosseno":
                taxa = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))
            elif tipo == "linear":
                taxa = lr * (1 - 0.9 * prog)
            else:                      # degraus
                taxa = lr * (1.0 if prog < 0.5 else (0.1 if prog < 0.8 else 0.01))
        for grupo in opt.param_groups:
            grupo["lr"] = taxa
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

    return perda(Xdev, Ydev)


print(f"  {'agendamento':>14s} {'val':>10s}")
res = {}
for tipo in ("constante", "cosseno", "linear", "degraus"):
    res[tipo] = treinar_agendamento(tipo)
    print(f"  {tipo:>14s} {res[tipo]:>10.4f}")
vencedor = min(res, key=res.get)
print(f"\n  melhor: {vencedor} ({res[vencedor]:.4f}); "
      f"pior: {max(res, key=res.get)} ({max(res.values()):.4f})")
print(f"  diferenca entre o melhor e o pior: {max(res.values())-min(res.values()):.4f}")
print("""
  Respostas:
  1 e 2. Decair ganha de nao decair, mas os agendamentos nao sao equivalentes
     entre si: o LINEAR e o COSSENO ficam praticamente empatados e claramente a'
     frente; os DEGRAUS ficam quase tao ruins quanto o constante.

     Faz sentido: com apenas 2.500 passos, o primeiro degrau (em 50%) demora
     demais a chegar. Agendamentos por degrau foram feitos para treinos longos,
     em que cada patamar tem tempo de render.

     E o cosseno ser o padrao da area e' mais convencao do que superioridade
     medida -- pelo menos nesta escala, o linear empata ou ganha.
  3. O agendamento por cosseno precisa saber max_steps DE ANTEMAO. Se voce
     quiser continuar um treino ja' terminado, a curva ja' chegou ao fim e nao
     ha' como estica-la sem estragar o formato. E' um incomodo real: por isso
     existem alternativas como o WSD (warmup-stable-decay), que mantem a lr
     constante por tempo indeterminado e so' decai no fim.""")
