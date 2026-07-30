"""
Gabarito executavel do Capitulo 09 — precisao.

Roda E2, E3, E4, E5 e E7. O E6 depende de GPU (veja gabarito.md).
Tudo aqui roda na CPU em segundos.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))

# ===========================================================================
print("=" * 74)
print("E2 — achando os limites na mao")
print("=" * 74)
for nome, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
    # subindo ate' estourar
    x, n = torch.tensor([1.0], dtype=dt), 0
    while torch.isfinite(x).all():
        x = x * 2
        n += 1
    maior_pot = n - 1
    # descendo ate' zerar
    y, m = torch.tensor([1.0], dtype=dt), 0
    while y.item() != 0.0:
        y = y / 2
        m += 1
    info = torch.finfo(dt)
    print(f"  {nome}: estoura em 2^{maior_pot} = {2.0**maior_pot:.3e}  "
          f"(finfo.max = {info.max:.3e})")
    print(f"        zera em 2^-{m} = {2.0**-m:.3e}  "
          f"(finfo.tiny = {info.tiny:.3e})")
print("""
  Respostas:
  1. O expoente maximo bate com finfo.max (o 2^n exato fica um pouco abaixo do
     max, que usa a mantissa cheia).
  2. Da' para ir MUITO abaixo do .tiny antes de zerar por causa dos numeros
     SUBNORMAIS: abaixo do menor normal, o formato abre mao de precisao para
     continuar representando valores cada vez menores, ate' acabar a mantissa.
     E' uma "rampa de saida" gradual em vez de um corte seco.
  3. O bf16 estoura e zera em expoentes MUITO maiores porque tem 8 bits de
     expoente contra 5 do fp16 -- o mesmo alcance do fp32.""")

# ===========================================================================
print("=" * 74)
print("E3 — o epsilon na pratica")
print("=" * 74)
print(f"  {'x':>10s} {'1+x em fp32':>14s} {'1+x em fp16':>14s} {'1+x em bf16':>14s}")
for x in (1e-1, 1e-2, 1e-3, 1e-4):
    linha = []
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        um = torch.tensor([1.0], dtype=dt)
        r = (um + torch.tensor([x], dtype=dt)).item()
        linha.append(f"{r:.6f}" + ("*" if r == 1.0 else " "))
    print(f"  {x:>10.0e} " + " ".join(f"{v:>14s}" for v in linha))
print("  (* = a soma NAO mudou nada: o x foi arredondado para fora)")
print(f"""
  Respostas:
  1. Em fp16 a soma para de mudar por volta de x = 1e-4 (epsilon
     {torch.finfo(torch.float16).eps:.2e}).
  2. Em bf16 o limite e' MAIOR ({torch.finfo(torch.bfloat16).eps:.2e}): ele tem
     3 bits menos de mantissa, entao distingue menos casas.
  3. E' exatamente por isso que os pesos mestres ficam em fp32. Uma atualizacao
     de treino tem ordem de 1e-4 a 1e-6; somada a um peso de ordem 1 em 16 bits,
     ela e' arredondada para fora e o peso NAO MUDA -- como a apostila mede
     (1.000000 depois de 100 atualizacoes).""")

# ===========================================================================
print("=" * 74)
print("E4 — a janela do loss scaling se move")
print("=" * 74)


def rede(dtype, escala_pesos):
    torch.manual_seed(1337)
    m = nn.Sequential(nn.Linear(64, 128), nn.GELU(),
                      nn.Linear(128, 128), nn.GELU(),
                      nn.Linear(128, 16)).to(dtype)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(escala_pesos)
    return m


def medir(escala_pesos, escala_loss):
    m = rede(torch.float16, escala_pesos)
    x = (torch.randn(32, 64) * 0.05).to(torch.float16)
    alvo = torch.randint(0, 16, (32,))
    loss = F.cross_entropy(m(x).float(), alvo)
    (loss * escala_loss).backward()
    g = torch.cat([p.grad.flatten().float() for p in m.parameters()])
    zerados = (g / escala_loss == 0).sum().item()
    infs = (~torch.isfinite(g)).sum().item()
    return zerados / g.numel(), infs


print(f"  {'pesos':>8s} {'escala':>10s} {'% zerados':>11s} {'infs':>8s}  situacao")
for escala_pesos in (0.02, 0.5):
    for escala_loss in (1, 1024, 2**20, 2**26):
        frac, infs = medir(escala_pesos, escala_loss)
        if infs:
            sit = "OVERFLOW"
        elif frac > 0.01:
            sit = "perde gradiente"
        else:
            sit = "ok"
        print(f"  {escala_pesos:>8.2f} {escala_loss:>10d} {frac:>10.1%} {infs:>8d}  {sit}")
    print()
print("""  Respostas (leia a tabela com cuidado -- o efeito nao e' o que parece):

  1 e 2. Com pesos MAIORES (0.5) o underflow praticamente desaparece: com escala
     1, os zerados caem de 50.3% para 0.1%. Ou seja, o limite INFERIOR da janela
     desce muito.

     Mas o limite SUPERIOR quase nao se mexe: os dois casos comecam a estourar
     por volta de 2^20. Entao a janela nao se DESLOCA -- ela ALARGA.

     Faz sentido: o underflow depende da magnitude dos gradientes (que muda com
     os pesos), enquanto o overflow depende do teto do formato (65504), que e'
     fixo. So' o piso se move.

  3. Ainda assim, o scaler DINAMICO continua se justificando: a magnitude dos
     gradientes muda MUITO ao longo do treino (cai conforme o modelo converge),
     e uma escala fixa escolhida no inicio pode ficar pequena demais depois --
     reintroduzindo underflow justamente na fase de ajuste fino, onde os
     gradientes sao menores.""")

# ===========================================================================
print("=" * 74)
print("E5 — melhorando o GradScaler")
print("=" * 74)


class ScalerComPiso:
    """Como o da apostila, mas com um PISO configuravel para a escala."""

    def __init__(self, escala=65536.0, fator=2.0, intervalo=5, piso=1.0):
        self.escala, self.fator, self.intervalo, self.piso = escala, fator, intervalo, piso
        self.bons, self.descartados = 0, 0

    def passo(self, params):
        tem_inf = any(not torch.isfinite(p.grad).all() for p in params if p.grad is not None)
        if tem_inf:
            self.escala = max(self.piso, self.escala / self.fator)
            self.bons = 0
            self.descartados += 1
            return False
        for p in params:
            if p.grad is not None:
                p.grad /= self.escala
        self.bons += 1
        if self.bons >= self.intervalo:
            self.escala *= self.fator
            self.bons = 0
        return True


print(f"  {'fator':>8s} {'escala final':>14s} {'descartados':>13s}")
for fator in (1.01, 2.0, 16.0):
    sc = ScalerComPiso(fator=fator)
    for passo in range(40):
        m = rede(torch.float16, 0.02)
        x = (torch.randn(32, 64) * 0.05 * (1 + passo * 0.3)).to(torch.float16)
        alvo = torch.randint(0, 16, (32,))
        loss = F.cross_entropy(m(x).float(), alvo)
        (loss * sc.escala).backward()
        sc.passo(list(m.parameters()))
    print(f"  {fator:>8.2f} {sc.escala:>14.0f} {sc.descartados:>12d}/40")
print("""
  Respostas:
  1. Fator GRANDE (16) reage rapido mas passa do ponto: sobe demais, estoura,
     desce demais -- e descarta mais passos. Fator PEQUENO (1.01) e' estavel
     mas demora para se ajustar quando a magnitude dos gradientes muda.
  2. O PISO evita que uma sequencia de overflows derrube a escala a um valor tao
     baixo que reintroduza underflow -- o scaler pode "se auto-sabotar" ao
     reagir a um pico isolado.
  3. growth_interval=2000 (o padrao do PyTorch) e' longo de proposito: subir a
     escala e' barato de errar (perde-se 1 passo), mas subir COM FREQUENCIA faz
     o treino descartar passos o tempo todo. Melhor subir raramente e ficar
     estavel.""")

# ===========================================================================
print("=" * 74)
print("E7 — treino em precisao mista (versao CPU)")
print("=" * 74)
print("""  A apostila mede que o ganho de VELOCIDADE nao aparece nesta maquina
  (CPU nao tem unidade de 16 bits; DirectML da 0.98x). O que da' para verificar
  aqui e' a QUALIDADE: autocast estraga o treino?
""")

torch.manual_seed(1337)
X = torch.randn(2000, 64)
Y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] > 0).long()


def treinar_amp(modo, passos=400):
    torch.manual_seed(1337)
    m = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 2))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1337)
    for _ in range(passos):
        ix = torch.randint(0, len(X), (64,), generator=g)
        if modo == "fp32":
            loss = F.cross_entropy(m(X[ix]), Y[ix])
        else:
            dt = torch.bfloat16 if modo == "bf16" else torch.float16
            with torch.autocast(device_type="cpu", dtype=dt):
                loss = F.cross_entropy(m(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.cross_entropy(m(X), Y).item()


print(f"  {'modo':>10s} {'loss final':>12s}")
for modo in ("fp32", "bf16", "fp16"):
    print(f"  {modo:>10s} {treinar_amp(modo):>12.4f}")
print("""
  Respostas (e a numero 2 exige honestidade sobre o que este teste NAO mostra):

  1 e 3. O autocast com bf16 chega a' MESMA loss do fp32. Ele nao estraga o
     treino -- resultado esperado, ja' que o bf16 tem o alcance do fp32.

  2. Aqui o fp16 SEM scaler tambem deu identico. Isso NAO significa que loss
     scaling seja dispensavel -- significa que ESTE problema nao provoca
     underflow.

     Compare com o loss_scaling.py da apostila: la' os pesos sao deliberadamente
     pequenos (p.mul_(0.02)), os gradientes ficam na casa de 1e-8, e 50.3% deles
     zeram em fp16. Aqui os pesos sao os da inicializacao padrao, os gradientes
     sao saudaveis, e nada zera.

     A licao metodologica: um teste que passa nao prova que a protecao e'
     desnecessaria -- prova que o teste nao exercita o caso que ela protege.
     (E' a mesma armadilha do E2 do Capitulo 4, com a mascara causal.)

     Modelos profundos de verdade tem gradientes muito menores nas camadas
     iniciais, e e' la' que o fp16 sem scaler quebra.

  4. A economia de memoria real fica ABAIXO dos 50% teoricos porque a copia
     MESTRA dos pesos e o estado do otimizador continuam em fp32 -- so' as
     ativacoes e as matmuls usam 16 bits.""")
