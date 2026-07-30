"""
loss_scaling.py — o truque que faz o fp16 funcionar, implementado do zero.

O problema (medido no floats.py): gradientes tipicos ficam entre 1e-4 e 1e-8, e o
fp16 zera tudo abaixo de ~6e-5 (ou ~6e-8 nos subnormais). Um gradiente zerado
nao atualiza o peso -- e o treino trava SEM DAR ERRO.

A solucao e' surpreendentemente simples. Como a derivada e' linear:

    d(k * L)/dw = k * dL/dw

...se multiplicarmos a LOSS por um fator k grande antes do backward, todos os
gradientes saem multiplicados por k -- longe da zona de underflow. Depois
dividimos por k antes de atualizar os pesos. O resultado e' identico ao que
teriamos em fp32, mas os numeros intermediarios ficaram representaveis.

    loss * 1024  ->  backward  ->  gradientes * 1024  ->  dividir por 1024  ->  usar

Este arquivo mede o problema e a solucao, e implementa um GradScaler dinamico.

Run:
    python loss_scaling.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


def modelinho(dtype):
    """Uma rede pequena, com pesos pequenos de proposito -- e' assim que os
    gradientes ficam minusculos, como num modelo profundo de verdade."""
    torch.manual_seed(1337)
    m = nn.Sequential(
        nn.Linear(64, 128), nn.GELU(),
        nn.Linear(128, 128), nn.GELU(),
        nn.Linear(128, 16),
    ).to(dtype)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.02)          # pesos pequenos -> gradientes pequenos
    return m


def gradientes(dtype, escala=1.0):
    """Faz um forward/backward e devolve todos os gradientes achatados."""
    m = modelinho(dtype)
    x = (torch.randn(32, 64) * 0.05).to(dtype)
    alvo = torch.randint(0, 16, (32,))

    saida = m(x)
    loss = F.cross_entropy(saida.float(), alvo)     # a loss em fp32, como manda o manual
    (loss * escala).backward()

    return torch.cat([p.grad.flatten().float() for p in m.parameters()])


# ---------------------------------------------------------------------------
# 1. Quantos gradientes o fp16 perde?
# ---------------------------------------------------------------------------
print("=== 1. gradientes que viram zero (underflow) ===")
g32 = gradientes(torch.float32)
g16 = gradientes(torch.float16)

zeros32 = (g32 == 0).sum().item()
zeros16 = (g16 == 0).sum().item()
total = g32.numel()

print(f"  total de gradientes: {total}")
print(f"  zerados em fp32: {zeros32:5d} ({zeros32/total:6.1%})")
print(f"  zerados em fp16: {zeros16:5d} ({zeros16/total:6.1%})   <- perdidos por underflow")
print(f"\n  magnitude mediana dos gradientes (fp32): {g32.abs().median().item():.2e}")
print(f"  menor valor normal do fp16:              {torch.finfo(torch.float16).tiny:.2e}")

# ---------------------------------------------------------------------------
# 2. O loss scaling recupera esses gradientes?
# ---------------------------------------------------------------------------
print("\n=== 2. efeito do loss scaling (fp16) ===")
print(f"{'escala':>8s} {'zerados':>9s} {'%':>7s} {'infs':>6s}  situacao")
for escala in (1, 16, 256, 1024, 65536, 2**22, 2**26, 2**30):
    g = gradientes(torch.float16, escala=escala)
    g_desescalado = g / escala          # desfaz a escala, como num treino real
    zer = (g_desescalado == 0).sum().item()
    infs = (~torch.isfinite(g)).sum().item()
    fracao = zer / total
    if infs:
        situacao = "OVERFLOW -- passo tem de ser descartado"
    elif fracao > 0.01:
        situacao = "perde MUITO gradiente"
    elif fracao > 0.001:
        situacao = "ainda perde um pouco"
    else:
        situacao = "ok"
    print(f"{escala:8d} {zer:9d} {zer/total:6.1%} {infs:6d}  {situacao}")

print("""
  Existe uma JANELA: escala pequena nao resolve o underflow, escala grande causa
  overflow (gradientes viram inf). E a janela se MOVE durante o treino, porque a
  magnitude dos gradientes muda conforme o modelo aprende.

  Por isso ninguem usa escala fixa: usa-se um scaler DINAMICO.""")

# ---------------------------------------------------------------------------
# 3. Um GradScaler dinamico, do zero.
# ---------------------------------------------------------------------------
print("=== 3. GradScaler dinamico (a versao do zero) ===")


class GradScalerSimples:
    """Ajusta a escala automaticamente durante o treino.

    Regra: se apareceu inf/nan, a escala esta' alta demais -- DIVIDE e descarta o
    passo. Se passaram N passos consecutivos sem problema, tenta MULTIPLICAR para
    aproveitar mais alcance. E' um controle de realimentacao simples que persegue
    a maior escala que ainda nao estoura.
    """

    def __init__(self, escala=65536.0, fator=2.0, intervalo=100):
        self.escala = escala
        self.fator = fator
        self.intervalo = intervalo
        self.bons_seguidos = 0
        self.descartados = 0

    def escalar(self, loss):
        return loss * self.escala

    def passo(self, parametros):
        """Desescala os gradientes e diz se o passo pode ser aplicado."""
        achou_inf = any(
            not torch.isfinite(p.grad).all() for p in parametros if p.grad is not None
        )
        if achou_inf:
            # passo perdido: NAO atualiza os pesos, e reduz a escala
            self.escala = max(1.0, self.escala / self.fator)
            self.bons_seguidos = 0
            self.descartados += 1
            return False

        for p in parametros:
            if p.grad is not None:
                p.grad /= self.escala

        self.bons_seguidos += 1
        if self.bons_seguidos >= self.intervalo:
            self.escala *= self.fator      # ousa mais
            self.bons_seguidos = 0
        return True


# simulacao: gradientes que crescem ao longo do "treino", para forcar o ajuste
scaler = GradScalerSimples(escala=65536.0, fator=2.0, intervalo=5)
print(f"  {'passo':>6s} {'escala':>10s} {'aplicado?':>10s}")
historico = []
for passo in range(1, 26):
    m = modelinho(torch.float16)
    x = (torch.randn(32, 64) * 0.05 * (1 + passo * 0.4)).to(torch.float16)  # cresce
    alvo = torch.randint(0, 16, (32,))
    loss = F.cross_entropy(m(x).float(), alvo)
    scaler.escalar(loss).backward()
    aplicado = scaler.passo(list(m.parameters()))
    historico.append((passo, scaler.escala, aplicado))
    if passo % 5 == 0 or not aplicado:
        print(f"  {passo:6d} {scaler.escala:10.0f} {'sim' if aplicado else 'DESCARTADO':>10s}")

print(f"\n  passos descartados: {scaler.descartados}/25")
print("""
  Descartar alguns passos e' aceitavel: com escala bem ajustada isso acontece em
  menos de 1% das iteracoes, e o custo e' irrelevante perto do ganho.

  Na pratica voce nao escreve isso: usa `torch.amp.GradScaler`, que faz exatamente
  este algoritmo (e mais alguns detalhes). Mas agora voce sabe o que ele faz.""")

# ---------------------------------------------------------------------------
# 4. A API de produção: autocast.
# ---------------------------------------------------------------------------
print("=== 4. como se escreve na pratica ===")
print("""
  Com fp16 (precisa de scaler, porque fp16 sofre underflow):

      scaler = torch.amp.GradScaler(device="cuda")
      for x, y in dados:
          with torch.autocast(device_type="cuda", dtype=torch.float16):
              loss = F.cross_entropy(modelo(x), y)      # matmuls em fp16
          opt.zero_grad(set_to_none=True)
          scaler.scale(loss).backward()                 # escala o backward
          scaler.step(opt)                              # desescala e aplica
          scaler.update()                               # ajusta a escala

  Com bf16 (NAO precisa de scaler -- o alcance do bf16 ja' cobre os gradientes):

      for x, y in dados:
          with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
              loss = F.cross_entropy(modelo(x), y)
          opt.zero_grad(set_to_none=True)
          loss.backward()
          opt.step()

  A segunda versao e' mais simples, e e' por isso que o bf16 virou o padrao onde o
  hardware o suporta: ele elimina uma engrenagem inteira do treino.""")

# verificacao de que o autocast funciona neste ambiente
print("  verificando o autocast aqui:")
for nome, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
    try:
        with torch.autocast(device_type="cpu", dtype=dt):
            r = torch.randn(32, 32) @ torch.randn(32, 32)
        print(f"    [OK   ] autocast(cpu, {nome}) -> matmul saiu em {r.dtype}")
    except Exception as e:
        print(f"    [FALHA] autocast(cpu, {nome}): {type(e).__name__}")
