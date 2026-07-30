"""
precision_bench.py — quanto se ganha, de verdade, com menos bits?

Mede duas coisas em fp32, fp16 e bf16:
  1. VELOCIDADE de matmul, na CPU e na GPU
  2. MEMORIA ocupada

E declara honestamente o que o seu dispositivo NAO suporta.

AVISO IMPORTANTE sobre o DirectML: ele nao implementa bfloat16 e, pior, ele
ABORTA O PROCESSO ao encontrar esse tipo -- nao lanca uma excecao que se possa
capturar com try/except:

    [F730 ...] Invalid or unsupported data type BFloat16.

Por isso o script consulta uma lista de suporte conhecido ANTES de tentar. E' um
lembrete pratico: em backends menos maduros, "nao suportado" pode significar
"seu programa morre", e nao "seu programa recebe um erro".

Run:
    python precision_bench.py
"""

import time

import torch

from device import pegar_device

dev, rotulo = pegar_device()
cpu = torch.device("cpu")

DTYPES = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]


def suportado(device, dtype):
    """Suporte CONHECIDO, consultado antes de tentar (ver aviso no topo)."""
    if device.type == "privateuseone" and dtype == torch.bfloat16:
        return False, "DirectML aborta o processo com bfloat16"
    return True, ""


def drenar(saida, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type != "cpu" and torch.is_tensor(saida):
        saida.float().cpu()      # .float() antes: alguns tipos nao copiam direto


def cronometrar(fn, device, orcamento=0.3, rodadas=3):
    """Mesma metodologia do Capitulo 8: aquecer, drenar pelo resultado, minimo.

    O numero de repeticoes e' ADAPTATIVO: medimos uma chamada e escolhemos quantas
    caber num orcamento de tempo. Sem isso, o caso patologico (matmul fp16 na CPU,
    que leva ~4 s por chamada) faria o benchmark inteiro levar minutos.
    """
    t0 = time.perf_counter()
    saida = fn()
    drenar(saida, device)
    uma = time.perf_counter() - t0

    repeticoes = max(1, min(20, int(orcamento / uma))) if uma > 0 else 20
    if uma < 0.05:                      # rapido: vale aquecer mais
        for _ in range(3):
            saida = fn()
        drenar(saida, device)

    melhor = uma
    for _ in range(rodadas):
        t0 = time.perf_counter()
        for _ in range(repeticoes):
            saida = fn()
        drenar(saida, device)
        melhor = min(melhor, (time.perf_counter() - t0) / repeticoes)
    return melhor


# ---------------------------------------------------------------------------
# 1. Velocidade de matmul.
# ---------------------------------------------------------------------------
N = 1024
print(f"=== 1. matmul {N}x{N}: tempo por precisao ===")
print(f"{'dispositivo':<12s} {'fp32':>10s} {'fp16':>10s} {'bf16':>10s}   observacao")

for nome_dev, device in (("CPU", cpu), (rotulo, dev)):
    if device.type == "cpu" and nome_dev != "CPU":
        continue
    tempos = {}
    notas = []
    for nome_dt, dt in DTYPES:
        ok, motivo = suportado(device, dt)
        if not ok:
            tempos[nome_dt] = None
            notas.append(f"{nome_dt}: {motivo}")
            continue
        a = torch.randn(N, N, device=device, dtype=dt)
        tempos[nome_dt] = cronometrar(lambda: a @ a, device)

    def fmt(v):
        return f"{v*1000:10.2f}" if v else f"{'n/d':>10s}"

    print(f"{nome_dev:<12s} {fmt(tempos['fp32'])} {fmt(tempos['fp16'])} "
          f"{fmt(tempos['bf16'])}   {'; '.join(notas)}")

    base = tempos["fp32"]
    ganhos = []
    for nome_dt in ("fp16", "bf16"):
        if tempos[nome_dt]:
            ganhos.append(f"{nome_dt} {base/tempos[nome_dt]:.2f}x")
    print(f"{'':12s} ganho vs fp32: {', '.join(ganhos) if ganhos else '-'}")

# ---------------------------------------------------------------------------
# 2. Memoria.
# ---------------------------------------------------------------------------
print("\n=== 2. memoria de um modelo hipotetico ===")
print("Um Transformer de 1 bilhao de parametros, so' os PESOS:")
for nome_dt, dt in DTYPES:
    bytes_por = torch.zeros(1, dtype=dt).element_size()
    print(f"  {nome_dt}: {bytes_por} bytes/param -> {1e9 * bytes_por / 1e9:5.1f} GB")

print("""
  E os pesos sao apenas uma parte. Num treino com AdamW voce guarda tambem:
    - os gradientes            (mesmo tamanho dos pesos)
    - o momento m do AdamW     (mesmo tamanho)
    - o momento v do AdamW     (mesmo tamanho)
  ou seja, ~4x o tamanho dos pesos, mais as ativacoes intermediarias.

  Em fp32, um modelo de 1B parametros pede ~16 GB so' de estado do otimizador --
  antes de qualquer ativacao. E' por isso que precisao reduzida nao e' luxo.""")

# ---------------------------------------------------------------------------
# 3. Precisao mista: o que fica em 16 bits e o que NAO fica.
# ---------------------------------------------------------------------------
print("=== 3. o que a 'precisao mista' realmente faz ===")
print("""
  A ideia nao e' converter tudo para 16 bits. E' escolher, operacao por operacao:

    EM 16 BITS (rapido, e o erro nao acumula):
      - matmul e convolucao  <- e' aqui que esta' 95% do tempo de calculo

    EM 32 BITS (onde a precisao importa):
      - a copia MESTRA dos pesos (as atualizacoes sao pequenas; em 16 bits
        muitas seriam absorvidas pelo arredondamento e simplesmente perdidas)
      - somas longas: softmax, LayerNorm, calculo da loss
      - o estado do otimizador

  O ganho vem do primeiro grupo; a estabilidade vem do segundo. Por isso se
  chama MISTA -- e por isso `torch.autocast` decide por operacao, em vez de
  voce converter o modelo inteiro com .half().""")

# ---------------------------------------------------------------------------
# 4. Demonstracao do problema da copia mestra.
# ---------------------------------------------------------------------------
print("=== 4. por que os pesos mestres ficam em fp32 ===")
peso = 1.0
atualizacao = 1e-4          # uma atualizacao tipica de um passo de treino
print(f"  peso = {peso}, atualizacao = {atualizacao:.0e}, aplicada 100 vezes:\n")
for nome_dt, dt in DTYPES:
    p = torch.tensor([peso], dtype=dt)
    u = torch.tensor([atualizacao], dtype=dt)
    for _ in range(100):
        p = p + u
    esperado = peso + 100 * atualizacao
    print(f"  {nome_dt}: {p.item():.6f}  (esperado {esperado:.6f}, "
          f"erro {abs(p.item()-esperado):.2e})")

print("""
  Em 16 bits, uma atualizacao pequena somada a um peso de ordem 1 cai abaixo do
  epsilon do formato e e' ARREDONDADA PARA FORA -- o peso simplesmente nao muda.
  Repetido por milhares de passos, o treino estagna.

  A solucao da precisao mista: manter os pesos em fp32 e usar 16 bits apenas no
  calculo. O melhor dos dois mundos, e o motivo de a economia de memoria real
  ser menor que os 50% que a aritmetica ingenua sugere.""")
