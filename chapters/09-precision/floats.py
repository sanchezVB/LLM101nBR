"""
floats.py — a anatomia de um numero de ponto flutuante.

Antes de treinar com menos bits, precisamos entender o que os bits fazem. Um
float e' guardado em tres partes:

    [sinal | expoente | mantissa]

  sinal    : 1 bit, positivo ou negativo
  expoente : define a ESCALA (a ordem de grandeza) -> quanto maior, mais ALCANCE
  mantissa : define os DIGITOS significativos      -> quanto maior, mais PRECISAO

Os tres formatos que importam em deep learning:

    fp32 (float)    : 1 + 8 + 23 = 32 bits   <- o padrao
    fp16 (half)     : 1 + 5 + 10 = 16 bits   <- metade da memoria, POUCO alcance
    bf16 (bfloat16) : 1 + 8 +  7 = 16 bits   <- metade da memoria, MESMO alcance do fp32

Repare no detalhe que decide tudo: o bf16 tem os MESMOS 8 bits de expoente do
fp32. Ele sacrifica precisao (7 bits de mantissa) para manter o alcance. O fp16
faz o contrario: guarda mais precisao, mas seu alcance e' estreito -- e e' por
isso que ele estoura durante o treino.

Run:
    python floats.py
"""

import struct

import torch

# ---------------------------------------------------------------------------
# 1. Os bits de um numero, nos tres formatos.
# ---------------------------------------------------------------------------
print("=== 1. como o numero 3.14159 e' guardado ===")


def bits_fp32(x):
    """Devolve os 32 bits de um float, agrupados em sinal/expoente/mantissa."""
    (inteiro,) = struct.unpack(">I", struct.pack(">f", x))
    b = f"{inteiro:032b}"
    return b[0], b[1:9], b[9:]


def bits_16(x, dtype):
    """Idem para fp16 ou bf16, via conversao do PyTorch."""
    t = torch.tensor([x], dtype=dtype)
    bruto = t.view(torch.int16).item() & 0xFFFF
    b = f"{bruto:016b}"
    if dtype == torch.float16:
        return b[0], b[1:6], b[6:]      # 1 + 5 + 10
    return b[0], b[1:9], b[9:]          # bf16: 1 + 8 + 7


valor = 3.14159
s, e, m = bits_fp32(valor)
print(f"  fp32: {s} {e} {m}   ({len(e)} bits de expoente, {len(m)} de mantissa)")
s, e, m = bits_16(valor, torch.float16)
print(f"  fp16: {s} {e:>8s} {m:<23s}   ({len(e)} exp, {len(m)} mantissa)")
s, e, m = bits_16(valor, torch.bfloat16)
print(f"  bf16: {s} {e} {m:<23s}   ({len(e)} exp, {len(m)} mantissa)")

print("\n  o valor que cada formato realmente guarda:")
for dt, nome in ((torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    guardado = torch.tensor([valor], dtype=dt).item()
    print(f"    {nome}: {guardado!r:22s} erro = {abs(guardado - valor):.2e}")

# Um valor so' pode enganar: se os bits extras do fp16 forem zeros, ele coincide
# com o bf16. Comparamos varios para o efeito medio ficar visivel.
print("\n  erro relativo em varios valores (a media e' o que importa):")
print(f"  {'valor':>12s} {'fp16':>11s} {'bf16':>11s}  razao bf16/fp16")
import math
amostras = [math.e, math.sqrt(2), 1 / 3, 0.1, 123.456, 0.007]
somas = {torch.float16: 0.0, torch.bfloat16: 0.0}
for v in amostras:
    erros = {}
    for dt in (torch.float16, torch.bfloat16):
        g = torch.tensor([v], dtype=dt).item()
        erros[dt] = abs(g - v) / abs(v)
        somas[dt] += erros[dt]
    razao = erros[torch.bfloat16] / erros[torch.float16] if erros[torch.float16] else float("inf")
    print(f"  {v:12.6f} {erros[torch.float16]:11.2e} {erros[torch.bfloat16]:11.2e}  {razao:7.1f}x")

m16 = somas[torch.float16] / len(amostras)
mb16 = somas[torch.bfloat16] / len(amostras)
print(f"\n  erro relativo MEDIO: fp16 {m16:.2e} | bf16 {mb16:.2e} "
      f"(bf16 e' ~{mb16/m16:.0f}x pior)")
print("  Confere com a teoria: 3 bits menos de mantissa = 2^3 = 8x menos resolucao.")

# ---------------------------------------------------------------------------
# 2. Alcance: qual o maior e o menor numero representavel?
# ---------------------------------------------------------------------------
print("\n=== 2. alcance (o expoente manda) ===")
print(f"{'formato':>8s} {'maior':>12s} {'menor normal':>14s} {'epsilon':>11s}")
for dt, nome in ((torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    info = torch.finfo(dt)
    print(f"{nome:>8s} {info.max:12.3e} {info.tiny:14.3e} {info.eps:11.3e}")

print("""
  Leia a tabela:
    - o fp16 estoura em 65504. Um valor maior vira INFINITO.
    - o bf16 alcanca 3.4e38, o mesmo do fp32 -- por ter os mesmos 8 bits de expoente.
    - em troca, o epsilon do bf16 (7.8e-03) e' 16x PIOR que o do fp16 (9.8e-04):
      ele distingue menos casas decimais.

  epsilon = a menor diferenca relativa que o formato consegue representar. Um
  epsilon de 7.8e-03 significa ~2 casas decimais significativas. Parece pouco --
  e no entanto o bf16 e' o formato preferido para treinar. A secao 4 explica.""")

# ---------------------------------------------------------------------------
# 3. Overflow e underflow: onde o fp16 quebra.
# ---------------------------------------------------------------------------
print("\n=== 3. overflow e underflow no fp16 ===")
for x in (1e4, 6e4, 7e4, 1e5):
    t16 = torch.tensor([x], dtype=torch.float16)
    tb16 = torch.tensor([x], dtype=torch.bfloat16)
    print(f"  {x:9.0e} -> fp16 {t16.item():>10} | bf16 {tb16.item():>12.3e}")

print()
for x in (1e-4, 1e-6, 1e-8, 1e-10):
    t16 = torch.tensor([x], dtype=torch.float16)
    tb16 = torch.tensor([x], dtype=torch.bfloat16)
    marca = "  <- virou ZERO" if t16.item() == 0 else ""
    print(f"  {x:9.0e} -> fp16 {t16.item():>12.3e} | bf16 {tb16.item():>12.3e}{marca}")

print("""
  Os dois lados sao problema real em treino:
    OVERFLOW  (numero grande demais -> inf): acontece em ativacoes e na loss.
    UNDERFLOW (pequeno demais -> zero):      acontece nos GRADIENTES, que sao
              tipicamente da ordem de 1e-4 a 1e-8. Um gradiente que vira zero
              simplesmente NAO ATUALIZA o peso -- o treino trava, sem erro.""")

# ---------------------------------------------------------------------------
# 4. Por que bf16 ganhou do fp16 para treinar.
# ---------------------------------------------------------------------------
print("=== 4. gradientes tipicos sobrevivem? ===")
print(f"{'gradiente':>11s} {'fp16':>14s} {'bf16':>14s}")
for g in (1e-3, 1e-5, 1e-7, 1e-8, 1e-9):
    f16 = torch.tensor([g], dtype=torch.float16).item()
    b16 = torch.tensor([g], dtype=torch.bfloat16).item()
    aviso16 = " (ZERO)" if f16 == 0 else ""
    aviso_b = " (ZERO)" if b16 == 0 else ""
    print(f"{g:11.0e} {f16:11.3e}{aviso16:7s} {b16:11.3e}{aviso_b}")

print("""
  Aqui esta' a resposta: gradientes pequenos SOBREVIVEM em bf16 e MORREM em fp16.
  Para o treino, ALCANCE importa mais que PRECISAO -- a direcao aproximada do
  gradiente e' util, um gradiente zerado nao e'.

  E' por isso que o hardware moderno (TPU, A100, H100) adotou o bf16 como formato
  padrao de treino, e por que o fp16 exige um truque extra: o LOSS SCALING, que
  veremos no proximo arquivo.""")

# ---------------------------------------------------------------------------
# 5. Memoria: o motivo pratico de tudo isso.
# ---------------------------------------------------------------------------
print("=== 5. memoria ===")
n = 1_000_000
for dt, nome in ((torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")):
    t = torch.zeros(n, dtype=dt)
    print(f"  {nome}: {t.element_size()} bytes/elemento -> "
          f"{t.numel() * t.element_size() / 1e6:.1f} MB por milhao de parametros")

print("""
  Metade da memoria significa: modelo 2x maior na mesma placa, ou batch 2x maior,
  ou ativacoes 2x mais longas. Em treino de LLM, memoria e' o recurso mais escasso
  -- e e' por isso que precisao reduzida nao e' um detalhe de otimizacao, e sim
  uma decisao de arquitetura.""")
