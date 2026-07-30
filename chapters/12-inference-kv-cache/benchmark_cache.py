"""
benchmark_cache.py — quanto o KV-cache acelera, e quanto de memoria custa.

Disciplina de medicao herdada do Capitulo 08:
  - aquecimento antes de cronometrar
  - MELHOR de varias rodadas, nao a media (interferencia so' sabe atrasar)
  - maquina ociosa; feche o resto antes de rodar

Run (a partir da pasta do capitulo):
    python benchmark_cache.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import carregar

m, _ = carregar()
prompt = torch.zeros((1, 1), dtype=torch.long)


def cronometrar(fn, rodadas=3):
    fn()                                   # aquecimento
    melhor = float("inf")
    for _ in range(rodadas):
        t0 = time.perf_counter()
        fn()
        melhor = min(melhor, time.perf_counter() - t0)
    return melhor


# ===========================================================================
print("=" * 74)
print("1. Velocidade da geracao, com e sem cache")
print("=" * 74)
print(f"  {'tokens':>7s} {'ingenuo (s)':>12s} {'cache (s)':>11s} {'speedup':>9s} "
      f"{'ms/token ing.':>14s} {'ms/token cache':>15s}")

for n in (16, 32, 64, 128):
    t_ing = cronometrar(lambda: m.gerar_ingenuo(prompt, n))
    t_cac = cronometrar(lambda: m.gerar_com_cache(prompt, n))
    print(f"  {n:>7d} {t_ing:>12.3f} {t_cac:>11.3f} {t_ing/t_cac:>8.2f}x "
          f"{t_ing/n*1000:>14.1f} {t_cac/n*1000:>15.1f}", flush=True)

print("""
  O que observar: o ms/token do caminho INGENUO cresce com o comprimento (cada
  token novo custa mais que o anterior, porque reprocessa um contexto maior). O
  do caminho com CACHE fica praticamente constante -- e' o ponto todo.""")

# ===========================================================================
print("=" * 74)
print("2. O custo por token, isolado: prefill vs decode")
print("=" * 74)
ctx = torch.zeros((1, 64), dtype=torch.long)
t_prefill = cronometrar(lambda: m(ctx))
_, cache = m(ctx)
um = torch.zeros((1, 1), dtype=torch.long)
t_decode = cronometrar(lambda: m(um, cache=cache))
print(f"  prefill de 64 tokens de uma vez : {t_prefill*1000:7.2f} ms "
      f"({t_prefill/64*1000:.2f} ms por token)")
print(f"  decode de 1 token com cache     : {t_decode*1000:7.2f} ms")
print(f"  razao: um decode custa {t_decode/(t_prefill/64):.1f}x o preco de um token no prefill")
print("""
  Sao duas fases com perfis OPOSTOS, e reconhecer isso organiza toda a
  inferencia de verdade:

    PREFILL - processa muitos tokens de uma vez. Limitado por CALCULO: e' uma
              matmul grande, que satura bem o processador.
    DECODE  - processa UM token de cada vez. Limitado por MEMORIA: para produzir
              um unico token e' preciso ler TODOS os pesos do modelo da memoria.

  Por isso um token de decode custa muito mais que um token de prefill: o
  trabalho de leitura dos pesos e' o mesmo, diluido em 1 token em vez de 64.
  E' a dicotomia latencia/vazao do Capitulo 08, de novo.""")

# ===========================================================================
print("=" * 74)
print("3. Quanto o cache ocupa de memoria")
print("=" * 74)
print(f"  {'contexto':>9s} {'cache (KB)':>12s}   {'cache/pesos':>12s}")
pesos = sum(p.nelement() for p in m.parameters()) * 4
for t in (128, 1024, 8192):
    b = m.bytes_do_cache(t)
    print(f"  {t:>9d} {b/1024:>12.0f} {b/pesos:>11.1%}")
print(f"\n  (os pesos do modelo ocupam {pesos/1024/1024:.1f} MB em float32)")

print(f"\n  E num modelo de verdade -- ordem de grandeza de um 7B "
      f"(32 camadas, 32 cabecas, head_size 128):")
for t, lote in ((4096, 1), (4096, 32), (32768, 32)):
    b = 2 * 32 * lote * 32 * t * 128 * 2          # 2 bytes: bf16
    print(f"    contexto {t:>6,}, batch {lote:>3d}: {b/1e9:>6.1f} GB de cache")
print("""
  Repare: com batch 32 e contexto longo, o CACHE passa dos pesos. Um modelo de
  7B em bf16 ocupa ~14 GB; o cache pode ocupar mais que isso.

  E' por isso que servir LLM e' caro de um jeito que treinar nao e': o cache
  cresce com o numero de usuarios simultaneos VEZES o comprimento de cada
  conversa. Metade da engenharia de inferencia moderna -- multi-query attention,
  grouped-query attention, paged attention -- existe para atacar esta tabela.""")
