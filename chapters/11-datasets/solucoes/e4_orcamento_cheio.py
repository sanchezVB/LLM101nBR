"""
E4 no orcamento CHEIO (3.000 passos, o mesmo da apostila).

Por que este script existe: a checagem em checagem_orcamento.py mostrou que o
ranking do E4 MUDA entre 400 e 1.200 passos --

    400 passos : 256 < 128 < 32   (256 e' o melhor)
   1200 passos : 128 < 256 < 32   (128 e' o melhor)

Ou seja, a resposta obtida com orcamento reduzido esta' errada como
generalizacao. A unica saida honesta e' medir no orcamento de verdade.

Custo: ~1 hora. Vale a hora, porque a alternativa e' publicar uma resposta que
a propria checagem do curso ja' desmentiu.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo_comum import treinar

PASSOS = 3000
resultados = {}

print("=" * 74)
print(f"E4 com {PASSOS} passos (orcamento da apostila)")
print("=" * 74)
print(f"  {'block_size':>11s} {'treino':>9s} {'val':>9s} {'minutos':>9s}")
for blk in (32, 128, 256):
    tr, va, dt, _ = treinar(block=blk, passos=PASSOS)
    resultados[blk] = va
    print(f"  {blk:>11d} {tr:>9.4f} {va:>9.4f} {dt/60:>9.1f}", flush=True)

ordem = sorted(resultados, key=lambda b: resultados[b])
print(f"\n  ranking (melhor -> pior): " + " < ".join(str(b) for b in ordem))
print(f"  melhor block_size neste orcamento: {ordem[0]}")
print("""
  Compare com os outros dois orcamentos:
     400 passos: 256 < 128 < 32
    1200 passos: 128 < 256 < 32
    3000 passos: (acima)""")
