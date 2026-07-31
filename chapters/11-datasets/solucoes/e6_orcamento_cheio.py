"""
E6 no orcamento CHEIO (3.000 passos) — o model collapse sobrevive?

Esta era a ultima conclusao do capitulo apoiada em orcamento curto. Num capitulo
onde o E4 e o E7 INVERTERAM ao passar de 400 para 3.000 passos, deixar a
terceira sem conferir era a aposta menos defensavel que restava.

O gabarito declarava a ressalva e argumentava que esta conclusao deveria
sobreviver, porque (a) o efeito medido e' grande e (b) o mecanismo e'
informacional, nao de otimizacao. Argumento nao e' medicao. Aqui esta' a
medicao.

Custo: ~45 min (dois treinos de 3.000 passos mais a geracao).

Run (a partir da pasta do capitulo):
    python solucoes/e6_orcamento_cheio.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo_comum import treinar, VAL

PASSOS = 3000
N_SINTETICOS = 60_000


@torch.no_grad()
def gerar_corpus(m, n_tokens, block=128, n_seq=64):
    """Mesma geracao paralela do gabarito.py."""
    m.eval()
    idx = torch.zeros((n_seq, 1), dtype=torch.long)
    for _ in range(math.ceil(n_tokens / n_seq)):
        logits, _ = m(idx[:, -block:])
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        prox = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, prox), dim=1)
    m.train()
    return idx.reshape(-1).numpy().astype(np.uint16)[:n_tokens]


print("=" * 74)
print(f"E6 com {PASSOS} passos — dados sinteticos e model collapse")
print("=" * 74)

print("  treinando o modelo A em dados reais...", flush=True)
tr_a, va_a, dt_a, modelo_a = treinar(block=128, passos=PASSOS)
print(f"    modelo A: treino {tr_a:.4f} | val {va_a:.4f}  ({dt_a/60:.1f} min)")

print(f"  gerando {N_SINTETICOS:,} tokens com o modelo A...", flush=True)
sintetico = gerar_corpus(modelo_a, N_SINTETICOS)

print("  treinando o modelo B no texto gerado pelo A...", flush=True)
tr_b, va_b, dt_b, _ = treinar(block=128, dados_tr=sintetico, dados_val=VAL,
                              passos=PASSOS)
print(f"    modelo B: treino {tr_b:.4f} | val {va_b:.4f}  ({dt_b/60:.1f} min)")

print(f"""
  {'modelo':>28s} {'treino':>9s} {'val real':>10s}
  {'A (corpus de Machado)':>28s} {tr_a:>9.4f} {va_a:>10.4f}
  {'B (texto gerado pelo A)':>28s} {tr_b:>9.4f} {va_b:>10.4f}

  degradacao na validacao real: {va_b - va_a:+.4f}
  perplexidade: {math.exp(va_a):.1f} -> {math.exp(va_b):.1f}
""")

print("  Com 400 passos a degradacao medida foi de +0.4496.")
if va_b > va_a:
    print(f"  Com {PASSOS} passos e' de {va_b-va_a:+.4f}. A CONCLUSAO SE MANTEM:")
    print("  treinar no proprio texto gerado degrada o modelo na validacao real.")
else:
    print(f"  Com {PASSOS} passos e' de {va_b-va_a:+.4f}. A CONCLUSAO NAO SE MANTEM --")
    print("  o gabarito precisa ser reescrito, como aconteceu com o E4 e o E7.")

print(f"""
  E a coluna de TREINO e' a parte perigosa. Com 400 passos o modelo B tinha loss
  de treino MENOR que a do A (4.30 contra 4.45): olhando so' a curva de treino,
  ele parecia estar aprendendo melhor. Aqui: {tr_b:.4f} contra {tr_a:.4f}
  ({'ainda menor' if tr_b < tr_a else 'agora maior'}).""")
