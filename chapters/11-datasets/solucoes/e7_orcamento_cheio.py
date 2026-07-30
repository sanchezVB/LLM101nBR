"""
E7 no orcamento CHEIO (3.000 passos), pelo mesmo motivo do e4_orcamento_cheio.

Com 400 passos o E7 deu: a validacao melhora de 4 para 6 obras e depois PIORA
com 8 e 11. Mas o E4 acabou de mostrar que 400 passos nao decide nada neste
capitulo -- o ranking dele virou DUAS vezes conforme o orcamento subiu (256
ganhava com 400 passos, 128 com 1.200, e 32 com 3.000). Seria incoerente
confiar no numero curto aqui logo depois de descobrir aquilo.

Custo: ~1 hora.

Run (a partir da pasta do capitulo):
    python solucoes/e7_orcamento_cheio.py
"""

import sys
from pathlib import Path

import numpy as np

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from prepare_data import baixar, limpar, deduplicar_paragrafos
from dataset import carregar, carregar_tokenizador
from bpe import BPETokenizer
from modelo_comum import treinar

EXTRAS = [
    (33056, "Historias sem Data"), (53101, "A Mao e a Luva"),
    (57001, "Papeis Avulsos"), (67162, "Helena"),
    (67780, "Iaia Garcia"), (67935, "Reliquias de Casa Velha"),
    (61653, "Poesias Completas"),
]
ORIGINAIS = [(55752, "Dom Casmurro"), (54829, "Memorias Postumas"),
             (55682, "Quincas Borba"), (56737, "Esau e Jaco")]

_merges, VOCAB = carregar_tokenizador()
tok = BPETokenizer()
tok.merges, tok.vocab = _merges, VOCAB
val = carregar("val")

print("=" * 74)
print("E7 com 3000 passos (orcamento da apostila)")
print("=" * 74)
print("  tokenizando (os livros ja' estao em cache do e7_mais_obras.py)...", flush=True)
obras = []
for id_livro, titulo in ORIGINAIS + EXTRAS:
    texto, _ = deduplicar_paragrafos(limpar(baixar(id_livro, titulo), titulo))
    obras.append((titulo, np.array(tok.encode(texto), dtype=np.uint16)))

print(f"\n  {'obras':>6s} {'tokens':>10s} {'treino':>9s} {'val':>9s} {'gap':>8s} {'min':>7s}")
res = {}
for n in (4, 6, 8, 11):
    corpus = np.concatenate([ids for _, ids in obras[:n]])
    tr, va, dt, _ = treinar(block=128, dados_tr=corpus, dados_val=val, passos=3000)
    res[n] = va
    print(f"  {n:>6d} {len(corpus):>10,} {tr:>9.4f} {va:>9.4f} {va-tr:>8.4f} {dt/60:>7.1f}",
          flush=True)

ordem = sorted(res, key=lambda k: res[k])
print("\n  ranking (melhor -> pior): " + " < ".join(f"{n} obras" for n in ordem))
print("  com 400 passos o ranking tinha sido: 6 < 8 < 4 < 11")
print(f"  {'MUDOU' if ordem[0] != 6 else 'nao mudou o vencedor'}.")
