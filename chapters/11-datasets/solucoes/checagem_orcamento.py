"""
Checagem de sensibilidade ao orcamento — o gabarito reduzido esta' mentindo?

Os gabaritos deste curso rodam com MENOS PASSOS que a apostila (400 em vez de
3.000, aqui). Isso e' uma aposta explicita: a de que a ORDEM entre as
configuracoes nao depende do orcamento, mesmo que os valores absolutos
dependam.

Essa aposta as vezes PERDE. Dois casos medidos neste curso:

  - Cap. 3 E4: a melhor learning rate com 4.000 passos e' 1.0; com 20.000 e' 0.1
  - Cap. 7 E5: 3e-3 ganha de 1e-3 com orcamento curto, e perde com o longo

Nos dois, a pergunta era sobre DINAMICA DE OTIMIZACAO -- e o orcamento e'
justamente a variavel que a dinamica de otimizacao consome. Ja' as perguntas
ESTRUTURAIS (quanto contexto, quantas posicoes previstas, quantos dados)
deveriam ser estaveis.

'Deveriam' nao e' medicao. Este script mede: roda o E4 com 400 e com 1.200
passos e compara os RANKINGS, nao os valores.

Run (a partir da pasta do capitulo):
    python solucoes/checagem_orcamento.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo_comum import treinar

BLOCOS = (32, 128, 256)
resultados = {}

print("=" * 74)
print("A conclusao do E4 sobrevive a um orcamento 3x maior?")
print("=" * 74)
for passos in (400, 1200):
    print(f"\n  {passos} passos:")
    print(f"    {'block_size':>11s} {'treino':>9s} {'val':>9s}")
    linha = {}
    for blk in BLOCOS:
        tr, va, dt, _ = treinar(block=blk, passos=passos)
        linha[blk] = va
        print(f"    {blk:>11d} {tr:>9.4f} {va:>9.4f}   ({dt/60:.1f} min)", flush=True)
    resultados[passos] = linha

print("\n" + "=" * 74)
ordem = {p: sorted(BLOCOS, key=lambda b: resultados[p][b]) for p in resultados}
for p, o in ordem.items():
    print(f"  ranking com {p:>5d} passos (melhor -> pior): "
          + " < ".join(str(b) for b in o))

if ordem[400] == ordem[1200]:
    print("""
  O RANKING NAO MUDOU. A conclusao do E4 ('mais contexto ajuda') e' estavel ao
  orcamento nesta faixa -- entao a resposta dada com 400 passos vale.

  Isso NAO prova que todo gabarito reduzido esta' certo. Prova que ESTE esta',
  e mostra como conferir os outros: rode a sua configuracao com o triplo dos
  passos e veja se a ordem se mantem. Se mantiver, o orcamento nao e' a
  variavel que decide.""")
else:
    print("""
  O RANKING MUDOU. O gabarito reduzido nao pode ser usado para responder este
  exercicio -- a resposta precisa vir do orcamento cheio. Cuidado ao
  generalizar qualquer conclusao obtida com poucos passos.""")
