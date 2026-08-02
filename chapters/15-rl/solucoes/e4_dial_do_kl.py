"""
e4_dial_do_kl.py — existe um beta que SEGURA o reward hacking?

O experimento central mostrou que beta=0.02 nao impede o hacking: a recompensa
'pontos' ainda vai a 0.958 e o modelo ainda responde '.'. O freio so' faz o
estrago demorar mais.

A pergunta que sobra e' quantitativa, e da' para responder: varrendo o beta,
onde fica a fronteira entre 'a politica hackeia' e 'a politica nao sai do
lugar'? E existe uma faixa util no meio?

Run (a partir da pasta do capitulo):
    python solucoes/e4_dial_do_kl.py
"""

import sys
from pathlib import Path

import numpy as np

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(CAP.parent / "14-sft"))
sys.path.insert(0, str(CAP.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reinforce import treinar, PASSOS
from experimento import loss_texto_real, amostra_de_texto
from reinforce import carregar_politica

base = carregar_politica().eval()
loss_base = loss_texto_real(base)

print("=" * 74)
print("O beta e' um DIAL, nao um interruptor")
print("=" * 74)
print(f"  recompensa 'pontos' (mal especificada), {PASSOS} passos")
print(f"  loss de partida em Machado: {loss_base:.4f}\n")
print(f"  {'beta':>8s} {'recompensa':>18s} {'KL':>8s} {'loss real':>11s} "
      f"{'custo':>9s}   amostra")

for beta in (0.0, 0.02, 0.1, 0.5, 2.0):
    pol, ref, hist, _ = treinar("pontos", beta=beta, passos=PASSOS, verbose=False)
    r0 = float(np.mean([h[0] for h in hist[:10]]))
    r1 = float(np.mean([h[0] for h in hist[-10:]]))
    kl = float(np.mean([h[1] for h in hist[-10:]]))
    lr_ = loss_texto_real(pol)
    amostra = amostra_de_texto(pol, ref, n=1)[0][:38]
    print(f"  {beta:>8.2f} {r0:>8.3f} -> {r1:<7.3f} {kl:>8.2f} {lr_:>11.4f} "
          f"{lr_-loss_base:>+9.4f}   {amostra!r}", flush=True)

print("""
  Como ler: a coluna 'custo' e' quanto o modelo perdeu de capacidade em
  portugues para ganhar pontuacao. Quanto mais perto de zero, melhor -- e
  beta alto demais leva a zero pelo motivo errado, porque a politica nao sai
  do lugar e nao aprende nada.

  O que se procura e' a faixa em que a recompensa sobe E o custo fica pequeno.
  Se ela nao existir para esta recompensa, a conclusao nao e' 'ajuste melhor o
  beta' -- e' que a RECOMPENSA esta' errada e nenhum beta conserta.""")
