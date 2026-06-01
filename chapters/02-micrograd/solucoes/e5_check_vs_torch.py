"""
Solucao do Exercicio E5 — nossos gradientes batem com os do PyTorch?

Monta a mesma expressao nos dois sistemas, roda backward() e compara data e
todos os gradientes. Tolerancia de 1e-6.

Run (a partir da pasta do capitulo):
    python solucoes/e5_check_vs_torch.py
"""
import sys
from pathlib import Path

# permite importar micrograd.py que esta na pasta do capitulo (pasta-mae)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from micrograd import Value


def build_ours():
    a = Value(2.0); b = Value(-3.0); c = Value(10.0); f = Value(-2.0)
    d = a * b + c
    g = d * f + d.tanh() + (a / b)
    g.backward()
    return g.data, {"a": a.grad, "b": b.grad, "c": c.grad, "f": f.grad}


def build_torch():
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([-3.0], requires_grad=True)
    c = torch.tensor([10.0], requires_grad=True)
    f = torch.tensor([-2.0], requires_grad=True)
    d = a * b + c
    g = d * f + torch.tanh(d) + (a / b)
    g.backward()
    return g.item(), {"a": a.grad.item(), "b": b.grad.item(),
                      "c": c.grad.item(), "f": f.grad.item()}


od, og = build_ours()
td, tg = build_torch()

print(f"{'qty':8s} {'ours':>14s} {'torch':>14s} {'match':>7s}")
print(f"{'g.data':8s} {od:14.8f} {td:14.8f} {str(abs(od-td)<1e-6):>7s}")
all_ok = abs(od - td) < 1e-6
for k in og:
    m = abs(og[k] - tg[k]) < 1e-6
    all_ok = all_ok and m
    print(f"{k+'.grad':8s} {og[k]:14.8f} {tg[k]:14.8f} {str(m):>7s}")

print("\nALL MATCH:", all_ok)
