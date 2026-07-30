"""
Gabarito executavel do Capitulo 02 — micrograd.

Roda os exercicios que pedem medicao (E2, E3, E4, E6, E7). As respostas
discursivas estao em gabarito.md.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import math
import random
import sys
from pathlib import Path

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))

from micrograd import Value
from nn import MLP

# ===========================================================================
print("=" * 70)
print("E2 — conferindo o gradiente com diferenca finita")
print("=" * 70)


def L(a_val, b_val, c_val):
    """A funcao que vamos derivar: L = (a*b + c).tanh()"""
    a, b, c = Value(a_val), Value(b_val), Value(c_val)
    return (a * b + c).tanh().data


a0, b0, c0 = 2.0, -3.0, 10.0

# gradiente pelo NOSSO autograd
a, b, c = Value(a0), Value(b0), Value(c0)
saida = (a * b + c).tanh()
saida.backward()

print(f"  {'variavel':>9s} {'autograd':>14s} {'dif. finita':>14s} {'erro relativo':>15s}")
h = 1e-6
for nome, grad, funcao in (
    ("a", a.grad, lambda d: L(a0 + d, b0, c0)),
    ("b", b.grad, lambda d: L(a0, b0 + d, c0)),
    ("c", c.grad, lambda d: L(a0, b0, c0 + d)),
):
    # diferenca CENTRAL: mais precisa que (f(x+h)-f(x))/h
    numerico = (funcao(h) - funcao(-h)) / (2 * h)
    erro = abs(grad - numerico) / (abs(numerico) + 1e-12)
    print(f"  {nome:>9s} {grad:14.8f} {numerico:14.8f} {erro:15.2e}")

print("""
  Resposta: batem ate' ~1e-8. A diferenca finita e' a forma classica de TESTAR
  um autograd -- e note que usamos a diferenca CENTRAL, (f(x+h)-f(x-h))/2h, que
  tem erro O(h^2) em vez de O(h) da versao ingenua. Com h muito pequeno o erro
  volta a crescer, por cancelamento em ponto flutuante: existe um h otimo.""")

# ===========================================================================
print("=" * 70)
print("E3 — implementando log()")
print("=" * 70)


def log_value(self):
    """d(ln x)/dx = 1/x"""
    out = Value(math.log(self.data), (self,), "log")

    def _backward():
        self.grad += (1.0 / self.data) * out.grad

    out._backward = _backward
    return out


Value.log = log_value      # acrescenta o metodo a' classe

x = Value(3.0)
y = x.log()
y.backward()
esperado = 1 / 3.0
print(f"  log(3.0)      = {y.data:.6f}  (math.log = {math.log(3.0):.6f})")
print(f"  d(log)/dx     = {x.grad:.6f}  (esperado 1/3 = {esperado:.6f})")

# confere com diferenca finita
h = 1e-6
num = (math.log(3.0 + h) - math.log(3.0 - h)) / (2 * h)
print(f"  dif. finita   = {num:.6f}  -> erro {abs(x.grad-num):.2e}")
print("""
  Resposta: a derivada local e' 1/x, e o _backward segue o padrao de sempre --
  derivada local vezes o gradiente que chega de fora (out.grad).""")

# ===========================================================================
print("=" * 70)
print("E4 — por que zerar o gradiente")
print("=" * 70)

xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]

for zerar in (True, False):
    random.seed(1337)
    m = MLP(3, [4, 4, 1])
    historico = []
    for passo in range(20):
        pred = [m(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(pred, ys))
        if zerar:
            m.zero_grad()
        loss.backward()
        for p in m.parameters():
            p.data -= 0.05 * p.grad
        historico.append(loss.data)
    rotulo = "COM zero_grad" if zerar else "SEM zero_grad"
    print(f"  {rotulo:16s}: passo 0={historico[0]:8.4f}  passo 5={historico[5]:10.4f}  "
          f"passo 19={historico[19]:12.4f}")

print("""
  Resposta (leia os numeros, nao a intuicao): sem o zero_grad o treino NAO
  explode neste caso -- ele ESTAGNA. A loss cai de 3.09 para ~0.6 e para ali,
  enquanto com zero_grad chega a 0.0001.

  Por que estagna em vez de explodir? Os gradientes sao ACUMULADOS com += a cada
  backward, entao o gradiente do passo N e' a soma de todos os anteriores. Isso
  tem dois efeitos que se combatem: o passo fica grande demais (empurra para
  divergir), mas tambem fica DESATUALIZADO -- ele carrega direcoes de pesos que
  ja' mudaram. Aqui o modelo entra num vai-e-vem e nao converge.

  Em outros problemas (ou com lr maior) o mesmo bug realmente diverge. A licao
  pratica e' a mesma nos dois casos: o resultado fica errado de um jeito que NAO
  se parece com um erro -- nao ha' excecao, so' um treino pior.""")

# ===========================================================================
print("=" * 70)
print("E6 — arquitetura e learning rate")
print("=" * 70)


def treinar(camadas, lr, passos=100):
    random.seed(1337)
    m = MLP(3, camadas)
    for _ in range(passos):
        pred = [m(x) for x in xs]
        loss = sum((p - y) ** 2 for p, y in zip(pred, ys))
        m.zero_grad()
        loss.backward()
        for p in m.parameters():
            p.data -= lr * p.grad
        if not math.isfinite(loss.data):
            return float("nan"), len(m.parameters())
    return loss.data, len(m.parameters())


def fmt(v):
    """Loss divergida vira um numero gigantesco; notacao cientifica e' o unico
    jeito de a tabela continuar legivel."""
    if not math.isfinite(v):
        return "    nao-numero"
    return f"{v:15.6f}" if v < 1e6 else f"{v:15.2e}"


print(f"  {'arquitetura':>18s} {'params':>7s} {'loss final':>15s}   situacao")
for camadas in ([4, 4, 1], [8, 1], [16, 16, 1]):
    L_fim, npar = treinar(camadas, 0.05)
    obs = "DIVERGIU" if (not math.isfinite(L_fim) or L_fim > 1) else "convergiu"
    print(f"  {str(camadas):>18s} {npar:7d} {fmt(L_fim)}   {obs}")

print(f"\n  {'learning rate':>18s} {'loss final':>15s}   situacao")
for lr in (0.5, 0.05, 0.001):
    L_fim, _ = treinar([4, 4, 1], lr)
    obs = "DIVERGIU" if (not math.isfinite(L_fim) or L_fim > 1) else (
        "lenta demais" if L_fim > 0.01 else "boa")
    print(f"  {lr:>18} {fmt(L_fim)}   {obs}")

print("""
  Respostas (e a primeira contraria o senso comum):

  1. [4,4,1] e [8,1] convergem -- e, por coincidencia, tem os MESMOS 41
     parametros. Mas [16,16,1], com 353 parametros, DIVERGE com a mesma
     learning rate. Rede maior nao e' automaticamente melhor: com mais
     parametros, os gradientes somados ficam maiores, e uma lr que servia para
     a rede pequena passa a ser grande demais.

     Esta e' a mesma licao do Capitulo 7 (boas praticas nao sao aditivas) e
     antecipa o Capitulo 7 de outro jeito: a lr adequada depende do TAMANHO do
     modelo, e e' por isso que existem inicializacao escalada e agendamento.

  2. lr=0.5 diverge; lr=0.05 converge; lr=0.001 quase nao anda no orcamento de
     100 passos. E' a curva em U que o Capitulo 7 mede de forma sistematica.""")

# ===========================================================================
print("=" * 70)
print("E7 — visualizando o grafo")
print("=" * 70)


def desenhar(raiz, prof=0, vistos=None):
    """Percorre _prev recursivamente e imprime a arvore do grafo."""
    if vistos is None:
        vistos = set()
    marca = "  " * prof + ("+-- " if prof else "")
    op = f" [{raiz._op}]" if raiz._op else ""
    print(f"  {marca}Value(data={raiz.data:.4f}, grad={raiz.grad:.4f}){op}")
    if id(raiz) in vistos:
        return
    vistos.add(id(raiz))
    for filho in raiz._prev:
        desenhar(filho, prof + 1, vistos)


a, b, c = Value(2.0), Value(-3.0), Value(10.0)
saida = (a * b + c).tanh()
saida.backward()
print("  grafo de (a*b + c).tanh():\n")
desenhar(saida)
print("""
  Cada nivel mostra um no' e os seus 'pais'. A backward() percorre exatamente
  esta estrutura, de cima para baixo, aplicando a regra da cadeia.""")
