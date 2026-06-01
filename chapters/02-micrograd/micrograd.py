"""
Micrograd — um motor de autograd (automatic differentiation) escalar, do zero.

Este e' o coracao do capitulo. No Capitulo 1, a linha `loss.backward()` do
PyTorch era magica: ela preenchia os gradientes sozinha. Aqui nos construimos
exatamente esse mecanismo, do zero, para um unico numero (escalar) de cada vez.

A ideia central: cada `Value` lembra de onde veio (quais Values e qual operacao
o produziram). Isso forma um grafo. A `backward()` percorre esse grafo de tras
para frente aplicando a regra da cadeia (chain rule) do calculo.

Inspirado no micrograd de Andrej Karpathy.
"""

import math


class Value:
    """Guarda um numero (`data`) e o seu gradiente (`grad`), e lembra como foi
    criado para poder fazer backpropagation."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0          # d(loss)/d(self): comeca em zero
        # --- internos para o autograd ---
        self._backward = lambda: None   # como propagar o grad para os pais
        self._prev = set(_children)      # os Values que geraram este
        self._op = _op                   # rotulo da operacao (para depurar/plotar)

    # ----------------------------------------------------------------------
    # Operacoes. Cada uma cria um novo Value e define como o gradiente
    # "flui de volta" para os operandos (a regra da cadeia local).
    # ----------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1 e d(out)/d(other) = 1; aplicamos a chain rule
            # (multiplicando pelo grad que chega de fora, out.grad) e ACUMULAMOS
            # com += para o caso de um Value ser usado mais de uma vez.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b  e  d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "so suportamos potencias int/float"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            # d(x**n)/dx = n * x**(n-1)
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        # funcao de ativacao classica; comprime qualquer numero para (-1, 1)
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():
            # d(e^x)/dx = e^x = out.data
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # derivada do ReLU: 0 se a entrada era negativa, 1 caso contrario
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # ----------------------------------------------------------------------
    # Backpropagation: ordena o grafo topologicamente e aplica _backward em
    # ordem reversa, partindo deste no' (geralmente a loss).
    # ----------------------------------------------------------------------
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # o gradiente da saida em relacao a ela mesma e' 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ----------------------------------------------------------------------
    # Acucar sintatico: faz -, -, / e ordem invertida (ex.: 2 * x) funcionarem
    # reaproveitando as operacoes acima.
    # ----------------------------------------------------------------------
    def __neg__(self):              # -self
        return self * -1

    def __radd__(self, other):      # other + self
        return self + other

    def __sub__(self, other):       # self - other
        return self + (-other)

    def __rsub__(self, other):      # other - self
        return other + (-self)

    def __rmul__(self, other):      # other * self
        return self * other

    def __truediv__(self, other):   # self / other
        return self * other ** -1

    def __rtruediv__(self, other):  # other / self
        return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


# ---------------------------------------------------------------------------
# Demonstracao rapida quando rodado direto: monta uma expressao, faz backward
# e mostra os gradientes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # expressao: L = (a*b + c) tanh
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    L = d.tanh()

    L.backward()

    print("L =", L)
    print("dL/da =", a.grad)
    print("dL/db =", b.grad)
    print("dL/dc =", c.grad)
