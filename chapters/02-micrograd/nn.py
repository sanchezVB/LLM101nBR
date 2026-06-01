"""
nn.py — uma pequena biblioteca de redes neurais construida sobre o Value.

Reusa o motor de autograd do micrograd.py para montar Neuron -> Layer -> MLP.
Cada peso e' um Value; treinar e' so chamar loss.backward() e empurrar cada
peso na direcao oposta ao seu gradiente -- exatamente como no Capitulo 1, mas
agora a `backward()` e' a NOSSA, escrita do zero.

Run:
    python nn.py
"""

import random
from micrograd import Value


class Module:
    """Base comum: zera gradientes e expoe os parametros."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """Um neuronio: soma ponderada das entradas + bias, com ativacao opcional."""

    def __init__(self, nin, nonlin=True):
        # pesos pequenos e aleatorios; bias comeca em zero
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # act = w . x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """Uma camada: varios neuronios em paralelo."""

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """Multi-Layer Perceptron: uma pilha de camadas.

    nin   = numero de entradas
    nouts = lista com o tamanho de cada camada, ex.: [4, 4, 1]
    A ultima camada e' linear (sem tanh) para poder produzir qualquer valor real.
    """

    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(nouts) - 1))
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ---------------------------------------------------------------------------
# Demo: treinar um MLP minusculo num problema de classificacao binaria
# (o classico "moons" simplificado, feito a mao com 4 pontos).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(1337)

    # 4 exemplos, 3 features cada
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]   # alvos desejados

    model = MLP(3, [4, 4, 1])     # 3 entradas -> 4 -> 4 -> 1 saida
    print(f"parametros no modelo: {len(model.parameters())}")

    for step in range(100):
        # forward: previsoes e loss (erro quadratico medio)
        ypred = [model(x) for x in xs]
        loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

        # backward: zera grads e roda a NOSSA backpropagation
        model.zero_grad()
        loss.backward()

        # update: gradient descent
        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 10 == 0:
            print(f"step {step:3d} | loss {loss.data:.6f}")

    print(f"\nloss final = {loss.data:.6f}")
    print("previsoes (alvo entre parenteses):")
    for x, yt in zip(xs, ys):
        print(f"  {model(x).data:+.3f}  ({yt:+.0f})")
