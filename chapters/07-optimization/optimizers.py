"""
Otimizadores do zero — de SGD ate AdamW.

Nos capitulos anteriores usamos duas coisas sem abrir: primeiro `p.data -= lr *
p.grad` (SGD na mao), depois `torch.optim.AdamW` (caixa-preta). Aqui construimos
a escada completa e verificamos que a nossa implementacao de AdamW bate com a do
PyTorch.

A escada:
    SGD          -> anda na direcao do gradiente
    + momentum   -> acumula inercia, atravessa ruido e vales estreitos
    + RMSProp    -> passo por parametro, escalado pelo tamanho tipico do gradiente
    = Adam       -> momentum + RMSProp juntos, com correcao de vies
    -> AdamW     -> Adam com weight decay DESACOPLADO

Run:
    python optimizers.py
"""

import torch

torch.manual_seed(1337)


# ---------------------------------------------------------------------------
# 1. SGD: o mais simples possivel.
# ---------------------------------------------------------------------------
class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# 2. SGD com momentum: guarda uma "velocidade" acumulada.
#    Se o gradiente aponta sempre para o mesmo lado, a velocidade cresce e o
#    passo fica maior; se ele oscila, as oscilacoes se cancelam.
# ---------------------------------------------------------------------------
class SGDMomentum:
    def __init__(self, params, lr, beta=0.9):
        self.params = list(params)
        self.lr, self.beta = lr, beta
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            v.mul_(self.beta).add_(p.grad, alpha=1 - self.beta)   # media movel
            p -= self.lr * v

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# 3. AdamW do zero.
#
#    O problema que ele resolve: parametros diferentes precisam de passos de
#    tamanhos diferentes. Um gradiente de 0.001 e um de 10 nao deveriam receber
#    o mesmo tratamento. O Adam normaliza cada passo pelo tamanho TIPICO do
#    gradiente daquele parametro -- entao o passo efetivo fica na escala da
#    learning rate, seja qual for a escala do gradiente.
#
#    m = media movel do gradiente          (momentum: "para onde ir")
#    v = media movel do gradiente AO QUADRADO ("quao grande costuma ser")
#    passo = lr * m_corrigido / (sqrt(v_corrigido) + eps)
#
#    O "W" de AdamW: weight decay DESACOPLADO. No Adam original, somar o decay
#    ao gradiente fazia ele passar pela normalizacao do v -- o que distorce a
#    regularizacao. No AdamW o decay e' aplicado direto no parametro.
# ---------------------------------------------------------------------------
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0                     # contador de passos (para a correcao de vies)

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad

            # weight decay desacoplado: encolhe o parametro direto, ANTES do
            # passo adaptativo (e sem passar pela normalizacao do v)
            if self.wd != 0:
                p.mul_(1 - self.lr * self.wd)

            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)              # momentum
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)       # 2o momento

            # CORRECAO DE VIES: m e v comecam em zero, entao nos primeiros
            # passos eles subestimam o valor real. Dividir por (1 - beta^t)
            # compensa isso -- o efeito desaparece quando t cresce.
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)

            p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ===========================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # A verificacao que importa: nosso AdamW == torch.optim.AdamW ?
    #
    # Usamos um problema de brinquedo com gradientes de escalas MUITO
    # diferentes, justamente onde um otimizador adaptativo se distingue.
    # -----------------------------------------------------------------------
    def problema():
        """Devolve parametros identicos para os dois otimizadores."""
        torch.manual_seed(42)
        base = [torch.randn(4, 3), torch.randn(3)]
        return [b.clone().requires_grad_(True) for b in base]

    def perda(ps):
        # escalas propositalmente desbalanceadas (1000x entre os termos)
        W, b = ps
        return (W ** 2).sum() * 1000.0 + (b ** 2).sum() * 0.001

    HP = dict(lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)

    nossos = problema()
    deles = problema()
    opt_nosso = AdamW(nossos, **HP)
    opt_deles = torch.optim.AdamW(deles, **HP)

    print("=== nosso AdamW vs torch.optim.AdamW (20 passos) ===")
    for passo in range(20):
        for opt, ps in ((opt_nosso, nossos), (opt_deles, deles)):
            opt.zero_grad()
            perda(ps).backward()
            opt.step()

        if passo in (0, 4, 19):
            dif = max((a - b).abs().max().item() for a, b in zip(nossos, deles))
            print(f"  passo {passo + 1:2d}: diferenca maxima = {dif:.2e}")

    igual = all(torch.allclose(a, b, atol=1e-6) for a, b in zip(nossos, deles))
    print(f"\n  resultados batem (atol=1e-6)? {igual}")

    # -----------------------------------------------------------------------
    # Comparacao pratica: quantos passos cada otimizador precisa?
    # -----------------------------------------------------------------------
    print("\n=== quantos passos para a loss cair abaixo de 1e-3? ===")
    for nome, fabrica in [
        ("SGD            ", lambda ps: SGD(ps, lr=1e-4)),
        ("SGD + momentum ", lambda ps: SGDMomentum(ps, lr=1e-4)),
        ("AdamW (nosso)  ", lambda ps: AdamW(ps, lr=1e-2, weight_decay=0.0)),
    ]:
        ps = problema()
        opt = fabrica(ps)
        n = None
        for passo in range(1, 3001):
            opt.zero_grad()
            L = perda(ps)
            L.backward()
            opt.step()
            if L.item() < 1e-3:
                n = passo
                break
        W, b = ps
        termo_W = (W ** 2).sum().item() * 1000.0
        termo_b = (b ** 2).sum().item() * 0.001
        alvo = f"{n} passos" if n else "nao chegou em 3000"
        print(f"  {nome}: {alvo:20s} | termo grande {termo_W:.1e} | termo pequeno {termo_b:.1e}")

    print("\n  Leia as duas ultimas colunas. As duas versoes de SGD resolvem bem o")
    print("  termo GRANDE e empacam no PEQUENO -- por isso elas param praticamente")
    print("  no mesmo lugar: o gargalo nao e' o momentum, e' a learning rate unica.")
    print("  Com lr pequena o bastante para o termo grande nao divergir, o termo")
    print("  pequeno quase nao anda. O AdamW normaliza o passo por parametro e")
    print("  escapa desse aperto -- resolve os dois na mesma corrida.")
