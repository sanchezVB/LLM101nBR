"""
LayerNorm do zero — e por que ela e' necessaria.

Layer normalization normaliza cada vetor de ativacoes INDIVIDUALMENTE: subtrai a
media e divide pelo desvio padrao, calculados ao longo das features daquela
posicao. Depois aplica um ganho (gamma) e um deslocamento (beta) aprendidos, para
a rede poder desfazer a normalizacao se isso for util.

Sem isso, ao empilhar muitas camadas as ativacoes tendem a crescer ou encolher
descontroladamente, e o treino fica instavel (ou simplesmente nao anda).

Run:
    python layernorm.py
"""

import torch
import torch.nn as nn

torch.manual_seed(1337)


class LayerNorm(nn.Module):
    """LayerNorm implementada do zero (equivalente a nn.LayerNorm)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))    # ganho (escala) aprendido
        self.beta = nn.Parameter(torch.zeros(dim))    # deslocamento aprendido

    def forward(self, x):
        # Normaliza ao longo da ULTIMA dimensao (as features), por posicao.
        # keepdim=True mantem o formato para o broadcasting funcionar.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # unbiased=False: divide por N
        xhat = (x - mean) / torch.sqrt(var + self.eps)      # eps evita divisao por zero
        return self.gamma * xhat + self.beta


# ---------------------------------------------------------------------------
# 1. A nossa implementacao bate com a do PyTorch?
# ---------------------------------------------------------------------------
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

nossa = LayerNorm(C)
deles = nn.LayerNorm(C)

out_nossa = nossa(x)
out_deles = deles(x)

print("=== nossa LayerNorm vs nn.LayerNorm ===")
print(f"formatos iguais? {out_nossa.shape == out_deles.shape}  {tuple(out_nossa.shape)}")
print(f"resultados batem? {torch.allclose(out_nossa, out_deles, atol=1e-6)}")
print(f"diferenca maxima: {(out_nossa - out_deles).abs().max().item():.2e}")

# ---------------------------------------------------------------------------
# 2. O que ela faz, na pratica: cada posicao fica com media ~0 e desvio ~1.
# ---------------------------------------------------------------------------
x_torto = torch.randn(B, T, C) * 15 + 40      # ativacoes "fora de escala"
y = nossa(x_torto)

print("\n=== efeito da normalizacao ===")
print(f"ANTES : media = {x_torto.mean():+.3f} | desvio = {x_torto.std():.3f}")
print(f"DEPOIS: media = {y.mean():+.3f} | desvio = {y.std():.3f}")
print("Note que a normalizacao e' por POSICAO, nao global:")
print(f"  media da primeira posicao depois: {y[0, 0].mean().item():+.6f} (~0)")
print(f"  desvio da primeira posicao depois: {y[0, 0].std(unbiased=False).item():.6f} (~1)")

# ---------------------------------------------------------------------------
# 3. Por que "Layer" e nao "Batch"? A normalizacao NAO mistura exemplos.
#    Cada posicao de cada sequencia e' normalizada com os seus proprios numeros.
#    Isso e' essencial para um modelo de linguagem: na hora de gerar texto
#    processamos UMA sequencia, e o resultado nao pode depender do batch.
# ---------------------------------------------------------------------------
um_so = nossa(x[:1])                      # so' o primeiro exemplo
do_batch = nossa(x)[:1]                   # o mesmo exemplo, dentro do batch
print("\n=== independencia do batch ===")
print(f"processar sozinho == processar no batch? {torch.allclose(um_so, do_batch, atol=1e-6)}")
print("(com BatchNorm isso seria FALSO -- e' por isso que Transformers usam LayerNorm)")
