"""
Attention — construindo o mecanismo de atencao em 4 versoes equivalentes.

A ideia central: cada posicao da sequencia precisa "olhar para tras" e combinar
informacao das posicoes anteriores. Comecamos com a forma mais burra (media
simples num loop) e chegamos na self-attention de verdade -- provando, a cada
passo, que a nova versao faz o mesmo que a anterior (so mais rapido/flexivel).

v1: loop explicito calculando a media dos vetores anteriores
v2: a mesma media, via multiplicacao de matrizes (matmul)
v3: a mesma media, via softmax de uma matriz mascarada
v4: self-attention: os PESOS deixam de ser uniformes e passam a ser
    calculados a partir do conteudo (query . key)

Run:
    python attention.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(1337)

# B = batch (quantas sequencias), T = time (posicoes), C = channels (dimensao)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
print(f"x tem formato {tuple(x.shape)}  (batch, time, channels)")

# ---------------------------------------------------------------------------
# v1 — a media "na mao": para cada posicao t, a media de x[0..t].
#      Isto e' um "bag of words": junta o passado sem dar peso a nada.
# ---------------------------------------------------------------------------
xbow1 = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]              # (t+1, C): tudo ate a posicao t
        xbow1[b, t] = xprev.mean(dim=0)    # media ao longo do tempo

# ---------------------------------------------------------------------------
# v2 — o mesmo resultado, com UMA matmul.
#      wei[t, i] = 1/(t+1) se i <= t, senao 0. Multiplicar por x faz a media.
#      tril = triangular inferior: e' o que impede olhar para o FUTURO.
# ---------------------------------------------------------------------------
wei = torch.tril(torch.ones(T, T))
wei = wei / wei.sum(dim=1, keepdim=True)   # normaliza cada linha para somar 1
xbow2 = wei @ x                             # (T,T) @ (B,T,C) -> (B,T,C)

# ---------------------------------------------------------------------------
# v3 — o mesmo resultado, via softmax.
#      Comecamos de zeros, marcamos o futuro com -inf e aplicamos softmax:
#      exp(0)=1 nas posicoes permitidas, exp(-inf)=0 no futuro; normalizado,
#      da' exatamente a media uniforme. Este formato e' o que permite o v4:
#      basta trocar os zeros por pontuacoes aprendidas.
# ---------------------------------------------------------------------------
tril = torch.tril(torch.ones(T, T))
wei3 = torch.zeros((T, T))
wei3 = wei3.masked_fill(tril == 0, float("-inf"))   # proibe ver o futuro
wei3 = F.softmax(wei3, dim=-1)
xbow3 = wei3 @ x

# As tres versoes devem dar o MESMO resultado. Usamos atol=1e-6 porque float32
# nao e' exato: somar numeros em ordens diferentes (loop vs matmul) da diferencas
# de ~1e-8. "Igual" em ponto flutuante quer dizer "igual dentro de uma tolerancia".
print(f"v1 == v2 ? {torch.allclose(xbow1, xbow2, atol=1e-6)}")
print(f"v2 == v3 ? {torch.allclose(xbow2, xbow3, atol=1e-6)}")
print(f"  (diferenca maxima v1 vs v2: {(xbow1 - xbow2).abs().max().item():.2e})")

# ---------------------------------------------------------------------------
# v4 — SELF-ATTENTION de verdade.
#
# Nas versoes anteriores todos os tokens do passado tinham o MESMO peso. Mas
# nem todo token importa igual: numa palavra, uma consoante pode "querer" uma
# vogal especifica. A atencao deixa cada posicao DECIDIR onde olhar:
#
#   query (q) : "o que eu estou procurando?"
#   key   (k) : "o que eu tenho a oferecer?"
#   value (v) : "o que eu de fato entrego, se voce me escolher"
#
# O peso entre a posicao t e a posicao i e' o produto escalar q[t] . k[i]:
# alto se o que t procura casa com o que i oferece.
# ---------------------------------------------------------------------------
head_size = 16
key = torch.nn.Linear(C, head_size, bias=False)
query = torch.nn.Linear(C, head_size, bias=False)
value = torch.nn.Linear(C, head_size, bias=False)

k = key(x)      # (B, T, head_size)
q = query(x)    # (B, T, head_size)
v = value(x)    # (B, T, head_size)

# pontuacoes de afinidade entre todas as posicoes: q . k
# transpose(-2,-1) troca as duas ultimas dimensoes para a matmul casar
wei4 = q @ k.transpose(-2, -1)              # (B, T, T)

# ESCALA: dividir por sqrt(head_size) mantem a variancia ~1. Sem isso, os
# valores crescem com head_size, o softmax satura e vira quase um one-hot
# (a atencao "trava" numa unica posicao e o gradiente morre).
wei4 = wei4 * head_size ** -0.5

# MASCARA causal: cada posicao so pode olhar para tras (e para si mesma).
wei4 = wei4.masked_fill(tril == 0, float("-inf"))
wei4 = F.softmax(wei4, dim=-1)

out = wei4 @ v                               # (B, T, head_size)
print(f"\nsaida da self-attention: {tuple(out.shape)}")

# ---------------------------------------------------------------------------
# Inspecao: os pesos de atencao da primeira sequencia do batch.
# Cada linha t mostra quanto a posicao t presta atencao em cada posicao i.
# Note o formato triangular (zeros no futuro) e que cada linha soma 1.
# ---------------------------------------------------------------------------
print("\npesos de atencao (sequencia 0) - linha = quem olha, coluna = quem e' olhado:")
torch.set_printoptions(precision=2, sci_mode=False)
print(wei4[0])
print(f"\ncada linha soma 1? {torch.allclose(wei4[0].sum(dim=-1), torch.ones(T))}")

# ---------------------------------------------------------------------------
# Por que a escala importa: sem ela, o softmax satura.
# ---------------------------------------------------------------------------
print("\n--- efeito da escala no softmax ---")
scores = torch.tensor([0.5, -0.2, 0.3, 0.1]) * 8    # pontuacoes "grandes"
print("sem escala (valores grandes):", F.softmax(scores, dim=-1))
print("com escala (divididos):      ", F.softmax(scores * 0.125, dim=-1))
print("Sem escala a distribuicao fica concentrada; com escala, mais suave.")
