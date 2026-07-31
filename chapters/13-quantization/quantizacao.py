"""
quantizacao.py — quantizacao int8 do zero.

Nao usa nada do torch.quantization. Sao ~30 linhas de aritmetica: achar uma
escala, dividir, arredondar, guardar em 8 bits.

Run (a partir da pasta do capitulo):
    python quantizacao.py
"""

import sys
from pathlib import Path

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ===========================================================================
# Simetrica: uma escala, sem deslocamento. O zero do float vira o zero do int.
# ===========================================================================
def quantizar_simetrica(x, bits=8, dim=None):
    """float -> (inteiros, escala).

    dim=None  : UMA escala para o tensor inteiro (per-tensor)
    dim=0     : uma escala por LINHA (per-channel)

    O intervalo de int8 e' [-128, 127]. Usamos [-127, 127] para ficar simetrico
    -- perder um valor de 256 custa 0,4% da resolucao e evita um caso especial.
    """
    qmax = 2 ** (bits - 1) - 1                       # 127 para 8 bits
    if dim is None:
        escala = x.abs().max() / qmax
    else:
        escala = x.abs().amax(dim=1 - dim, keepdim=True) / qmax
    escala = torch.clamp(escala, min=1e-12)          # tensor todo zero
    q = torch.round(x / escala).clamp(-qmax, qmax).to(torch.int8)
    return q, escala


def desquantizar_simetrica(q, escala):
    return q.to(torch.float32) * escala


# ===========================================================================
# Assimetrica: escala + zero-point. Aproveita todo o intervalo quando os dados
# nao sao centrados em zero (saida de ReLU, por exemplo).
# ===========================================================================
def quantizar_assimetrica(x, bits=8):
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1     # -128, 127
    xmin, xmax = x.min(), x.max()

    # O INTERVALO PRECISA CONTER O ZERO. Sem isto o zero-point necessario cai
    # fora de [qmin, qmax] e o clamp destroi o mapeamento -- foi o primeiro bug
    # deste capitulo, e ele CUSTOU: para dados com media 5, a assimetrica ficou
    # 12x PIOR que a simetrica, o oposto do esperado. O zero-point ideal era
    # -170,7, nao cabia em int8, virava -128 no clamp, e os valores grandes
    # saturavam no teto.
    #
    # PyTorch, TFLite e ONNX fazem exatamente esta extensao. Ela custa um pouco
    # de resolucao (parte do intervalo representa valores que nao ocorrem) e
    # garante que o float 0.0 tenha um inteiro exato -- o que importa para
    # padding e para mascaras.
    xmin = torch.minimum(xmin, torch.zeros_like(xmin))
    xmax = torch.maximum(xmax, torch.zeros_like(xmax))

    escala = torch.clamp((xmax - xmin) / (qmax - qmin), min=1e-12)
    zero = torch.round(qmin - xmin / escala).clamp(qmin, qmax)
    q = torch.round(x / escala + zero).clamp(qmin, qmax).to(torch.int8)
    return q, escala, zero


def desquantizar_assimetrica(q, escala, zero):
    return (q.to(torch.float32) - zero) * escala


# ===========================================================================
def erro_relativo(original, recuperado):
    """Erro de reconstrucao, normalizado pela magnitude do tensor."""
    return ((recuperado - original).norm() / original.norm()).item()


# ===========================================================================
if __name__ == "__main__":
    torch.manual_seed(1337)

    print("=" * 74)
    print("1. O que a quantizacao faz com um tensor")
    print("=" * 74)
    x = torch.randn(4, 6)
    q, s = quantizar_simetrica(x)
    r = desquantizar_simetrica(q, s)
    print(f"  original (primeira linha) : {[f'{v:+.4f}' for v in x[0].tolist()]}")
    print(f"  int8                      : {q[0].tolist()}")
    print(f"  de volta a float          : {[f'{v:+.4f}' for v in r[0].tolist()]}")
    print(f"\n  escala = max|x| / 127 = {s.item():.6f}")
    print(f"  erro relativo: {erro_relativo(x, r):.4%}")
    print(f"  bytes: {x.numel()*4} (float32) -> {q.numel()} (int8) + "
          f"{s.numel()*4} da escala")

    print("=" * 74)
    print("2. Simetrica vs assimetrica: depende da distribuicao")
    print("=" * 74)
    casos = {
        "normal (centrada em 0)": torch.randn(1000),
        "so' positiva (pos-ReLU)": torch.randn(1000).relu(),
        "deslocada (media 5)": torch.randn(1000) + 5.0,
    }
    print(f"  {'distribuicao':>24s} {'simetrica':>12s} {'assimetrica':>13s}")
    for nome, t in casos.items():
        _, s1 = quantizar_simetrica(t)
        e1 = erro_relativo(t, desquantizar_simetrica(*quantizar_simetrica(t)))
        qa, sa, za = quantizar_assimetrica(t)
        e2 = erro_relativo(t, desquantizar_assimetrica(qa, sa, za))
        print(f"  {nome:>24s} {e1:>11.3%} {e2:>13.3%}")
    print("""
  Para dados centrados em zero as duas empatam. Para dados deslocados, a
  simetrica desperdica metade do intervalo -- ela reserva os inteiros negativos
  para valores que nunca aparecem.

  PESOS de rede costumam ser centrados em zero, entao a simetrica basta e sai
  mais barata (nao ha' zero-point para somar em cada operacao). ATIVACOES,
  depois de ReLU ou GELU, nao sao -- e ai' a assimetrica ganha.""")

    print("=" * 74)
    print("3. Per-tensor vs per-channel, e por que a diferenca aparece")
    print("=" * 74)
    # matriz onde as linhas tem escalas MUITO diferentes -- comum em redes reais
    w = torch.randn(8, 64)
    w[0] *= 100.0                      # uma linha com valores enormes
    r_tensor = desquantizar_simetrica(*quantizar_simetrica(w))
    qc, sc = quantizar_simetrica(w, dim=0)
    r_canal = desquantizar_simetrica(qc, sc)
    print("  matriz 8x64 com UMA linha 100x maior que as outras:\n")
    print(f"  {'metrica':>34s} {'per-tensor':>12s} {'per-channel':>13s}")
    print(f"  {'erro na matriz inteira':>34s} "
          f"{erro_relativo(w, r_tensor):>11.2%} {erro_relativo(w, r_canal):>13.2%}")
    print(f"  {'erro nas 7 linhas NORMAIS':>34s} "
          f"{erro_relativo(w[1:], r_tensor[1:]):>11.2%} "
          f"{erro_relativo(w[1:], r_canal[1:]):>13.2%}")
    print(f"  {'erro na linha grande':>34s} "
          f"{erro_relativo(w[0], r_tensor[0]):>11.2%} "
          f"{erro_relativo(w[0], r_canal[0]):>13.2%}")
    print(f"\n  custo extra do per-channel: {sc.numel()} floats de escala em vez "
          f"de 1 ({(sc.numel()-1)*4} bytes)")
    print("""
  ATENCAO A' METRICA -- e' o ponto do exercicio. Olhando so' a matriz inteira,
  o per-tensor parece aceitavel. Ele NAO e': o erro esta' concentrado nas linhas
  pequenas, e a norma global esconde isso porque e' dominada pela linha grande.

  A razao: uma escala unica e' refem do MAIOR valor da matriz. A linha 100x
  maior define a escala, e as outras sete passam a caber em pouquissimos niveis
  inteiros -- perto de zero, elas viram zero.

  Per-channel custa alguns bytes e resolve. E' por isso que praticamente toda
  quantizacao de peso na pratica e' per-channel.""")

    print("=" * 74)
    print("4. Quantos bits ainda dao um resultado utilizavel?")
    print("=" * 74)
    w = torch.randn(256, 256)
    print(f"  {'bits':>5s} {'niveis':>8s} {'erro relativo':>15s} {'bytes/peso':>12s}")
    for bits in (8, 7, 6, 5, 4, 3, 2):
        q, s = quantizar_simetrica(w, bits=bits, dim=0)
        print(f"  {bits:>5d} {2**bits:>8d} "
              f"{erro_relativo(w, desquantizar_simetrica(q, s)):>14.2%} "
              f"{bits/8:>12.2f}")
    print("""
  O erro APROXIMADAMENTE DOBRA a cada bit removido, e a tabela deixa isso ver:
  cada bit a menos corta o numero de niveis pela metade, entao o passo de
  quantizacao dobra.

  (Cuidado ao ler tabelas que pulam bits: de 8 para 6 bits o erro cresce ~4x,
  nao ~2x, porque sao DOIS bits.)

  O que NAO da' para ler nesta tabela e' se o modelo ainda funciona. Erro de
  reconstrucao do peso nao e' erro do modelo: e' preciso medir a LOSS. E' o que
  o quantizar_modelo.py faz.""")
