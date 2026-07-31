"""
Gabarito executavel do Capitulo 13 — quantizacao.

Roda E2, E3, E4, E5, E6 e E7. (O E1 e' conceitual.)

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(CAP.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(CAP.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from quantizacao import (quantizar_simetrica, desquantizar_simetrica,
                         quantizar_assimetrica, desquantizar_assimetrica,
                         erro_relativo)
from quantizar_modelo import (quantizar_modelo, loss_val, cronometrar,
                              modelo, LinearQuantizada)

torch.manual_seed(1337)

# ===========================================================================
print("=" * 74)
print("E2 — reproduzindo a armadilha do zero-point")
print("=" * 74)


def assimetrica_sem_extensao(x, bits=8):
    """A versao COM o bug: nao estende o intervalo para conter o zero."""
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    xmin, xmax = x.min(), x.max()
    escala = torch.clamp((xmax - xmin) / (qmax - qmin), min=1e-12)
    zero_ideal = qmin - xmin / escala
    zero = torch.round(zero_ideal).clamp(qmin, qmax)
    q = torch.round(x / escala + zero).clamp(qmin, qmax)
    return q, escala, zero, zero_ideal, (q >= qmax).sum().item()


x = torch.randn(1000) + 5.0
q, s, z, z_ideal, saturados = assimetrica_sem_extensao(x)
e_sim = erro_relativo(x, desquantizar_simetrica(*quantizar_simetrica(x)))
e_bug = erro_relativo(x, desquantizar_assimetrica(q, s, z))
qa, sa, za = quantizar_assimetrica(x)
e_ok = erro_relativo(x, desquantizar_assimetrica(qa, sa, za))

print(f"  distribuicao randn(1000) + 5, valores em [{x.min():.2f}, {x.max():.2f}]\n")
print(f"  {'variante':>34s} {'erro':>10s}")
print(f"  {'simetrica':>34s} {e_sim:>9.3%}")
print(f"  {'assimetrica SEM estender o zero':>34s} {e_bug:>9.3%}  <- pior!")
print(f"  {'assimetrica COM estender o zero':>34s} {e_ok:>9.3%}")
print(f"\n  zero-point ideal    : {z_ideal:.1f}")
print(f"  intervalo do int8   : [-128, 127]  ->  o ideal NAO cabe")
print(f"  clamp forca para    : {z.item():.0f}")
print(f"  valores saturados no teto (+127): {saturados} de {len(x)}")
print("""
  Respostas:
  1. Com o bug, a SIMETRICA ganha -- o oposto do esperado, e o oposto do que a
     teoria diz. E' o sintoma que denuncia o problema.
  2. Nao cabe. O zero-point ideal fica bem abaixo de -128, porque os dados nao
     contem o zero: para representar o float 0.0 seria preciso um inteiro que
     nao existe no tipo.
  3. A cadeia causal completa:

       dados em [1.2, 8.6], que nao contem o zero
         -> zero-point ideal = qmin - xmin/escala fica fora de [-128, 127]
         -> o clamp o move para -128
         -> o mapeamento inteiro se desloca em relacao ao pretendido
         -> q = x/escala + zero estoura o teto para os valores grandes
         -> eles saturam em +127 e perdem a informacao
         -> o erro medido dispara

     A correcao (estender o intervalo para conter o zero) e' o que PyTorch,
     TFLite e ONNX fazem. Ela custa resolucao e garante que o zero-point caiba
     -- e que o float 0.0 tenha um inteiro exato, o que importa para padding e
     mascaras.""")

# ===========================================================================
print("=" * 74)
print("E3 — a metrica que mente")
print("=" * 74)
w = torch.randn(8, 64)
w[0] *= 100.0
r_t = desquantizar_simetrica(*quantizar_simetrica(w))
r_c = desquantizar_simetrica(*quantizar_simetrica(w, dim=0))


def erro_por_elemento(orig, rec):
    """Mediana do erro relativo de CADA elemento -- nao deixa o grande dominar."""
    return ((rec - orig).abs() / orig.abs().clamp(min=1e-12)).median().item()


print(f"  {'metrica':>38s} {'per-tensor':>12s} {'per-channel':>13s}")
print(f"  {'erro relativo global (norma)':>38s} "
      f"{erro_relativo(w, r_t):>11.2%} {erro_relativo(w, r_c):>13.2%}")
print(f"  {'... so nas 7 linhas normais':>38s} "
      f"{erro_relativo(w[1:], r_t[1:]):>11.2%} {erro_relativo(w[1:], r_c[1:]):>13.2%}")
print(f"  {'MEDIANA do erro por elemento':>38s} "
      f"{erro_por_elemento(w, r_t):>11.2%} {erro_por_elemento(w, r_c):>13.2%}")
n_zerados = (r_t[1:] == 0).float().mean().item()
print(f"\n  fracao das linhas normais que virou ZERO exato (per-tensor): {n_zerados:.1%}")
print(f"""
  Respostas:
  1. Compare as duas primeiras linhas da tabela: {erro_relativo(w, r_t):.1%} de erro
     global contra {erro_relativo(w[1:], r_t[1:]):.1%} nas linhas normais. Sao tao
     diferentes porque a norma global e' dominada pela linha grande, que
     quantiza bem.
  2. A conta: ||w|| e' dominado pela linha 0, cujos valores sao 100x maiores.
     Como a norma soma QUADRADOS, essa linha contribui ~10.000x mais que cada
     uma das outras. O denominador de erro_relativo e' praticamente so' ela,
     enquanto o numerador (o erro) esta' espalhado -- entao o erro das linhas
     pequenas fica dividido por um numero que nao tem nada a ver com elas.
  3. A metrica que denuncia sem desagregar: MEDIANA do erro relativo POR
     ELEMENTO. Ela nao tem denominador global, entao nenhum elemento pode
     esconder os outros. Veja na tabela: ela ja' acusa a diferenca.

     Uma alternativa igualmente boa: a fracao de pesos que virou ZERO EXATO.
     Acima ela mostra {n_zerados:.0%} das linhas normais -- elas nao foram
     'degradadas', foram APAGADAS. Uma mediana de erro por elemento de
     {erro_por_elemento(w, r_t):.0%} diz a mesma coisa: mais da metade dos pesos
     perdeu todo o valor.""")

# ===========================================================================
print("=" * 74)
print("E4 — onde fica o precipicio")
print("=" * 74)
w256 = torch.randn(256, 256)
base = loss_val(modelo)
print(f"  {'bits':>5s} {'erro no PESO':>14s} {'piora da LOSS':>15s} {'perplexidade':>13s}")
print(f"  {'fp32':>5s} {'--':>14s} {'--':>15s} {math.exp(base):>13.1f}")
pioras = {}
for bits in (8, 6, 4, 3, 2):
    e_peso = erro_relativo(w256, desquantizar_simetrica(
        *quantizar_simetrica(w256, bits=bits, dim=0)))
    l = loss_val(quantizar_modelo(modelo, bits=bits))
    pioras[bits] = l - base
    print(f"  {bits:>5d} {e_peso:>13.2%} {l-base:>+15.4f} {math.exp(l):>13.1f}",
          flush=True)
print(f"""
  Respostas:
  1. Uma NAO prevê a outra. O erro no peso cresce de forma suave e regular
     (dobra a cada bit). A piora da loss fica desprezivel ate' 4 bits e depois
     dispara. Sao curvas de formatos diferentes.

     A razao e' que a rede tem folga: pequenas perturbacoes nos pesos sao
     absorvidas, ate' o ponto em que deixam de ser pequenas.

  2. O joelho esta' entre 4 e 3 BITS. A piora salta de {pioras[4]:+.4f} para
     {pioras[3]:+.4f} -- cerca de {pioras[3]/pioras[4]:.0f}x. E de 3 para 2 o modelo
     e' destruido: a perplexidade sai da casa das dezenas para a dos milhares.

  3. Para servir este modelo eu escolheria int8 -- e a justificativa precisa dos
     DOIS eixos:
       - qualidade: int8 e' de graca (a loss nao muda na 4a casa)
       - tamanho  : 4x menor, o que resolve o gargalo do Capitulo 12
     int4 tambem seria defensavel (custa 0,06 de loss e da' 8x), e a escolha
     entre os dois depende de quanta memoria voce precisa economizar. O que NAO
     e' defensavel e' escolher por um eixo so'.""")

# ===========================================================================
print("=" * 74)
print("E5 — quantizando os embeddings tambem")
print("=" * 74)
lineares = sum(m.weight.numel() for m in modelo.modules() if isinstance(m, nn.Linear))
total = sum(p.numel() for p in modelo.parameters())
outros = total - lineares
for bits in (8, 4):
    l_sem = loss_val(quantizar_modelo(modelo, bits=bits))
    l_com = loss_val(quantizar_modelo(modelo, bits=bits, incluir_embeddings=True))
    mb_sem = (lineares * bits / 8 + outros * 4) / 1e6
    mb_com = (total * bits / 8) / 1e6
    print(f"  int{bits}: so' Linear   -> {mb_sem:5.2f} MB | loss {l_sem:.4f}")
    print(f"        + embeddings -> {mb_com:5.2f} MB | loss {l_com:.4f} "
          f"({l_com-l_sem:+.4f})", flush=True)
print("""
  Respostas:
  1. O modelo encolhe mais e a loss quase nao muda em int8. Em int4 o custo de
     incluir os embeddings ja' aparece.
  2. Vale mais a pena onde eles sao 10% (este modelo) do que onde sao 2% (um
     7B) -- mas a resposta NAO e' 'no que tem a maior fracao', e sim: vale onde
     a fracao e' grande O BASTANTE para mover o total.

     Num 7B, quantizar os embeddings muda o tamanho em 2%. Nao compensa o risco
     de qualidade nem a complexidade. Aqui muda em 10%, o que ja' e' visivel. A
     pergunta e' sempre 'quanto isso move o numero que me importa', nao 'qual e'
     maior'.
  3. A camada de saida merece cuidado extra por uma razao estrutural: ela
     produz os LOGITS, e a softmax e' sensivel a diferencas pequenas entre eles.
     Um erro de quantizacao que seria inofensivo numa camada intermediaria pode
     reordenar os tokens mais provaveis. Muitos esquemas de producao deixam a
     camada de saida (e a primeira) em precisao maior justamente por isso.""")

# ===========================================================================
print("=" * 74)
print("E6 — o que a quantizacao faz com a velocidade")
print("=" * 74)
prompt = torch.zeros((1, 1), dtype=torch.long)
mq8 = quantizar_modelo(modelo, bits=8)
t_f = cronometrar(lambda: modelo.gerar_com_cache(prompt, 64, semente=1))
t_q = cronometrar(lambda: mq8.gerar_com_cache(prompt, 64, semente=1))
print(f"  float32 : {t_f:.3f}s")
print(f"  int8    : {t_q:.3f}s   ({t_f/t_q:.2f}x)")
print("""
  Respostas:
  1. O float32 e' MAIS RAPIDO. A quantizacao, desta forma, custa tempo.
  2. Onde o tempo vai: no forward da LinearQuantizada,

         F.linear(x, desquantizar_simetrica(self.q, self.escala), self.bias)

     a desquantizacao reconstroi a matriz inteira em float32 -- uma conversao de
     tipo mais uma multiplicacao por escala, sobre TODOS os pesos, a CADA
     chamada. Depois disso a matmul e' exatamente a mesma de antes. Ou seja:
     trabalho extra, mesmo trabalho antigo.
  3. Para acelerar seria preciso um kernel que multiplicasse int8 x int8
     acumulando em int32, sem nunca materializar a matriz em float. Nao se faz
     em PyTorch puro porque o operador nao existe: F.linear espera floats. E'
     preciso descer para C++/CUDA (ou usar llama.cpp, bitsandbytes, ONNX
     Runtime, que ja' fizeram isso).

     A licao: 'menor' e 'mais rapido' sao independentes. A primeira decorre da
     representacao; a segunda exige que alguem escreva o kernel.""")

# ===========================================================================
print("=" * 74)
print("E7 — quantizacao por grupos")
print("=" * 74)


def quantizar_grupos(x, bits=8, G=64):
    """Uma escala a cada G valores consecutivos da linha (o esquema do GGUF/GPTQ)."""
    saida, entrada = x.shape
    assert entrada % G == 0, "para simplificar, entrada precisa ser multiplo de G"
    xg = x.reshape(saida, entrada // G, G)
    qmax = 2 ** (bits - 1) - 1
    escala = torch.clamp(xg.abs().amax(dim=2, keepdim=True) / qmax, min=1e-12)
    q = torch.round(xg / escala).clamp(-qmax, qmax)
    return q, escala, (saida, entrada)


def desquantizar_grupos(q, escala, forma):
    return (q * escala).reshape(forma)


w = torch.randn(256, 256)
w[:, :32] *= 50.0          # um bloco de colunas com escala muito maior
NORMAIS = slice(32, None)  # as colunas que NAO sao outlier

print("  matriz 256x256 com as 32 primeiras colunas 50x maiores (int4)\n")
print(f"  {'esquema':>28s} {'erro global':>12s} {'erro nas normais':>18s} "
      f"{'bits/peso':>11s}")
r_pc = desquantizar_simetrica(*quantizar_simetrica(w, bits=4, dim=0))
print(f"  {'per-channel (1/linha)':>28s} {erro_relativo(w, r_pc):>11.2%} "
      f"{erro_relativo(w[:, NORMAIS], r_pc[:, NORMAIS]):>17.2%} {4 + 16/256:>11.2f}")
for G in (128, 64, 32):
    q, s, f = quantizar_grupos(w, bits=4, G=G)
    r = desquantizar_grupos(q, s, f)
    print(f"  {f'grupos de {G}':>28s} {erro_relativo(w, r):>11.2%} "
          f"{erro_relativo(w[:, NORMAIS], r[:, NORMAIS]):>17.2%} {4 + 16/G:>11.2f}")

print(f"""
  ATENCAO -- eu caí na armadilha do proprio E3 ao escrever este exercicio.

  Na primeira versao eu media so' o erro GLOBAL e concluí que o ganho dos grupos
  era pequeno (de 11,2% para 9,9%, contra 11% mais bits: praticamente empate).
  Errado, e pelo motivo que o E3 acabou de ensinar: a norma global e' dominada
  pelas 32 colunas outlier, que quantizam bem em qualquer esquema.

  Olhando a coluna que importa -- o erro nas colunas NORMAIS, que sao 87,5% da
  matriz -- o ganho aparece.

  Respostas:
  1 e 2. Grupos menores reduzem o erro onde ele existe, porque cada escala
     precisa cobrir menos variacao. Com G=32 o bloco de outliers cai inteiro num
     grupo so' e para de contaminar os demais -- exatamente o que se quer.

  3. A conta de bits/peso REAIS e' a parte que costuma ser esquecida:

         bits_reais = bits + (bits_da_escala / G)

     Com int4 e G=64, guardando a escala em fp16: 4 + 16/64 = 4,25 bits por
     peso. Com G=32: 4,5 bits. Ou seja, 'int4' com grupos pequenos nao e' 4 bits
     -- e' ate' 12% mais.

     E' exatamente esse compromisso que os nomes dos formatos GGUF expoem: Q4_0,
     Q4_K_M e afins diferem no tamanho do grupo e em quantos bits gastam com
     metadados. Escolher entre eles e' escolher um ponto nesta curva -- e a
     escolha so' faz sentido se voce medir o erro ONDE ELE ESTA'.""")
