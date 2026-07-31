"""
Gabarito executavel do Capitulo 14 — SFT.

Roda E4, E5, E6 e E7. (O E1 e' conceitual; o E2 tem script proprio,
e2_quando_a_mascara_importa.py; o E3 esta' documentado no gabarito.md.)

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(CAP.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(CAP.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import sft as S
from sft import finetune, aumentar_vocabulario
from preparar_sft import PEDIDO, RESPOSTA, FIM, VOCAB_SFT
from avaliar import gerar_resposta, medir
from modelo import carregar
from dataset import carregar as carregar_dados, pegar_batch

PASSOS = 400
# MATERIALIZAR AGORA. np.load devolve um NpzFile PREGUICOSO, que le' do disco a
# cada acesso -- e o E5 sobrescreve este mesmo arquivo. Sem o dict() abaixo, o
# segundo acesso le' um arquivo que ja' foi trocado ("BadZipFile: Truncated file
# header"). Foi assim que este script quebrou na primeira execucao.
with np.load(CAP / "sft_dados.npz") as _z:
    DADOS = {k: _z[k] for k in _z.files}

# GUARDA CONTRA ESTADO CORROMPIDO, e ela existe por um motivo concreto.
#
# O E5 abaixo SOBRESCREVE o sft_dados.npz com subconjuntos menores e o restaura
# no fim. Se o script morrer no meio (foi o que aconteceu na primeira execucao,
# por causa da leitura preguicosa do np.load), o arquivo fica truncado -- e a
# execucao SEGUINTE o carrega sem reclamar.
#
# Foi assim que uma rodada inteira do E4, E5 e E6 saiu com 500 exemplos em vez
# de 8.000, produzindo numeros plausiveis e errados. Nada avisou.
ESPERADO = 8000
if DADOS["Xtr"].shape[0] != ESPERADO:
    raise SystemExit(
        f"sft_dados.npz tem {DADOS['Xtr'].shape[0]} exemplos de treino, "
        f"esperados {ESPERADO}.\n"
        f"Provavelmente uma execucao anterior morreu no meio do E5 e deixou o "
        f"arquivo truncado.\n"
        f"Rode:  python preparar_sft.py"
    )

pedidos = [DADOS["Xva"][i, 1:25].tolist() for i in range(30)]

# ===========================================================================
print("=" * 74)
print("E4 — a learning rate do finetuning")
print("=" * 74)
print(f"  {'lr':>10s} {'loss na resposta':>18s} {'taxa de parada':>16s} "
      f"{'loss no texto puro':>20s}")

val_puro = carregar_dados("val")


@torch.no_grad()
def loss_texto_puro(m, n=20, semente=11):
    """Loss no corpus ORIGINAL do Cap. 11, sem formato de instrucao.

    Mede se o modelo ainda sabe escrever portugues comum -- o eixo que o SFT
    poderia ter destruido.
    """
    g = torch.Generator().manual_seed(semente)
    m.eval()
    tot = 0.0
    for _ in range(n):
        x, y = pegar_batch(val_puro, 16, m.block_size, generator=g)
        logits, _ = m(x)
        # so' os 1024 tokens originais competem: os especiais nao existem no texto
        tot += F.cross_entropy(logits[..., :1024].reshape(-1, 1024),
                               y.reshape(-1)).item()
    return tot / n


base, _ = carregar()
base_puro = loss_texto_puro(aumentar_vocabulario(copy.deepcopy(base), VOCAB_SFT))
lr_original = S.LR
resultados_lr = {}
for lr in (3e-5, 3e-4, 1e-3):
    S.LR = lr
    m, l_resp, _ = finetune(passos=PASSOS, verbose=False)
    r = medir(m, pedidos, "")
    lp = loss_texto_puro(m)
    resultados_lr[lr] = (l_resp, r["taxa"], lp)
    print(f"  {lr:>10.0e} {l_resp:>18.4f} {r['taxa']:>15.0%} {lp:>20.4f}", flush=True)
S.LR = lr_original
print(f"  {'(base)':>10s} {'--':>18s} {'0%':>16s} {base_puro:>20.4f}")
print(f"""
  Respostas:
  1. NAO sao as tres. Com lr 3e-5 a taxa de parada e' 0% -- o modelo nao aprende
     a emitir o token novo. Passos pequenos demais nao chegam a instalar um
     comportamento que nao existia; eles so' ajustam o que ja' estava la'.

     (Eu tinha escrito aqui 'as tres aprendem a parar'. A medicao desmentiu.)

     A loss na resposta e' melhor no MEIO (3e-4), nao no extremo: 4.2448 com
     lr baixa, 3.9749 no meio, 4.0181 com lr alta. Learning rate boa nao e' a
     maior nem a menor.
  2. Olhe a ULTIMA coluna -- e' a que responde a pergunta. Ela mede a loss no
     texto corrido do Capitulo 11, sem formato de instrucao: a capacidade
     original do modelo. O modelo-base marca {base_puro:.4f}.

     Quanto maior a learning rate, mais essa coluna se afasta do valor original.
     O modelo esta' trocando a competencia antiga pela nova.

  3. CATASTROPHIC FORGETTING. O mecanismo: o finetuning ve' um conjunto pequeno
     e homogeneo, e o gradiente empurra TODOS os pesos na direcao que serve
     aquele conjunto. Nada no objetivo preserva o que foi aprendido antes -- a
     unica protecao e' dar passos pequenos (lr baixa) e poucos.

     E' por isso que lr de finetuning e' tipicamente 3 a 10 vezes menor que a
     do pre-treino. Nao e' cautela: e' o unico mecanismo disponivel.""")

# ===========================================================================
print("=" * 74)
print("E5 — quantos exemplos bastam?")
print("=" * 74)
Xtr_full, Ytr_full = DADOS["Xtr"], DADOS["Ytr"]
print(f"  {'exemplos':>10s} {'loss na resposta':>18s} {'taxa de parada':>16s}")
for n_ex in (500, 2000, 8000):
    np.savez_compressed(CAP / "sft_dados.npz",
                        Xtr=Xtr_full[:n_ex], Ytr=Ytr_full[:n_ex],
                        Xva=DADOS["Xva"], Yva=DADOS["Yva"])
    m, l, _ = finetune(passos=PASSOS, verbose=False)
    r = medir(m, pedidos, "")
    print(f"  {n_ex:>10,} {l:>18.4f} {r['taxa']:>15.0%}", flush=True)
np.savez_compressed(CAP / "sft_dados.npz", **DADOS)   # devolve o original
print("""
  Respostas:
  1. AS DUAS METRICAS SE SEPARAM, e essa e' a resposta.

     A taxa de parada satura em 500 exemplos -- ja' e' 100%. O formato e' barato
     de instalar: sao tres tokens especiais e uma convencao de onde cada um
     entra, e centenas de exemplos bastam.

     A loss na resposta NAO satura: 4.8662 -> 4.1922 -> 3.9749, quase 0,9 de
     melhora. (Eu tinha escrito aqui que ela 'melhora um pouco mais'. Melhora
     muito.)

     Sao coisas diferentes: uma mede se o modelo respeita o FORMATO, a outra se
     ele escreve uma resposta BOA. So' a primeira e' barata.

  2. Por isso a comparacao com o Capitulo 11 precisa ser dividida. Para o
     FORMATO, a curva satura cedo -- ao contrario do capitulo 11, onde mais
     dados sempre ajudaram. Para a QUALIDADE da resposta, a curva se parece com
     a de la', porque a tarefa volta a ser modelar texto.

  3. Resultados como o do LIMA ('mil exemplos bem escolhidos bastam') falam do
     primeiro eixo, nao do segundo. Se o que voce quer e' instalar
     comportamento -- formato, tom, recusa, uso de ferramenta -- mil exemplos
     variados bastam mesmo, e a medicao acima mostra por que: e' pouca
     informacao a transmitir.

     O que mil exemplos NAO compram e' capacidade de gerar conteudo melhor. Essa
     vem do pre-treino, e nenhum SFT a substitui.""")

# ===========================================================================
print("=" * 74)
print("E6 — o alignment tax")
print("=" * 74)
m_sft, _, _ = finetune(passos=PASSOS, verbose=False)
sft_puro = loss_texto_puro(m_sft)
print(f"  loss no texto corrido do Cap. 11 (sem formato de instrucao):")
print(f"    modelo-base      : {base_puro:.4f}")
print(f"    depois do SFT    : {sft_puro:.4f}   ({sft_puro - base_puro:+.4f})")
print(f"""
  Respostas:
  1 e 2. O SFT {'PIOROU' if sft_puro > base_puro else 'nao piorou'} a capacidade de modelar
     portugues comum: {sft_puro - base_puro:+.4f} de loss.
  3. E' o ALIGNMENT TAX -- o custo, na competencia original, de instalar um
     comportamento novo.

     Ele nao e' inevitavel, e ha' tres formas conhecidas de reduzi-lo:
       - MISTURAR dados de pre-treino no conjunto de SFT, para que o gradiente
         continue vendo texto comum;
       - congelar parte do modelo, ou treinar so' adaptadores de baixo posto
         (LoRA), limitando quanto os pesos originais podem se mover;
       - lr menor e menos passos, que e' o que o E4 mede.

     Todas atacam a mesma causa: nada no objetivo do SFT pede que o modelo
     lembre do que sabia.""")

# ===========================================================================
print("=" * 74)
print("E7 — um formato de conversa com varios turnos")
print("=" * 74)
print("""  A pergunta 2 e' a que decide o desenho, e da' para responder sem treinar.

  Se o <|fim|> aparecer SO' no fim da conversa, a geracao nao tem onde parar ao
  terminar um turno -- o modelo continuaria inventando a fala do usuario. Foi
  exatamente esse o comportamento do modelo-base que o capitulo inteiro
  combateu, so' que agora dentro do formato.

  Entao o token de fim tem de vir NO FIM DE CADA RESPOSTA. E' por isso que
  modelos reais usam <|im_end|> (ou equivalente) por turno, e nao um marcador
  unico de conversa.

  E a mascara muda: em vez de UMA faixa treinada, sao VARIAS -- uma por
  resposta. Todos os turnos do usuario ficam mascarados, inclusive os do meio.

    <|pedido|> A <|resposta|> B <|fim|> <|pedido|> C <|resposta|> D <|fim|>
    \\__ mascara __/\\__ treina __/      \\__ mascara __/\\__ treina __/

  Repare que isso torna a conta do E2 mais importante, nao menos: numa conversa
  longa, o historico (mascarado) cresce e as respostas nao. A fracao de posicoes
  que seriam desperdicadas sem mascara so' aumenta com o numero de turnos.""")
