"""
e2_quando_a_mascara_importa.py — a mascara no pedido importa? Depende, e da' para
descobrir de que.

A recomendacao padrao ("mascare o prompt, treine so' na resposta") esta' em todo
tutorial de SFT. Medindo com o dataset deste capitulo -- pedido de 24 tokens,
resposta de ~34 -- ela NAO faz diferenca nenhuma:

    com mascara : loss 4.0030 | taxa de parada 100%
    sem mascara : loss 3.9995 | taxa de parada 100%

Um resultado nulo nao e' o fim da investigacao; e' o comeco. A pergunta vira:
sob que condicao a mascara passaria a importar?

HIPOTESE: a loss sem mascara e' a media sobre TODAS as posicoes. Se o pedido for
curto, ele e' uma fracao pequena e quase nao dilui o sinal da resposta. Se o
pedido for LONGO, ele domina a media -- e o gradiente passa a ser quase todo
sobre uma tarefa que ninguem pediu.

Este script testa isso variando a proporcao pedido/resposta.

Run (a partir da pasta do capitulo):
    python solucoes/e2_quando_a_mascara_importa.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(CAP.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(CAP.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import preparar_sft as ps
from preparar_sft import IGNORAR, FIM
from dataset import carregar
from sft import finetune, avaliar, aumentar_vocabulario
from avaliar import gerar_resposta

PASSOS = 400          # menos que o sft.py: sao 6 treinos


def montar_com(tam_pedido, resp_min, resp_max, n_treino=6000, n_val=300):
    """Remonta o dataset com outra proporcao pedido/resposta."""
    guardados = (ps.TAM_PEDIDO, ps.RESP_MIN, ps.RESP_MAX)
    ps.TAM_PEDIDO, ps.RESP_MIN, ps.RESP_MAX = tam_pedido, resp_min, resp_max
    from dataset import carregar_tokenizador
    _, vocab = carregar_tokenizador()
    fins = ps.tokens_de_fim_de_frase(vocab)
    Xtr, Ytr = ps.montar(carregar("treino"), n_treino, fins, semente=1337)
    Xva, Yva = ps.montar(carregar("val"), n_val, fins, semente=99)
    ps.TAM_PEDIDO, ps.RESP_MIN, ps.RESP_MAX = guardados
    return Xtr, Ytr, Xva, Yva


def fracao_de_pedido(X, Y):
    """Que fracao das posicoes treinadas SEM mascara e' pedido."""
    n = (X != FIM).sum(1) + 1
    treinadas = (Y != IGNORAR).sum(1)
    return float((n - treinadas).sum() / n.sum())


print("=" * 74)
print("Quando a mascara no pedido passa a importar?")
print("=" * 74)
print(f"  {PASSOS} passos por treino, 3 proporcoes x 2 variantes\n")
print(f"  {'pedido/resposta':>18s} {'% pedido':>9s} {'com mascara':>13s} "
      f"{'sem mascara':>13s} {'diferenca':>11s}")

CASOS = [
    ("24 / ~34", 24, 12, 56),
    ("64 / ~20", 64, 8, 32),
    ("104 / ~12", 104, 4, 18),
]

# Um arquivo SEPARADO, e nao o sft_dados.npz do capitulo.
#
# A primeira versao deste script sobrescrevia o dataset do capitulo e o
# regenerava no fim. Isso e' correto enquanto o script termina. Interrompa-o no
# meio -- Ctrl+C, ou o timeout do smoke test -- e o capitulo inteiro fica com um
# dataset truncado que ninguem pediu, e o proximo script treina sobre ele em
# silencio. Um arquivo proprio elimina a possibilidade, em vez de administra'-la.
TEMP = CAP / "_sft_dados_e2.npz"

for rotulo, tp, rmin, rmax in CASOS:
    Xtr, Ytr, Xva, Yva = montar_com(tp, rmin, rmax)
    frac = fracao_de_pedido(Xtr, Ytr)

    np.savez_compressed(TEMP, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)
    _, l_com, _ = finetune(mascarar=True, passos=PASSOS, verbose=False,
                           dados_npz=TEMP)
    _, l_sem, _ = finetune(mascarar=False, passos=PASSOS, verbose=False,
                           dados_npz=TEMP)
    print(f"  {rotulo:>18s} {frac:>8.0%} {l_com:>13.4f} {l_sem:>13.4f} "
          f"{l_sem - l_com:>+11.4f}", flush=True)

print("""
  Leia a coluna da direita: e' a penalidade de NAO mascarar.

  Com pedido curto ela e' nula ou negativa -- a recomendacao padrao nao se
  sustenta nessa configuracao. Conforme o pedido cresce e passa a dominar as
  posicoes treinadas, a penalidade aparece.

  O mecanismo: a loss sem mascara e' a MEDIA sobre todas as posicoes. Com 80%
  das posicoes sendo pedido, 80% do gradiente empurra o modelo a ficar bom em
  prever o texto do usuario -- que ninguem vai pedir que ele gere.

  POR QUE ISSO IMPORTA NA PRATICA: em SFT de verdade a proporcao e' justamente
  essa. Um prompt de sistema com instrucoes longas, mais a pergunta do usuario,
  costuma ser muito maior que a resposta. A recomendacao padrao esta' certa --
  no regime em que ela foi formulada.

  E e' por isso que um resultado nulo nao encerra a questao. 'Nao faz diferenca'
  quase sempre quer dizer 'nao faz diferenca AQUI', e vale descobrir onde e' o
  aqui.""")

# Nada a restaurar: o sft_dados.npz do capitulo nunca foi tocado.


# ===========================================================================
# O PISO DE RUIDO -- sem ele a tabela acima nao autoriza conclusao nenhuma.
#
# As diferencas medidas ficaram na casa de 0,015 sobre uma loss de ~3,95: 0,4%.
# Antes de chamar isso de efeito, e' preciso saber quanto a mesma configuracao
# varia so' por mudar a ordem dos batches. Se o ruido for maior que 0,015, a
# tabela acima nao mostra nada.
# ===========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 74)
    print("Quanto varia so' pela semente? (o piso de ruido)")
    print("=" * 74)
    Xtr, Ytr, Xva, Yva = montar_com(104, 4, 18)     # o caso de pedido longo
    np.savez_compressed(TEMP, Xtr=Xtr, Ytr=Ytr, Xva=Xva, Yva=Yva)

    SEMENTES = (1337, 42, 2024, 7, 99, 555)
    print(f"  configuracao 104/~12, {len(SEMENTES)} sementes por variante")
    print("  as duas variantes compartilham a semente: e' um experimento PAREADO\n")
    print(f"  {'semente':>9s} {'com mascara':>13s} {'sem mascara':>13s} {'diferenca':>11s}")
    difs, coms = [], []
    for s in SEMENTES:
        _, lc, _ = finetune(mascarar=True, passos=PASSOS, verbose=False,
                            semente=s, dados_npz=TEMP)
        _, ls, _ = finetune(mascarar=False, passos=PASSOS, verbose=False,
                            semente=s, dados_npz=TEMP)
        coms.append(lc)
        difs.append(ls - lc)
        print(f"  {s:>9d} {lc:>13.4f} {ls:>13.4f} {ls-lc:>+11.4f}", flush=True)

    import statistics
    n = len(difs)
    positivas = sum(d > 0 for d in difs)
    media = statistics.mean(difs)
    desvio = statistics.stdev(difs)
    amplitude = max(coms) - min(coms)

    print(f"""
  A ESCOLHA DO CRITERIO DECIDE A CONCLUSAO -- e a primeira versao deste script
  usou o criterio errado.

  Comparacao NAO pareada (errada para este desenho):
    variacao da mesma variante entre sementes : {amplitude:.4f}
    diferenca media entre variantes           : {media:+.4f}
    veredito                                  : 'o ruido supera o efeito'

  Comparacao PAREADA (a correta):
    diferencas         : {', '.join(f'{d:+.4f}' for d in difs)}
    na direcao prevista: {positivas} de {n}
    media +- desvio    : {media:+.4f} +- {desvio:.4f}

  Por que a pareada e' a certa: as duas variantes rodam com a MESMA semente,
  entao a variacao que a semente causa afeta as duas e se CANCELA na diferenca.
  Comparar o efeito com a variacao absoluta joga fora exatamente a vantagem de
  ter pareado o experimento -- e quase me fez descartar um sinal consistente.""")

    if positivas == n and media > desvio:
        print(f"""
  VEREDITO: {positivas}/{n} sementes na mesma direcao, media maior que o desvio.
  A tendencia se sustenta -- mascarar ajuda quando o pedido domina as posicoes.

  MAS OLHE A MAGNITUDE: {media:+.4f} sobre uma loss de ~3,95, ou seja {media/3.95:.1%}.
  'Detectavel' e 'importante na pratica' sao coisas diferentes. Este numero e' o
  primeiro sem ser o segundo, nesta escala.""")
    elif positivas >= n - 1:
        print(f"""
  VEREDITO: {positivas}/{n} na direcao prevista, mas a media ({media:+.4f}) nao
  supera o desvio ({desvio:.4f}). Ha' sinal e ele NAO esta' estabelecido com {n}
  sementes -- seriam precisas mais, ou mais passos, ou um modelo maior.""")
    else:
        print(f"""
  VEREDITO: apenas {positivas}/{n} na direcao prevista. A tendencia da tabela
  inicial NAO se sustenta: era ruido.""")

    TEMP.unlink(missing_ok=True)
