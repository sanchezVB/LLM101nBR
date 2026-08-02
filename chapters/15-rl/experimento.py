"""
experimento.py — o experimento central do capitulo, em tres configuracoes.

  1. recompensa BEM ESPECIFICADA (comprimento), com freio de KL
  2. recompensa MAL ESPECIFICADA (pontos), SEM freio     -> reward hacking
  3. a mesma mal especificada, COM freio                 -> o freio segura?

Tres numeros por configuracao, e os tres sao necessarios:

  RECOMPENSA -- o que o RL esta' otimizando
  KL         -- quanto a politica se afastou do modelo de referencia
  LOSS REAL  -- a loss em texto de Machado, que o RL NAO ve'

O terceiro e' o juiz. A recompensa sempre sobe -- e' o que o algoritmo faz. A
pergunta e' se o modelo continua sendo um modelo de portugues enquanto isso.

Run (a partir da pasta do capitulo):
    python experimento.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "14-sft"))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from reinforce import treinar, amostrar, carregar_politica
from preparar_sft import FIM, NOMES_ESPECIAIS, TAM_PEDIDO
from dataset import carregar as carregar_dados, pegar_batch, carregar_tokenizador
from recompensa import _comprimentos

PASSOS = 400
val = carregar_dados("val")
_, vocab = carregar_tokenizador()


@torch.no_grad()
def loss_texto_real(m, n=20, semente=11):
    """A loss em Machado. O RL nunca ve' este numero -- por isso ele julga."""
    g = torch.Generator().manual_seed(semente)
    m.eval()
    tot = 0.0
    for _ in range(n):
        x, y = pegar_batch(val, 16, m.block_size, generator=g)
        logits, _ = m(x)
        tot += F.cross_entropy(logits[..., :1024].reshape(-1, 1024),
                               y.reshape(-1)).item()
    m.train()
    return tot / n


def amostra_de_texto(politica, referencia, n=2):
    dados = np.load(AQUI.parent / "14-sft" / "sft_dados.npz")
    pedidos = torch.from_numpy(dados["Xva"][:n, 1:1 + TAM_PEDIDO])
    dados.close()
    with torch.no_grad():
        seqs, _, _ = amostrar(politica, referencia, pedidos)
    out = []
    for i in range(seqs.shape[0]):
        s = []
        for t in seqs[i]:
            t = int(t)
            if t == FIM:
                break
            s.append(NOMES_ESPECIAIS.get(
                t, vocab[t].decode("utf-8", errors="replace") if t < 1024 else "?"))
        out.append("".join(s).replace("\n", "\\n"))
    return out


if __name__ == "__main__":
    # ===========================================================================
    print("=" * 74)
    print("O experimento central: recompensa, KL e o juiz que o RL nao ve'")
    print("=" * 74)

    base = carregar_politica().eval()
    loss_base = loss_texto_real(base)
    print(f"  modelo de partida (SFT do Cap. 14): loss em Machado = {loss_base:.4f}\n")

    CONFIGS = [
        ("comprimento bem especificada, com freio", "comprimento", 0.02),
        ("pontos MAL especificada, SEM freio", "pontos", 0.0),
        ("pontos MAL especificada, com freio", "pontos", 0.02),
    ]

    print(f"  {'configuracao':>42s} {'recompensa':>18s} {'KL':>8s} {'loss real':>11s}")
    resultados = []
    for rotulo, nome_r, beta in CONFIGS:
        politica, referencia, hist, _ = treinar(nome_r, beta, PASSOS, verbose=False)
        r0 = float(np.mean([h[0] for h in hist[:10]]))
        r1 = float(np.mean([h[0] for h in hist[-10:]]))
        kl = float(np.mean([h[1] for h in hist[-10:]]))
        lr_ = loss_texto_real(politica)
        resultados.append((rotulo, nome_r, beta, r0, r1, kl, lr_,
                           amostra_de_texto(politica, referencia)))
        print(f"  {rotulo:>42s} {r0:>8.3f} -> {r1:<7.3f} {kl:>8.2f} {lr_:>11.4f}",
              flush=True)

    print(f"  {'(partida, sem RL)':>42s} {'--':>18s} {'0.00':>8s} {loss_base:>11.4f}")

    # ---------------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("O que as respostas viraram")
    print("=" * 74)
    for rotulo, _, _, _, _, _, _, amostras in resultados:
        print(f"\n  {rotulo}:")
        for a in amostras:
            print(f"    {a[:110]!r}")

    print(f"""

      COMO LER A TABELA, e a ordem importa:

      1. A recompensa SOBE nas tres. Isso nao e' informacao -- e' o que o algoritmo
         faz. Uma recompensa que nao sobe indica bug, nao sucesso.

      2. A coluna 'loss real' e' o juiz. Ela mede a capacidade de modelar Machado,
         que o RL nunca observa. Se ela se afasta de {loss_base:.4f}, o modelo esta'
         pagando com competencia o que ganha em pontuacao.

      3. A KL diz QUANTO a politica se afastou do ponto de partida. Ela e' o
         mecanismo, e a loss real e' a consequencia.

      A configuracao do meio existe para falhar, e falha: recompensa 0.996 com o
      modelo respondendo '. ' a qualquer pedido. Otimizar uma metrica razoavel,
      sem freio, produz um modelo que maximiza a metrica e abandona a tarefa.

      E A TERCEIRA TAMBEM FALHA -- eu tinha escrito aqui que ela mostraria o
      freio segurando. Nao mostra. Com beta=0.02 a recompensa ainda vai a 0.958
      e a resposta ainda e' '.', so' que o custo em portugues cai pela metade
      (+0.32 contra +0.74). O freio nao IMPEDE o hacking; ele o torna mais
      lento e mais barato.

      Isso reformula a licao, para melhor: a penalidade de KL e' um DIAL, nao um
      interruptor. Ela compra tempo e limita o estrago, e nao substitui uma
      recompensa correta. O e4_dial_do_kl.py varre o beta para achar a fronteira
      -- e a pergunta que ele responde e' se existe faixa util, ou se para esta
      recompensa nenhum beta serve.""")
