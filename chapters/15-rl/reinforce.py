"""
reinforce.py — policy gradient do zero, com penalidade de KL.

O SFT do Capitulo 14 aprende IMITANDO respostas que alguem escreveu. Aqui nao
ha' resposta certa para imitar: ha' uma pontuacao. O modelo gera, e' avaliado, e
os gradientes empurram na direcao do que pontuou bem.

O algoritmo e' o REINFORCE, e cabe numa linha:

    loss = -(R - baseline) * log P(resposta gerada)

Se a recompensa foi acima da media, a loss empurra para AUMENTAR a probabilidade
daquela resposta. Se foi abaixo, para diminuir. Nada mais.

  O BASELINE (a media do batch) nao muda o gradiente esperado -- ele so' reduz a
  variancia. Sem ele, TODAS as respostas com recompensa positiva sao reforcadas,
  inclusive as ruins, e o aprendizado fica lento e instavel. O E3 mede.

  A PENALIDADE DE KL amarra a politica ao modelo de referencia (o SFT, congelado):

    R_efetiva = R - beta * log( P_politica / P_referencia )

  Ela nao esta' ali por elegancia. E' o unico freio contra o modelo abandonar o
  portugues para perseguir a pontuacao -- o que a recompensa 'pontos' provoca em
  poucas dezenas de passos.

Run (a partir da pasta do capitulo):
    python reinforce.py                          # recompensa de comprimento
    python reinforce.py --recompensa pontos      # a mal especificada
    python reinforce.py --recompensa pontos --beta 0.0    # sem freio
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "14-sft"))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import GPT
from preparar_sft import (PEDIDO, RESPOSTA, FIM, NOMES_ESPECIAIS,
                          TAM_PEDIDO, tokens_de_fim_de_frase)
from dataset import carregar as carregar_dados, pegar_batch, carregar_tokenizador
from recompensa import RECOMPENSAS, ALVO

PASSOS = 400
BATCH = 32
MAX_TOKENS = 48
LR = 3e-5              # RL e' instavel; passos menores que o SFT
BETA = 0.02            # peso da penalidade de KL

# SOBRE ESTE ORCAMENTO. A primeira versao usava 150 passos e lr 1e-5, e o
# reward hacking mal aparecia: a recompensa subia de 0.05 para 0.16 e o texto
# continuava parecendo portugues. Eu quase publiquei aquilo como 'demonstracao
# de reward hacking' -- nao era, era um empurraozinho na direcao certa.
#
# Com 400 passos e lr 3e-5 o fenomeno e' total: a recompensa vai a 0.996 e o
# modelo passa a responder '. ' e mais nada.
#
# A licao vale alem deste capitulo: quando um fenomeno NAO aparece, a primeira
# pergunta e' se voce deu orcamento para ele aparecer.


def carregar_politica():
    """Parte do modelo do Capitulo 14 -- ja' sabe o formato e sabe parar."""
    caminho = AQUI.parent / "14-sft" / "modelo_sft.pt"
    if not caminho.exists():
        raise SystemExit(
            f"Checkpoint do SFT nao encontrado em {caminho}.\n"
            f"Rode antes:  cd ../14-sft && python preparar_sft.py && python sft.py"
        )
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m = GPT(ck["config"])
    m.load_state_dict(ck["state_dict"])
    return m


def amostrar(politica, referencia, pedidos, max_tokens=MAX_TOKENS, gerador=None):
    """Gera respostas e devolve tudo que o gradiente precisa.

    Devolve (sequencias, logp_politica, logp_referencia), onde os log-prob sao
    somados sobre os tokens gerados -- so' eles, nao o pedido.
    """
    B = pedidos.shape[0]
    idx = torch.cat([
        torch.full((B, 1), PEDIDO, dtype=torch.long),
        pedidos,
        torch.full((B, 1), RESPOSTA, dtype=torch.long),
    ], dim=1)
    n_prompt = idx.shape[1]

    logp_pol = torch.zeros(B)
    logp_ref = torch.zeros(B)
    vivo = torch.ones(B, dtype=torch.bool)

    for _ in range(max_tokens):
        recorte = idx[:, -politica.block_size:]
        logits, _ = politica(recorte)
        logits = logits[:, -1, :]
        with torch.no_grad():
            logits_ref, _ = referencia(recorte)
            logits_ref = logits_ref[:, -1, :]

        dist = F.log_softmax(logits, dim=-1)
        dist_ref = F.log_softmax(logits_ref, dim=-1)
        prox = torch.multinomial(dist.exp(), 1, generator=gerador)

        # so' acumula log-prob de sequencias que ainda nao terminaram
        lp = dist.gather(1, prox).squeeze(1)
        lpr = dist_ref.gather(1, prox).squeeze(1)
        logp_pol = logp_pol + lp * vivo
        logp_ref = logp_ref + lpr * vivo

        vivo = vivo & (prox.squeeze(1) != FIM)
        idx = torch.cat((idx, prox), dim=1)
        if not vivo.any():
            break

    return idx[:, n_prompt:], logp_pol, logp_ref


def treinar(nome_recompensa="comprimento", beta=BETA, passos=PASSOS,
            usar_baseline=True, semente=1337, verbose=True):
    torch.manual_seed(semente)
    g = torch.Generator().manual_seed(semente)

    politica = carregar_politica()
    referencia = copy.deepcopy(politica).eval()
    for p in referencia.parameters():
        p.requires_grad_(False)

    _, vocab = carregar_tokenizador()
    fins = tokens_de_fim_de_frase(vocab)
    fn_recompensa = RECOMPENSAS[nome_recompensa]

    dados = np.load(AQUI.parent / "14-sft" / "sft_dados.npz")
    Xtr = torch.from_numpy(dados["Xtr"])
    dados.close()

    opt = torch.optim.AdamW(politica.parameters(), lr=LR)
    historico = []

    for passo in range(passos):
        ix = torch.randint(0, Xtr.shape[0], (BATCH,), generator=g)
        pedidos = Xtr[ix][:, 1:1 + TAM_PEDIDO]

        seqs, logp_pol, logp_ref = amostrar(politica, referencia, pedidos, gerador=g)

        with torch.no_grad():
            R = fn_recompensa(seqs, fins, FIM)
            # KL estimada por amostra: log P_pol - log P_ref na trajetoria gerada
            kl = logp_pol - logp_ref
            R_efetiva = R - beta * kl

        vantagem = R_efetiva - R_efetiva.mean() if usar_baseline else R_efetiva
        loss = -(vantagem * logp_pol).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(politica.parameters(), 1.0)
        opt.step()

        historico.append((R.mean().item(), kl.mean().item()))
        if verbose and (passo % 30 == 0 or passo == passos - 1):
            print(f"    passo {passo:>4d} | recompensa {R.mean():.3f} | "
                  f"KL {kl.mean():+.2f}", flush=True)

    return politica, referencia, historico, fins


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RL com policy gradient.")
    ap.add_argument("--recompensa", default="comprimento", choices=list(RECOMPENSAS))
    ap.add_argument("--beta", type=float, default=BETA)
    ap.add_argument("--passos", type=int, default=PASSOS)
    ap.add_argument("--sem-baseline", action="store_true")
    args = ap.parse_args()

    print("=" * 74)
    print(f"REINFORCE | recompensa '{args.recompensa}' | beta {args.beta} | "
          f"{args.passos} passos")
    print("=" * 74)

    politica, referencia, hist, fins = treinar(
        args.recompensa, args.beta, args.passos, not args.sem_baseline)

    r0 = np.mean([h[0] for h in hist[:10]])
    r1 = np.mean([h[0] for h in hist[-10:]])
    kl1 = np.mean([h[1] for h in hist[-10:]])
    print(f"\n  recompensa: {r0:.3f} -> {r1:.3f}   ({r1-r0:+.3f})")
    print(f"  KL final da referencia: {kl1:+.2f}")

    # como ficaram as respostas
    _, vocab = carregar_tokenizador()
    dados = np.load(AQUI.parent / "14-sft" / "sft_dados.npz")
    pedidos = torch.from_numpy(dados["Xva"][:3, 1:1 + TAM_PEDIDO])
    dados.close()
    with torch.no_grad():
        seqs, _, _ = amostrar(politica, referencia, pedidos)

    def txt(ts):
        out = []
        for t in ts:
            t = int(t)
            if t == FIM:
                break
            out.append(NOMES_ESPECIAIS.get(t, vocab[t].decode("utf-8", errors="replace")
                                           if t < 1024 else "?"))
        return "".join(out).replace("\n", "\\n")

    print("\n  respostas depois do RL:")
    for i in range(seqs.shape[0]):
        print(f"    {txt(seqs[i])!r}")

    torch.save({"state_dict": politica.state_dict(), "config": politica.cfg},
               AQUI / f"modelo_rl_{args.recompensa}_beta{args.beta}.pt")
