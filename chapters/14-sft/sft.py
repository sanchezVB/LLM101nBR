"""
sft.py — supervised finetuning do modelo do Capitulo 11.

Duas coisas tecnicas acontecem aqui, e as duas sao o capitulo:

  1. AUMENTAR O VOCABULARIO. O modelo-base tem 1024 tokens de saida e precisa de
     1027, porque acrescentamos <|pedido|>, <|resposta|> e <|fim|>. Isso exige
     redimensionar a tabela de embeddings E a camada de saida -- e decidir com o
     que inicializar as linhas novas.

  2. MASCARAR A LOSS. So' os tokens da resposta contam. O PyTorch faz isso de
     graca: cross_entropy(ignore_index=-100) descarta os alvos marcados.

Run (a partir da pasta do capitulo):
    python sft.py                  # finetuning com mascara (o certo)
    python sft.py --sem-mascara    # treina no pedido tambem (o E2)
"""

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import carregar
from preparar_sft import PEDIDO, RESPOSTA, FIM, IGNORAR, VOCAB_SFT, BLOCO

PASSOS = 600
BATCH = 16
LR = 3e-4               # 3x menor que o do pre-treino: nao queremos destruir o que ja' existe


# ===========================================================================
def aumentar_vocabulario(m, novo_tamanho):
    """Estica a tabela de embeddings e a camada de saida.

    A decisao que importa e' COM O QUE preencher as linhas novas.

    Zeros parecem naturais e sao ruins na camada de saida: um logit igual para
    todos os tokens novos significa que eles comecam com a MESMA probabilidade
    dos tokens reais mais improvaveis -- e a rede leva tempo para separa-los.

    O que se faz na pratica, e o que fazemos aqui: inicializar com a MEDIA das
    linhas existentes. O token novo comeca como 'um token medio' e se afasta
    dali conforme aprende. E' mais rapido e mais estavel que zero ou aleatorio.
    """
    ne = m.cfg["n_embd"]
    velho = m.te.weight.shape[0]
    if velho >= novo_tamanho:
        return m

    te = nn.Embedding(novo_tamanho, ne)
    te.weight.data[:velho] = m.te.weight.data
    te.weight.data[velho:] = m.te.weight.data.mean(0, keepdim=True)
    m.te = te

    lm = nn.Linear(ne, novo_tamanho)
    lm.weight.data[:velho] = m.lm.weight.data
    lm.weight.data[velho:] = m.lm.weight.data.mean(0, keepdim=True)
    lm.bias.data[:velho] = m.lm.bias.data
    lm.bias.data[velho:] = m.lm.bias.data.mean()
    m.lm = lm

    m.cfg["vocab_size"] = novo_tamanho
    return m


def loss_mascarada(logits, alvos):
    """cross_entropy ignorando as posicoes marcadas com IGNORAR (-100)."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           alvos.reshape(-1), ignore_index=IGNORAR)


def batches(X, Y, batch, gerador):
    ix = torch.randint(0, X.shape[0], (batch,), generator=gerador)
    return X[ix], Y[ix]


@torch.no_grad()
def avaliar(m, X, Y, n=20, semente=7):
    g = torch.Generator().manual_seed(semente)
    m.eval()
    tot = 0.0
    for _ in range(n):
        x, y = batches(X, Y, BATCH, g)
        logits, _ = m(x)
        tot += loss_mascarada(logits, y).item()
    m.train()
    return tot / n


# ===========================================================================
def finetune(mascarar=True, passos=PASSOS, verbose=True, semente=1337,
             dados_npz=None):
    # `dados_npz` existe para que os exercicios possam treinar sobre datasets
    # ALTERNATIVOS sem tocar no arquivo do capitulo. A primeira versao nao tinha
    # esse parametro: os exercicios sobrescreviam sft_dados.npz e o restauravam
    # no fim. Funciona ate' a execucao ser interrompida no meio -- e ai' o
    # dataset do capitulo fica truncado, e o proximo script treina sobre ele sem
    # avisar nada. Foi o que o smoke test provocou, matando o E2 no meio do laco.
    dados = np.load(dados_npz or (AQUI / "sft_dados.npz"))
    Xtr = torch.from_numpy(dados["Xtr"])
    Ytr = torch.from_numpy(dados["Ytr"])
    Xva = torch.from_numpy(dados["Xva"])
    Yva = torch.from_numpy(dados["Yva"])

    # A VALIDACAO NUNCA MUDA. Guardamos os alvos mascarados antes de qualquer
    # coisa, porque as duas variantes precisam ser julgadas pela MESMA metrica:
    # a loss na RESPOSTA, que e' o que interessa nos dois casos.
    #
    # A primeira versao deste script alterava Yva junto com Ytr, e as duas
    # variantes saiam com losses de 2,08 e 4,02 -- numeros que pareciam uma
    # descoberta e mediam coisas diferentes. Comparacao invalida.
    Yva_metrica = Yva.clone()

    if not mascarar:
        # SEM mascara: no TREINO, o alvo passa a ser 'o proximo token' em toda
        # posicao, inclusive dentro do pedido. E' o que o E2 mede.
        Ytr = torch.where(Ytr == IGNORAR, torch.roll(Xtr, -1, dims=1), Ytr)

    base, _ = carregar()
    m = aumentar_vocabulario(copy.deepcopy(base), VOCAB_SFT)
    m.train()

    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(semente)
    t0 = time.perf_counter()

    for passo in range(passos):
        for grupo in opt.param_groups:
            grupo["lr"] = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * passo / passos)))
        x, y = batches(Xtr, Ytr, BATCH, g)
        logits, _ = m(x)
        loss = loss_mascarada(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if verbose and (passo % 150 == 0 or passo == passos - 1):
            print(f"    passo {passo:>4d} | loss {loss.item():.4f}", flush=True)

    dt = time.perf_counter() - t0
    m.eval()
    return m, avaliar(m, Xva, Yva_metrica), dt


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Finetuning supervisionado.")
    ap.add_argument("--sem-mascara", action="store_true",
                    help="treina tambem nos tokens do pedido (ver E2)")
    ap.add_argument("--passos", type=int, default=PASSOS)
    args = ap.parse_args()

    mascarar = not args.sem_mascara
    print("=" * 74)
    print(f"SFT {'COM' if mascarar else 'SEM'} mascara no pedido "
          f"({args.passos} passos, lr {LR})")
    print("=" * 74)

    base, _ = carregar()
    print(f"  modelo-base: {sum(p.nelement() for p in base.parameters()):,} parametros, "
          f"vocabulario {base.cfg['vocab_size']}")
    m_teste = aumentar_vocabulario(copy.deepcopy(base), VOCAB_SFT)
    print(f"  apos aumentar: {sum(p.nelement() for p in m_teste.parameters()):,} "
          f"parametros, vocabulario {m_teste.cfg['vocab_size']} "
          f"(+{VOCAB_SFT - base.cfg['vocab_size']} tokens especiais)\n")

    m, loss_val, dt = finetune(mascarar=mascarar, passos=args.passos)
    print(f"\n  loss de validacao (so' na resposta): {loss_val:.4f}")
    print(f"  tempo: {dt/60:.1f} min")

    saida = AQUI / ("modelo_sft.pt" if mascarar else "modelo_sft_sem_mascara.pt")
    torch.save({"state_dict": m.state_dict(), "config": m.cfg,
                "loss_val": loss_val, "mascarado": mascarar}, saida)
    print(f"  salvo em {saida.name}")
    print("\n  Rode agora:  python avaliar.py")
