"""
Solucao do Exercicio E4 — registrando a loss corretamente em treino distribuido.

O problema: cada rank calcula a loss sobre o SEU pedaco do batch. Se voce
registrar `loss.item()` direto, esta' registrando a loss de 1/N dos dados -- um
numero mais ruidoso que a loss real, e diferente em cada processo.

A correcao e' fazer all_reduce da loss tambem, exatamente como se faz com os
gradientes. Este script mostra os dois lado a lado.

Run (a partir da pasta do capitulo):
    python solucoes/e4_loss_correta.py
"""

import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dist_utils import iniciar, encerrar, limpar_rendezvous, apenas_no_rank0
from ddp_train import Modelo, carregar_dados, BATCH_TOTAL

PASSOS = 30


def loss_global(loss_local, world_size):
    """A loss media sobre TODO o batch efetivo, e nao so' sobre o pedaco local."""
    t = loss_local.detach().clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / world_size).item()


def worker(rank, world_size):
    import os
    os.chdir(Path(__file__).resolve().parent.parent)   # names.txt esta' no capitulo
    iniciar(rank, world_size)
    torch.manual_seed(1337)

    X, Y, vocab = carregar_dados()
    ddp = DDP(Modelo(vocab))
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1337 + rank)
    por_processo = BATCH_TOTAL // world_size

    apenas_no_rank0(rank, f"\n{'passo':>6s} | {'loss LOCAL por rank':^34s} | {'loss GLOBAL':>11s}")
    apenas_no_rank0(rank, f"{'-'*6}-+-{'-'*34}-+-{'-'*11}")

    for passo in range(PASSOS):
        ix = torch.randint(0, X.shape[0], (por_processo,), generator=g)
        loss = F.cross_entropy(ddp(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if passo % 10 == 0 or passo == PASSOS - 1:
            # reune as losses locais so' para PODER MOSTRAR a diferenca
            locais = [torch.zeros(1) for _ in range(world_size)]
            dist.all_gather(locais, loss.detach().reshape(1))
            glob = loss_global(loss, world_size)
            if rank == 0:
                txt = " ".join(f"{l.item():7.4f}" for l in locais)
                print(f"{passo:6d} | {txt:^34s} | {glob:11.4f}", flush=True)

    dist.barrier()
    apenas_no_rank0(rank, """
Observe as colunas:

  As losses LOCAIS variam bastante entre os ranks -- e essa variacao e' ruido de
  amostragem, nao sinal. Cada rank viu apenas 1/N dos exemplos daquele passo.

  A loss GLOBAL e' a media sobre o batch efetivo inteiro. E' ela que corresponde
  a' loss que voce teria treinando num processo so' com o batch completo, e e' ela
  que deve ir para o grafico.

Erro comum: registrar `loss.item()` do rank 0 e concluir que "o treino esta' mais
ruidoso em multi-GPU". Nao esta' -- a MEDICAO e' que ficou mais ruidosa.

Custo: um all_reduce de UM escalar por passo, desprezivel perto do all-reduce dos
gradientes. Nao ha' motivo para nao fazer.""")

    encerrar()


if __name__ == "__main__":
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    limpar_rendezvous()
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
