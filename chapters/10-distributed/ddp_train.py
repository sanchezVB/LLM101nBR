"""
ddp_train.py — treino de verdade com DistributedDataParallel (DDP).

A ideia do paralelismo de dados:

    - todo processo tem uma COPIA COMPLETA do modelo (identica no inicio)
    - cada um recebe um PEDACO DIFERENTE do batch
    - cada um calcula gradientes sobre o seu pedaco
    - all-reduce faz a MEDIA dos gradientes
    - todos aplicam a mesma atualizacao -> os modelos continuam identicos

O resultado matematico e' o mesmo de treinar com o batch inteiro num processo so'
-- e este arquivo PROVA isso, comparando os gradientes numero a numero.

Run:
    python ddp_train.py           # 4 processos
    python ddp_train.py 2         # 2 processos
"""

import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from dist_utils import iniciar, encerrar, limpar_rendezvous, apenas_no_rank0

BLOCK_SIZE = 8
N_EMBD = 64
BATCH_TOTAL = 256          # batch efetivo, dividido entre os processos
PASSOS = 50


def carregar_dados():
    words = open("names.txt", encoding="utf-8").read().splitlines()
    words = [w.strip() for w in words if w.strip()]
    chars = sorted(set("".join(words)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    X, Y = [], []
    for w in words[:20000]:            # subconjunto: aqui medimos mecanica, nao qualidade
        ctx = [0] * BLOCK_SIZE
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y), len(stoi)


class Modelo(nn.Module):
    """Um modelo simples -- o foco do capitulo e' a distribuicao, nao a arquitetura."""

    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, N_EMBD)
        self.rede = nn.Sequential(
            nn.Linear(BLOCK_SIZE * N_EMBD, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, vocab),
        )

    def forward(self, idx):
        x = self.emb(idx).view(idx.shape[0], -1)
        return self.rede(x)


def worker(rank, world_size):
    iniciar(rank, world_size, verbose=True)
    torch.manual_seed(1337)            # MESMA semente -> modelos identicos no inicio

    X, Y, vocab = carregar_dados()
    modelo = Modelo(vocab)
    apenas_no_rank0(rank, f"\nmodelo: {sum(p.nelement() for p in modelo.parameters())} parametros")

    # -----------------------------------------------------------------------
    # 1. PROVA: o gradiente do DDP e' igual ao do batch inteiro num processo so'?
    # -----------------------------------------------------------------------
    apenas_no_rank0(rank, f"\n=== 1. o DDP calcula o gradiente certo? ===")
    apenas_no_rank0(rank, f"  batch total {BATCH_TOTAL}, dividido em {world_size} "
                          f"pedacos de {BATCH_TOTAL // world_size}")

    # gradiente de referencia: batch INTEIRO, sem distribuicao
    ref = Modelo(vocab)
    ref.load_state_dict(modelo.state_dict())
    lote_x, lote_y = X[:BATCH_TOTAL], Y[:BATCH_TOTAL]
    F.cross_entropy(ref(lote_x), lote_y).backward()
    grad_ref = torch.cat([p.grad.flatten() for p in ref.parameters()])

    # agora com DDP: cada rank pega a sua fatia
    ddp = DDP(modelo)
    inicio = rank * (BATCH_TOTAL // world_size)
    fim = inicio + (BATCH_TOTAL // world_size)
    meu_x, meu_y = X[inicio:fim], Y[inicio:fim]

    # o DDP dispara o all-reduce automaticamente durante o backward
    F.cross_entropy(ddp(meu_x), meu_y).backward()
    grad_ddp = torch.cat([p.grad.flatten() for p in ddp.module.parameters()])

    dif = (grad_ref - grad_ddp).abs().max().item()
    print(f"  [rank {rank}] diferenca maxima vs batch unico: {dif:.2e} "
          f"{'OK' if dif < 1e-5 else 'ERRO'}", flush=True)

    dist.barrier()
    apenas_no_rank0(rank, """
  Os gradientes sao IGUAIS (a menos de arredondamento). Isso confirma a
  propriedade central: N processos com batch/N cada equivalem a um processo com
  o batch inteiro. O DDP nao aproxima -- ele reproduz exatamente.""")

    # -----------------------------------------------------------------------
    # 2. Os modelos continuam sincronizados ao longo do treino?
    # -----------------------------------------------------------------------
    dist.barrier()
    apenas_no_rank0(rank, "=== 2. treino: os pesos continuam identicos? ===")

    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(1337 + rank)      # cada rank sorteia dados diferentes
    por_processo = BATCH_TOTAL // world_size

    t0 = time.perf_counter()
    for passo in range(PASSOS):
        ix = torch.randint(0, X.shape[0], (por_processo,), generator=g)
        loss = F.cross_entropy(ddp(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()                 # <- all-reduce acontece aqui, automatico
        opt.step()
    tempo = time.perf_counter() - t0

    # checagem: os pesos de todos os ranks batem?
    primeiro = next(ddp.module.parameters()).detach().clone()
    reunidos = [torch.zeros_like(primeiro) for _ in range(world_size)]
    dist.all_gather(reunidos, primeiro)
    iguais = all(torch.allclose(reunidos[0], r, atol=1e-6) for r in reunidos)

    print(f"  [rank {rank}] loss final {loss.item():.4f} | {tempo:.1f}s | "
          f"pesos identicos entre ranks: {iguais}", flush=True)

    dist.barrier()
    apenas_no_rank0(rank, f"""
  Os pesos continuam identicos em todos os processos depois de {PASSOS} passos --
  apesar de cada um ter visto dados DIFERENTES. E' o all-reduce fazendo o seu
  trabalho a cada backward.

  Note tambem que a loss impressa por cada rank e' diferente: ela e' calculada
  sobre o pedaco LOCAL. Para registrar a loss real do treino, e' preciso fazer
  all_reduce dela tambem (e e' um erro comum esquecer disso).""")

    # -----------------------------------------------------------------------
    # 3. O que muda no hiperparametro: batch efetivo.
    # -----------------------------------------------------------------------
    dist.barrier()
    apenas_no_rank0(rank, f"""=== 3. cuidado com o batch efetivo ===

  Com {world_size} processos processando {por_processo} exemplos cada, o batch
  EFETIVO e' {BATCH_TOTAL} -- e nao {por_processo}. Isso importa porque:

    - gradiente de batch maior e' menos ruidoso
    - logo, tolera (e costuma pedir) learning rate MAIOR
    - a regra pratica mais usada: multiplicar a lr por sqrt(world_size),
      ou linearmente com warmup mais longo

  Trocar 1 GPU por 8 sem tocar na learning rate e' um erro classico: o treino
  fica mais lento em NUMERO DE PASSOS, mesmo sendo mais rapido por passo.""")

    encerrar()


if __name__ == "__main__":
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    limpar_rendezvous()
    print(f"iniciando {world_size} processos...")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nconcluido.")
