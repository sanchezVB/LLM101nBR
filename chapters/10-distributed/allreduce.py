"""
allreduce.py — a operacao que sustenta todo o treino distribuido.

O problema: N processos treinam o MESMO modelo, cada um com um pedaco diferente
do batch. Cada um calcula um gradiente diferente. Para os modelos continuarem
identicos, todos precisam aplicar a MEDIA dos N gradientes.

    all-reduce = todos contribuem, e todos recebem o resultado combinado

Este arquivo faz tres coisas:
  1. usa o all_reduce do PyTorch e confere o resultado
  2. IMPLEMENTA um ring all-reduce a mao, para mostrar o algoritmo
  3. mede quanto tempo a comunicacao custa

Run:
    python allreduce.py            # 4 processos (padrao)
    python allreduce.py 2          # 2 processos
"""

import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dist_utils import iniciar, encerrar, limpar_rendezvous, apenas_no_rank0


def ring_allreduce_manual(tensor, rank, world_size):
    """Ring all-reduce implementado com send/recv, do zero.

    O algoritmo tem duas fases, e a chave e' que o tensor e' FATIADO em N pedacos:

      Fase 1 (reduce-scatter): em N-1 passos, cada processo passa um pedaco para
        o vizinho da direita e soma o pedaco que recebe da esquerda. Ao final,
        cada processo tem UM pedaco completamente somado (mas pedacos diferentes).

      Fase 2 (all-gather): em mais N-1 passos, esses pedacos completos circulam
        pelo anel ate todos terem todos.

    Por que em anel, e nao "todos mandam para o rank 0 e ele devolve"? BANDA.
    No esquema ingenuo, o rank 0 recebe (N-1) x tamanho e devolve (N-1) x tamanho
    -- ele vira um gargalo que piora conforme N cresce. No anel, cada processo
    envia e recebe sempre a MESMA quantidade (~2 x tamanho), independentemente de
    N. E' por isso que o ring all-reduce escala.
    """
    pedacos = list(tensor.chunk(world_size))
    esquerda = (rank - 1) % world_size
    direita = (rank + 1) % world_size

    # --- Fase 1: reduce-scatter ---
    for passo in range(world_size - 1):
        idx_envio = (rank - passo) % world_size
        idx_recebe = (rank - passo - 1) % world_size

        recebido = torch.zeros_like(pedacos[idx_recebe])
        # ordem par/impar evita deadlock: se todos enviassem primeiro, e o buffer
        # do sistema enchesse, todos ficariam esperando o outro receber
        if rank % 2 == 0:
            dist.send(pedacos[idx_envio].contiguous(), dst=direita)
            dist.recv(recebido, src=esquerda)
        else:
            dist.recv(recebido, src=esquerda)
            dist.send(pedacos[idx_envio].contiguous(), dst=direita)
        pedacos[idx_recebe] += recebido

    # --- Fase 2: all-gather ---
    for passo in range(world_size - 1):
        idx_envio = (rank + 1 - passo) % world_size
        idx_recebe = (rank - passo) % world_size

        recebido = torch.zeros_like(pedacos[idx_recebe])
        if rank % 2 == 0:
            dist.send(pedacos[idx_envio].contiguous(), dst=direita)
            dist.recv(recebido, src=esquerda)
        else:
            dist.recv(recebido, src=esquerda)
            dist.send(pedacos[idx_envio].contiguous(), dst=direita)
        pedacos[idx_recebe] = recebido        # substitui (ja' esta' somado)

    return torch.cat(pedacos)


def worker(rank, world_size):
    iniciar(rank, world_size, verbose=True)

    # -----------------------------------------------------------------------
    # 1. O all_reduce do PyTorch.
    # -----------------------------------------------------------------------
    apenas_no_rank0(rank, f"\n=== 1. all_reduce com {world_size} processos ===")
    dist.barrier()

    t = torch.tensor([float(rank + 1)])
    original = t.item()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    esperado = world_size * (world_size + 1) / 2
    print(f"  [rank {rank}] tinha {original:.0f}, depois do all_reduce SUM: "
          f"{t.item():.0f} (esperado {esperado:.0f}) "
          f"{'OK' if abs(t.item() - esperado) < 1e-6 else 'ERRO'}", flush=True)

    dist.barrier()
    apenas_no_rank0(rank, "\n  Note: TODOS os processos terminam com o MESMO valor.")
    apenas_no_rank0(rank, "  E' isso que mantem os modelos sincronizados.\n")

    # -----------------------------------------------------------------------
    # 2. A media -- que e' o que o treino realmente usa.
    # -----------------------------------------------------------------------
    dist.barrier()
    apenas_no_rank0(rank, "=== 2. media dos gradientes (o caso real) ===")
    grad = torch.tensor([float(rank + 1) * 0.1])
    antes = grad.item()
    dist.all_reduce(grad, op=dist.ReduceOp.SUM)
    grad /= world_size                     # SUM + divisao = media
    print(f"  [rank {rank}] gradiente local {antes:.2f} -> medio {grad.item():.4f}", flush=True)

    # -----------------------------------------------------------------------
    # 3. O nosso ring all-reduce bate com o do PyTorch?
    # -----------------------------------------------------------------------
    dist.barrier()
    apenas_no_rank0(rank, "\n=== 3. ring all-reduce implementado a mao ===")

    n = 8 * world_size                     # divisivel pelo numero de processos
    base = torch.arange(n, dtype=torch.float32) + rank * 100

    nosso = ring_allreduce_manual(base.clone(), rank, world_size)

    deles = base.clone()
    dist.all_reduce(deles, op=dist.ReduceOp.SUM)

    bate = torch.allclose(nosso, deles, atol=1e-5)
    print(f"  [rank {rank}] nosso == PyTorch? {bate}  "
          f"(primeiros 3: nosso {nosso[:3].tolist()}, pytorch {deles[:3].tolist()})",
          flush=True)

    # -----------------------------------------------------------------------
    # 4. Quanto custa comunicar?
    # -----------------------------------------------------------------------
    dist.barrier()
    apenas_no_rank0(rank, "\n=== 4. custo da comunicacao ===")
    if rank == 0:
        print(f"  {'tamanho':>12s} {'MB':>7s} {'tempo (ms)':>11s} {'MB/s':>9s}")

    for numel in (10_000, 1_000_000, 10_000_000):
        x = torch.randn(numel)
        mb = x.numel() * 4 / 1e6
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(5):
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        dt = (time.perf_counter() - t0) / 5
        if rank == 0:
            print(f"  {numel:12d} {mb:7.1f} {dt*1000:11.2f} {mb/dt:9.0f}")

    dist.barrier()
    apenas_no_rank0(rank, """
  Este e' o limite do treino distribuido: a cada passo, TODO o gradiente do
  modelo atravessa a rede. Um modelo de 1 bilhao de parametros gera 4 GB de
  gradientes por passo -- e se a comunicacao demorar mais que o calculo, aumentar
  o numero de maquinas para de ajudar. E' por isso que existem NVLink e InfiniBand.""")

    encerrar()


if __name__ == "__main__":
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    limpar_rendezvous()
    print(f"iniciando {world_size} processos...")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nconcluido.")
