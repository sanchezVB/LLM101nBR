"""
Worker do E3: demonstra o deadlock do send/recv sem alternancia par/impar.

Rodado como SUBPROCESSO pelo gabarito, com timeout -- porque o objetivo
literal deste exercicio e' fazer o programa TRAVAR, e um gabarito que trava
para sempre nao serve para ninguem.

Uso:
    python _worker_deadlock.py <rank> <world_size> <numel> <alternar 0|1>
"""

import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dist_utils import iniciar, encerrar

rank = int(sys.argv[1])
world = int(sys.argv[2])
numel = int(sys.argv[3])
alternar = bool(int(sys.argv[4]))

iniciar(rank, world)

x = torch.randn(numel)
recebido = torch.zeros_like(x)
esquerda = (rank - 1) % world
direita = (rank + 1) % world

if alternar and rank % 2 == 1:
    # ordem SEGURA: os impares recebem primeiro
    dist.recv(recebido, src=esquerda)
    dist.send(x, dst=direita)
else:
    # todos enviam primeiro -- se o buffer do sistema nao aguentar, TRAVA
    dist.send(x, dst=direita)
    dist.recv(recebido, src=esquerda)

print(f"[rank {rank}] terminou", flush=True)
encerrar()
