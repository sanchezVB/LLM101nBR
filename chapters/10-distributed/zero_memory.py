"""
zero_memory.py — o desperdicio do DDP, e como o ZeRO o elimina.

O DDP resolve o problema de VELOCIDADE (mais maquinas, mais exemplos por
segundo), mas nao o de MEMORIA: cada processo guarda uma copia COMPLETA de tudo.
Com 8 GPUs, o estado do otimizador esta' duplicado 8 vezes -- e ele e' a maior
parte do consumo.

Contabilidade de um treino com AdamW, por parametro:

    pesos            4 bytes (fp32)
    gradientes       4 bytes
    momento m        4 bytes   <- estado do AdamW
    momento v        4 bytes   <- estado do AdamW
    -----------------------------
    total           16 bytes por parametro, em CADA processo

O ZeRO (Zero Redundancy Optimizer) observa que essa duplicacao e' desnecessaria e
FATIA o estado entre os processos, em tres estagios:

    ZeRO-1: fatia o estado do otimizador (m, v)
    ZeRO-2: fatia tambem os gradientes
    ZeRO-3: fatia tambem os pesos (buscados sob demanda durante o forward)

Run:
    python zero_memory.py           # a conta + demo com 4 processos
"""

import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer

from dist_utils import iniciar, encerrar, limpar_rendezvous, apenas_no_rank0


# ---------------------------------------------------------------------------
# A conta de memoria (nao precisa de processos: e' aritmetica).
# ---------------------------------------------------------------------------
def tabela_memoria():
    print("=== memoria por processo, treinando com AdamW em fp32 ===\n")
    print("Um modelo de 1 bilhao de parametros:\n")
    print(f"  {'componente':<22s} {'bytes/param':>12s} {'total':>10s}")
    print(f"  {'-'*22} {'-'*12} {'-'*10}")
    itens = [("pesos", 4), ("gradientes", 4), ("AdamW: momento m", 4), ("AdamW: momento v", 4)]
    for nome, b in itens:
        print(f"  {nome:<22s} {b:>12d} {b:>9.0f} GB")
    print(f"  {'TOTAL':<22s} {16:>12d} {16:>9.0f} GB   <- em CADA processo\n")

    print(f"  {'processos':>10s} {'DDP':>10s} {'ZeRO-1':>10s} {'ZeRO-2':>10s} {'ZeRO-3':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for n in (1, 2, 4, 8, 64):
        ddp = 16.0                                   # tudo duplicado
        z1 = 4 + 4 + 8 / n                           # pesos + grads + (m,v)/n
        z2 = 4 + 4 / n + 8 / n                       # pesos + (grads + m,v)/n
        z3 = 16 / n                                  # tudo fatiado
        print(f"  {n:>10d} {ddp:>8.1f} GB {z1:>8.1f} GB {z2:>8.1f} GB {z3:>8.1f} GB")

    print("""
  Leia a linha de 64 processos: o DDP continua exigindo 16 GB por GPU -- o mesmo
  de rodar sozinho. O ZeRO-3 pede 0,25 GB. E' a diferenca entre "nao cabe" e
  "cabe folgado", e e' o que permite treinar modelos maiores que uma placa.

  O preco: mais COMUNICACAO. O ZeRO-3 precisa buscar os pesos de outros processos
  durante o forward e o backward. Troca-se memoria por banda de rede -- e por isso
  o estagio certo depende do seu gargalo.""")


# ---------------------------------------------------------------------------
# Demonstracao real: ZeRO-1 via ZeroRedundancyOptimizer.
# ---------------------------------------------------------------------------
class Modelo(nn.Module):
    def __init__(self, dim=512, camadas=4):
        super().__init__()
        camadas_lista = []
        for _ in range(camadas):
            camadas_lista += [nn.Linear(dim, dim), nn.GELU()]
        self.rede = nn.Sequential(*camadas_lista, nn.Linear(dim, 32))

    def forward(self, x):
        return self.rede(x)


def estado_do_otimizador_em_bytes(opt):
    """Quantos bytes de estado (m, v) este processo esta' realmente guardando."""
    total = 0
    for estado in opt.state.values():
        for v in estado.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total


def worker(rank, world_size):
    iniciar(rank, world_size, verbose=True)
    torch.manual_seed(1337)

    x = torch.randn(64, 512)
    alvo = torch.randint(0, 32, (64,))

    # --- AdamW comum: cada processo guarda o estado INTEIRO ---
    m1 = Modelo()
    opt_normal = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    F.cross_entropy(m1(x), alvo).backward()
    opt_normal.step()
    bytes_normal = estado_do_otimizador_em_bytes(opt_normal)

    # --- ZeRO-1: o estado e' fatiado entre os processos ---
    torch.manual_seed(1337)
    m2 = Modelo()
    opt_zero = ZeroRedundancyOptimizer(
        m2.parameters(), optimizer_class=torch.optim.AdamW, lr=1e-3
    )
    F.cross_entropy(m2(x), alvo).backward()
    opt_zero.step()
    bytes_zero = estado_do_otimizador_em_bytes(opt_zero.optim)

    nparams = sum(p.nelement() for p in m1.parameters())
    apenas_no_rank0(rank, f"\n=== ZeRO-1 medido ({world_size} processos) ===")
    apenas_no_rank0(rank, f"  modelo: {nparams} parametros\n")
    print(f"  [rank {rank}] estado do otimizador: AdamW normal "
          f"{bytes_normal/1e6:7.2f} MB | ZeRO-1 {bytes_zero/1e6:7.2f} MB "
          f"({bytes_normal/max(bytes_zero,1):.1f}x menor)", flush=True)

    dist.barrier()
    apenas_no_rank0(rank, f"""
  Cada processo guarda cerca de 1/{world_size} do estado -- que e' exatamente a
  promessa do ZeRO-1. Os pesos e os gradientes continuam duplicados (isso e' o
  que os estagios 2 e 3 atacam).

  Na hora do `step`, cada processo atualiza a sua fatia e depois um all-gather
  redistribui os pesos atualizados. Custo extra de comunicacao, em troca de
  memoria.""")

    encerrar()


if __name__ == "__main__":
    tabela_memoria()
    world_size = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    limpar_rendezvous()
    print(f"\niniciando {world_size} processos para a demonstracao...")
    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nconcluido.")
