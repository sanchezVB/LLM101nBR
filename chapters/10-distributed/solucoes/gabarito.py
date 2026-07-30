"""
Gabarito executavel do Capitulo 10 — treino distribuido.

Roda E2, E3, E5 e E6 (o E4 ja' tem solucao propria). Tudo na CPU, com
multiplos processos -- nao precisa de GPU.

O E3 provoca um DEADLOCK de proposito, e por isso roda em subprocesso com
timeout: um gabarito que trava para sempre nao serve para ninguem.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

AQUI = Path(__file__).resolve().parent
CAP = AQUI.parent
sys.path.insert(0, str(CAP))

from dist_utils import iniciar, encerrar, limpar_rendezvous, apenas_no_rank0


# ===========================================================================
# E2 — quanto o all-reduce custa conforme cresce o numero de processos
# ===========================================================================
def worker_e2(rank, world_size, retorno):
    iniciar(rank, world_size)
    tempos = {}
    for numel in (10_000, 1_000_000, 10_000_000):
        x = torch.randn(numel)
        for _ in range(3):
            dist.all_reduce(x)          # aquecimento
        # MELHOR de varias rodadas, nao a media -- a licao do E2 do Capitulo 08.
        # A media inclui interferencia de outros processos; o minimo se aproxima
        # do custo real, porque interferencia so' sabe ATRASAR.
        melhor = float("inf")
        reps = 10 if numel <= 1_000_000 else 5
        for _ in range(3):
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                dist.all_reduce(x, op=dist.ReduceOp.SUM)
            melhor = min(melhor, (time.perf_counter() - t0) / reps)
        tempos[numel] = melhor
    if rank == 0:
        retorno.update(tempos)
    encerrar()


# ===========================================================================
# E5 — batch efetivo e learning rate
# ===========================================================================
class Modelinho(nn.Module):
    def __init__(self, vocab, block=8, ne=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, ne)
        self.rede = nn.Sequential(nn.Linear(block * ne, 256), nn.GELU(),
                                  nn.Linear(256, vocab))

    def forward(self, idx):
        return self.rede(self.emb(idx).view(idx.shape[0], -1))


def carregar_dados(block=8):
    palavras = (CAP / "names.txt").read_text(encoding="utf-8").split()[:20000]
    chars = sorted(set("".join(palavras)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    X, Y = [], []
    for w in palavras:
        ctx = [0] * block
        for ch in w + ".":
            ix = stoi[ch]
            X.append(ctx)
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y), len(stoi)


def worker_e5(rank, world_size, lr, retorno):
    iniciar(rank, world_size)
    torch.manual_seed(1337)
    X, Y, V = carregar_dados()
    ddp = DDP(Modelinho(V))
    opt = torch.optim.AdamW(ddp.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1337 + rank)
    POR_PROCESSO = 64
    for _ in range(300):
        ix = torch.randint(0, X.shape[0], (POR_PROCESSO,), generator=g)
        loss = F.cross_entropy(ddp(X[ix]), Y[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    # loss GLOBAL (a licao do E4): all_reduce antes de reportar
    with torch.no_grad():
        aval = F.cross_entropy(ddp(X[:8192]), Y[:8192])
    dist.all_reduce(aval, op=dist.ReduceOp.SUM)
    if rank == 0:
        retorno["loss"] = (aval / world_size).item()
    encerrar()


# ===========================================================================
# E6 — ZeRO em varios tamanhos e com otimizador sem estado
# ===========================================================================
def worker_e6(rank, world_size, usar_sgd, retorno):
    iniciar(rank, world_size)
    torch.manual_seed(1337)
    m = nn.Sequential(*[c for _ in range(4)
                        for c in (nn.Linear(512, 512), nn.GELU())], nn.Linear(512, 32))
    x, alvo = torch.randn(64, 512), torch.randint(0, 32, (64,))

    classe = torch.optim.SGD if usar_sgd else torch.optim.AdamW
    kwargs = dict(lr=1e-3)
    normal = classe(m.parameters(), **kwargs)
    F.cross_entropy(m(x), alvo).backward()
    normal.step()
    bytes_normal = sum(v.numel() * v.element_size()
                       for e in normal.state.values() for v in e.values()
                       if torch.is_tensor(v))

    torch.manual_seed(1337)
    m2 = nn.Sequential(*[c for _ in range(4)
                         for c in (nn.Linear(512, 512), nn.GELU())], nn.Linear(512, 32))
    z = ZeroRedundancyOptimizer(m2.parameters(), optimizer_class=classe, **kwargs)
    F.cross_entropy(m2(x), alvo).backward()
    z.step()
    bytes_zero = sum(v.numel() * v.element_size()
                     for e in z.optim.state.values() for v in e.values()
                     if torch.is_tensor(v))

    if rank == 0:
        retorno["normal"] = bytes_normal
        retorno["zero"] = bytes_zero
    encerrar()


def rodar(fn, world_size, *args):
    """Roda uma funcao em N processos e devolve o que o rank 0 escreveu."""
    limpar_rendezvous()
    with mp.Manager() as ger:
        d = ger.dict()
        mp.spawn(fn, args=(world_size, *args, d), nprocs=world_size, join=True)
        return dict(d)


# ===========================================================================
if __name__ == "__main__":
    print("=" * 74)
    print("E2 — escalando o numero de processos")
    print("=" * 74)
    print(f"  {'processos':>10s} {'10 mil':>12s} {'1 milhao':>12s} {'10 milhoes':>12s}")
    for ws in (2, 4, 8):
        r = rodar(worker_e2, ws)
        print(f"  {ws:>10d} " + " ".join(f"{r.get(n, 0)*1000:>11.2f}ms"
                                         for n in (10_000, 1_000_000, 10_000_000)),
              flush=True)
    print(f"\n  nucleos de CPU nesta maquina: {os.cpu_count()}")
    print("""
  Respostas:
  1. NAO HA' UMA RESPOSTA SO' -- ela depende do TAMANHO do tensor, e essa e' a
     descoberta do exercicio. Olhe as tres colunas separadamente:

       10 mil elementos : o tempo CRESCE ~10x de 2 para 8 processos
       10 milhoes       : o tempo CAI quase 3x de 2 para 4 processos

     Sim, CAI. Com mais processos, o all-reduce do tensor grande fica mais
     RAPIDO. Eu tinha escrito que ele sempre cresce; a medicao (repetida em duas
     passadas, com variacao de so' 1.02x a 1.13x) mostrou o contrario.

     A explicacao e' a mesma dicotomia do Capitulo 08, LATENCIA vs VAZAO:

       - tensor pequeno: quase nao ha' dado para transferir, entao o custo e'
         a LATENCIA dos N-1 saltos do anel. Mais processos = mais saltos = pior.

       - tensor grande: o anel divide o trabalho em pedacos de tamanho/N. Com
         N=2 cada processo lida com 20 MB por etapa; com N=4, com 10 MB. Pedacos
         menores cabem melhor na cache, e a SOMA (que tambem e' trabalho real,
         nao so' transferencia) se espalha por mais nucleos. Mais processos =
         mais paralelismo = melhor.

     Onde os dois efeitos se cruzam depende da maquina. Nesta, com 12 nucleos, o
     melhor ponto para 10 milhoes foi N=4; em N=8 o tempo piora de novo, porque
     ai' os processos passam a disputar CPU (item 3).

  2. No esquema ingenuo (todos -> rank 0 -> todos), o rank 0 receberia
     (N-1) x tamanho: o tempo cresceria LINEARMENTE com N, e ele viraria um
     gargalo cada vez pior. Nenhum ganho como o de N=4 acima seria possivel --
     o rank 0 e' um so', e nao paraleliza.

  3. Acima do numero de nucleos os processos passam a DISPUTAR CPU em vez de
     somar -- e' o que faz N=8 perder para N=4 nas colunas grandes. O tempo
     piora por contencao, nao por comunicacao. Numa maquina so', 'mais
     processos' tem um teto fisico, e ele fica perto do numero de nucleos.

     ATENCAO ao generalizar: tudo isto foi medido em UMA maquina, com o gloo
     em CPU. Num cluster de verdade, com GPUs e rede dedicada, o equilibrio
     entre latencia e vazao e' outro -- mas a PERGUNTA a fazer e' a mesma.""")

    # -----------------------------------------------------------------------
    print("=" * 74)
    print("E3 — provocando o deadlock (com timeout, para nao travar o gabarito)")
    print("=" * 74)
    print(f"  {'tensor':>12s} {'alternancia':>12s} {'resultado':>28s}")
    for numel, rotulo in ((1, "1 elemento"), (100, "pequeno"), (5_000_000, "grande")):
        for alternar in (1, 0):
            limpar_rendezvous()
            procs = [
                subprocess.Popen([sys.executable, str(AQUI / "worker_deadlock.py"),
                                  str(r), "2", str(numel), str(alternar)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for r in range(2)
            ]
            travou = False
            try:
                for p in procs:
                    p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                travou = True
                for p in procs:
                    p.kill()
            res = "TRAVOU (deadlock)" if travou else "terminou"
            print(f"  {rotulo:>12s} {'sim' if alternar else 'NAO':>12s} {res:>28s}",
                  flush=True)
    print("""
  Respostas:
  1 e 2. TRAVA SEMPRE, em qualquer tamanho -- ate' com UM UNICO elemento.

     ATENCAO, porque aqui a resposta 'de livro' esta' ERRADA para este backend.
     A explicacao classica do deadlock em send/recv e' o BUFFER DE SOCKET: o
     sistema operacional aceita mensagens pequenas e devolve o controle na hora,
     entao so' mensagens grandes travariam. Essa explicacao vale para sockets
     crus e para MPI -- e NAO vale para o gloo.

     Ao escrever este gabarito eu previ que o tensor pequeno passaria. A medicao
     mostrou o contrario, e um teste com numel=1 confirmou: o dist.send do gloo
     so' retorna quando o recv correspondente foi postado do outro lado, custe o
     que custar. Nao ha' tamanho pequeno o bastante para escapar.

  3. Duas licoes, e a segunda e' a que importa.

     A primeira e' sobre o gloo: essa escolha de projeto e' BOA. Um deadlock que
     acontece SEMPRE e' muito melhor que um que so' aparece em producao com
     tensores grandes. O gloo trocou 'as vezes funciona' por 'nunca funciona' --
     e falhar de forma deterministica e' um recurso, nao um defeito.

     A segunda e' sobre voce: eu tinha um modelo mental correto (o do buffer de
     socket) e o apliquei a uma implementacao que nao o segue. O modelo estava
     certo EM GERAL e errado NAQUELE CASO. E' o mesmo padrao que aparece no E2
     do Capitulo 4 e no E3 do Capitulo 5 -- a unica defesa e' medir.

     E note o modo de falha: nao ha' excecao, nao ha' mensagem. O programa
     simplesmente para -- exatamente como aconteceu no desenvolvimento deste
     capitulo, com o hostname que resolvia para um IP publico (Secao 8).""")

    # -----------------------------------------------------------------------
    print("=" * 74)
    print("E5 — batch efetivo e learning rate (300 passos, batch local fixo = 64)")
    print("=" * 74)
    print(f"  {'processos':>10s} {'batch efetivo':>14s} {'lr':>10s} {'loss global':>12s}")
    base_lr = 1e-3
    for ws in (1, 2, 4):
        r = rodar(worker_e5, ws, base_lr)
        print(f"  {ws:>10d} {64*ws:>14d} {base_lr:>10.0e} {r.get('loss', 0):>12.4f}",
              flush=True)
    print()
    for ws in (2, 4):
        lr = base_lr * (ws ** 0.5)
        r = rodar(worker_e5, ws, lr)
        print(f"  {ws:>10d} {64*ws:>14d} {lr:>10.1e} {r.get('loss', 0):>12.4f}  "
              f"(lr x raiz de {ws})", flush=True)
    print("""
  Respostas:
  1. Com o batch LOCAL fixo, mais processos = batch efetivo maior. Como o
     numero de PASSOS e' o mesmo, o modelo ve' mais dados -- mas da' o mesmo
     numero de atualizacoes.
  2. Ajustar a lr por raiz de world_size costuma recuperar parte da diferenca:
     gradiente menos ruidoso tolera (e pede) passo maior.
  3. Por isso 'mais GPUs' pode deixar o treino mais lento EM NUMERO DE PASSOS:
     voce processa mais exemplos por passo, mas se nao aumentar a lr esta'
     andando com o mesmo tamanho de passo -- e precisa de mais passos para
     chegar ao mesmo lugar.""")

    # -----------------------------------------------------------------------
    print("=" * 74)
    print("E6 — ZeRO em varios tamanhos, e com otimizador SEM estado")
    print("=" * 74)
    print(f"  {'otimizador':>12s} {'processos':>10s} {'normal (MB)':>12s} "
          f"{'ZeRO-1 (MB)':>12s} {'reducao':>9s}")
    for usar_sgd in (False, True):
        for ws in (2, 4):
            r = rodar(worker_e6, ws, usar_sgd)
            nm, zr = r.get("normal", 0) / 1e6, r.get("zero", 0) / 1e6
            razao = f"{nm/zr:.1f}x" if zr else "n/d"
            print(f"  {'SGD' if usar_sgd else 'AdamW':>12s} {ws:>10d} {nm:>12.2f} "
                  f"{zr:>12.2f} {razao:>9s}", flush=True)
    print("""
  Respostas:
  1. Sim: o estado por processo cai para ~1/N, como a apostila mede com 4
     processos.
  2. Com SGD SEM momentum o otimizador nao guarda estado NENHUM -- entao nao ha'
     o que fatiar, e o ZeRO-1 nao tem efeito. O ZeRO-1 economiza exatamente
     aquilo que o otimizador guarda; se ele nao guarda nada, a economia e' zero.
     (Os estagios 2 e 3, que fatiam gradientes e pesos, continuariam ajudando.)
  3. Com bf16 nos pesos (Capitulo 9), a conta da apostila muda: pesos e
     gradientes passam de 4 para 2 bytes por parametro, mas o estado do AdamW
     costuma ficar em fp32 por precisao. Com 8 processos:
        DDP    : 2 + 2 + 8      = 12 GB/GPU
        ZeRO-1 : 2 + 2 + 8/8    =  5 GB/GPU
     O peso relativo do estado do otimizador AUMENTA quando os pesos encolhem --
     ou seja, precisao reduzida torna o ZeRO ainda mais valioso.""")

    print("=" * 74)
    print("E7 — quando distribuir NAO compensa")
    print("=" * 74)
    print("""  Junte os numeros medidos:

    tempo de calculo de um passo (modelo pequeno, CPU, Capitulo 8): ~12.8 ms
    all-reduce de ~150 mil parametros (0.6 MB): veja a coluna '10 mil' do E2,
      escalada -- fica na casa de alguns milissegundos

  Respostas:
  1 e 2. Para o modelo pequeno deste curso, comunicacao e calculo ficam na MESMA
     ordem de grandeza. Distribuir em 4 processos dividiria o calculo por 4 mas
     acrescentaria comunicacao a cada passo -- o ganho liquido seria pequeno ou
     negativo.
  3. A regra geral: distribuir compensa quando

        tempo_de_calculo_por_passo  >>  tempo_de_comunicacao_por_passo

     Como o calculo cresce com o TAMANHO DO MODELO e com o BATCH, enquanto a
     comunicacao cresce so' com o tamanho do MODELO, a razao melhora com batches
     maiores. E' por isso que treino distribuido de verdade usa batches enormes:
     nao e' so' por velocidade, e' para que a comunicacao valha a pena.""")
