"""
Gabarito executavel do Capitulo 08 — dispositivos.

Roda E2, E3, E4, E5 e E7 (o E6 ja' tem solucao propria).

IMPORTANTE: para exercitar a GPU, rode com o Python que tem torch-directml:
    C:\\dml312\\Scripts\\python.exe solucoes/gabarito.py
Sem GPU o script roda mesmo assim e informa o que nao pode ser medido.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
import time
from pathlib import Path

import torch

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))

from device import pegar_device

dev, rotulo = pegar_device()
cpu = torch.device("cpu")
TEM_GPU = dev.type != "cpu"


def drenar(saida, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type != "cpu" and torch.is_tensor(saida):
        saida.cpu()


def cronometrar(fn, device, repeticoes=20, rodadas=3):
    for _ in range(5):
        s = fn()
    drenar(s, device)
    melhor = float("inf")
    for _ in range(rodadas):
        t0 = time.perf_counter()
        for _ in range(repeticoes):
            s = fn()
        drenar(s, device)
        melhor = min(melhor, (time.perf_counter() - t0) / repeticoes)
    return melhor


# ===========================================================================
print("=" * 74)
print("E2 — encontre o SEU ponto de virada")
print("=" * 74)
if not TEM_GPU:
    print("  (sem GPU neste ambiente -- veja gabarito.md para os numeros medidos)")
else:
    print(f"  dispositivo: {rotulo}")
    print(f"  {'tamanho':>8s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'speedup':>9s} {'GFLOP/s GPU':>12s}")
    virada = None
    for n in (128, 256, 512, 1024, 2048):
        # DUAS matrizes distintas, igual ao benchmark.py da apostila. Usar a
        # mesma matriz duas vezes (a @ a) melhora a localidade de cache na CPU e
        # da' numeros diferentes -- detalhe pequeno que muda a conclusao nos
        # tamanhos pequenos.
        a, b = torch.randn(n, n), torch.randn(n, n)
        t_cpu = cronometrar(lambda: a @ b, cpu, repeticoes=(20 if n <= 1024 else 5))
        ag, bg = a.to(dev), b.to(dev)
        t_gpu = cronometrar(lambda: ag @ bg, dev, repeticoes=(20 if n <= 1024 else 5))
        sp = t_cpu / t_gpu
        if virada is None and sp > 1:
            virada = n
        print(f"  {n:>8d} {t_cpu*1000:>10.2f} {t_gpu*1000:>10.2f} {sp:>8.2f}x "
              f"{2*n**3/t_gpu/1e9:>12.0f}")
    print(f"\n  ponto de virada: ~{virada}x{virada}")
print("""
  Respostas:
  1 e 2. A GPU passa a ganhar a partir de matrizes de algumas centenas, e o
     speedup CRESCE com o tamanho ate' estabilizar.
  3. O pico de GFLOP/s medido fica bem abaixo da especificacao da placa -- e' o
     normal. A especificacao supoe uso perfeito das unidades, precisao reduzida
     e nenhum gargalo de memoria. Os capitulos 9 e 10 atacam essa diferenca.

  SOBRE OS NUMEROS QUE VOCE VAI VER: eles NAO vao bater com os da apostila, e
  isso e' esperado. Ao preparar este gabarito eu rodei a medicao tres vezes e
  obtive speedups de 1.08x, 1.52x e 2.09x para 128x128 -- a variacao vinha de
  outros processos disputando a CPU (o proprio gabarito de outro capitulo
  estava treinando ao fundo).

  Medicao de tempo e' sensivel a: carga da maquina, estado termico, frequencia
  do processador e ate' se voce multiplica 'a @ b' ou 'a @ a' (a segunda tem
  melhor localidade de cache).

  O que E' estavel, e o que voce deve verificar:
    - a FORMA da curva: speedup pequeno nos tamanhos pequenos, crescendo ate'
      estabilizar
    - a ordem de grandeza do ganho maximo (uma a duas dezenas de vezes)
    - o fato de que a CPU satura e a GPU escala

  Se voce quiser numeros comparaveis, feche tudo e rode com a maquina ociosa.""")

# ===========================================================================
print("=" * 74)
print("E3 — a armadilha do .item()")
print("=" * 74)
import torch.nn as nn
import torch.nn.functional as F

modelo = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256)).to(dev)
x = torch.randn(256, 256).to(dev)
opt = torch.optim.SGD(modelo.parameters(), lr=1e-3)


def passo(ler_loss):
    saida = modelo(x)
    loss = saida.square().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if ler_loss:
        return loss.item()          # <- forca transferencia E sincronizacao
    return loss


t_sem = cronometrar(lambda: passo(False), dev, repeticoes=20)
t_com = cronometrar(lambda: passo(True), dev, repeticoes=20)
print(f"  sem ler a loss  : {t_sem*1000:.2f} ms/passo")
print(f"  lendo .item()   : {t_com*1000:.2f} ms/passo  ({100*(t_com/t_sem-1):+.1f}%)")
print("""
  Respostas:
  1 e 2. Ler a loss a cada passo custa: o .item() faz DUAS coisas -- transfere
     um valor da GPU para a CPU e, para poder faze-lo, ESPERA a GPU terminar
     tudo o que estava na fila. A segunda e' a cara: ela destroi o
     paralelismo entre CPU e GPU.
  3. Registre a cada N passos (100, por exemplo), ou acumule os valores num
     tensor NA GPU e leia so' no fim. Voce raramente precisa da loss de todos os
     passos -- precisa da CURVA.""")

# ===========================================================================
print("=" * 74)
print("E4 — tamanho do batch")
print("=" * 74)
print(f"  {'batch':>7s} {'ms/passo':>10s} {'ms por exemplo':>16s}")
for bs in (32, 128, 512, 2048):
    xb = torch.randn(bs, 256).to(dev)

    def op():
        return modelo(xb)

    t = cronometrar(op, dev, repeticoes=10)
    print(f"  {bs:>7d} {t*1000:>10.3f} {t*1000/bs:>16.5f}")
print("""
  Respostas:
  1 e 2. O tempo por passo cresce MENOS que proporcionalmente ao batch, entao o
     tempo POR EXEMPLO cai. E' a mesma licao da Secao 4 da apostila: a GPU quer
     trabalho grande de uma vez.
  3. O limite e' a memoria da placa (voce vera' um erro de alocacao).
  4. CUIDADO com a conclusao: batch maior processa mais exemplos por segundo,
     mas cada passo continua sendo UM passo de gradiente. Se voce dobrar o batch
     sem ajustar a learning rate, o treino faz metade do progresso por exemplo
     visto -- releia o E5 do Capitulo 7.""")

# ===========================================================================
print("=" * 74)
print("E5 — meca errado de proposito")
print("=" * 74)
if not TEM_GPU:
    print("  (precisa de GPU)")
else:
    a = torch.randn(1024, 1024).to(dev)

    # medicao ERRADA: sem drenar
    for _ in range(5):
        a @ a
    t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    t_errado = (time.perf_counter() - t0) / 20

    t_certo = cronometrar(lambda: a @ a, dev, repeticoes=20)
    b = torch.randn(1024, 1024)
    t_cpu = cronometrar(lambda: b @ b, cpu, repeticoes=20)

    print(f"  GPU medida SEM drenar : {t_errado*1000:8.3f} ms -> "
          f"'speedup' {t_cpu/t_errado:7.1f}x")
    print(f"  GPU medida COM drenar : {t_certo*1000:8.3f} ms -> "
          f"speedup real {t_cpu/t_certo:7.1f}x")
    print(f"  a medicao errada exagera em {t_certo/t_errado:.0f}x")
print("""
  Respostas:
  1. Sem drenar, a GPU parece absurdamente rapida -- porque medimos o tempo de
     ENFILEIRAR a operacao, nao o de executa-la.
  2. A CPU nao e' afetada porque nela a execucao e' SINCRONA: quando a linha
     termina, a conta terminou.
  3. Detecta-se pelo formato da curva: speedups que nao crescem de forma
     monotonica com o tamanho, ou valores fisicamente implausiveis (centenas de
     vezes), sao sinal de medicao contaminada.""")

# ===========================================================================
print("=" * 74)
print("E7 — quando vale a pena mudar para GPU?")
print("=" * 74)
print("""  Junte os numeros da apostila (Secao 7):

    modelo pequeno (153 mil params) : 0.30x  -> a GPU PERDE
    modelo medio   (3.2 M params)   : 2.65x
    modelo grande  (18.9 M params)  : 6.82x

  Respostas:
  1. O ponto de equilibrio fica entre 153 mil e 3.2 milhoes de parametros --
     mais perto do limite inferior, ja' que 3.2M ja' da' 2.65x.
  2. O custo NAO e' so' de execucao: instalar um ambiente separado, descobrir
     operacoes nao suportadas (o lerp do E6) e depurar travamentos consomem
     horas. Para um treino de 6 minutos, nao compensa; para um de 6 horas,
     compensa muito.
  3. Para o modelo pequeno deste curso a recomendacao e' CPU -- e essa e' a
     resposta contra-intuitiva do capitulo. 'Usar GPU' nao e' automaticamente
     certo; e' uma decisao que depende de escala e se mede.""")
