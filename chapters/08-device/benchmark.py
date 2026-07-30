"""
benchmark.py — quando a GPU ganha da CPU, e por que.

Tres medicoes:
  1. matmul de tamanhos crescentes -> onde esta' o PONTO DE VIRADA
  2. custo de TRANSFERIR dados entre CPU e GPU
  3. o efeito do tamanho do BATCH com o mesmo total de trabalho

A conclusao que os numeros vao mostrar: a GPU nao e' "mais rapida", ela e' mais
PARALELA. Para trabalho pequeno ela perde, porque tem um custo fixo por operacao
(lancar o kernel, mover dados) que a CPU nao tem.

Run:
    python benchmark.py
"""

import time

import torch

from device import pegar_device, sincronizar

dev, rotulo = pegar_device()
cpu = torch.device("cpu")


def drenar(saida, device):
    """Espera a GPU terminar, LENDO o resultado de verdade.

    Cuidado sutil e importante: nao basta criar outro tensor e copia-lo para a
    CPU -- a fila da GPU pode nao garantir que a operacao que nos interessa ja'
    terminou. A forma confiavel e' tocar no PROPRIO resultado, o que obriga o
    dispositivo a produzi-lo. Medir GPU errado e' facil: a primeira versao deste
    arquivo dava speedups nao-monotonicos (0.62x em 512 e 18x em 1024) porque
    media o enfileiramento, e nao a execucao.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type != "cpu" and torch.is_tensor(saida):
        saida.cpu()


def cronometrar(fn, device, repeticoes=20, aquecimentos=5, rodadas=3):
    """Mede o tempo de fn. fn DEVE devolver o tensor de resultado.

    Tomamos o MINIMO de varias rodadas, e nao a media. Essa e' a pratica padrao
    em microbenchmark: o tempo real da operacao e' um piso, e qualquer medicao
    acima dele foi contaminada por outra coisa disputando a maquina (o sistema
    operacional, outro processo, o proprio Python). A media incorpora esse ruido;
    o minimo o descarta.
    """
    for _ in range(aquecimentos):      # aquecimento: aloca buffers, compila kernels
        saida = fn()
    drenar(saida, device)

    melhor = float("inf")
    for _ in range(rodadas):
        t0 = time.perf_counter()
        for _ in range(repeticoes):
            saida = fn()
        drenar(saida, device)
        melhor = min(melhor, (time.perf_counter() - t0) / repeticoes)
    return melhor


# ---------------------------------------------------------------------------
# 1. Matmul: onde a GPU passa a ganhar?
# ---------------------------------------------------------------------------
print(f"\n=== 1. matmul: CPU vs {rotulo} ===")
print(f"{'tamanho':>8s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'speedup':>9s} {'GFLOP/s CPU':>12s} {'GFLOP/s GPU':>12s}")

virada = None
for n in (128, 256, 512, 1024, 2048, 4096):
    a, b = torch.randn(n, n), torch.randn(n, n)
    t_cpu = cronometrar(lambda: a @ b, cpu, repeticoes=(20 if n <= 1024 else 5))

    ag, bg = a.to(dev), b.to(dev)
    t_gpu = cronometrar(lambda: ag @ bg, dev, repeticoes=(20 if n <= 1024 else 5))

    # uma matmul n x n faz 2*n^3 operacoes de ponto flutuante
    flops = 2 * n ** 3
    sp = t_cpu / t_gpu
    if virada is None and sp > 1:
        virada = n
    print(f"{n:8d} {t_cpu*1000:10.2f} {t_gpu*1000:10.2f} {sp:8.2f}x "
          f"{flops/t_cpu/1e9:12.1f} {flops/t_gpu/1e9:12.1f}")

print(f"\n  ponto de virada: a GPU passa a ganhar a partir de ~{virada}x{virada}")
print("  Abaixo disso, o custo fixo de acionar a GPU e' maior que o trabalho.")

# ---------------------------------------------------------------------------
# 2. O custo de transferir dados.
# ---------------------------------------------------------------------------
print(f"\n=== 2. custo de transferencia CPU <-> {rotulo} ===")
print(f"{'tamanho':>12s} {'MB':>7s} {'CPU->GPU':>10s} {'GPU->CPU':>10s} {'GB/s':>7s}")
for n in (256, 1024, 2048):
    x = torch.randn(n, n)
    mb = x.numel() * 4 / 1e6
    t_ida = cronometrar(lambda: x.to(dev), dev)
    xg = x.to(dev)
    t_volta = cronometrar(lambda: xg.cpu(), dev)
    print(f"{n:6d}x{n:<5d} {mb:7.1f} {t_ida*1000:9.2f}ms {t_volta*1000:9.2f}ms "
          f"{mb/1e3/t_ida:7.1f}")

print("""
  Licao pratica: transferir custa tempo. O padrao correto e' mover os dados
  UMA VEZ para a GPU e deixa-los la', em vez de ficar mandando de ida e volta a
  cada passo. Um `.cpu()` ou um `.item()` dentro do laco de treino forca uma
  transferencia E uma sincronizacao -- e vira gargalo.""")

# ---------------------------------------------------------------------------
# 3. Mesmo trabalho total, batch diferente.
#    A GPU quer trabalho GRANDE de uma vez. Fazer 64 matmuls pequenas e' muito
#    pior que fazer 1 matmul 64x maior -- mesmo somando o mesmo total de FLOPs.
# ---------------------------------------------------------------------------
print(f"\n=== 3. mesmo trabalho, batches diferentes ({rotulo}) ===")
DIM = 256
TOTAL = 4096          # numero total de vetores a processar
W = torch.randn(DIM, DIM).to(dev)
print(f"{'batch':>7s} {'chamadas':>9s} {'tempo (ms)':>11s}")
for batch in (1, 16, 64, 256, 1024, 4096):
    chamadas = TOTAL // batch
    x = torch.randn(batch, DIM).to(dev)

    def op():
        for _ in range(chamadas):
            r = x @ W
        return r

    t = cronometrar(op, dev, repeticoes=5, aquecimentos=2)
    print(f"{batch:7d} {chamadas:9d} {t*1000:11.2f}")

print("""
  O total de multiplicacoes e' IGUAL em todas as linhas -- so' muda como ele e'
  entregue. Batches grandes ganham porque a GPU tem milhares de nucleos esperando
  trabalho: entregar pouco de cada vez deixa a maioria deles ociosa, e ainda paga
  o custo fixo de lancamento em cada chamada.""")
