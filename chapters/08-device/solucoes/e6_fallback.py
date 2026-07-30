"""
Solucao do Exercicio E6 — operacoes que caem de volta para a CPU.

O aviso que aparece ao treinar no DirectML:

    UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported
    on the DML backend and will fall back to run on the CPU.

...vem do AdamW. Ele usa `lerp` (interpolacao linear) para atualizar as medias
moveis, e o DirectML nao implementa essa operacao. O PyTorch entao executa essa
parte na CPU, atravessando a fronteira CPU<->GPU a cada passo.

Este script mede o custo disso: compara AdamW (com fallback) contra SGD (sem
fallback), no mesmo modelo e dispositivo.

Run (a partir da pasta do capitulo):
    python solucoes/e6_fallback.py
"""

import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from device import pegar_device, sincronizar
from train_device import Modelo, Xtr, Ytr, V   # reusa o modelo e os dados

dev, rotulo = pegar_device()
cpu = torch.device("cpu")
PASSOS = 60

# Configuracao do modelo "medio", onde a GPU ja' ganha
N_EMBD, N_HEAD, N_LAYER, BATCH = 256, 8, 4, 256


def medir(device, fabrica_opt, nome_opt):
    torch.manual_seed(1337)
    m = Modelo(N_EMBD, N_HEAD, N_LAYER).to(device)
    opt = fabrica_opt(m.parameters())
    Xd, Yd = Xtr.to(device), Ytr.to(device)
    g = torch.Generator().manual_seed(1337)

    avisos = []

    def um_passo():
        ix = torch.randint(0, Xtr.shape[0], (BATCH,), generator=g).to(device)
        loss = F.cross_entropy(m(Xd[ix]), Yd[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # captura os avisos de fallback do primeiro passo
    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        um_passo()
        for w in capturados:
            msg = str(w.message)
            if "fall back" in msg or "not currently supported" in msg:
                # extrai so' o nome do operador
                if "'" in msg:
                    avisos.append(msg.split("'")[1])

    for _ in range(4):
        um_passo()
    sincronizar(device)

    t0 = time.perf_counter()
    for _ in range(PASSOS):
        um_passo()
    sincronizar(device)
    return (time.perf_counter() - t0) / PASSOS * 1000, sorted(set(avisos))


OTIMIZADORES = [
    ("AdamW", lambda ps: torch.optim.AdamW(ps, lr=1e-3)),
    ("SGD+momentum", lambda ps: torch.optim.SGD(ps, lr=1e-3, momentum=0.9)),
    ("SGD puro", lambda ps: torch.optim.SGD(ps, lr=1e-3)),
]

print(f"modelo medio: n_embd={N_EMBD}, {N_LAYER} blocos, batch={BATCH}")
print(f"dispositivo: {rotulo}\n")
print(f"{'otimizador':<16s} {'CPU (ms)':>10s} {'GPU (ms)':>10s} {'speedup':>9s}  operadores em fallback")
print("-" * 78)

for nome, fabrica in OTIMIZADORES:
    t_cpu, _ = medir(cpu, fabrica, nome)
    t_gpu, fb = medir(dev, fabrica, nome)
    lista = ", ".join(fb) if fb else "(nenhum)"
    print(f"{nome:<16s} {t_cpu:10.1f} {t_gpu:10.1f} {t_cpu/t_gpu:8.2f}x  {lista}", flush=True)

print("""
Como interpretar:

  Se o SGD nao tem fallback e mostra speedup MAIOR que o AdamW, a diferenca e' o
  preco de atravessar a fronteira CPU<->GPU a cada passo -- e isso significa que
  os numeros da apostila (medidos com AdamW) sao um PISO, nao um teto.

  Em CUDA esse problema nao existe: todas as operacoes do AdamW sao implementadas.
  A licao geral: um backend menos maduro pode ter lacunas em lugares inesperados,
  e elas nao aparecem como ERRO -- aparecem como lentidao silenciosa. Sempre rode
  uma vez com os avisos ligados:

      python -W always::UserWarning seu_script.py
""")
