"""
preparar_dados.py — baixa o MNIST e grava em .npy.

Por que MNIST e nao um dataset de imagem-e-texto: o Capitulo 11 gastou um
capitulo inteiro montando um corpus de 1,6 MB. Um dataset pareado de
imagem-legenda util comeca em dezenas de gigabytes -- fora do que este curso
consegue rodar honestamente.

MNIST resolve o que interessa aqui: sao imagens de verdade, pequenas
(28x28, tons de cinza), universalmente conhecidas, e cabem em 11 MB. O que o
capitulo demonstra -- pixels virando tokens discretos que entram no MESMO
Transformer -- nao depende de as imagens serem complexas.

Run (a partir da pasta do capitulo):
    python preparar_dados.py
"""

import gzip
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
BRUTOS = AQUI / "_brutos"
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
ARQUIVOS = {
    "treino_x": "train-images-idx3-ubyte.gz",
    "treino_y": "train-labels-idx1-ubyte.gz",
    "val_x": "t10k-images-idx3-ubyte.gz",
    "val_y": "t10k-labels-idx1-ubyte.gz",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def baixar(nome):
    """Com cache em disco, e I/O binario -- a licao do Capitulo 11."""
    BRUTOS.mkdir(exist_ok=True)
    destino = BRUTOS / nome
    if destino.exists():
        return destino.read_bytes()
    print(f"  baixando {nome}...", flush=True)
    req = urllib.request.Request(BASE + nome,
                                 headers={"User-Agent": "llm101n-br (material didatico)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        bruto = r.read()
    destino.write_bytes(bruto)
    return bruto


def ler_idx(bruto):
    """Formato IDX: magic, dimensoes, e os bytes crus."""
    dados = gzip.decompress(bruto)
    magic, = struct.unpack(">I", dados[:4])
    n_dims = magic & 0xFF
    dims = struct.unpack(">" + "I" * n_dims, dados[4:4 + 4 * n_dims])
    corpo = np.frombuffer(dados[4 + 4 * n_dims:], dtype=np.uint8)
    return corpo.reshape(dims)


if __name__ == "__main__":
    print("=" * 74)
    print("Preparando o MNIST")
    print("=" * 74)
    saida = {}
    for chave, nome in ARQUIVOS.items():
        saida[chave] = ler_idx(baixar(nome))
        print(f"  {chave:>9s}: {saida[chave].shape} {saida[chave].dtype}")

    np.savez_compressed(AQUI / "mnist.npz", **saida)
    tam = (AQUI / "mnist.npz").stat().st_size
    print(f"\n  gravado em mnist.npz ({tam/1e6:.1f} MB)")

    x = saida["treino_x"]
    print(f"""
  O que temos: {x.shape[0]:,} imagens de {x.shape[1]}x{x.shape[2]} em tons de cinza.

  E ja' da' para ver o problema que o capitulo resolve. Cada imagem tem
  {x.shape[1]*x.shape[2]} pixels, cada pixel e' um numero de 0 a 255. Para um
  Transformer engolir isso como texto, seriam {x.shape[1]*x.shape[2]} 'tokens'
  por imagem, de um vocabulario de 256 -- uma sequencia mais longa que o
  contexto inteiro do modelo do Capitulo 11 ({x.shape[1]*x.shape[2]} contra 128),
  para UMA imagem.

  Alem de longo, e' desperdicio: pixels vizinhos sao quase sempre parecidos, e
  gastar um token em cada um e' como tokenizar texto por byte -- que e'
  exatamente o que o Capitulo 6 mostrou ser ruim.

  O VQ-VAE e' o BPE das imagens.""")
