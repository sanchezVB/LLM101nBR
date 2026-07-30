"""
dataset.py — como alimentar o modelo sem carregar tudo na memoria.

Nos capitulos anteriores o dataset cabia num tensor. Com texto de verdade isso
deixa de ser verdade rapido: o corpus do GPT-3 tem ~570 GB, e nenhuma maquina
carrega isso na RAM.

A solucao padrao (a mesma do nanoGPT) tem tres partes:

  1. os tokens vivem em disco, como uint16 puro
  2. abrimos o arquivo com np.memmap -- o sistema operacional traz do disco so'
     as paginas que forem realmente lidas
  3. cada exemplo do batch e' um PEDACO ALEATORIO contiguo desse fluxo

Note a mudanca em relacao aos capitulos 3-7: la', cada exemplo era uma janela
com preenchimento e havia UM alvo (o proximo caractere). Aqui o texto e' um fluxo
continuo, e cada posicao tem um alvo -- o modelo aprende com T previsoes por
exemplo, em vez de uma. E' assim que se treina de verdade, e e' bem mais eficiente.

Run:
    python dataset.py
"""

import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).parent


def carregar(split):
    """Abre train.bin ou val.bin como memmap (nao le' o arquivo agora)."""
    caminho = AQUI / f"{split}.bin"
    if not caminho.exists():
        raise SystemExit(f"{caminho.name} nao existe. Rode primeiro: python prepare_data.py")
    # mode="r": somente leitura. O array se comporta como um vetor normal, mas as
    # paginas so' sao lidas do disco quando tocadas.
    return np.memmap(caminho, dtype=np.uint16, mode="r")


def pegar_batch(dados, batch_size, block_size, generator=None):
    """Sorteia `batch_size` pedacos contiguos de tamanho `block_size`.

    x = tokens [i : i+block]
    y = tokens [i+1 : i+block+1]   <- o alvo e' o proprio texto deslocado de 1

    Esse deslocamento e' toda a supervisao de que um LLM precisa: o texto ja' e'
    a resposta. Nao existe rotulo humano aqui -- e' por isso que se chama
    aprendizado AUTO-supervisionado, e por isso da' para treinar com a internet
    inteira sem ninguem anotar nada.
    """
    ix = torch.randint(len(dados) - block_size - 1, (batch_size,), generator=generator)
    # .astype(np.int64) porque o PyTorch indexa com int64; uint16 nao serve
    x = torch.stack([torch.from_numpy(dados[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(dados[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x, y


def carregar_tokenizador():
    import pickle
    with open(AQUI / "tokenizador.pkl", "rb") as f:
        d = pickle.load(f)
    return d["merges"], d["vocab"]


def decodificar(ids, vocab):
    bs = b"".join(vocab[int(i)] for i in ids)
    return bs.decode("utf-8", errors="replace")


# ===========================================================================
if __name__ == "__main__":
    print("=== 1. abrindo os dados ===")
    t0 = time.perf_counter()
    treino = carregar("treino")
    val = carregar("val")
    print(f"  memmap aberto em {(time.perf_counter()-t0)*1000:.1f} ms")
    print(f"  treino: {len(treino):>9,} tokens")
    print(f"  val   : {len(val):>9,} tokens")
    print(f"  vocabulario usado: 0..{int(max(treino[:100000].max(), val[:10000].max()))}")

    print("""
  Repare no tempo: abrir o memmap e' instantaneo porque NADA foi lido ainda.
  O custo aparece so' quando tocamos nos dados -- e mesmo assim, so' nas paginas
  tocadas. Um arquivo de 100 GB abre igualmente rapido.""")

    # -----------------------------------------------------------------------
    print("=== 2. um batch de exemplo ===")
    g = torch.Generator().manual_seed(1337)
    x, y = pegar_batch(treino, batch_size=4, block_size=64, generator=g)
    print(f"  x: {tuple(x.shape)}  y: {tuple(y.shape)}")

    _, vocab = carregar_tokenizador()
    print("\n  o primeiro exemplo do batch, em texto:")
    print(f"    entrada : {decodificar(x[0][:40], vocab)!r}")
    print(f"    alvo    : {decodificar(y[0][:40], vocab)!r}")
    print("\n  Note que o alvo e' a entrada DESLOCADA DE UM token. Prever o proximo")
    print("  token e' aprender com o proprio texto, sem rotulo nenhum.")

    # -----------------------------------------------------------------------
    print("\n=== 3. memmap vs carregar tudo ===")
    t0 = time.perf_counter()
    tudo = np.fromfile(AQUI / "treino.bin", dtype=np.uint16)
    t_tudo = time.perf_counter() - t0
    mb = tudo.nbytes / 1e6

    t0 = time.perf_counter()
    mm = np.memmap(AQUI / "treino.bin", dtype=np.uint16, mode="r")
    t_mm = time.perf_counter() - t0

    print(f"  np.fromfile (carrega tudo): {t_tudo*1000:7.1f} ms, {mb:.1f} MB na RAM")
    print(f"  np.memmap   (so' mapeia)  : {t_mm*1000:7.1f} ms, ~0 MB na RAM")
    print(f"""
  Com {mb:.0f} MB a diferenca e' irrelevante -- caberia na memoria de qualquer jeito.
  A questao e' o que acontece quando o corpus tem 100 GB: o fromfile simplesmente
  nao roda, e o memmap continua funcionando igual. Escrever assim desde o inicio
  custa nada e evita reescrever o pipeline depois.""")

    # -----------------------------------------------------------------------
    print("=== 4. quantos tokens por segundo? ===")
    for bs, blk in ((32, 64), (64, 128), (128, 256)):
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            pegar_batch(treino, bs, blk, generator=g)
        dt = (time.perf_counter() - t0) / n
        print(f"  batch={bs:4d} block={blk:4d}: {dt*1000:6.2f} ms/batch, "
              f"{bs*blk/dt/1e6:6.2f} M tokens/s")

    print("""
  Um detalhe pratico: se o carregamento de dados for mais lento que o passo de
  treino, a GPU fica ESPERANDO -- e voce paga por uma placa ociosa. Por isso
  frameworks usam workers em paralelo (DataLoader com num_workers) e prefetch.
  Aqui, como cada batch e' so' uma leitura de fatias contiguas, o custo e'
  desprezivel perto do forward/backward.""")
