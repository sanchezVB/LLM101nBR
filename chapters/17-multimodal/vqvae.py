"""
vqvae.py — transformar imagens em tokens discretos, do zero.

O PROBLEMA. Um Transformer come tokens: inteiros de um vocabulario finito.
Imagens sao numeros continuos, e muitos -- 784 por imagem no MNIST. Passar pixel
por pixel daria sequencias longas demais e desperdicadas, do mesmo jeito que
tokenizar texto byte a byte (Capitulo 6).

A SOLUCAO. Aprender um "alfabeto" de pedacos de imagem, exatamente como o BPE
aprendeu um alfabeto de pedacos de texto. E' o VQ-VAE:

    imagem --[encoder]--> vetores continuos --[quantizacao]--> INTEIROS
                                                                  |
    imagem' <--[decoder]-- vetores do codebook <------------------+

A peca nova e' a QUANTIZACAO: um "codebook" de K vetores aprendidos. Cada vetor
que sai do encoder e' substituido pelo vetor MAIS PROXIMO do codebook -- e o
indice desse vetor e' o token.

O TRUQUE que faz isso treinar: escolher-o-mais-proximo nao tem derivada. A
solucao (straight-through estimator) e' fingir, no backward, que a quantizacao
foi a identidade. Uma linha:

    z_q = z + (z_q - z).detach()

No forward vale z_q; no backward o gradiente passa direto para z como se nada
tivesse acontecido. E' aproximado, e funciona.

Run (a partir da pasta do capitulo):
    python vqvae.py                    # treina e mede
    python vqvae.py --codebook 64      # outro tamanho de alfabeto
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CODEBOOK = 128          # tamanho do "alfabeto" de imagem
D_LATENTE = 32          # dimensao de cada vetor do codebook
PASSOS = 1500
BATCH = 128
LR = 2e-3
BETA_COMP = 0.25        # peso do "commitment loss"


class Quantizador(nn.Module):
    """O codebook: K vetores aprendidos, e a busca pelo mais proximo."""

    def __init__(self, k=CODEBOOK, d=D_LATENTE):
        super().__init__()
        self.codebook = nn.Embedding(k, d)
        self.codebook.weight.data.uniform_(-1.0 / k, 1.0 / k)
        self.k = k

    def forward(self, z):
        # z: (B, d, H, W) -> (B*H*W, d), um vetor por posicao do mapa
        B, d, H, W = z.shape
        plano = z.permute(0, 2, 3, 1).reshape(-1, d)

        # distancia ao quadrado para cada vetor do codebook, sem loop:
        #   ||a - b||^2 = ||a||^2 - 2ab + ||b||^2
        dist = (plano.pow(2).sum(1, keepdim=True)
                - 2 * plano @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        indices = dist.argmin(1)
        z_q = self.codebook(indices).view(B, H, W, d).permute(0, 3, 1, 2)

        # DUAS perdas, e as duas sao necessarias:
        #   codebook loss  -- puxa os vetores do codebook para perto do encoder
        #   commitment     -- puxa o encoder para perto do codebook escolhido
        # Sem a segunda, o encoder foge livremente e o codebook nunca alcanca.
        perda_codebook = F.mse_loss(z_q, z.detach())
        perda_commit = F.mse_loss(z, z_q.detach())
        perda = perda_codebook + BETA_COMP * perda_commit

        # STRAIGHT-THROUGH: forward usa z_q, backward passa direto para z.
        z_q = z + (z_q - z).detach()
        return z_q, perda, indices.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, k=CODEBOOK, d=D_LATENTE):
        super().__init__()
        # 28x28 -> 14x14 -> 7x7. Cada imagem vira 49 tokens em vez de 784.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, d, 3, padding=1),
        )
        self.quant = Quantizador(k, d)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z_q, perda_vq, indices = self.quant(z)
        return self.decoder(z_q), perda_vq, indices

    @torch.no_grad()
    def para_tokens(self, x):
        _, _, idx = self.quant(self.encoder(x))
        return idx.reshape(x.shape[0], -1)          # (B, 49)

    @torch.no_grad()
    def de_tokens(self, tokens, lado=7):
        B = tokens.shape[0]
        z_q = self.quant.codebook(tokens.view(B, lado, lado))
        return self.decoder(z_q.permute(0, 3, 1, 2))


# ===========================================================================
def carregar_mnist():
    caminho = AQUI / "mnist.npz"
    if not caminho.exists():
        raise SystemExit("mnist.npz nao encontrado. Rode antes: python preparar_dados.py")
    d = np.load(caminho)
    tr = torch.from_numpy(d["treino_x"]).float().unsqueeze(1) / 255.0
    va = torch.from_numpy(d["val_x"]).float().unsqueeze(1) / 255.0
    return tr, va


def treinar(k=CODEBOOK, passos=PASSOS, verbose=True, semente=1337):
    torch.manual_seed(semente)
    tr, va = carregar_mnist()
    m = VQVAE(k)
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    g = torch.Generator().manual_seed(semente)
    t0 = time.perf_counter()

    for passo in range(passos):
        ix = torch.randint(0, tr.shape[0], (BATCH,), generator=g)
        x = tr[ix]
        rec, perda_vq, _ = m(x)
        perda_rec = F.mse_loss(rec, x)
        perda = perda_rec + perda_vq
        opt.zero_grad(set_to_none=True)
        perda.backward()
        opt.step()
        if verbose and (passo % 300 == 0 or passo == passos - 1):
            print(f"    passo {passo:>4d} | reconstrucao {perda_rec.item():.4f} "
                  f"| vq {perda_vq.item():.4f}", flush=True)

    dt = time.perf_counter() - t0
    m.eval()
    with torch.no_grad():
        rec, _, idx = m(va[:2000])
        erro = F.mse_loss(rec, va[:2000]).item()
        usados = len(torch.unique(idx))
    return m, erro, usados, dt


def desenhar(x, largura=28):
    """Arte ASCII -- para ver a imagem sem depender de matplotlib."""
    escala = " .:-=+*#%@"
    linhas = []
    img = x.squeeze().clamp(0, 1)
    for i in range(0, img.shape[0], 2):          # pula linhas: fonte e' mais alta que larga
        linhas.append("".join(escala[min(9, int(v * 9.99))] for v in img[i]))
    return linhas


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VQ-VAE do zero.")
    ap.add_argument("--codebook", type=int, default=CODEBOOK)
    ap.add_argument("--passos", type=int, default=PASSOS)
    args = ap.parse_args()

    print("=" * 74)
    print(f"VQ-VAE | codebook de {args.codebook} | {args.passos} passos")
    print("=" * 74)
    m, erro, usados, dt = treinar(args.codebook, args.passos)
    print(f"\n  erro de reconstrucao (validacao): {erro:.4f}")
    print(f"  codigos usados: {usados} de {args.codebook} ({usados/args.codebook:.0%})")
    print(f"  tempo: {dt/60:.1f} min")

    _, va = carregar_mnist()
    tokens = m.para_tokens(va[:1])
    print(f"\n  uma imagem 28x28 = 784 pixels vira {tokens.shape[1]} tokens:")
    print(f"    {tokens[0].tolist()}")
    print(f"  compressao: {784/tokens.shape[1]:.0f}x menos posicoes")

    print("\n  original vs reconstruida:")
    orig = desenhar(va[0])
    with torch.no_grad():
        rec, _, _ = m(va[:1])
    recd = desenhar(rec[0])
    for a, b in zip(orig, recd):
        print(f"    {a}   {b}")

    torch.save({"state_dict": m.state_dict(), "codebook": args.codebook,
                "d_latente": D_LATENTE, "erro": erro}, AQUI / "vqvae.pt")
    print(f"\n  salvo em vqvae.pt")
