"""
gerar_imagens.py — o MESMO Transformer do Capitulo 5, gerando imagens.

Este arquivo tem uma linha que e' o capitulo inteiro:

    from modelo import GPT

E' a mesma classe que escreveu prosa de Machado. Nao ha' versao "visual" dela,
nao ha' camada nova, nao ha' atencao especial para imagens. O que mudou foi o
que entra: em vez de tokens de BPE, tokens do codebook do VQ-VAE.

    Capitulo 11:  texto  --[BPE]-->    tokens (vocab 1024, 128 por amostra)
    Capitulo 17:  imagem --[VQ-VAE]--> tokens (vocab 128,   49 por amostra)

A arquitetura nao sabe a diferenca. Ela recebe inteiros e aprende a prever o
proximo -- e "o proximo" pode ser a proxima palavra ou o proximo pedaco de
imagem.

E' por isso que modelos multimodais existem: uma vez que tudo vira token, o
mesmo modelo serve.

Run (a partir da pasta do capitulo):
    python gerar_imagens.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import GPT                      # <- a MESMA classe dos capitulos de texto
from vqvae import VQVAE, carregar_mnist, desenhar

PASSOS = 2000
BATCH = 64
LR = 1e-3
LADO = 7                                    # o mapa latente e' 7x7
N_TOKENS = LADO * LADO                      # 49 tokens por imagem


def carregar_vqvae():
    caminho = AQUI / "vqvae.pt"
    if not caminho.exists():
        raise SystemExit("vqvae.pt nao encontrado. Rode antes: python vqvae.py")
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m = VQVAE(ck["codebook"], ck["d_latente"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["codebook"]


@torch.no_grad()
def tokenizar_tudo(vq, imagens, lote=512):
    """Converte o dataset inteiro em tokens. E' o equivalente do treino.bin."""
    saida = []
    for i in range(0, imagens.shape[0], lote):
        saida.append(vq.para_tokens(imagens[i:i + lote]))
    return torch.cat(saida, 0)


def treinar_gpt(tokens_tr, vocab, passos=PASSOS, verbose=True, semente=1337):
    torch.manual_seed(semente)
    # MESMA classe, MESMO formato de config -- so' os numeros mudam.
    cfg = {"vocab_size": vocab, "block_size": N_TOKENS,
           "n_embd": 128, "n_head": 4, "n_layer": 4}
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    g = torch.Generator().manual_seed(semente)
    t0 = time.perf_counter()

    for passo in range(passos):
        for grupo in opt.param_groups:
            grupo["lr"] = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * passo / passos)))
        ix = torch.randint(0, tokens_tr.shape[0], (BATCH,), generator=g)
        seq = tokens_tr[ix]
        logits, _ = m(seq[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), seq[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if verbose and (passo % 400 == 0 or passo == passos - 1):
            print(f"    passo {passo:>4d} | loss {loss.item():.4f}", flush=True)

    return m, time.perf_counter() - t0


@torch.no_grad()
def amostrar_imagens(gpt, vq, n=3, semente=42, temperatura=1.0):
    """Gera tokens autorregressivamente e devolve as imagens decodificadas."""
    g = torch.Generator().manual_seed(semente)
    idx = torch.randint(0, gpt.cfg["vocab_size"], (n, 1), generator=g)
    for _ in range(N_TOKENS - 1):
        logits, _ = gpt(idx)
        lg = logits[:, -1, :] / temperatura
        prox = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=g)
        idx = torch.cat((idx, prox), dim=1)
    return vq.de_tokens(idx, LADO), idx


# ===========================================================================
if __name__ == "__main__":
    print("=" * 74)
    print("O mesmo Transformer, agora gerando imagens")
    print("=" * 74)

    vq, vocab = carregar_vqvae()
    tr, va = carregar_mnist()
    print(f"  tokenizando {tr.shape[0]:,} imagens com o VQ-VAE...", flush=True)
    tok_tr = tokenizar_tudo(vq, tr)
    tok_va = tokenizar_tudo(vq, va)
    print(f"    {tok_tr.shape[0]:,} sequencias de {tok_tr.shape[1]} tokens "
          f"(vocabulario {vocab})")
    print(f"    total: {tok_tr.numel():,} tokens de treino\n")

    gpt, dt = treinar_gpt(tok_tr, vocab)
    n_par = sum(p.nelement() for p in gpt.parameters())
    print(f"\n  GPT: {n_par:,} parametros | {dt/60:.1f} min")

    with torch.no_grad():
        logits, _ = gpt(tok_va[:1000, :-1])
        loss_val = F.cross_entropy(logits.reshape(-1, vocab),
                                   tok_va[:1000, 1:].reshape(-1)).item()
    print(f"  loss de validacao: {loss_val:.4f} "
          f"(perplexidade {math.exp(loss_val):.1f} entre {vocab} tokens)")

    imagens, tokens = amostrar_imagens(gpt, vq, n=3)
    print("\n  tres imagens geradas do zero, token por token:\n")
    desenhos = [desenhar(im) for im in imagens]
    for linhas in zip(*desenhos):
        print("    " + "   ".join(linhas))

    print(f"""
  Nenhuma dessas imagens estava no dataset. O modelo gerou 49 tokens
  autorregressivamente -- exatamente como gerou palavras nos capitulos
  anteriores -- e o decoder do VQ-VAE os transformou em pixels.

  E VALE OLHAR COM HONESTIDADE: os digitos saem tortos, alguns nao sao digitos
  nenhum. E' esperado. O GPT aqui tem {n_par/1e6:.1f} M de parametros e viu
  {tok_tr.numel()/1e6:.1f} M de tokens -- e a mesma conta do Capitulo 11 se aplica.
  O que o capitulo demonstra e' o MECANISMO, nao a qualidade.""")

    torch.save({"state_dict": gpt.state_dict(), "config": gpt.cfg,
                "loss_val": loss_val}, AQUI / "gpt_imagem.pt")
    print(f"\n  salvo em gpt_imagem.pt")
