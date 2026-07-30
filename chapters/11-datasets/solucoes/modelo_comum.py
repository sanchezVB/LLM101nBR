"""
Modelo e laco de treino compartilhados pelos scripts de solucao do Capitulo 11.

Fica num modulo separado para que `gabarito.py` e `e7_mais_obras.py` usem
EXATAMENTE o mesmo codigo -- se cada um tivesse a sua copia, uma divergencia
silenciosa entre elas invalidaria a comparacao entre os dois scripts.

Arquitetura IDENTICA a' da apostila (n_embd=192, 6 cabecas, 4 blocos, 2.2 M
parametros). O que muda nas solucoes e' o numero de PASSOS, nao o modelo:
reduzir passos e' defensavel para perguntas comparativas, trocar a arquitetura
nao seria, porque mudaria o objeto em estudo.
"""

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))

from dataset import carregar, pegar_batch, carregar_tokenizador

PASSOS = 400
N_EMBD, N_HEAD, N_LAYER = 192, 6, 4      # identico a' apostila

TREINO = carregar("treino")
VAL = carregar("val")
_TOK, VOCAB = carregar_tokenizador()
V = max(VOCAB.keys()) + 1


class Bloco(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.hs = N_EMBD // N_HEAD
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.fi = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.fo = nn.Linear(4 * N_EMBD, N_EMBD)
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.register_buffer("tril", torch.tril(torch.ones(block, block)))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        k = k.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        v = v.view(B, T, N_HEAD, self.hs).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, block, so_ultima=False, vocab=V):
        super().__init__()
        self.block, self.so_ultima = block, so_ultima
        self.te = nn.Embedding(vocab, N_EMBD)
        self.pe = nn.Embedding(block, N_EMBD)
        self.blocos = nn.ModuleList([Bloco(block) for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(N_EMBD)
        self.lm = nn.Linear(N_EMBD, vocab)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T, device=idx.device))
        for b in self.blocos:
            x = b(x)
        x = self.lnf(x)
        if self.so_ultima:
            logits = self.lm(x[:, -1, :])
            if alvos is None:
                return logits, None
            return logits, F.cross_entropy(logits, alvos[:, -1])
        logits = self.lm(x)
        if alvos is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), alvos.reshape(-1))


def treinar(block=128, so_ultima=False, dados_tr=None, dados_val=None,
            passos=PASSOS, vocab=V, semente=1337):
    """Devolve (loss de treino, loss de validacao, segundos, modelo)."""
    dados_tr = TREINO if dados_tr is None else dados_tr
    dados_val = VAL if dados_val is None else dados_val
    torch.manual_seed(semente)
    m = GPT(block, so_ultima, vocab)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(semente)
    t0 = time.perf_counter()
    for passo in range(passos):
        for grupo in opt.param_groups:
            grupo["lr"] = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * passo / passos)))
        x, y = pegar_batch(dados_tr, 32, block, generator=g)
        _, loss = m(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    dt = time.perf_counter() - t0

    @torch.no_grad()
    def avaliar(dados):
        m.eval()
        tot = 0.0
        for _ in range(20):
            x, y = pegar_batch(dados, 32, block, generator=g)
            _, l = m(x, y)
            tot += l.item()
        m.train()
        return tot / 20

    return avaliar(dados_tr), avaliar(dados_val), dt, m
