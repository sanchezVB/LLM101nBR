"""
train_text.py — o mesmo Transformer, agora treinado em PROSA de verdade.

Duas mudancas em relacao aos capitulos anteriores, e as duas importam:

  1. Os dados sao TEXTO CORRIDO (Machado de Assis), tokenizado com BPE. Nao sao
     mais nomes soltos.

  2. O modelo preve em TODAS as posicoes, nao so' na ultima. Antes, cada exemplo
     de batch gerava 1 previsao; agora gera `block_size` previsoes. Com o mesmo
     custo de forward, o sinal de treino fica ~128x maior -- e' assim que se
     treina LLM de verdade.

AVISO SOBRE A LOSS: ela NAO e' comparavel com o 1.776 dos capitulos anteriores.
Mudou a tarefa (prosa em vez de nomes), mudou a unidade (tokens BPE em vez de
caracteres) e mudou o vocabulario (1024 em vez de 27). Prever entre 1024 opcoes e'
muito mais dificil que entre 27. O benchmark antigo esta' aposentado; este e' o
novo ponto de partida.

Run:
    python train_text.py
"""

import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import carregar, pegar_batch, carregar_tokenizador, decodificar

# Sem isto, redirecionar a saida para um arquivo no Windows usa cp1252 e os
# acentos do texto gerado viram '?'. O modelo esta' certo; a IMPRESSAO e' que
# quebra -- e e' um erro facil de confundir com bug do modelo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
block_size = 128           # bem maior que os 8 dos capitulos anteriores
n_embd = 192
n_head = 6
n_layer = 4
max_steps = 3000
batch_size = 32
base_lr = 1e-3
warmup = 200
# ---------------------------------------------------------------------------

treino = carregar("treino")
val = carregar("val")
_, vocab_bpe = carregar_tokenizador()
vocab_size = max(vocab_bpe.keys()) + 1
print(f"treino: {len(treino):,} tokens | val: {len(val):,} tokens | vocab: {vocab_size}")


class Bloco(nn.Module):
    def __init__(self):
        super().__init__()
        self.hs = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.fi = nn.Linear(n_embd, 4 * n_embd)
        self.fo = nn.Linear(4 * n_embd, n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, n_head, self.hs).transpose(1, 2)
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.te = nn.Embedding(vocab_size, n_embd)
        self.pe = nn.Embedding(block_size, n_embd)
        self.blocos = nn.ModuleList([Bloco() for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, alvos=None):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T, device=idx.device))
        for b in self.blocos:
            x = b(x)
        logits = self.lm(self.lnf(x))          # (B, T, vocab) -- TODAS as posicoes
        if alvos is None:
            return logits, None
        # achata (B,T,V) -> (B*T,V) para a cross_entropy
        loss = F.cross_entropy(logits.view(-1, vocab_size), alvos.view(-1))
        return logits, loss

    @torch.no_grad()
    def gerar(self, idx, n_tokens, temperatura=0.8, top_k=40):
        """Gera texto, um token por vez, realimentando o proprio resultado."""
        for _ in range(n_tokens):
            recorte = idx[:, -block_size:]      # so' cabe block_size de contexto
            logits, _ = self(recorte)
            logits = logits[:, -1, :] / temperatura
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            prox = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, prox), dim=1)
        return idx


modelo = GPT()
nparams = sum(p.nelement() for p in modelo.parameters())
print(f"parametros: {nparams:,} ({n_layer} blocos, {n_head} cabecas, n_embd={n_embd}, "
      f"contexto {block_size})")

opt = torch.optim.AdamW(modelo.parameters(), lr=base_lr, weight_decay=0.01)
g = torch.Generator().manual_seed(1337)


def lr_do_passo(step):
    """Warmup + cosine -- o unico ajuste que ajudou na ablacao do Capitulo 7."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, max_steps - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))


@torch.no_grad()
def avaliar(dados, n=40):
    modelo.eval()
    total = 0.0
    for _ in range(n):
        x, y = pegar_batch(dados, batch_size, block_size, generator=g)
        _, loss = modelo(x, y)
        total += loss.item()
    modelo.train()
    return total / n


print("\n=== treinando ===")
t0 = time.perf_counter()
for step in range(max_steps):
    for grupo in opt.param_groups:
        grupo["lr"] = lr_do_passo(step)

    x, y = pegar_batch(treino, batch_size, block_size, generator=g)
    _, loss = modelo(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
    opt.step()

    if step % 500 == 0:
        l_tr, l_val = avaliar(treino, 10), avaliar(val, 10)
        print(f"  passo {step:5d} | treino {l_tr:.4f} | val {l_val:.4f} | "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

l_tr, l_val = avaliar(treino), avaliar(val)
print(f"\n  FINAL: treino {l_tr:.4f} | validacao {l_val:.4f} | "
      f"{time.perf_counter()-t0:.0f}s de treino")
print(f"  perplexidade de validacao: {math.exp(l_val):.1f}")
print("""
  A PERPLEXIDADE e' exp(loss), e tem uma leitura intuitiva: e' o numero medio de
  opcoes entre as quais o modelo esta' "em duvida" a cada token. Um modelo que
  chutasse uniformemente entre 1024 tokens teria perplexidade 1024. Quanto menor,
  mais o modelo sabe.""")

# ---------------------------------------------------------------------------
# Salvar o modelo. Sem isto, 18 minutos de treino sao descartados ao fim do
# script -- e o Capitulo 12 (KV-cache) precisa de um modelo ja' treinado.
CKPT = "modelo.pt"
torch.save({
    "state_dict": modelo.state_dict(),
    "config": dict(block_size=block_size, n_embd=n_embd, n_head=n_head,
                   n_layer=n_layer, vocab_size=vocab_size),
    "loss_treino": l_tr,
    "loss_val": l_val,
}, CKPT)
import os
print(f"\nmodelo salvo em {CKPT} ({os.path.getsize(CKPT)/1e6:.1f} MB)")

# ---------------------------------------------------------------------------
print("=== texto gerado ===")
for temp in (0.5, 0.8, 1.0):
    inicio = torch.zeros((1, 1), dtype=torch.long)
    saida = modelo.gerar(inicio, 220, temperatura=temp)
    texto = decodificar(saida[0].tolist(), vocab_bpe)
    print(f"\n--- temperatura {temp} ---")
    print(texto.strip()[:400])

print("""

  Compare com os capitulos anteriores: la' o modelo gerava NOMES ('jandir',
  'valdinia'). Aqui ele gera PROSA -- com pontuacao, espacos, quebras de
  paragrafo e estrutura de frase. Nao e' Machado, mas ja' e' portugues escrito.""")
