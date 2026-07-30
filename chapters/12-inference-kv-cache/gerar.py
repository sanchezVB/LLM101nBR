"""
gerar.py — usa o modelo do Capitulo 11 para escrever prosa, agora com KV-cache.

E' o script "para usar": carrega os pesos, aceita um prompt e escreve.

Run (a partir da pasta do capitulo):
    python gerar.py
    python gerar.py "Havia em mim uma ideia fixa"
    python gerar.py "Capitu" --tokens 400 --temp 0.7
"""

import argparse
import sys
import time
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import carregar, carregar_tokenizador, decodificar
from bpe import BPETokenizer

ap = argparse.ArgumentParser(description="Gera texto com o modelo do curso.")
ap.add_argument("prompt", nargs="?", default="", help="texto inicial (opcional)")
ap.add_argument("--tokens", type=int, default=300, help="quantos tokens gerar")
ap.add_argument("--temp", type=float, default=0.8, help="temperatura")
ap.add_argument("--top-k", type=int, default=40)
ap.add_argument("--semente", type=int, default=1337)
ap.add_argument("--sem-cache", action="store_true",
                help="usa o caminho ingenuo do Cap. 11, para comparar")
args = ap.parse_args()

modelo, _ = carregar()
merges, vocab = carregar_tokenizador()

if args.prompt:
    tok = BPETokenizer()
    tok.merges, tok.vocab = merges, vocab
    ids = tok.encode(args.prompt)
    contexto = torch.tensor([ids], dtype=torch.long)
else:
    contexto = torch.zeros((1, 1), dtype=torch.long)   # 0 = inicio de texto

metodo = modelo.gerar_ingenuo if args.sem_cache else modelo.gerar_com_cache
t0 = time.perf_counter()
saida = metodo(contexto, args.tokens, temperatura=args.temp,
               top_k=args.top_k, semente=args.semente)
dt = time.perf_counter() - t0

print(decodificar(saida[0].tolist(), vocab))
print("\n" + "-" * 70)
print(f"{args.tokens} tokens em {dt:.2f}s  ({dt/args.tokens*1000:.1f} ms/token, "
      f"{'SEM' if args.sem_cache else 'com'} cache)")
