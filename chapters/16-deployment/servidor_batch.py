"""
servidor_batch.py — o mesmo servidor, agrupando requisicoes num batch.

O carga.py mostrou o problema: com uma trava em volta do modelo, oito clientes
entregam 25% mais vazao que um, e esperam 3,6x mais. As requisicoes se revezam
quando deveriam viajar juntas.

A correcao vem do Capitulo 12: o decode e' limitado por MEMORIA. Ler os pesos do
modelo custa o mesmo para 1 ou para 16 sequencias -- entao processar 16 juntas
custa quase o mesmo que processar 1, e a vazao multiplica.

COMO FUNCIONA. Um unico thread e' dono do modelo. As requisicoes HTTP nao geram
nada: elas colocam um pedido numa fila e esperam os tokens chegarem de volta por
outra fila. O thread-dono junta o que estiver esperando, gera UM token para todo
mundo de uma vez, distribui, e repete.

    cliente A --\\                             /--> tokens de A
    cliente B ---+--> [fila] --> laco --------+---> tokens de B
    cliente C --/               (1 thread)     \\--> tokens de C

Isso e' CONTINUOUS BATCHING em miniatura -- quem termina sai do lote, quem chega
entra no proximo passo, sem esperar o lote inteiro fechar. E' o que vLLM e TGI
fazem, com muito mais cuidado.

Run:
    python servidor_batch.py --porta 8001
"""

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import carregar, carregar_tokenizador
from bpe import BPETokenizer

MODELO, _ = carregar()
_merges, VOCAB = carregar_tokenizador()
TOK = BPETokenizer()
TOK.merges, TOK.vocab = _merges, VOCAB

LOTE_MAX = 16          # quantas sequencias no maximo por passo
ESPERA_MS = 8          # quanto esperar juntando pedidos antes de comecar


class Pedido:
    """Uma requisicao viva dentro do lote."""

    def __init__(self, ids, n_tokens, semente):
        self.idx = torch.tensor([ids or [0]], dtype=torch.long)
        self.restantes = n_tokens
        self.saida = queue.Queue()          # o handler HTTP le' daqui
        self.g = torch.Generator()
        self.g.manual_seed(semente if semente is not None else torch.seed() % (2**31))


FILA = queue.Queue()


@torch.no_grad()
def laco_do_modelo():
    """O unico lugar do processo que toca o modelo."""
    ativos = []
    while True:
        # 1. Recolhe quem chegou.
        #
        # A REGRA E' NUNCA BLOQUEAR QUANDO HA' TRABALHO. A primeira versao deste
        # laco esperava ESPERA_MS por novos pedidos a CADA passo de decode:
        #
        #     FILA.get(timeout=None if not ativos else ESPERA_MS/1000)
        #
        # Com 8 ms de espera e 40 tokens, isso somava 320 ms de espera pura por
        # requisicao -- e o servidor "com batching" ficou MAIS LENTO que o
        # simples (0,62s contra 0,15s com um cliente). O ganho do agrupamento
        # existia e estava sendo comido pela espera.
        #
        # Agora: bloqueia so' quando esta' ocioso, e drena sem esperar quando ja'
        # tem o que fazer.
        if not ativos:
            ativos.append(FILA.get())          # ocioso: pode bloquear
        while len(ativos) < LOTE_MAX:
            try:
                ativos.append(FILA.get_nowait())   # ocupado: nunca espera
            except queue.Empty:
                break

        # 2. UM passo para todo o lote.
        #
        # Simplificacao honesta: recortamos todas as sequencias no mesmo
        # comprimento (a mais curta) para poder empilhar num tensor. Um servidor
        # de verdade usa padding com mascara, ou paged attention. Aqui o objetivo
        # e' mostrar o ganho do agrupamento, nao competir com o vLLM.
        menor = min(p.idx.shape[1] for p in ativos)
        lote = torch.cat([p.idx[:, -min(menor, MODELO.block_size):] for p in ativos], 0)
        logits, _ = MODELO(lote)

        for i, p in enumerate(ativos):
            lg = logits[i, -1, :] / 0.8
            v, _ = torch.topk(lg, 40)
            lg = lg.masked_fill(lg < v[-1], -float("inf"))
            prox = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=p.g)
            p.idx = torch.cat((p.idx, prox.view(1, 1)), dim=1)
            p.saida.put(int(prox.item()))
            p.restantes -= 1

        # 3. quem acabou sai do lote AGORA, sem esperar os outros -- e' isso que
        #    faz o batching ser "continuous" em vez de "estatico".
        for p in [p for p in ativos if p.restantes <= 0]:
            p.saida.put(None)
        ativos = [p for p in ativos if p.restantes > 0]


def decodificar(ids):
    return b"".join(VOCAB[int(i)] for i in ids).decode("utf-8", errors="replace")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            pedido = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self.send_error(400); return

        ids = TOK.encode(pedido.get("prompt", "")) if pedido.get("prompt") else [0]
        p = Pedido(ids, int(pedido.get("tokens", 60)), pedido.get("semente"))
        FILA.put(p)

        streaming = self.path.endswith("/stream")
        if streaming:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        colhidos = []
        try:
            while True:
                t = p.saida.get(timeout=180)
                if t is None:
                    break
                if streaming:
                    b = decodificar([t]).encode("utf-8")
                    self.wfile.write(f"{len(b):X}\r\n".encode() + b + b"\r\n")
                    self.wfile.flush()
                else:
                    colhidos.append(t)
            if streaming:
                self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
            else:
                corpo = json.dumps({"texto": decodificar(colhidos),
                                    "tokens": len(colhidos)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(corpo)))
                self.end_headers()
                self.wfile.write(corpo)
        except (BrokenPipeError, ConnectionAbortedError, queue.Empty):
            p.restantes = 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Servidor com batching.")
    ap.add_argument("--porta", type=int, default=8001)
    ap.add_argument("--lote", type=int, default=LOTE_MAX)
    args = ap.parse_args()
    LOTE_MAX = args.lote

    threading.Thread(target=laco_do_modelo, daemon=True).start()
    print(f"servidor COM BATCHING em http://127.0.0.1:{args.porta}")
    print(f"  lote maximo: {LOTE_MAX} | espera para juntar: {ESPERA_MS} ms")
    print("  POST /gerar  e  POST /gerar/stream\n  Ctrl-C para parar.")
    ThreadingHTTPServer(("127.0.0.1", args.porta), Handler).serve_forever()
