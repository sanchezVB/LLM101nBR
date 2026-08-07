"""
servidor.py — servir o modelo por HTTP, com streaming.

Biblioteca padrao apenas: http.server. Sem FastAPI, sem uvicorn. Nao e' teimosia
-- e' que o assunto do capitulo sao as decisoes de SERVICO (streaming, batching,
concorrencia), e um framework as esconde atras de decoradores.

Dois endpoints, e a diferenca entre eles e' o capitulo:

    POST /gerar            devolve a resposta inteira, de uma vez
    POST /gerar/stream     devolve token por token, conforme sao produzidos

O texto final e' o MESMO. O que muda e' quando o primeiro byte chega -- e e' isso
que o usuario sente.

Run (a partir da pasta do capitulo):
    python servidor.py
    python servidor.py --porta 8080 --batching
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

MODELO, _ck = carregar()
_merges, VOCAB = carregar_tokenizador()
TOK = BPETokenizer()
TOK.merges, TOK.vocab = _merges, VOCAB

# O modelo NAO e' thread-safe para treino, mas para inferencia com no_grad ele e'
# -- nao ha' estado mutavel. Ainda assim serializamos o acesso quando o batching
# esta' ligado, porque ai' ha' uma fila compartilhada.
TRAVA = threading.Lock()


def decodificar(ids):
    return b"".join(VOCAB[int(i)] for i in ids).decode("utf-8", errors="replace")


# ===========================================================================
@torch.no_grad()
def gerar_tokens(prompt_ids, n_tokens=60, temperatura=0.8, top_k=40, semente=None):
    """Gerador Python: produz um token por vez, com KV-cache (Capitulo 12).

    Ser um GERADOR e' o que torna o streaming possivel. A funcao nao devolve o
    texto pronto -- ela cede cada token no instante em que existe, e quem chamou
    decide o que fazer com ele.
    """
    g = torch.Generator()
    g.manual_seed(semente if semente is not None else torch.seed() % (2**31))
    idx = torch.tensor([prompt_ids or [0]], dtype=torch.long)

    logits, cache = MODELO(idx[:, -MODELO.block_size:])
    for _ in range(n_tokens):
        lg = logits[:, -1, :] / temperatura
        v, _ = torch.topk(lg, min(top_k, lg.size(-1)))
        lg = lg.masked_fill(lg < v[:, [-1]], -float("inf"))
        prox = torch.multinomial(F.softmax(lg, dim=-1), 1, generator=g)
        yield int(prox.item())
        idx = torch.cat((idx, prox), dim=1)
        if cache[0][0].shape[2] >= MODELO.block_size:
            logits, cache = MODELO(idx[:, -MODELO.block_size:])
        else:
            logits, cache = MODELO(prox, cache=cache)


def tokenizar(texto):
    return TOK.encode(texto) if texto else [0]


# ===========================================================================
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                       # silencia o log padrao, que polui a medicao

    def _corpo(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_POST(self):
        try:
            pedido = self._corpo()
        except Exception:
            self.send_error(400, "JSON invalido")
            return
        ids = tokenizar(pedido.get("prompt", ""))
        n = int(pedido.get("tokens", 60))
        semente = pedido.get("semente")

        if self.path == "/gerar":
            self._resposta_inteira(ids, n, semente)
        elif self.path == "/gerar/stream":
            self._resposta_streaming(ids, n, semente)
        else:
            self.send_error(404)

    # -----------------------------------------------------------------------
    def _resposta_inteira(self, ids, n, semente):
        """Junta tudo e envia no fim. O cliente espera calado."""
        t0 = time.perf_counter()
        with TRAVA:
            saida = list(gerar_tokens(ids, n, semente=semente))
        corpo = json.dumps({
            "texto": decodificar(saida),
            "tokens": len(saida),
            "segundos": round(time.perf_counter() - t0, 3),
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def _resposta_streaming(self, ids, n, semente):
        """Envia cada token assim que ele existe, em chunked encoding.

        O primeiro byte sai depois de UM token, nao depois de sessenta. O tempo
        total nao muda; o tempo ate' o usuario ver algo, sim.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("X-Accel-Buffering", "no")     # nginx nao deve bufferizar
        self.end_headers()
        try:
            with TRAVA:
                for t in gerar_tokens(ids, n, semente=semente):
                    pedaco = decodificar([t]).encode("utf-8")
                    self.wfile.write(f"{len(pedaco):X}\r\n".encode())
                    self.wfile.write(pedaco + b"\r\n")
                    self.wfile.flush()      # sem isto, o SO junta tudo e o
                                            # streaming vira nao-streaming
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError):
            pass                            # cliente desistiu no meio


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Servidor HTTP do modelo.")
    ap.add_argument("--porta", type=int, default=8000)
    args = ap.parse_args()

    n = sum(p.nelement() for p in MODELO.parameters())
    print(f"modelo: {n:,} parametros, contexto {MODELO.block_size}")
    print(f"servindo em http://127.0.0.1:{args.porta}\n")
    print("  POST /gerar          -> resposta inteira")
    print("  POST /gerar/stream   -> token por token\n")
    print("  teste:")
    print(f'    curl -X POST http://127.0.0.1:{args.porta}/gerar '
          f'-d \'{{"prompt":"Havia em mim","tokens":40}}\'')
    print("\n  Ctrl-C para parar.")

    srv = ThreadingHTTPServer(("127.0.0.1", args.porta), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado.")
