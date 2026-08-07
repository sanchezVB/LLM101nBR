"""
carga.py — mede o servidor sob carga. Tres perguntas, tres numeros.

  TTFT       tempo ate' o primeiro token. E' o que o usuario sente como
             "o sistema travou" ou "o sistema respondeu".
  LATENCIA   tempo ate' a resposta completa.
  VAZAO      tokens por segundo somando TODOS os clientes.

As duas ultimas puxam em direcoes opostas quando ha' concorrencia, e o capitulo
inteiro e' sobre esse conflito.

Run (com o servidor de pe' em outro terminal):
    python servidor.py                    # terminal 1
    python carga.py                       # terminal 2
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

URL = "http://127.0.0.1:8000"
TOKENS = 40
PROMPT = "Havia em mim, leitor amigo, uma ideia"


def pedir(caminho, dados, timeout=180):
    req = urllib.request.Request(URL + caminho, data=json.dumps(dados).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def uma_requisicao(streaming, tokens=TOKENS, semente=None):
    """Devolve (ttft, latencia_total, n_bytes)."""
    t0 = time.perf_counter()
    corpo = {"prompt": PROMPT, "tokens": tokens, "semente": semente}
    if not streaming:
        r = pedir("/gerar", corpo)
        dados = r.read()
        t = time.perf_counter() - t0
        # sem streaming, o primeiro byte chega junto com o ultimo
        return t, t, len(dados)

    r = pedir("/gerar/stream", corpo)
    ttft, total = None, 0
    while True:
        pedaco = r.read(64)
        if not pedaco:
            break
        if ttft is None:
            ttft = time.perf_counter() - t0
        total += len(pedaco)
    return ttft, time.perf_counter() - t0, total


def aquecer():
    """Primeira requisicao carrega caches e aloca buffers -- nao vale medir."""
    for _ in range(2):
        uma_requisicao(streaming=False, tokens=8)


# ===========================================================================
def medir_streaming(n=5):
    print("=" * 74)
    print("1. Streaming muda o que o usuario sente, nao o que o servidor faz")
    print("=" * 74)
    print(f"  {n} requisicoes de {TOKENS} tokens, uma por vez\n")
    print(f"  {'endpoint':>16s} {'TTFT (s)':>10s} {'total (s)':>11s} {'tokens/s':>10s}")
    for rotulo, streaming in (("/gerar", False), ("/gerar/stream", True)):
        ttfts, totais = [], []
        for i in range(n):
            a, b, _ = uma_requisicao(streaming, semente=i)
            ttfts.append(a); totais.append(b)
        print(f"  {rotulo:>16s} {statistics.median(ttfts):>10.3f} "
              f"{statistics.median(totais):>11.3f} "
              f"{TOKENS/statistics.median(totais):>10.1f}")
    print("""
  O TOTAL e' praticamente o mesmo -- e tem de ser: o servidor faz o mesmo
  trabalho nos dois casos. O que muda e' o TTFT.

  Sem streaming, o usuario olha para uma tela parada durante toda a geracao e
  recebe tudo de uma vez. Com streaming, ele ve' a primeira palavra quase
  imediatamente e le' enquanto o resto e' produzido.

  E' a otimizacao mais barata deste capitulo: nao acelera nada, e muda a
  experiencia por completo. Repare que ela so' e' possivel porque a geracao e'
  AUTORREGRESSIVA -- os tokens existem um a um, em ordem. Um modelo que
  produzisse a resposta inteira de uma vez nao teria o que transmitir aos
  poucos.

  MAS OLHE A MAGNITUDE ANTES DE SE ANIMAR. A diferenca medida aqui e' de ~0,04s
  (0,079 contra 0,115), uns 30%. E' pouco, e o motivo e' que este modelo e'
  RAPIDO DEMAIS para o streaming brilhar: 40 tokens saem em 0,12s, entao o TTFT
  ja' era baixo e boa parte dele e' custo de HTTP, nao de geracao.

  O ganho do streaming e' proporcional ao TEMPO TOTAL de geracao. Num modelo de
  verdade, com 500 tokens a 30 tok/s, o total e' ~17 segundos -- e o TTFT cai de
  17s para uns 0,3s. E' a diferenca entre um sistema que parece travado e um que
  parece instantaneo.

  Este capitulo mede o MECANISMO. A magnitude vem da escala, e nesta escala ela
  e' pequena.""")


def medir_concorrencia(niveis=(1, 2, 4, 8)):
    print("=" * 74)
    print("2. O que acontece quando varios clientes chegam juntos")
    print("=" * 74)
    print(f"  cada cliente pede {TOKENS} tokens, todos ao mesmo tempo\n")
    print(f"  {'clientes':>9s} {'latencia mediana':>18s} {'p95':>8s} "
          f"{'vazao total':>13s}")
    for n in niveis:
        resultados = []
        trava = threading.Lock()

        def trabalho(i):
            r = uma_requisicao(streaming=True, semente=i)
            with trava:
                resultados.append(r)

        threads = [threading.Thread(target=trabalho, args=(i,)) for i in range(n)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        parede = time.perf_counter() - t0

        latencias = sorted(r[1] for r in resultados)
        p95 = latencias[min(len(latencias) - 1, int(0.95 * len(latencias)))]
        vazao = n * TOKENS / parede
        print(f"  {n:>9d} {statistics.median(latencias):>17.2f}s {p95:>7.2f}s "
              f"{vazao:>10.1f} tok/s", flush=True)
    print("""
  Leia as duas colunas juntas, porque nenhuma sozinha conta a historia.

  A VAZAO quase nao cresce: 268 -> 334 tok/s ao passar de 1 para 8 clientes.
  Oito vezes mais gente, 25% mais trabalho entregue. Se o servidor escalasse, a
  vazao teria multiplicado por algo perto de 8.

  A LATENCIA cresce 3,6x (0,15s -> 0,54s), e o p95 vai a 0,96s. Quem chega por
  ultimo espera a fila inteira.

  A causa esta' no servidor.py: ha' uma trava (threading.Lock) em volta da
  geracao, entao as requisicoes se REVEZAM no modelo. Os 25% de ganho que
  aparecem vem das partes que rodam FORA da trava -- HTTP, tokenizacao, JSON --
  e essas sim se sobrepoem.

  Vale ser preciso sobre o que isso mostra: o servidor nao esta' quebrado nem a
  trava esta' errada. Sem ela, varias threads chamariam o modelo ao mesmo tempo
  e disputariam os mesmos nucleos, o que costuma sair PIOR. O problema e' que
  requisicoes concorrentes estao sendo tratadas como independentes quando
  poderiam viajar JUNTAS.

  E' o que o Capitulo 12 ja' media: gerar 16 sequencias em paralelo custou 9,1x
  menos que 16 geracoes separadas, porque o decode e' limitado por MEMORIA e ler
  os pesos custa o mesmo para 1 ou para 16. Cada cliente sozinho desperdicia
  essa leitura; agrupados, eles a dividem.""")


# ===========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Teste de carga do servidor.")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--so-streaming", action="store_true")
    args = ap.parse_args()
    URL = args.url

    try:
        aquecer()
    except Exception as e:
        raise SystemExit(
            f"Nao consegui falar com {URL}: {e}\n"
            f"Suba o servidor antes:  python servidor.py"
        )

    medir_streaming()
    if not args.so_streaming:
        medir_concorrencia()
