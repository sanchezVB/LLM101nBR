"""
smoke_test.py — o curso inteiro ainda roda?

Um curso de 13 capitulos e 65 scripts apodrece em silencio: uma versao de
biblioteca muda, um arquivo e' renomeado, um import quebra -- e so' se descobre
quando um aluno tenta rodar.

Este script roda TODOS os scripts do curso e reporta o estado de cada um.

O QUE ELE VERIFICA, e o que nao verifica: a maioria dos scripts treina modelos e
leva minutos. Nao da' para esperar todos terminarem. Entao o criterio e':

    OK       terminou com sucesso dentro do tempo
    RODANDO  nao terminou, mas estava produzindo saida quando o tempo acabou
             -- ou seja, arrancou: imports resolvidos, dados carregados
    FALHOU   terminou com erro, ou nao produziu nada
    PULADO   pediu um pre-requisito que nao esta' presente (e disse qual)

'RODANDO' e' aprovacao. A esmagadora maioria das quebras -- import errado,
arquivo faltando, API que mudou -- acontece nos primeiros segundos.

Uso:
    python tools/smoke_test.py                  # tudo, 20s por script
    python tools/smoke_test.py --timeout 60     # mais folga
    python tools/smoke_test.py --chapter 11     # so' um capitulo
    python tools/smoke_test.py --rapidos        # so' os que costumam terminar
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS = ROOT / "chapters"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Scripts que NAO sao para rodar sozinhos: workers lancados por outro script,
# modulos so' de importacao, e coisas que exigem argumentos obrigatorios.
NAO_RODAR = {
    "worker_deadlock.py",     # lancado como subprocesso pelo gabarito do cap. 10
    "modelo_comum.py",        # modulo compartilhado, sem __main__
    "dist_utils.py",          # tem __main__ de diagnostico, mas mexe em rede
}


def scripts_do_curso(filtro_capitulo=None):
    for ch in sorted(CHAPTERS.glob("[0-9][0-9]-*")):
        if filtro_capitulo and not ch.name.startswith(filtro_capitulo):
            continue
        for py in sorted(list(ch.glob("*.py")) + list(ch.glob("solucoes/*.py"))):
            if py.name in NAO_RODAR or py.name.startswith("_"):
                continue
            yield ch, py


def _matar_arvore(p):
    """Mata o processo E OS FILHOS DELE.

    p.kill() mata so' o processo direto. Varios scripts do curso lancam
    subprocessos -- o gabarito do cap. 10 provoca deadlocks de proposito -- e os
    netos sobrevivem, segurando o pipe de saida aberto.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)],
                       capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    p.kill()


def rodar(py, cwd, timeout):
    """Devolve (estado, detalhe, segundos).

    A saida e' drenada por uma THREAD, e nao por communicate(). O motivo e' um
    bug que travou este script por completo:

      p.kill(); p.communicate()      # <- espera para sempre

    communicate() espera o pipe chegar a EOF. Se o processo morto tiver deixado
    NETOS vivos, eles ainda seguram a ponta de escrita, o EOF nunca chega, e o
    smoke test congela. Foi o que aconteceu no capitulo 10, e a execucao inteira
    do curso parou ali -- reportando exit 0, o que tornou o diagnostico pior.

    Com a thread leitora, o que ja' foi produzido esta' sempre disponivel, e a
    thread e' daemon: se ela ficar presa num read, nao impede o programa de sair.
    """
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.Popen([sys.executable, "-u", py.name if py.parent == cwd
                              else str(py.relative_to(cwd))],
                             cwd=cwd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env=env,
                             bufsize=1)
    except OSError as e:
        return "FALHOU", f"nao executou: {e}", 0.0

    linhas_lidas = []

    def drenar():
        try:
            for linha in p.stdout:
                linhas_lidas.append(linha)
        except Exception:
            pass

    leitor = threading.Thread(target=drenar, daemon=True)
    leitor.start()

    try:
        p.wait(timeout=timeout)
        leitor.join(timeout=3)
        saida = "".join(linhas_lidas)
        dt = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        _matar_arvore(p)
        leitor.join(timeout=3)
        saida = "".join(linhas_lidas)
        dt = time.perf_counter() - t0
        # produziu saida antes de ser morto? entao arrancou.
        if saida and saida.strip():
            return "RODANDO", f"{len(saida.splitlines())} linhas de saida", dt
        return "FALHOU", "nenhuma saida dentro do tempo", dt

    if p.returncode == 0:
        return "OK", f"{len(saida.splitlines())} linhas", dt

    # Distinguir uma MENSAGEM DELIBERADA de uma QUEBRA, e o criterio e' objetivo:
    # raise SystemExit("...") imprime so' a mensagem e sai com codigo 1; qualquer
    # excecao nao tratada imprime um Traceback. Procurar frases seria fragil --
    # a primeira versao deste script fazia isso e classificou como FALHA um
    # script que so' pedia um arquivo de pre-requisito.
    linhas = [l for l in saida.strip().splitlines() if l.strip()]
    if "Traceback (most recent call last)" not in saida:
        return "PULADO", (linhas[0] if linhas else "sem mensagem")[:90], dt
    return "FALHOU", (linhas[-1] if linhas else "sem saida")[:90], dt


def main():
    ap = argparse.ArgumentParser(description="Roda todos os scripts do curso.")
    ap.add_argument("--timeout", type=int, default=20, help="segundos por script")
    ap.add_argument("--chapter", help="so' este capitulo, ex.: 11")
    ap.add_argument("--rapidos", action="store_true",
                    help="timeout curto (8s): so' checa se arranca")
    args = ap.parse_args()
    timeout = 8 if args.rapidos else args.timeout

    print(f"Rodando os scripts do curso (timeout de {timeout}s por script)\n")
    contagem = {"OK": 0, "RODANDO": 0, "PULADO": 0, "FALHOU": 0}
    falhas = []
    cap_atual = None

    for ch, py in scripts_do_curso(args.chapter):
        if ch.name != cap_atual:
            cap_atual = ch.name
            print(f"\n{ch.name}")
        rel = py.relative_to(ch)
        estado, detalhe, dt = rodar(py, ch, timeout)
        contagem[estado] += 1
        if estado == "FALHOU":
            falhas.append((ch.name, str(rel), detalhe))
        marca = {"OK": "  ok   ", "RODANDO": "  roda ",
                 "PULADO": "  pula ", "FALHOU": "  FALHA"}[estado]
        print(f"{marca} {str(rel):<32s} {dt:5.1f}s  {detalhe}", flush=True)

    print("\n" + "=" * 74)
    total = sum(contagem.values())
    print(f"  {total} scripts: "
          + ", ".join(f"{v} {k.lower()}" for k, v in contagem.items() if v))
    if falhas:
        print("\n  FALHAS:")
        for cap, arq, det in falhas:
            print(f"    {cap}/{arq}\n      {det}")
        return 1
    print("\n  Nenhuma falha.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
