"""
Solucao do Exercicio E5 — o imposto do portugues nos tokenizadores.

Treina DOIS tokenizadores BPE com o mesmo tamanho de vocabulario:
  (a) em nomes sem acento  -- como o bpe.py do capitulo
  (b) em texto portugues de verdade, com acentos

...e compara quantos tokens cada um gasta na mesma frase. Como corpus portugues
usamos as proprias apostilas do curso (os README.md), que sao texto corrido em
portugues, com acentuacao.

A licao pratica: um tokenizador so' comprime bem o que ele viu no treino. APIs de
LLM cobram POR TOKEN -- se o tokenizador foi treinado majoritariamente em ingles,
escrever em portugues custa mais caro pelo mesmo conteudo.

Run (a partir da pasta do capitulo):
    python solucoes/e5_bpe_portugues.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bpe import BPETokenizer

VOCAB_SIZE = 512
CAP = Path(__file__).resolve().parent.parent
CURSO = CAP.parent.parent          # raiz do repositorio

# ---------------------------------------------------------------------------
# Corpus (a): nomes sem acento.
# ---------------------------------------------------------------------------
nomes = (CAP / "names.txt").read_text(encoding="utf-8")[:150_000]

# ---------------------------------------------------------------------------
# Corpus (b): portugues com acentos -- as apostilas do proprio curso.
# ---------------------------------------------------------------------------
partes = []
for md in sorted(CURSO.glob("chapters/*/README.md")) + sorted(CURSO.glob("docs/*.md")):
    partes.append(md.read_text(encoding="utf-8"))
portugues = "\n".join(partes)
print(f"corpus portugues: {len(portugues)} caracteres, de {len(partes)} arquivos")
if len(portugues) < 20_000:
    print("AVISO: corpus pequeno; os numeros ficam menos representativos.")

# ---------------------------------------------------------------------------
# Treina os dois.
# ---------------------------------------------------------------------------
print("\ntreinando os dois tokenizadores (aguarde ~20s)...")
tok_nomes = BPETokenizer()
tok_nomes.train(nomes, VOCAB_SIZE)

tok_pt = BPETokenizer()
tok_pt.train(portugues[:150_000], VOCAB_SIZE)

# ---------------------------------------------------------------------------
# 1. O tokenizador portugues aprendeu pedacos com acento?
# ---------------------------------------------------------------------------
print("\n=== 1. tokens COM ACENTO aprendidos pelo tokenizador portugues ===")
com_acento = []
for i, b in tok_pt.vocab.items():
    if i < 256:
        continue
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if any(c in s for c in "áàâãéêíóôõúüç"):
        com_acento.append((i, s))
com_acento.sort(key=lambda kv: -len(kv[1]))
if com_acento:
    for i, s in com_acento[:15]:
        print(f"  token {i:4d} = {s!r}")
    print(f"  (total: {len(com_acento)} tokens com acento)")
else:
    print("  nenhum -- aumente VOCAB_SIZE ou o corpus")

print("\n  no tokenizador de NOMES (sem acento), tokens com acento:", end=" ")
n_ac = 0
for i, b in tok_nomes.vocab.items():
    if i < 256:
        continue
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        continue
    if any(c in s for c in "áàâãéêíóôõúüç"):
        n_ac += 1
print(n_ac)

# ---------------------------------------------------------------------------
# 2. Comparacao na mesma frase.
# ---------------------------------------------------------------------------
frases = [
    "A informação sobre a ação de coração está na página.",
    "A função de ativação é aplicada em cada posição.",
    "O modelo de linguagem prevê o próximo token.",
]

print("\n=== 2. tokens gastos na mesma frase ===")
print(f"  {'frase':52s} {'bytes':>6s} {'nomes':>7s} {'pt':>5s} {'ganho':>7s}")
tot_n = tot_p = 0
for f in frases:
    n = len(tok_nomes.encode(f))
    p = len(tok_pt.encode(f))
    tot_n += n
    tot_p += p
    print(f"  {f[:50]:52s} {len(f.encode('utf-8')):6d} {n:7d} {p:5d} {(n - p) / n:6.0%}")

print(f"\n  TOTAL: {tot_n} tokens (treinado em nomes) vs {tot_p} (treinado em portugues)")
print(f"  o tokenizador adequado ao idioma gasta {(tot_n - tot_p) / tot_n:.0%} menos tokens")

# ---------------------------------------------------------------------------
# 3. A frase fatiada pelos dois, lado a lado.
# ---------------------------------------------------------------------------
def mostrar(tok, tid):
    bs = tok.vocab[tid]
    try:
        return bs.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{b:02X}>" for b in bs)


frase = frases[0]
print("\n=== 3. a mesma frase, fatiada pelos dois ===")
for nome, tok in (("nomes  ", tok_nomes), ("portugues", tok_pt)):
    ids = tok.encode(frase)
    print(f"\n  [{nome}] {len(ids)} tokens:")
    print("   " + " | ".join(mostrar(tok, i) for i in ids))

print()
print("Conclusao: tokenizador e idioma andam juntos. O mesmo texto custa mais")
print("tokens -- logo mais dinheiro e mais contexto gasto -- quando o tokenizador")
print("nao foi treinado no idioma em que voce escreve.")
