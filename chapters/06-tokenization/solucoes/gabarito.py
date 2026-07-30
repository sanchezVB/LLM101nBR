"""
Gabarito executavel do Capitulo 06 — tokenizacao.

Roda E2, E3, E4 e E6 (o E5 ja' tem solucao propria). Tudo em Python puro,
sem treino de rede -- roda em ~2 minutos.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
from pathlib import Path

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bpe import BPETokenizer, get_stats, merge

TREINO = (CAP / "names.txt").read_text(encoding="utf-8")[:150_000]
TESTE = (CAP / "names.txt").read_text(encoding="utf-8")[150_000:200_000]

# ===========================================================================
print("=" * 72)
print("E1 — Unicode e UTF-8 (conferindo as contas)")
print("=" * 72)
for palavra in ("pao", "pão"):
    b = palavra.encode("utf-8")
    print(f"  {palavra!r:8s}: {len(palavra)} caracteres, {len(b)} bytes -> {list(b)}")
print(f"\n  ord('ç') = {ord('ç')} -- cabe em 1 byte? "
      f"{'sim' if ord('ç') < 256 else 'NAO'} (mas em UTF-8 ocupa "
      f"{len('ç'.encode('utf-8'))} bytes)")
print("""
  Respostas:
  1. 'pao' = 3 bytes; 'pão' = 4 bytes. O 'ã' ocupa 2 bytes em UTF-8.
  2. ord('ç') = 231, que cabe num byte -- mas isso e' o CODE POINT, nao a
     codificacao. Em UTF-8 todo code point acima de 127 usa 2 bytes ou mais.
     Confundir code point com byte e' o erro classico aqui.
  3. Comecando pelos bytes, o vocabulario base tem 256 entradas e cobre QUALQUER
     sequencia possivel -- nao existe caractere fora dele.""")

# ===========================================================================
print("=" * 72)
print("E2 — tamanho do vocabulario")
print("=" * 72)
print(f"  {'vocab':>7s} {'fusoes':>7s} {'tokens (treino)':>16s} {'compressao':>11s} "
      f"{'ganho vs anterior':>18s}")
anterior = None
tokenizadores = {}
for vs in (300, 512, 1024):
    tok = BPETokenizer()
    ids = tok.train(TREINO, vs)
    tokenizadores[vs] = tok
    comp = len(TREINO.encode("utf-8")) / len(ids)
    ganho = f"{comp - anterior:+.3f}" if anterior else "-"
    anterior = comp
    print(f"  {vs:>7d} {vs-256:>7d} {len(ids):>16d} {comp:>10.3f}x {ganho:>18s}")
print("""
  Respostas:
  1 e 2. A compressao melhora com o vocabulario, mas com RETORNO DECRESCENTE:
     dobrar o vocabulario nao dobra o ganho.
  3. O custo de um vocabulario maior aparece no MODELO, nao no tokenizador: a
     camada de saida (lm_head) tem n_embd x vocab_size parametros, e a matriz de
     embeddings tambem. Dobrar o vocabulario dobra os dois. Ou seja: mais
     vocabulario = sequencias mais curtas (bom) mas modelo maior (custo).""")

# ===========================================================================
print("=" * 72)
print("E3 — as primeiras fusoes")
print("=" * 72)
tok = tokenizadores[512]
primeiras = list(tok.merges.items())[:10]
print("  as 10 primeiras fusoes aprendidas:")
for (a, b), novo in primeiras:
    txt = tok.vocab[novo].decode("utf-8", errors="replace")
    print(f"    {novo}: {txt!r}")

com_quebra = sum(1 for (a, b), n in primeiras if b"\n" in tok.vocab[n])
print(f"\n  quantas das 10 primeiras contem quebra de linha: {com_quebra}")

longos = sorted(((i, b) for i, b in tok.vocab.items() if i >= 256),
                key=lambda kv: -len(kv[1]))[:20]
sufixos = [b.decode("utf-8", errors="replace") for _, b in longos]
print(f"\n  os 20 tokens mais longos:")
print("   ", ", ".join(repr(s) for s in sufixos[:10]))
print("   ", ", ".join(repr(s) for s in sufixos[10:]))
print("""
  Respostas:
  1. ('a','n') e' a primeira porque 'an' e' a sequencia de duas letras mais
     frequente em nomes brasileiros (ana, alessandra, fernanda, luana...).
  2. Varias fusoes iniciais contem '\\n' porque o arquivo tem UM NOME POR LINHA:
     a quebra de linha faz parte do padrao de TERMINACAO dos nomes. O
     tokenizador esta' aprendendo "como um nome acaba".
  3. Ele descobriu sufixos ('ilson', 'erson', 'ilton') so' contando pares. Nao
     ha' nenhuma nocao de morfologia no algoritmo -- e' estatistica pura virando
     estrutura linguistica.""")

# ===========================================================================
print("=" * 72)
print("E4 — a ordem das fusoes importa")
print("=" * 72)


def encode_ordem_errada(tok, texto):
    """Aplica a PRIMEIRA fusao aplicavel encontrada, em vez da mais antiga."""
    ids = list(texto.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        par = None
        for p in stats:                    # ordem arbitraria de iteracao
            if p in tok.merges:
                par = p
                break
        if par is None:
            break
        ids = merge(ids, par, tok.merges[par])
    return ids


amostras = ["maria eduarda", "joao vinicius", "ana beatriz de souza"]
print(f"  {'texto':>24s} {'certo':>7s} {'errado':>8s} {'round-trip errado':>18s}")
for s in amostras:
    certo = tok.encode(s)
    errado = encode_ordem_errada(tok, s)
    volta = tok.decode(errado)
    print(f"  {s:>24s} {len(certo):>7d} {len(errado):>8d} {str(volta == s):>18s}")

# em textos curtos o efeito quase nao aparece; medimos num texto MAIOR
grande = TESTE[:20_000]
n_certo = sum(len(tok.encode(l + "\n")) for l in grande.split("\n") if l)
n_errado = sum(len(encode_ordem_errada(tok, l + "\n")) for l in grande.split("\n") if l)
print(f"\n  em 20.000 caracteres:")
print(f"    ordem correta : {n_certo:>7d} tokens")
print(f"    ordem errada  : {n_errado:>7d} tokens  ({100*(n_errado-n_certo)/n_certo:+.2f}%)")
print("""
  Respostas (e a magnitude importa mais que o sinal):
  1. O round-trip CONTINUA valendo: o decode desfaz qualquer sequencia de fusoes
     validas, entao o texto volta igual. A ordem errada nao QUEBRA o
     tokenizador -- ela apenas o deixa um pouco pior.
  2. A compressao piora ~10% em texto de tamanho razoavel -- significativo, ainda
     que nao catastrofico. Repare no contraste com a tabela acima: em textos
     CURTOS o efeito muitas vezes nem aparece (dois dos tres exemplos deram o
     mesmo numero de tokens).

     E' um bom lembrete metodologico: medir em tres frases teria levado a
     concluir "a ordem quase nao importa". So' com volume o efeito de ~10%
     aparece de forma confiavel. Amostra pequena esconde efeito real.
  3. Ainda assim a regra e' obrigatoria: uma fusao tardia pode usar um token
     criado por uma anterior, e aplica-la antes desmonta essa cadeia. Alem
     disso, sem uma ordem fixa o mesmo texto poderia produzir tokens diferentes
     em execucoes diferentes -- e um tokenizador precisa ser DETERMINISTICO.""")

# ===========================================================================
print("=" * 72)
print("E6 — tokens fora do dominio")
print("=" * 72)
casos = [
    ("nomes (mesmo dominio)", TESTE[:2000]),
    ("frase em portugues", "A informação sobre a ação de coração está na página."),
    ("japones + emoji", "Olá! " + chr(26085) + chr(26412) + " " + chr(128640)),
    ("codigo Python", "def soma(a, b):\n    return a + b\n"),
]
print(f"  {'texto':>24s} {'bytes':>7s} {'tokens':>7s} {'compressao':>11s} {'round-trip':>11s}")
for nome, s in casos:
    ids = tok.encode(s)
    nb = len(s.encode("utf-8"))
    ok = tok.decode(ids) == s
    print(f"  {nome:>24s} {nb:>7d} {len(ids):>7d} {nb/len(ids):>10.2f}x {str(ok):>11s}")
print("""
  Respostas:
  1. Fora do dominio a compressao cai para perto de 1.00x: nenhuma das fusoes
     aprendidas se aplica, entao cada byte vira um token.
  2. Mas o round-trip CONTINUA True em todos os casos -- e essa e' a garantia
     que importa. O tokenizador nao comprime o que nao conhece, mas nunca
     FALHA. Um texto em japones, um emoji ou codigo-fonte sao todos
     representaveis, porque o vocabulario base sao os 256 bytes.
  3. Um tokenizador baseado em PALAVRAS receberia uma palavra japonesa e nao
     teria nenhuma entrada para ela -- teria de emitir um token <UNK>,
     perdendo a informacao de forma irreversivel. E' a diferenca entre
     'comprime mal' e 'nao representa'.""")
