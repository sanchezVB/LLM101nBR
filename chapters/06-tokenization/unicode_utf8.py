"""
Unicode e UTF-8 — o chao onde a tokenizacao pisa.

Antes de construir um tokenizador, precisamos entender como texto vira numeros.
Este arquivo mostra os tres niveis:

    caractere  ->  code point (Unicode)  ->  bytes (UTF-8)

E responde a pergunta pratica do capitulo: por que um tokenizador serio comeca
pelos BYTES, e nao pelos caracteres.

Run:
    python unicode_utf8.py
"""

import sys

# O terminal do Windows usa cp1252 por padrao e QUEBRA ao imprimir caracteres
# fora dessa tabela (japones, emoji...). Forcamos UTF-8 na saida para o script
# rodar em qualquer plataforma. E' o primeiro exemplo pratico do capitulo: a
# codificacao nao e' um detalhe, ela decide o que o seu programa consegue fazer.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. Code point: o numero que o Unicode atribui a cada caractere.
# ---------------------------------------------------------------------------
print("=== 1. code points (Unicode) ===")
for ch in ["a", "z", "A", "0", "e", "é", "ã", "ç", "€", "日", "🚀"]:
    print(f"  {ch!r:6s} -> code point {ord(ch):7d}  (U+{ord(ch):04X})")

print(f"\n  chr(97) = {chr(97)!r}   chr(233) = {chr(233)!r}   chr(128640) = {chr(128640)!r}")
print("  O Unicode define ~150 mil caracteres. Usar um token por caractere daria")
print("  um vocabulario gigantesco -- e ainda assim incompleto.")

# ---------------------------------------------------------------------------
# 2. UTF-8: como o code point e' gravado em bytes.
#    Tamanho variavel: 1 byte para ASCII, 2 para acentuados, 3-4 para o resto.
# ---------------------------------------------------------------------------
print("\n=== 2. UTF-8 (code point -> bytes) ===")
for ch in ["a", "é", "ã", "€", "日", "🚀"]:
    bs = ch.encode("utf-8")
    lista = " ".join(str(b) for b in bs)
    print(f"  {ch!r:6s} -> {len(bs)} byte(s): [{lista}]")

# ---------------------------------------------------------------------------
# 3. O IMPOSTO DO PORTUGUES.
#    Caracteres acentuados custam 2 bytes. Num modelo que opera sobre bytes,
#    texto em portugues ocupa mais espaco que o equivalente em ingles.
# ---------------------------------------------------------------------------
print("\n=== 3. o custo dos acentos em portugues ===")
pares = [
    ("jose", "josé"),
    ("acao", "ação"),
    ("informacao", "informação"),
    ("coracao", "coração"),
]
for sem, com in pares:
    b_sem, b_com = len(sem.encode("utf-8")), len(com.encode("utf-8"))
    print(f"  {sem:12s} {b_sem:2d} bytes  |  {com:12s} {b_com:2d} bytes  (+{b_com-b_sem})")

frase = "A ação começa às três horas."
print(f"\n  frase: {frase!r}")
print(f"  {len(frase)} caracteres, mas {len(frase.encode('utf-8'))} bytes em UTF-8")

# ---------------------------------------------------------------------------
# 4. Por que comecar pelos bytes?
#    Porque existem apenas 256 valores possiveis de byte. Isso da' um
#    vocabulario BASE pequeno e, principalmente, COMPLETO: qualquer texto de
#    qualquer idioma (ou emoji, ou codigo) e' representavel. Nunca ha' um
#    caractere "desconhecido".
# ---------------------------------------------------------------------------
print("\n=== 4. bytes: vocabulario base de 256, sem 'desconhecidos' ===")
texto = "Olá! 日本 🚀"
bs = texto.encode("utf-8")
print(f"  texto : {texto!r}")
print(f"  bytes : {list(bs)}")
print(f"  volta : {bs.decode('utf-8')!r}")
print(f"  todos os bytes estao em 0..255? {all(0 <= b <= 255 for b in bs)}")

# ---------------------------------------------------------------------------
# 5. O problema que sobra: sequencias longas.
#    Um token por byte funciona, mas gasta muitos tokens. E' isso que o BPE
#    (proximo arquivo) resolve, juntando os pares mais frequentes.
# ---------------------------------------------------------------------------
print("\n=== 5. o problema que o BPE vai resolver ===")
exemplo = "maria eduarda"
print(f"  {exemplo!r}")
print(f"  como bytes  : {len(exemplo.encode('utf-8'))} tokens")
print("  Sequencias longas custam caro: a atencao e' O(T^2) no tamanho do")
print("  contexto (Cap. 4). Menos tokens = contexto efetivo maior e treino mais")
print("  barato. O BPE junta pares frequentes ('ma', 'ria', ...) em um token so'.")

# ---------------------------------------------------------------------------
# 6. Cuidado: decodificar bytes arbitrarios pode falhar.
#    Nem toda sequencia de bytes e' UTF-8 valido. O tokenizador precisa lidar
#    com isso (usamos errors="replace" no decode do bpe.py).
# ---------------------------------------------------------------------------
print("\n=== 6. nem todo byte solto e' UTF-8 valido ===")
ruim = bytes([0x80])          # byte de continuacao sem inicio: invalido
try:
    ruim.decode("utf-8")
except UnicodeDecodeError as e:
    print(f"  bytes([0x80]).decode('utf-8') -> UnicodeDecodeError: {e.reason}")
print(f"  com errors='replace': {ruim.decode('utf-8', errors='replace')!r}")
print("  Por isso o decode do nosso tokenizador usa errors='replace': um token")
print("  pode conter meio caractere, e o programa nao pode quebrar por isso.")
