"""
BPE (Byte Pair Encoding) — um tokenizador do zero.

O algoritmo, em uma frase: comece com um token por BYTE (256 tokens) e, repetidas
vezes, encontre o par de tokens vizinhos mais frequente e funda-o num token novo.

    "aa ab aa ab"  -> o par ('a','a') e' frequente -> vira o token X
    "X ab X ab"    -> o par ('a','b') e' frequente -> vira o token Y
    "X Y X Y"      -> e assim por diante

Isso da' o melhor dos dois mundos: o vocabulario base de bytes garante que
QUALQUER texto e' representavel (nunca ha' "desconhecido"), e as fusoes encurtam
as sequencias, deixando o contexto do modelo render mais.

E' o mesmo algoritmo usado pelo GPT-2/GPT-4 (com refinamentos).

Run:
    python bpe.py
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VOCAB_SIZE = 512          # 256 bytes + 256 fusoes aprendidas
TRAIN_BYTES = 150_000     # subconjunto do dataset (BPE em Python puro e' lento)


# ---------------------------------------------------------------------------
# As duas funcoes que fazem todo o trabalho.
# ---------------------------------------------------------------------------
def get_stats(ids):
    """Conta quantas vezes cada par de tokens vizinhos aparece."""
    counts = {}
    for pair in zip(ids, ids[1:]):        # todos os pares vizinhos
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    """Substitui toda ocorrencia de `pair` pelo token `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        # se achamos o par nesta posicao, escreve o token novo e SALTA dois
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        self.merges = {}      # (id_a, id_b) -> id_novo   (a ordem importa!)
        self.vocab = {}       # id -> bytes

    # -----------------------------------------------------------------------
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))       # ponto de partida: bytes crus
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for k in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break
            # o par mais frequente e' o proximo a ser fundido (o coracao do BPE)
            pair = max(stats, key=stats.get)
            new_id = 256 + k
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            # o token novo "significa" a concatenacao dos bytes dos dois pais
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and k < 10:
                trecho = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"    fusao {k + 1:3d}: {pair} -> {new_id}  ({trecho!r}), {stats[pair]}x")
        return ids

    # -----------------------------------------------------------------------
    def encode(self, text):
        """texto -> lista de ids."""
        ids = list(text.encode("utf-8"))
        # Aplica as fusoes na MESMA ORDEM em que foram aprendidas. Isso e'
        # essencial: uma fusao tardia pode depender de tokens criados antes.
        while len(ids) >= 2:
            stats = get_stats(ids)
            # entre os pares presentes, escolhe o que foi aprendido mais cedo
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                      # nao ha' mais nada para fundir
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        """lista de ids -> texto."""
        bs = b"".join(self.vocab[i] for i in ids)
        # errors="replace": um token pode conter meio caractere multibyte
        return bs.decode("utf-8", errors="replace")


# ===========================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # 1. Treino sobre os nomes.
    # -----------------------------------------------------------------------
    # Este arquivo e' o mesmo do Capitulo 6, e a demo dele usava names.txt.
    # O Capitulo 11 nao tem names.txt (o corpus aqui e' Machado), entao a demo
    # busca o do Capitulo 6. Se nem ele existir, avisa em vez de estourar.
    nomes = Path(__file__).parent.parent / "06-tokenization" / "names.txt"
    if not nomes.exists():
        raise SystemExit(
            f"Demo do BPE precisa de {nomes}, que nao foi encontrado.\n"
            f"Para usar o tokenizador ja' treinado neste capitulo:\n"
            f"    python -c \"from dataset import carregar_tokenizador; "
            f"print(len(carregar_tokenizador()[0]))\""
        )
    texto_completo = nomes.read_text(encoding="utf-8")
    treino = texto_completo[:TRAIN_BYTES]

    print(f"=== 1. treinando BPE (vocab_size={VOCAB_SIZE}) ===")
    print(f"  texto de treino: {len(treino)} caracteres")
    tok = BPETokenizer()
    ids_final = tok.train(treino, VOCAB_SIZE, verbose=True)

    bytes_orig = len(treino.encode("utf-8"))
    print(f"\n  bytes originais : {bytes_orig}")
    print(f"  tokens depois   : {len(ids_final)}")
    print(f"  taxa de compressao: {bytes_orig / len(ids_final):.2f}x")

    # -----------------------------------------------------------------------
    # 2. O que ele aprendeu? Os tokens mais longos sao os mais reveladores.
    # -----------------------------------------------------------------------
    print("\n=== 2. tokens aprendidos (os 15 mais longos) ===")
    aprendidos = [(i, b) for i, b in tok.vocab.items() if i >= 256]
    aprendidos.sort(key=lambda kv: -len(kv[1]))
    for i, b in aprendidos[:15]:
        print(f"  token {i:4d} = {b.decode('utf-8', errors='replace')!r}")

    # -----------------------------------------------------------------------
    # 3. A propriedade que NAO pode falhar: decode(encode(x)) == x
    # -----------------------------------------------------------------------
    print("\n=== 3. round-trip: decode(encode(x)) == x ? ===")
    casos = [
        "maria eduarda",
        "vinicius",
        "josé da conceição",          # acentos (nao estavam no treino!)
        "A ação começa às três.",
        "Olá! 日本 🚀",                # idioma e emoji fora do treino
        "",                            # caso limite: vazio
        "xyzkw",                       # letras raras
    ]
    todos_ok = True
    for s in casos:
        ids = tok.encode(s)
        volta = tok.decode(ids)
        ok = volta == s
        todos_ok = todos_ok and ok
        n_bytes = len(s.encode("utf-8"))
        taxa = f"{n_bytes / len(ids):.2f}x" if ids else "-"
        print(f"  [{'OK ' if ok else 'ERRO'}] {s!r:26s} {n_bytes:3d} bytes -> {len(ids):3d} tokens ({taxa})")
    print(f"\n  TODOS os round-trips passaram? {todos_ok}")

    # -----------------------------------------------------------------------
    # 4. Compressao em texto NAO visto no treino.
    # -----------------------------------------------------------------------
    print("\n=== 4. compressao fora do treino ===")
    teste = texto_completo[TRAIN_BYTES : TRAIN_BYTES + 50_000]
    ids_teste = tok.encode(teste)
    b_teste = len(teste.encode("utf-8"))
    print(f"  nomes nao vistos : {b_teste} bytes -> {len(ids_teste)} tokens "
          f"({b_teste / len(ids_teste):.2f}x)")

    frase = "A informação sobre a ação de coração está na página."
    ids_frase = tok.encode(frase)
    b_frase = len(frase.encode("utf-8"))
    print(f"  frase em portugues: {b_frase} bytes -> {len(ids_frase)} tokens "
          f"({b_frase / len(ids_frase):.2f}x)")
    print("  A compressao cai fora do dominio de treino -- o tokenizador foi")
    print("  treinado em NOMES, nao em frases. Tokenizador e dados andam juntos.")

    # -----------------------------------------------------------------------
    # 5. Como a frase e' fatiada, token por token.
    #
    #    Tokens que nao formam UTF-8 valido sozinhos aparecem como <0xNN>.
    #    Isso acontece porque um caractere acentuado ocupa 2 bytes e o
    #    tokenizador (treinado em nomes SEM acento) nunca aprendeu a juntar
    #    esses bytes -- entao cada um vira um token separado, que isolado nao
    #    e' um caractere. E' o mesmo motivo do errors="replace" no decode.
    # -----------------------------------------------------------------------
    print("\n=== 5. a frase fatiada em tokens ===")

    def mostrar(token_id):
        bs = tok.vocab[token_id]
        try:
            return bs.decode("utf-8")            # token "limpo"
        except UnicodeDecodeError:
            return "".join(f"<0x{b:02X}>" for b in bs)   # fragmento de byte

    print("  " + " | ".join(mostrar(i) for i in ids_frase))
    quebrados = sum(1 for i in ids_frase if mostrar(i).startswith("<0x"))
    print(f"\n  {quebrados} dos {len(ids_frase)} tokens sao fragmentos de bytes (<0xNN>),")
    print("  todos vindos de caracteres acentuados. Um tokenizador treinado em")
    print("  portugues com acentos aprenderia 'ção' como um token so'.")
