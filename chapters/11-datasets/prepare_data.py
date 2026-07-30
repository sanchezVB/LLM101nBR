"""
prepare_data.py — do texto bruto ate os tokens prontos para treinar.

Este e' o pipeline que todo LLM tem, e quase nunca aparece nos tutoriais:

    baixar -> limpar -> DIVIDIR (treino/val) -> tokenizar -> gravar em binario

A ordem importa, e dois passos sao onde a maioria erra:

  1. DIVIDIR ANTES DE TOKENIZAR. O tokenizador e' treinado nos dados -- se ele
     ver o texto de validacao, ele aprende o vocabulario de la', e a avaliacao
     fica otimista. E' vazamento (leakage) sutil, mas e' vazamento.

  2. DIVIDIR POR DOCUMENTO, nao por linha sorteada. Se voce embaralhar as linhas
     de um livro e mandar 10% para validacao, o modelo vai ver o capitulo
     inteiro no treino e ser avaliado em frases do MESMO capitulo -- praticamente
     as mesmas palavras, o mesmo assunto, o mesmo estilo. A loss de validacao
     fica boa e nao significa nada.

Corpus: obras de Machado de Assis (1839-1908), em dominio publico, obtidas do
Project Gutenberg. Escolha deliberada: e' portugues literario de verdade, e a
licenca e' limpa.

Run:
    python prepare_data.py
"""

import os
import re
import time
import urllib.request
from pathlib import Path

import numpy as np

from bpe import BPETokenizer

AQUI = Path(__file__).parent
BRUTOS = AQUI / "_brutos"
VOCAB_SIZE = 1024
BYTES_TREINO_TOKENIZADOR = 300_000     # BPE em Python puro e' lento; treina num subconjunto

# (id no Gutenberg, titulo). Os quatro primeiros vao para TREINO, o ultimo para
# VALIDACAO -- divisao por OBRA, nao por linha.
LIVROS = [
    (55752, "Dom Casmurro"),
    (54829, "Memorias Postumas de Bras Cubas"),
    (55682, "Quincas Borba"),
    (56737, "Esau e Jaco"),
    (55797, "Memorial de Aires"),          # <- validacao
]


def baixar(id_livro, titulo):
    """Baixa uma obra, com cache em disco (nao rebaixa se ja' existe).

    O cache usa I/O BINARIO de proposito. Se usassemos write_text/read_text, o
    Windows traduziria as quebras de linha: o texto baixado ja' vem com \\r\\n, o
    write_text transformaria cada \\n em \\r\\n, e o arquivo ficaria com \\r\\r\\n.
    O resultado seria um corpus diferente na segunda execucao -- um bug silencioso
    e chato de achar (aconteceu aqui: 4220 ocorrencias no primeiro cache).
    Bytes crus nao sofrem traducao nenhuma.
    """
    BRUTOS.mkdir(exist_ok=True)
    destino = BRUTOS / f"{id_livro}.txt"
    if destino.exists():
        return destino.read_bytes().decode("utf-8", errors="replace")

    url = f"https://www.gutenberg.org/cache/epub/{id_livro}/pg{id_livro}.txt"
    print(f"  baixando '{titulo}' ({url})...")
    req = urllib.request.Request(url, headers={"User-Agent": "llm101n-br (material didatico)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        bruto = r.read()
    destino.write_bytes(bruto)
    return bruto.decode("utf-8", errors="replace")


def limpar(texto, titulo):
    """Remove o cabecalho/rodape do Gutenberg e normaliza o espacamento.

    Os arquivos do Gutenberg vem com licenca, creditos e metadados em ingles.
    Deixar isso no corpus ensinaria o modelo a escrever termos de licenca em
    ingles -- lixo que compete com o texto que interessa.
    """
    inicio = re.search(r"\*\*\*\s*START OF TH[EIS] PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    fim = re.search(r"\*\*\*\s*END OF TH[EIS] PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    if inicio:
        texto = texto[inicio.end():]
    if fim:
        # o 'fim' foi encontrado no texto original; refaz a busca no texto ja' cortado
        fim2 = re.search(r"\*\*\*\s*END OF TH[EIS] PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
        if fim2:
            texto = texto[:fim2.start()]

    # normaliza quebras de linha de qualquer origem (\r\n do Windows, \r solto do
    # Mac antigo) ANTES de qualquer regex que dependa de \n
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r"\n{3,}", "\n\n", texto)      # no maximo uma linha em branco
    texto = re.sub(r"[ \t]+", " ", texto)          # espacos repetidos
    texto = "\n".join(linha.strip() for linha in texto.split("\n"))
    return texto.strip()


def deduplicar_paragrafos(texto):
    """Remove paragrafos repetidos, preservando a ordem.

    Corpus real tem repeticao: cabecalhos, sumarios, trechos citados. Texto
    duplicado faz o modelo MEMORIZAR em vez de generalizar, e ainda contamina a
    avaliacao se a mesma passagem cair nos dois lados da divisao.
    """
    vistos = set()
    saida = []
    removidos = 0
    for par in texto.split("\n\n"):
        chave = par.strip().lower()
        if len(chave) < 40:            # paragrafos curtos repetem legitimamente
            saida.append(par)
            continue
        if chave in vistos:
            removidos += 1
            continue
        vistos.add(chave)
        saida.append(par)
    return "\n\n".join(saida), removidos


def main():
    # -----------------------------------------------------------------------
    print("=== 1. baixando e limpando ===")
    textos = {}
    for id_livro, titulo in LIVROS:
        bruto = baixar(id_livro, titulo)
        limpo = limpar(bruto, titulo)
        textos[titulo] = limpo
        print(f"  {titulo:36s} {len(bruto):>8d} -> {len(limpo):>8d} chars "
              f"({100*(1-len(limpo)/len(bruto)):4.1f}% removido)")

    # -----------------------------------------------------------------------
    print("\n=== 2. divisao por OBRA (sem vazamento) ===")
    titulos_treino = [t for _, t in LIVROS[:-1]]
    titulo_val = LIVROS[-1][1]

    texto_treino = "\n\n".join(textos[t] for t in titulos_treino)
    texto_val = textos[titulo_val]

    texto_treino, dup_tr = deduplicar_paragrafos(texto_treino)
    texto_val, dup_val = deduplicar_paragrafos(texto_val)

    print(f"  treino: {len(titulos_treino)} obras, {len(texto_treino):>8d} chars "
          f"({dup_tr} paragrafos duplicados removidos)")
    print(f"  val   : '{titulo_val}', {len(texto_val):>8d} chars "
          f"({dup_val} duplicados removidos)")
    print(f"  proporcao: {100*len(texto_val)/(len(texto_treino)+len(texto_val)):.1f}% para validacao")

    # -----------------------------------------------------------------------
    print(f"\n=== 3. treinando o tokenizador (vocab={VOCAB_SIZE}) ===")
    print(f"  IMPORTANTE: treinado SO' no texto de treino, nunca no de validacao")
    t0 = time.perf_counter()
    tok = BPETokenizer()
    tok.train(texto_treino[:BYTES_TREINO_TOKENIZADOR], VOCAB_SIZE)
    print(f"  pronto em {time.perf_counter()-t0:.0f}s "
          f"(treinado em {BYTES_TREINO_TOKENIZADOR} chars, por velocidade)")

    aprendidos = [(i, b) for i, b in tok.vocab.items() if i >= 256]
    aprendidos.sort(key=lambda kv: -len(kv[1]))
    amostra = []
    for i, b in aprendidos[:12]:
        try:
            amostra.append(repr(b.decode("utf-8")))
        except UnicodeDecodeError:
            pass
    print(f"  tokens mais longos: {', '.join(amostra[:10])}")

    # -----------------------------------------------------------------------
    print("\n=== 4. tokenizando (paragrafo a paragrafo, por velocidade) ===")

    def tokenizar(texto, nome):
        t0 = time.perf_counter()
        ids = []
        pars = texto.split("\n\n")
        for k, par in enumerate(pars):
            ids.extend(tok.encode(par + "\n\n"))
            if k % 500 == 0 and k:
                print(f"    {nome}: {k}/{len(pars)} paragrafos...", flush=True)
        print(f"  {nome}: {len(texto)} chars -> {len(ids)} tokens "
              f"({len(texto.encode('utf-8'))/len(ids):.2f}x compressao) "
              f"em {time.perf_counter()-t0:.0f}s")
        return ids

    ids_treino = tokenizar(texto_treino, "treino")
    ids_val = tokenizar(texto_val, "val")

    # -----------------------------------------------------------------------
    print("\n=== 5. gravando em binario (uint16) ===")
    # uint16 basta para vocab <= 65536, e ocupa METADE do uint32. Gravar em
    # binario permite ler com memmap depois -- sem carregar tudo na RAM.
    assert VOCAB_SIZE <= 65536, "uint16 nao serve para vocabularios maiores"
    for nome, ids in (("treino", ids_treino), ("val", ids_val)):
        arr = np.array(ids, dtype=np.uint16)
        caminho = AQUI / f"{nome}.bin"
        arr.tofile(caminho)
        print(f"  {caminho.name}: {len(arr)} tokens, {caminho.stat().st_size/1e6:.2f} MB")

    # o vocabulario precisa ser salvo junto -- sem ele os tokens nao viram texto
    import pickle
    with open(AQUI / "tokenizador.pkl", "wb") as f:
        pickle.dump({"merges": tok.merges, "vocab": tok.vocab}, f)
    print(f"  tokenizador.pkl: {len(tok.merges)} fusoes")

    print("""
=== pronto ===
  train.bin / val.bin guardam os tokens como uint16 puro. E' o formato que o
  nanoGPT usa, e a razao e' pratica: da' para abrir com np.memmap e ler pedacos
  aleatorios SEM carregar o arquivo inteiro na memoria. Com corpus de gigabytes,
  isso deixa de ser otimizacao e vira requisito.""")


if __name__ == "__main__":
    main()
