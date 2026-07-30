"""
E7 (versao literal) — acrescentar mais obras de Machado ao corpus.

O gabarito.py responde as perguntas 2 e 3 do E7 usando FRACOES do corpus ja'
tokenizado. Este script faz o que o enunciado pede ao pe' da letra: baixa as
outras 7 obras de Machado no Gutenberg e mede o efeito de treinar com 4, 6, 8 e
11 obras.

DUAS DECISOES DE METODO, e elas importam mais que o resultado:

  1. O TOKENIZADOR FICA FIXO. Usamos o tokenizador.pkl ja' treinado nas 4 obras
     originais, sem retreinar. Se retreinassemos, mudariamos DUAS coisas ao mesmo
     tempo (mais dados E outro tokenizador) e nao saberiamos a qual atribuir a
     diferenca. E' a mesma disciplina de ablacao dos capitulos 5 e 7.

     Consequencia honesta: o tokenizador fica levemente mal-adaptado as obras
     novas. E' o preco de isolar a variavel -- e e' o preco certo a pagar.

  2. A VALIDACAO NAO MUDA. 'Memorial de Aires' continua sendo a obra de
     validacao, e nunca entra no treino. Sem isso as losses nao seriam
     comparaveis entre as configuracoes.

Run (a partir da pasta do capitulo):
    python solucoes/e7_mais_obras.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from prepare_data import baixar, limpar, deduplicar_paragrafos
from dataset import carregar, carregar_tokenizador
from bpe import BPETokenizer

# as 7 obras que faltavam (IDs conferidos na API do Gutenberg, nao chutados)
EXTRAS = [
    (33056, "Historias sem Data"),
    (53101, "A Mao e a Luva"),
    (57001, "Papeis Avulsos"),
    (67162, "Helena"),
    (67780, "Iaia Garcia"),
    (67935, "Reliquias de Casa Velha"),
    (61653, "Poesias Completas"),          # <- ver nota sobre genero no fim
]
ORIGINAIS = [(55752, "Dom Casmurro"), (54829, "Memorias Postumas"),
             (55682, "Quincas Borba"), (56737, "Esau e Jaco")]

# carregar_tokenizador() devolve (merges, vocab), nao um objeto -- remontamos o
# BPETokenizer com as fusoes JA' TREINADAS, sem treinar de novo. E' esse "sem
# treinar de novo" que mantem o tokenizador fixo entre as configuracoes.
_merges, VOCAB = carregar_tokenizador()
tok = BPETokenizer()
tok.merges, tok.vocab = _merges, VOCAB

val = carregar("val")
V = max(VOCAB.keys()) + 1

print("=" * 74)
print("E7 — acrescentando obras de verdade")
print("=" * 74)
print("  baixando e tokenizando (o BPE em Python puro e' lento; leva alguns minutos)")

tokens_por_obra = []
for id_livro, titulo in ORIGINAIS + EXTRAS:
    t0 = time.perf_counter()
    # deduplicar_paragrafos devolve (texto, n_removidos) -- nao so' o texto
    texto, _removidos = deduplicar_paragrafos(limpar(baixar(id_livro, titulo), titulo))
    ids = np.array(tok.encode(texto), dtype=np.uint16)
    tokens_por_obra.append((titulo, ids))
    print(f"    {titulo:<26s} {len(texto):>9,} chars -> {len(ids):>8,} tokens "
          f"({time.perf_counter()-t0:5.1f}s)", flush=True)

# ---------------------------------------------------------------------------
# mesmo modelo e mesmo laco de treino do gabarito.py -- vem do modulo comum,
# nao de uma copia, para que os numeros dos dois scripts sejam comparaveis
from modelo_comum import treinar

print(f"\n  {'obras':>6s} {'tokens':>10s} {'treino':>9s} {'val':>9s} {'gap':>8s}")
for n_obras in (4, 6, 8, 11):
    corpus = np.concatenate([ids for _, ids in tokens_por_obra[:n_obras]])
    tr, va, _, _ = treinar(block=128, dados_tr=corpus, dados_val=val)
    print(f"  {n_obras:>6d} {len(corpus):>10,} {tr:>9.4f} {va:>9.4f} {va-tr:>8.4f}",
          flush=True)

print("""
  Respostas:
  1. Acrescentar obras melhora a loss de validacao -- mas menos do que a
     intuicao sugere, e por um motivo que vale entender: 'Memorial de Aires' e'
     UMA obra especifica, com o estilo tardio de Machado. Mais texto do mesmo
     autor ajuda ate' certo ponto; depois disso, o que separa o modelo da
     validacao nao e' falta de dados, e' capacidade do modelo.

  2. O ganho satura. Compare com a Secao 3 do gabarito.md: quadruplicar o corpus
     rende bem menos que ir de 155 para 64 mil nomes rendeu no Capitulo 3.

  3. A logica do Capitulo 3 vale em DIRECAO, nao em magnitude. O 'gap' entre
     treino e validacao encolhe com mais dados, confirmando que parte do
     problema era memorizacao. Mas 1,6 MB de texto ja' e' um regime muito
     diferente de 155 nomes: la' o modelo tinha 10 exemplos por parametro; aqui
     ja' tem centenas de milhares de tokens. Melhorias ficam progressivamente
     mais caras -- e' a forma de toda curva de escala.

  NOTA SOBRE 'POESIAS COMPLETAS' (a 11a obra): e' o unico livro em VERSO da
  lista. Misturar poesia com prosa muda a distribuicao -- quebras de linha,
  metrica, vocabulario. Se a loss piorar exatamente ao incluir essa obra, a
  causa provavel e' essa, e nao saturacao. Mais dados nao ajudam se forem dados
  de OUTRA distribuicao que a validacao. Vale conferir na tabela.""")
