"""
Gabarito executavel do Capitulo 11 — datasets.

Roda E3, E4, E5, E6 e E7 (o E2 ja' tem solucao propria).

ORCAMENTO: usamos a MESMA arquitetura da apostila (n_embd=192, 6 cabecas, 4
blocos, 2.2 M params) com menos PASSOS -- 400 em vez de 3.000. Reduzir passos e'
defensavel para perguntas comparativas; trocar a arquitetura nao seria, porque
mudaria o objeto em estudo.

ATENCAO A UMA LIMITACAO REAL: para exercicios que dependem da DINAMICA DE
OTIMIZACAO (melhor learning rate, warmup, agendamento), o orcamento reduzido pode
mudar a RESPOSTA, nao so' a precisao dela -- isso foi medido no E4 do Capitulo 3,
onde a melhor lr com 4.000 passos e' 1.0 e com 20.000 e' 0.1. Os exercicios deste
capitulo sao estruturais (contexto, posicoes previstas, dados), e por isso menos
sensiveis. Onde houver duvida, ela esta' declarada na resposta.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from bpe import BPETokenizer
from dataset import carregar_tokenizador

# modelo (identico a' apostila) e laco de treino vem do modulo comum, para que
# este script e o e7_mais_obras.py usem exatamente o mesmo codigo
from _modelo import PASSOS, treinar, TREINO as treino, VAL as val, VOCAB, V


# ===========================================================================
print("=" * 74)
print("E3 — o tokenizador e' do dominio")
print("=" * 74)
longos = sorted(((i, b) for i, b in VOCAB.items() if i >= 256),
                key=lambda kv: -len(kv[1]))[:20]
nomes_proprios = 0
print("  os 20 tokens mais longos do tokenizador treinado em Machado:")
amostra = []
for i, b in longos:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        s = repr(b)
    amostra.append(s)
    if any(p in s for p in ("José", "Capitú", "Justin", "Aires", "Rubião", "Sofia")):
        nomes_proprios += 1
print("   ", " | ".join(repr(s) for s in amostra[:10]))
print("   ", " | ".join(repr(s) for s in amostra[10:]))
print(f"\n  contendo nome de personagem: {nomes_proprios} de 20")

# tokenizador treinado em NOMES, aplicado a Machado
nomes_txt = (CAP.parent / "06-tokenization" / "names.txt").read_text(encoding="utf-8")[:150_000]
tok_nomes = BPETokenizer()
tok_nomes.train(nomes_txt, 1024)

# o tokenizador de Machado, remontado a partir das fusoes salvas (nao retreinado)
_merges, _ = carregar_tokenizador()
tok_machado = BPETokenizer()
tok_machado.merges, tok_machado.vocab = _merges, VOCAB

# MEDIR OS DOIS NO MESMO TRECHO. A primeira versao deste gabarito comparava
# 'tokens no trecho' (de um) com 'chars/token no corpus inteiro' (do outro) --
# duas grandezas diferentes. A comparacao parecia razoavel e nao era.
trecho = """Havia em mim, leitor amigo, uma ideia fixa, e essa ideia
levou-me a crer que a vida e' um espetaculo curioso."""
n_bytes = len(trecho.encode("utf-8"))
print(f"\n  o MESMO trecho de prosa ({n_bytes} bytes), por dois BPEs de vocab 1024:")
contagens = {}
for rotulo, tk in (("MACHADO", tok_machado), ("NOMES", tok_nomes)):
    ids = tk.encode(trecho)
    contagens[rotulo] = len(ids)
    print(f"    treinado em {rotulo:>7s}: {len(ids):>4d} tokens "
          f"({n_bytes/len(ids):.2f} chars/token)")
print(f"    o de nomes gasta {contagens['NOMES']/contagens['MACHADO']:.2f}x "
      f"mais tokens para dizer a mesma coisa")
print("""
  Respostas:
  1. Boa parte dos tokens mais longos sao NOMES DE PERSONAGENS -- o tokenizador
     descobriu 'José Dias', 'Capitú' e afins so' contando pares de bytes.
  2. E' otimo para ESTE corpus e pessimo fora dele. Um tokenizador treinado em
     Machado gastaria muitos tokens num artigo de medicina, porque nenhuma das
     fusoes aprendidas se aplicaria.
  3. O BPE treinado em NOMES gasta bem mais tokens no mesmo trecho de prosa: ele
     aprendeu terminacoes de nome ('ilson', 'erson'), que quase nao aparecem em
     texto corrido.""")

# ===========================================================================
print("=" * 74)
print(f"E4 — tamanho do contexto ({PASSOS} passos)")
print("=" * 74)
print(f"  {'block_size':>11s} {'treino':>9s} {'val':>9s} {'seg/treino':>11s} "
      f"{'ms/passo':>10s}")
tempos = {}
for blk in (32, 128, 256):
    tr, va, dt, _ = treinar(block=blk)
    tempos[blk] = dt
    print(f"  {blk:>11d} {tr:>9.4f} {va:>9.4f} {dt:>11.1f} {dt/PASSOS*1000:>10.1f}",
          flush=True)
print(f"\n  de 128 para 256, o tempo por passo cresceu "
      f"{tempos[256]/tempos[128]:.2f}x (o contexto dobrou)")
print("""
  Respostas:
  1. Mais contexto melhora a loss -- ao contrario do que acontecia com NOMES no
     Capitulo 4 (onde 16 era pior que 8). Aqui o texto e' longo de verdade, e
     ha' informacao util a 256 tokens de distancia.
  2. O custo NAO quadruplica ao dobrar o contexto, apesar de a atencao ser
     O(T^2). Motivo: a atencao e' so' UMA parte do custo -- as camadas lineares
     (qkv, projecao, feedforward) crescem LINEARMENTE com T, e neste modelo elas
     dominam. O termo quadratico so' passa a mandar em contextos bem maiores.
  3. Para prosa o contexto ajuda mais que para nomes porque a dependencia
     linguistica e' longa: concordancia, referencia a personagens, estrutura de
     frase. Nomes tem ~7 letras -- nao ha' o que lembrar de longe.""")

# ===========================================================================
print("=" * 74)
print(f"E5 — prever em todas as posicoes vs so' na ultima ({PASSOS} passos)")
print("=" * 74)
BLK = 128
tr_todas, va_todas, dt_todas, _ = treinar(block=BLK, so_ultima=False)
tr_ult, va_ult, dt_ult, _ = treinar(block=BLK, so_ultima=True)
print(f"  {'modo':>22s} {'treino':>9s} {'val':>9s} {'previsoes/batch':>17s} "
      f"{'ms/passo':>10s}")
print(f"  {'todas as posicoes':>22s} {tr_todas:>9.4f} {va_todas:>9.4f} "
      f"{32*BLK:>17d} {dt_todas/PASSOS*1000:>10.1f}")
print(f"  {'so a ultima':>22s} {tr_ult:>9.4f} {va_ult:>9.4f} "
      f"{32:>17d} {dt_ult/PASSOS*1000:>10.1f}")
print("""
  Respostas:
  1. Com batch 32 e block 128: 4.096 previsoes por batch contra 32. E' 128x mais
     sinal de treino.
  2. Treinando o mesmo numero de PASSOS, a versao que preve so' na ultima
     posicao fica MUITO pior -- ela recebe 128x menos informacao por passo.
  3. E o custo do forward e' praticamente o MESMO: as duas processam as 128
     posicoes de qualquer forma; a diferenca e' so' quantas saidas da camada
     final sao usadas na loss. Ou seja, a versao 'so' a ultima' joga fora 99%
     do calculo que ja' fez.

     E' o tipo de mudanca que todo mundo deveria querer: 128x mais sinal, custo
     igual. Os capitulos 3 a 7 usaram a versao ineficiente de proposito, porque
     ela mantinha a metrica comparavel entre capitulos.""")

# ===========================================================================
print("=" * 74)
print(f"E6 — dados sinteticos e model collapse ({PASSOS} passos)")
print("=" * 74)
tr_real, va_real, _, modelo_real = treinar(block=128)
print(f"  modelo A (dados reais): treino {tr_real:.4f} | val {va_real:.4f}")


@torch.no_grad()
def gerar_corpus(m, n_tokens, block=128, n_seq=64):
    """Gera em PARALELO: n_seq sequencias de uma vez.

    Gerar 60 mil tokens um a um custaria ~30 min (cada token e' um forward
    inteiro). Gerando 64 sequencias em paralelo, sao 60000/64 = 938 forwards em
    vez de 60000 -- o custo por forward quase nao muda, porque a matmul com
    batch 64 aproveita muito melhor o processador. Mesma licao do Capitulo 08.
    """
    m.eval()
    idx = torch.zeros((n_seq, 1), dtype=torch.long)
    passos = math.ceil(n_tokens / n_seq)
    for _ in range(passos):
        logits, _ = m(idx[:, -block:])
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        prox = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat((idx, prox), dim=1)
    m.train()
    return idx.reshape(-1).numpy().astype(np.uint16)[:n_tokens]


print("  gerando 60.000 tokens sinteticos com o modelo A...", flush=True)
sintetico = gerar_corpus(modelo_real, 60_000)
tr_sint, va_sint, _, _ = treinar(block=128, dados_tr=sintetico, dados_val=val)
print(f"  modelo B (dados do A)  : treino {tr_sint:.4f} | val {va_sint:.4f} "
      f"(avaliado no MESMO conjunto de validacao real)")
print(f"\n  degradacao: {va_sint - va_real:+.4f} na validacao real")
print("""
  Respostas:
  1 e 2. O modelo B, treinado no texto gerado pelo A, fica PIOR na validacao
     real. O fenomeno tem nome: MODEL COLLAPSE.

     E REPARE NA COLUNA DE TREINO, que e' a parte mais perigosa: a loss de
     TREINO do modelo B e' MENOR que a do A (4.30 contra 4.45). Se voce so'
     olhasse a curva de treino, concluiria que o B esta' aprendendo melhor.

     Nao esta'. Texto gerado por um modelo e' mais PREVISIVEL que texto humano
     -- tem menos vocabulario, menos construcoes raras, menos surpresa. Ficar
     bom em prever texto facil nao e' ficar bom em prever Machado.

     O colapso chega DISFARCADO DE PROGRESSO. E' por isso que a validacao
     precisa vir de dados reais e intocados (Capitulo 3, e o E2 deste
     capitulo).

     A razao e' informacional: o texto gerado por A e' uma AMOSTRA da
     distribuicao que A aprendeu -- e amostrar perde a cauda. Padroes raros
     aparecem pouco ou nao aparecem, e o B aprende uma versao ainda mais
     concentrada. Repetindo o ciclo, a diversidade colapsa.

     Nenhum modelo pode aprender com os proprios dados MAIS do que ja' sabe: a
     geracao nao cria informacao nova sobre o mundo.

  3. Dados sinteticos ajudam quando ha' uma fonte de VERDADE externa para
     filtrar: matematica (da' para conferir a resposta), codigo (da' para
     rodar os testes), jogos (da' para saber quem ganhou). Ai' o modelo gera
     muitas tentativas, um verificador seleciona as boas, e a informacao que
     entra vem do VERIFICADOR, nao do modelo.""")

# ===========================================================================
print("=" * 74)
print(f"E7 — escalando o corpus ({PASSOS} passos)")
print("=" * 74)
print(f"  {'corpus':>18s} {'tokens':>10s} {'treino':>9s} {'val':>9s} {'gap':>8s}")
for frac, rot in ((0.25, "25% do treino"), (0.5, "50% do treino"), (1.0, "corpus inteiro")):
    sub = treino[: int(len(treino) * frac)]
    tr, va, _, _ = treinar(block=128, dados_tr=sub)
    print(f"  {rot:>18s} {len(sub):>10,} {tr:>9.4f} {va:>9.4f} {va-tr:>8.4f}",
          flush=True)
print("""
  Respostas:
  1 e 2. Mais dados melhoram a loss de validacao, com retorno decrescente.
  3. A logica do Capitulo 3 vale, e o 'gap' entre treino e validacao mostra:
     com menos dados, o modelo memoriza mais (gap maior). Com o corpus inteiro
     o gap encolhe.

     Mas note a diferenca de escala: no Capitulo 3, ir de 155 para 64.000 nomes
     mudou o gap de 5.7 para ~0. Aqui, quadruplicar o corpus muda o gap bem
     menos -- porque 1.6 MB de texto ja' e' um regime muito diferente de 155
     nomes. Melhorias ficam progressivamente mais caras.""")
