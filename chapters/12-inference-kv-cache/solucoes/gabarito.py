"""
Gabarito executavel do Capitulo 12 — KV-cache.

Roda E2, E3, E5, E6 e E7. (O E1 e' conceitual; o E4 e' o benchmark_cache.py.)

Os E2 e E3 introduzem os bugs DE PROPOSITO, em subclasses, para que voce veja o
modo de falha sem precisar editar o modelo bom -- e para que o gabarito continue
rodando depois.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
sys.path.insert(0, str(CAP.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from modelo import GPT, Bloco, carregar, carregar_tokenizador, decodificar

modelo, ck = carregar()
_, VOCAB = carregar_tokenizador()
PROMPT = torch.zeros((1, 1), dtype=torch.long)
REFERENCIA = modelo.gerar_ingenuo(PROMPT, 80, semente=7)


def primeira_divergencia(a, b):
    n = min(a.shape[1], b.shape[1])
    dif = (a[0, :n] != b[0, :n]).nonzero()
    return dif[0].item() if len(dif) else None


# ===========================================================================
print("=" * 74)
print("E2 — quebrando a posicao de proposito")
print("=" * 74)


class GPTPosicaoErrada(GPT):
    """O bug: usa arange(T) em vez da posicao absoluta."""

    def forward(self, idx, cache=None):
        B, T = idx.shape
        x = self.te(idx) + self.pe(torch.arange(T, device=idx.device))   # <- BUG
        cache_novo = []
        for i, b in enumerate(self.blocos):
            x, kv = b(x, None if cache is None else cache[i])
            cache_novo.append(kv)
        return self.lm(self.lnf(x)), cache_novo


ruim = GPTPosicaoErrada(modelo.cfg)
ruim.load_state_dict(modelo.state_dict())
ruim.eval()
saida_ruim = ruim.gerar_com_cache(PROMPT, 80, semente=7)

d = primeira_divergencia(REFERENCIA, saida_ruim)
print(f"  levantou excecao?          nao")
print(f"  gerou texto?               sim, {saida_ruim.shape[1]} tokens")
print(f"  primeira divergencia:      token {d}")
print(f"\n  referencia : {decodificar(REFERENCIA[0, :40].tolist(), VOCAB)!r}")
print(f"  com o bug  : {decodificar(saida_ruim[0, :40].tolist(), VOCAB)!r}")
print("""
  Respostas:
  1. Nenhuma excecao, e o texto continua saindo com aparencia normal. E' o pior
     tipo de bug: ele nao AVISA.
  2. A divergencia comeca no primeiro token gerado pela fase de DECODE. O
     prefill processa o prompt inteiro de uma vez, e ali T == T_total, entao
     arange(T) por acaso da' a posicao certa. O erro so' aparece quando passamos
     UM token: o modelo acha que ele esta' na posicao 0, isto e', que e' o comeco
     do texto. Dai' em diante todo token gerado carrega a posicao errada.
  3. Sem a referencia ao lado, os dois textos parecem igualmente plausiveis --
     este modelo e' pequeno e escreve mal nas duas versoes. E' exatamente por
     isso que o criterio do capitulo e' SAIDA IDENTICA, e nao 'parece bom'.
     Numa otimizacao de inferencia, 'parece bom' nao e' evidencia de nada.""")

# ===========================================================================
print("=" * 74)
print("E3 — quebrando a mascara de proposito")
print("=" * 74)


class BlocoMascaraErrada(Bloco):
    """O bug: aplica a mascara tambem no decode."""

    def forward(self, x, cache=None):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        novo = (k, v)
        T_total = k.shape[2]
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5
        w = w.masked_fill(self.tril[:T, :T_total] == 0, float("-inf"))   # <- BUG
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x)))), novo


ruim2 = GPT(modelo.cfg)
ruim2.blocos = torch.nn.ModuleList([BlocoMascaraErrada(modelo.cfg)
                                    for _ in range(modelo.cfg["n_layer"])])
ruim2.load_state_dict(modelo.state_dict())
ruim2.eval()
saida_ruim2 = ruim2.gerar_com_cache(PROMPT, 80, semente=7)
d2 = primeira_divergencia(REFERENCIA, saida_ruim2)
unicos = len(set(saida_ruim2[0, 1:].tolist()))
print(f"  primeira divergencia:  token {d2}")
print(f"  tokens distintos na saida: {unicos} de {saida_ruim2.shape[1]-1}")
print(f"\n  com o bug  : {decodificar(saida_ruim2[0, :40].tolist(), VOCAB)!r}")

# O QUE O MODELO QUEBRADO VIROU -- medido, nao suposto.
# Duas historias com o MESMO primeiro token e continuacoes diferentes. Se os
# logits sairem iguais, o modelo esta' ignorando tudo no meio.
torch.manual_seed(0)
h1 = torch.cat([torch.tensor([[7]]), torch.randint(1, 1000, (1, 19))], dim=1)
h2 = torch.cat([torch.tensor([[7]]), torch.randint(1, 1000, (1, 19))], dim=1)
tok = torch.tensor([[500]])
with torch.no_grad():
    _, c1 = ruim2(h1)
    _, c2 = ruim2(h2)
    l1, _ = ruim2(tok, cache=c1)
    l2, _ = ruim2(tok, cache=c2)
    _, b1 = modelo(h1)
    _, b2 = modelo(h2)
    m1, _ = modelo(tok, cache=b1)
    m2, _ = modelo(tok, cache=b2)
print(f"\n  mesmo token na mesma posicao, mesmo 1o token, 19 tokens do meio DIFERENTES:")
print(f"    modelo com bug : logits identicos? {torch.allclose(l1, l2, atol=1e-4)}"
      f"   (diferenca maxima {(l1-l2).abs().max():.2e})")
print(f"    modelo correto : logits identicos? {torch.allclose(m1, m2, atol=1e-4)}"
      f"  (diferenca maxima {(m1-m2).abs().max():.2e})")
print("""
  Respostas:
  1. O texto fica ruim, mas ATENCAO: nao e' repeticao. Foi o que eu previ ao
     escrever este gabarito, e a contagem de tokens distintos acima desmente --
     a saida tem variedade. A amostragem continua estocastica; o que se perdeu
     foi o CONTEXTO, nao a diversidade.

     O que o modelo virou esta' medido logo acima: com o mesmo primeiro token e
     o mesmo token atual, os logits saem BIT A BIT IDENTICOS por mais que os 19
     tokens do meio sejam outros. Ou seja, o modelo passou a prever a partir de
     (primeiro token, token atual, posicao) e mais nada.

     Ele virou, literalmente, o modelo do Capitulo 1: um BIGRAMA. Toda a
     maquinaria de atencao continua rodando e nao transporta informacao nenhuma.

  2. No decode T == 1, entao tril[:1, :T_total] seleciona a PRIMEIRA linha da
     matriz triangular: [1, 0, 0, ..., 0]. Ela permite olhar so' a posicao 0 --
     e e' por isso que o primeiro token ainda influencia, enquanto o resto do
     contexto desaparece.

  3. Eu tambem previ que este bug seria MAIS FACIL de notar que o do E2, por
     falhar de forma escandalosa. Errado de novo: leia os dois textos acima. Sao
     igualmente ruins a olho nu, porque um modelo de 2,2 M parametros ja' escreve
     mal quando esta' CERTO. A degradacao se esconde no ruido de base.

     E ai' esta' a licao dos dois exercicios juntos: em modelo pequeno, nenhum
     dos dois bugs se denuncia pela leitura. Um deles derruba o modelo ao nivel
     do Capitulo 1 e voce nao percebe olhando. So' o teste de EQUIVALENCIA
     denuncia -- e por isso ele nao e' burocracia, e' o unico instrumento que
     funciona.""")

# ===========================================================================
print("=" * 74)
print("E5 — a conta de memoria")
print("=" * 74)
c = modelo.cfg
print(f"  formula: 2 x n_layer x batch x n_head x T x head_size x bytes")
print(f"  conferindo contra bytes_do_cache() para T=128:")
manual = 2 * c["n_layer"] * 1 * c["n_head"] * 128 * (c["n_embd"] // c["n_head"]) * 4
print(f"    manual = {manual:,} bytes | metodo = {modelo.bytes_do_cache(128):,} bytes")

print(f"\n  modelo 7B (32 camadas, 32 cabecas, hs 128) em bf16, contexto 8192:")
por_usuario = 2 * 32 * 1 * 32 * 8192 * 128 * 2
print(f"    por usuario: {por_usuario/1e9:.1f} GB")
livre = 80 - 14
print(f"    placa de 80 GB - 14 GB de pesos = {livre} GB livres")
print(f"    usuarios simultaneos: {int(livre*1e9/por_usuario)}")
print("""
  Resposta 3 -- a que importa:
     n_head x head_size = n_embd. Entao a formula vira

         2 x n_layer x batch x T x n_embd x bytes

     O cache NAO depende de quantas cabecas voce usa, nem do tamanho delas --
     depende da LARGURA do modelo e da PROFUNDIDADE. Redistribuir n_embd entre
     mais ou menos cabecas nao muda nada no custo de servir.

     O que muda e' quebrar a simetria: se as cabecas de Q forem muitas e as de
     K/V forem poucas, o produto n_head_kv x head_size fica menor que n_embd. E'
     exatamente isso que a MQA e a GQA fazem (E7).""")

# ===========================================================================
print("=" * 74)
print("E6 — cache com batch")
print("=" * 74)
B = 16
lote = torch.zeros((B, 1), dtype=torch.long)
t0 = time.perf_counter()
saida_lote = modelo.gerar_com_cache(lote, 64, semente=99)
t_lote = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(B):
    modelo.gerar_com_cache(PROMPT, 64, semente=99)
t_sep = time.perf_counter() - t0

print(f"  {B} sequencias de 64 tokens em PARALELO : {t_lote:6.2f}s")
print(f"  {B} sequencias de 64 tokens SEPARADAS   : {t_sep:6.2f}s")
print(f"  ganho: {t_sep/t_lote:.1f}x")
print("""
  Respostas:
  1. Nada muda no codigo. A dimensao de batch ja' esta' na forma do cache
     (B, n_head, T, hs) e todas as operacoes sao em lote. Foi so' passar um
     prompt com B linhas -- o que e' um bom sinal sobre a implementacao.
  2. O ganho e' grande, e pela razao do Capitulo 08: o decode e' limitado por
     MEMORIA, nao por calculo. Ler os pesos do modelo custa o mesmo para 1 ou
     para 16 sequencias, entao o custo se dilui. E' por isso que servico de
     inferencia agrupa requisicoes de usuarios diferentes no mesmo batch.
  3. Comprimentos diferentes sao o problema de verdade. Se uma sequencia termina
     no token 10 e outra no 200, o batch inteiro continua rodando ate' a mais
     longa acabar -- as posicoes ja' terminadas gastam calculo a' toa, e o cache
     delas continua ocupando memoria.

     Estrategias: (a) preenchimento com mascara de terminados, simples e
     desperdicador; (b) CONTINUOUS BATCHING -- retirar a sequencia pronta e
     colocar outra da fila no lugar, que e' o que servidores modernos fazem;
     (c) paged attention (vLLM), que aloca o cache em paginas e evita reservar
     o pior caso para cada sequencia.""")

# ===========================================================================
print("=" * 74)
print("E7 — multi-query attention sobre pesos treinados com multi-head")
print("=" * 74)
from dataset import carregar as carregar_dados, pegar_batch

val = carregar_dados("val")


@torch.no_grad()
def loss_val(m, n=20, semente=1234):
    g = torch.Generator().manual_seed(semente)
    tot = 0.0
    for _ in range(n):
        x, y = pegar_batch(val, 16, m.block_size, generator=g)
        logits, _ = m(x)
        tot += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
    return tot / n


class BlocoMQA(Bloco):
    """MQA improvisada: MEDIA das cabecas de K e V, compartilhada por todas as de Q."""

    def forward(self, x, cache=None):
        Bsz, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(Bsz, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(Bsz, T, self.n_head, self.hs).transpose(1, 2).mean(1, keepdim=True)
        v = v.view(Bsz, T, self.n_head, self.hs).transpose(1, 2).mean(1, keepdim=True)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        novo = (k, v)
        T_total = k.shape[2]
        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5      # broadcast: 1 cabeca de K
        if T == T_total:
            w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(Bsz, T, C)
        x = x + self.proj(y)
        return x + self.fo(F.gelu(self.fi(self.ln2(x)))), novo


mqa = GPT(modelo.cfg)
mqa.blocos = torch.nn.ModuleList([BlocoMQA(modelo.cfg)
                                  for _ in range(modelo.cfg["n_layer"])])
mqa.load_state_dict(modelo.state_dict())
mqa.eval()

l_mha, l_mqa = loss_val(modelo), loss_val(mqa)
print(f"  loss de validacao, multi-head original : {l_mha:.4f}")
print(f"  loss de validacao, MQA improvisada     : {l_mqa:.4f}   ({l_mqa-l_mha:+.4f})")
print(f"  cache: cai por um fator de {c['n_head']}x "
      f"({modelo.bytes_do_cache(128)/1024:.0f} KB -> "
      f"{modelo.bytes_do_cache(128)/c['n_head']/1024:.0f} KB para T=128)")
print(f"  num 7B (32 cabecas), a mesma conta daria 32x menos cache")
print("""
  Resposta 3 -- e e' a licao do exercicio:
     A loss piora, e MUITO. Nao ha' surpresa nisso: o modelo foi TREINADO com 6
     cabecas de K/V independentes, e cada cabeca de Q aprendeu a perguntar para a
     SUA. Substituir as seis por uma media destroi essa correspondencia.

     MQA nao e' uma otimizacao que se liga na hora de servir. E' uma decisao de
     ARQUITETURA: o modelo precisa ser treinado assim desde o inicio (ou passar
     por um 'uptraining' com dados). Modelos que usam GQA -- Llama 2 70B, Mistral
     -- foram projetados com ela.

     A licao geral vale alem deste exercicio: 'economiza memoria' e 'da' para
     aplicar depois' sao afirmacoes independentes. O KV-cache deste capitulo e'
     gratuito porque nao muda a conta; a MQA nao e', porque muda.""")
