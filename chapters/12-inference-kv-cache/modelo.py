"""
modelo.py — o mesmo GPT do Capitulo 11, agora com KV-CACHE.

A arquitetura NAO muda. Os pesos sao os mesmos, carregados do checkpoint que o
Capitulo 11 salvou. O que muda e' so' COMO calculamos a geracao -- e o teste de
que a mudanca esta' certa e' o texto sair IDENTICO ao de antes.

O modelo aceita os dois modos:

    logits, _ = m(idx)                  # modo treino/prefill: processa T posicoes
    logits, cache = m(idx, cache=c)     # modo decode: processa 1 posicao

Run (a partir da pasta do capitulo):
    python modelo.py            # confere que os dois modos dao o MESMO resultado
"""

import pickle
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
CAP11 = AQUI.parent / "11-datasets"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ===========================================================================
# O bloco, com e sem cache
# ===========================================================================
class Bloco(nn.Module):
    """Bloco identico ao do Capitulo 11, com um caminho extra para o cache.

    A diferenca esta' toda no forward: quando recebe um cache, ele CONCATENA as
    chaves e valores novos aos que ja' existiam, em vez de recalcular tudo.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.hs = cfg["n_embd"] // cfg["n_head"]
        ne = cfg["n_embd"]
        self.qkv = nn.Linear(ne, 3 * ne, bias=False)
        self.proj = nn.Linear(ne, ne)
        self.fi = nn.Linear(ne, 4 * ne)
        self.fo = nn.Linear(4 * ne, ne)
        self.ln1, self.ln2 = nn.LayerNorm(ne), nn.LayerNorm(ne)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
        )

    def forward(self, x, cache=None):
        """cache = (k_anterior, v_anterior) ou None.

        Devolve (saida, (k_novo, v_novo)).
        """
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)   # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)

        if cache is not None:
            # AQUI ESTA' O CAPITULO INTEIRO, em duas linhas.
            # As chaves e valores das posicoes anteriores NAO mudam quando um
            # token novo chega -- elas dependem so' do proprio token e da sua
            # posicao. Entao basta concatenar as novas ao que ja' foi calculado.
            k_ant, v_ant = cache
            k = torch.cat((k_ant, k), dim=2)
            v = torch.cat((v_ant, v), dim=2)

        novo_cache = (k, v)
        T_total = k.shape[2]                    # quantas posicoes ha' no total

        w = (q @ k.transpose(-2, -1)) * self.hs ** -0.5          # (B, nh, T, T_total)

        # A MASCARA, e a sutileza que derruba muita gente:
        #
        # No modo prefill (T == T_total) a mascara e' a de sempre: cada posicao
        # so' pode olhar para tras.
        #
        # No modo decode (T == 1) a UNICA query e' a do ultimo token, e ela PODE
        # olhar para todas as posicoes anteriores -- que e' exatamente o que a
        # ultima linha da matriz triangular ja' dizia. Nao ha' o que mascarar.
        # Aplicar tril[:1, :T_total] aqui seria um BUG: ele deixaria o token so'
        # enxergar a posicao 0.
        if T == T_total:
            w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        y = (F.softmax(w, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.fo(F.gelu(self.fi(self.ln2(x))))
        return x, novo_cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg["block_size"]
        self.te = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.pe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.blocos = nn.ModuleList([Bloco(cfg) for _ in range(cfg["n_layer"])])
        self.lnf = nn.LayerNorm(cfg["n_embd"])
        self.lm = nn.Linear(cfg["n_embd"], cfg["vocab_size"])

    def forward(self, idx, cache=None):
        """cache = lista de (k, v) por bloco, ou None.

        Devolve (logits, cache_novo).
        """
        B, T = idx.shape

        # A POSICAO E' ABSOLUTA, NAO RELATIVA AO PEDACO QUE ESTAMOS PROCESSANDO.
        # No modo decode passamos UM token, mas ele nao esta' na posicao 0 -- esta'
        # na posicao (quantos tokens ja' vieram antes). Errar isto e' o bug mais
        # comum de KV-cache: o texto continua saindo, com aparencia plausivel, e
        # so' um pouco pior. Nao ha' excecao para avisar.
        ja_vistos = 0 if cache is None else cache[0][0].shape[2]
        pos = torch.arange(ja_vistos, ja_vistos + T, device=idx.device)

        x = self.te(idx) + self.pe(pos)
        cache_novo = []
        for i, b in enumerate(self.blocos):
            x, kv = b(x, None if cache is None else cache[i])
            cache_novo.append(kv)
        logits = self.lm(self.lnf(x))
        return logits, cache_novo

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def gerar_ingenuo(self, idx, n_tokens, temperatura=0.8, top_k=40, semente=1337):
        """A geracao do Capitulo 11: reprocessa TODO o contexto a cada token."""
        g = torch.Generator(device=idx.device).manual_seed(semente)
        for _ in range(n_tokens):
            recorte = idx[:, -self.block_size:]
            logits, _ = self(recorte)                 # sem cache: T posicoes
            idx = torch.cat((idx, self._amostrar(logits[:, -1, :], temperatura,
                                                 top_k, g)), dim=1)
        return idx

    @torch.no_grad()
    def gerar_com_cache(self, idx, n_tokens, temperatura=0.8, top_k=40, semente=1337):
        """Duas fases, e a distincao entre elas organiza toda a inferencia real.

        PREFILL: processa o prompt inteiro de uma vez e monta o cache.
        DECODE : a partir dai', um token por vez, cada um vendo so' a si mesmo.
        """
        g = torch.Generator(device=idx.device).manual_seed(semente)

        # --- prefill
        logits, cache = self(idx[:, -self.block_size:])
        prox = self._amostrar(logits[:, -1, :], temperatura, top_k, g)
        idx = torch.cat((idx, prox), dim=1)

        # --- decode
        for _ in range(n_tokens - 1):
            if cache[0][0].shape[2] >= self.block_size:
                # O contexto encheu. O modelo so' tem embeddings posicionais ate'
                # block_size, entao precisamos descartar o comeco -- e, como as
                # posicoes de todos os tokens mudam, o cache inteiro perde a
                # validade. E' o custo de um contexto fixo; o Capitulo 13 discute
                # as alternativas.
                logits, cache = self(idx[:, -self.block_size:])
            else:
                logits, cache = self(prox, cache=cache)   # UM token so'
            prox = self._amostrar(logits[:, -1, :], temperatura, top_k, g)
            idx = torch.cat((idx, prox), dim=1)
        return idx

    @staticmethod
    def _amostrar(logits, temperatura, top_k, g):
        logits = logits / temperatura
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1, generator=g)

    def bytes_do_cache(self, n_tokens, batch=1):
        """Memoria do cache: 2 (K e V) x camadas x batch x cabecas x T x head_size."""
        c = self.cfg
        return (2 * c["n_layer"] * batch * c["n_head"] * n_tokens
                * (c["n_embd"] // c["n_head"]) * 4)      # 4 bytes por float32


# ===========================================================================
def carregar(caminho=None):
    """Carrega os pesos treinados no Capitulo 11."""
    caminho = Path(caminho) if caminho else CAP11 / "modelo.pt"
    if not caminho.exists():
        raise SystemExit(
            f"Checkpoint nao encontrado em {caminho}.\n"
            f"Rode antes:  cd ../11-datasets && python train_text.py"
        )
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    m = GPT(ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def carregar_tokenizador():
    with open(CAP11 / "tokenizador.pkl", "rb") as f:
        d = pickle.load(f)
    return d["merges"], d["vocab"]


def decodificar(ids, vocab):
    return b"".join(vocab[int(i)] for i in ids).decode("utf-8", errors="replace")


# ===========================================================================
if __name__ == "__main__":
    m, ck = carregar()
    n = sum(p.nelement() for p in m.parameters())
    print(f"modelo do Capitulo 11: {n:,} parametros, contexto {m.block_size}")
    print(f"  loss de validacao no treino: {ck['loss_val']:.4f}\n")

    print("O teste que importa: os dois caminhos dao o MESMO texto?")
    prompt = torch.zeros((1, 1), dtype=torch.long)
    a = m.gerar_ingenuo(prompt, 60, semente=42)
    b = m.gerar_com_cache(prompt, 60, semente=42)
    igual = torch.equal(a, b)
    print(f"  ingenuo    : {a[0, :12].tolist()}")
    print(f"  com cache  : {b[0, :12].tolist()}")
    print(f"  identicos  : {igual}")
    if not igual:
        difs = (a != b).nonzero()
        print(f"  primeira divergencia na posicao {difs[0, 1].item()}")
    print("""
  Um KV-cache que muda a saida nao esta' 'quase certo' -- esta' ERRADO. Ele nao
  e' uma aproximacao nem uma troca de qualidade por velocidade: e' a MESMA conta,
  reorganizada para nao repetir trabalho. Se o texto mudou, ha' um bug.""")
