"""
quantizar_modelo.py — quantiza o modelo do Capitulo 11 e mede o que isso custa.

Fazemos WEIGHT-ONLY QUANTIZATION: os pesos ficam em int8, as ativacoes continuam
em float32. E' a forma mais usada para LLM, e o motivo esta' no Capitulo 12 -- o
decode e' limitado por MEMORIA, e quem ocupa memoria sao os pesos.

Tres coisas sao medidas, e as tres importam:
    TAMANHO   -- quantos bytes o modelo ocupa
    QUALIDADE -- a loss de validacao (erro no peso NAO e' erro no modelo)
    VELOCIDADE-- se acelerou de verdade

Run (a partir da pasta do capitulo):
    python quantizar_modelo.py
"""

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent / "12-inference-kv-cache"))
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from quantizacao import quantizar_simetrica, desquantizar_simetrica
from modelo import carregar
from dataset import carregar as carregar_dados, pegar_batch


# ===========================================================================
class LinearQuantizada(nn.Module):
    """Guarda o peso em int8 e o reconstroi na hora de usar.

    Isto NAO acelera a conta -- a matmul continua em float32, e ainda pagamos a
    desquantizacao. O que ele economiza e' MEMORIA. Ver a Secao 3 do resultado.

    Uma implementacao de produção faria a matmul em int8 diretamente, com kernel
    dedicado. Aqui o objetivo e' enxergar a aritmetica, nao competir com o
    llama.cpp.
    """

    def __init__(self, linear, bits=8):
        super().__init__()
        q, escala = quantizar_simetrica(linear.weight.data, bits=bits, dim=0)
        self.register_buffer("q", q)
        self.register_buffer("escala", escala)
        self.bias = linear.bias
        self.bits = bits

    def forward(self, x):
        return F.linear(x, desquantizar_simetrica(self.q, self.escala), self.bias)

    def bytes(self):
        # int8 guarda 1 byte por peso independentemente de bits<8 -- ver nota no
        # relatorio sobre empacotamento
        return self.q.numel() + self.escala.numel() * 4 + \
            (self.bias.numel() * 4 if self.bias is not None else 0)


def quantizar_modelo(m, bits=8, incluir_embeddings=False):
    """Troca todo nn.Linear por LinearQuantizada. Devolve uma copia."""
    m = copy.deepcopy(m)
    for pai in m.modules():
        for nome, filho in list(pai.named_children()):
            if isinstance(filho, nn.Linear):
                setattr(pai, nome, LinearQuantizada(filho, bits=bits))
    if incluir_embeddings:
        for pai in m.modules():
            for nome, filho in list(pai.named_children()):
                if isinstance(filho, nn.Embedding):
                    q, e = quantizar_simetrica(filho.weight.data, bits=bits, dim=0)
                    filho.weight.data = desquantizar_simetrica(q, e)
    return m


def bytes_do_modelo(m):
    total = 0
    for mod in m.modules():
        if isinstance(mod, LinearQuantizada):
            total += mod.bytes()
    for nome, p in m.named_parameters():
        if not any(f"{n}." in nome for n in ("q", "escala")):
            # parametros que sobraram em float32 (embeddings, LayerNorm, bias)
            if not isinstance(dict(m.named_modules()).get(nome.rsplit(".", 1)[0]),
                              LinearQuantizada):
                total += p.numel() * 4
    return total


# ===========================================================================
modelo, ck = carregar()
val = carregar_dados("val")


@torch.no_grad()
def loss_val(m, n=25, semente=1234):
    g = torch.Generator().manual_seed(semente)
    m.eval()
    tot = 0.0
    for _ in range(n):
        x, y = pegar_batch(val, 16, m.block_size, generator=g)
        logits, _ = m(x)
        tot += F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1)).item()
    return tot / n


def cronometrar(fn, rodadas=3):
    fn()
    return min((lambda: (lambda t0: (fn(), time.perf_counter() - t0)[1])
                (time.perf_counter()))() for _ in range(rodadas))


# ===========================================================================
if __name__ == "__main__":
    print("=" * 74)
    print("1. Qualidade: quantos bits o modelo aguenta?")
    print("=" * 74)
    base = loss_val(modelo)
    print(f"  float32 (referencia): loss {base:.4f}\n")
    print(f"  {'bits':>5s} {'loss':>9s} {'piora':>9s} {'perplexidade':>13s}")
    print(f"  {'fp32':>5s} {base:>9.4f} {'--':>9s} {math.exp(base):>13.1f}")
    resultados = {}
    for bits in (8, 6, 4, 3, 2):
        mq = quantizar_modelo(modelo, bits=bits)
        l = loss_val(mq)
        resultados[bits] = l
        print(f"  {bits:>5d} {l:>9.4f} {l-base:>+9.4f} {math.exp(l):>13.1f}", flush=True)

    print("=" * 74)
    print("2. Tamanho: quanto o modelo encolhe")
    print("=" * 74)
    fp32 = sum(p.numel() for p in modelo.parameters()) * 4
    lineares = sum(m.weight.numel() for m in modelo.modules() if isinstance(m, nn.Linear))
    outros = sum(p.numel() for p in modelo.parameters()) - lineares
    print(f"  parametros totais      : {sum(p.numel() for p in modelo.parameters()):,}")
    print(f"    em camadas Linear    : {lineares:,} ({lineares/(lineares+outros):.0%})")
    print(f"    embeddings/LayerNorm : {outros:,} ({outros/(lineares+outros):.0%})")
    print(f"\n  {'config':>22s} {'MB':>8s} {'reducao':>9s}")
    print(f"  {'tudo em float32':>22s} {fp32/1e6:>8.2f} {'--':>9s}")
    for bits in (8, 4):
        mb = (lineares * bits / 8 + outros * 4) / 1e6
        print(f"  {f'Linear em int{bits}':>22s} {mb:>8.2f} {fp32/1e6/mb:>8.2f}x")
    mb_tudo = (sum(p.numel() for p in modelo.parameters()) * 1) / 1e6
    print(f"  {'TUDO em int8':>22s} {mb_tudo:>8.2f} {fp32/1e6/mb_tudo:>8.2f}x")
    print("""
      Duas leituras desta tabela, e a segunda e' uma ressalva de honestidade:

      1. Quantizar so' as camadas Linear da' 3,04x, nao 4x. A diferenca e'
         exatamente os 10% de parametros que ficam em float32 (embeddings e
         LayerNorm). Nao e' um deficit grande -- e' aritmetica: 0,9/4 + 0,1 = 0,325
         do tamanho original, ou seja 3,08x. Bate com o medido.

         Eu tinha escrito aqui que 'os embeddings sao uma fracao enorme dos
         parametros deste modelo'. Sao 10%. A frase estava errada.

      2. A linha do int4 supoe EMPACOTAMENTO: dois pesos por byte. A implementacao
         deste capitulo NAO empacota -- ela guarda cada valor num int8, mesmo quando
         usa so' 4 bits de resolucao. Entao o tamanho da tabela e' o TEORICO, o que
         um formato de arquivo de verdade (GGUF, por exemplo) entregaria.

         Empacotar e' trabalho de serializacao, nao de quantizacao, e mantê-los
         separados deixa o codigo legivel. Mas o numero na tabela e' uma promessa,
         nao uma medicao -- e vale dizer qual e' qual.""")

    print("=" * 74)
    print("3. Velocidade: a pergunta que quase ninguem faz")
    print("=" * 74)
    mq8 = quantizar_modelo(modelo, bits=8)
    prompt = torch.zeros((1, 1), dtype=torch.long)
    t_fp32 = cronometrar(lambda: modelo.gerar_com_cache(prompt, 64, semente=1))
    t_int8 = cronometrar(lambda: mq8.gerar_com_cache(prompt, 64, semente=1))
    print(f"  gerar 64 tokens, float32 : {t_fp32:.3f}s")
    print(f"  gerar 64 tokens, int8    : {t_int8:.3f}s   ({t_fp32/t_int8:.2f}x)")
    print("""
      Se o int8 saiu MAIS LENTO, esta' correto -- e e' a licao da secao.

      Esta implementacao guarda o peso em int8 e o RECONSTROI em float32 a cada
      forward. Ou seja: ela paga a desquantizacao e faz a mesma matmul de antes. O
      ganho de memoria e' real; o de velocidade nao existe, porque a conta nao mudou.

      Para a quantizacao acelerar de verdade e' preciso um kernel que multiplique
      int8 x int8 acumulando em int32, sem passar por float. Isso nao se escreve em
      PyTorch puro -- e' o que bibliotecas como llama.cpp, bitsandbytes e ONNX
      Runtime fornecem.

      A licao geral: 'o modelo ficou 4x menor' e 'o modelo ficou mais rapido' sao
      afirmacoes INDEPENDENTES. A primeira decorre da representacao; a segunda exige
      que alguem tenha escrito o kernel.""")
