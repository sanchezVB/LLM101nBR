"""
Inicializacao — por que os pesos iniciais decidem se a rede treina.

No Capitulo 3 escrevemos, sem explicar:

    W1 = torch.randn(...) * (5 / 3) / (n_embd * block_size) ** 0.5
                            ^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                            ganho     1 / sqrt(fan_in)

Este arquivo mostra de onde vem cada peca, medindo o que acontece com as
ativacoes e os gradientes ao atravessar varias camadas.

A ideia central: cada camada multiplica a escala do sinal por um fator. Se esse
fator for < 1, o sinal encolhe a cada camada e MORRE. Se for > 1, ele cresce e
SATURA (ou explode). Queremos fator ~1: o sinal atravessa a rede intacto.

Run:
    python initialization.py
"""

import torch

torch.manual_seed(1337)

N_LAYERS = 8
DIM = 256
BATCH = 1024


def rodar(gain, ativacao="tanh", n_layers=N_LAYERS, dim=DIM):
    """Passa um sinal por n_layers camadas e devolve o desvio padrao em cada uma.

    A inicializacao e' `randn * gain / sqrt(fan_in)`, o esquema padrao.
    """
    x = torch.randn(BATCH, dim)
    pesos = [
        (torch.randn(dim, dim) * gain / dim ** 0.5).requires_grad_(True)
        for _ in range(n_layers)
    ]

    stds = []
    h = x
    for W in pesos:
        h = h @ W
        h = torch.tanh(h) if ativacao == "tanh" else torch.relu(h)
        stds.append(h.std().item())

    # gradiente: mede o que chega de volta na PRIMEIRA camada
    h.sum().backward()
    grad_primeira = pesos[0].grad.std().item()
    return stds, grad_primeira


print(f"=== ativacoes atraves de {N_LAYERS} camadas (tanh, dim={DIM}) ===")
print("Desvio padrao da saida de cada camada. Queremos ~constante.\n")
print(f"{'ganho':>7s}  " + "  ".join(f"c{i+1:<5d}" for i in range(N_LAYERS)) + "   grad 1a camada")

for gain, rotulo in [(0.5, "pequeno"), (1.0, "um"), (5 / 3, "5/3"), (3.0, "grande")]:
    stds, g = rodar(gain)
    linha = "  ".join(f"{s:.3f} " for s in stds)
    print(f"{gain:7.3f}  {linha}   {g:.2e}")

print("""
Como ler a tabela:

  ganho 0.50  -> as ativacoes ENCOLHEM ate ~0. O sinal morre; as camadas do fim
                 recebem quase nada e o gradiente que volta e' minusculo.
  ganho 1.00  -> tambem encolhe, mais lentamente. A tanh e' o motivo: ela
                 COMPRIME o sinal (|tanh(x)| < |x|), entao ganho 1 nao basta
                 para compensar a perda.
  ganho 5/3   -> as ativacoes ficam ESTAVEIS ao longo das camadas. Esse 5/3
                 (~1.667) e' exatamente o valor que compensa a compressao da
                 tanh -- e' o "ganho de Kaiming" tabelado para essa ativacao.
  ganho 3.00  -> as ativacoes SATURAM perto de 1 (a tanh grudou nos extremos).
                 Lembre do Capitulo 2: tanh saturada tem derivada ~0, entao o
                 gradiente morre por outro caminho.
""")

# ---------------------------------------------------------------------------
# O ganho depende da ativacao. Para ReLU (que zera metade dos valores) o ganho
# certo e' sqrt(2) -- ela corta metade da variancia, e sqrt(2) recompoe.
# ---------------------------------------------------------------------------
print(f"=== o ganho correto depende da ATIVACAO (dim={DIM}) ===")
print(f"{'ativacao':>10s} {'ganho':>8s}  std na camada 1  std na camada 8")
for ativacao, gains in [("tanh", [1.0, 5 / 3]), ("relu", [1.0, 2 ** 0.5])]:
    for gain in gains:
        stds, _ = rodar(gain, ativacao=ativacao)
        print(f"{ativacao:>10s} {gain:8.3f}  {stds[0]:15.3f}  {stds[-1]:15.3f}")

print("""
Regra pratica (inicializacao de Kaiming):

    W ~ randn * gain / sqrt(fan_in)

  fan_in = numero de entradas do neuronio. Dividir por sqrt(fan_in) mantem a
  variancia estavel independentemente do TAMANHO da camada; o gain corrige o
  efeito da ATIVACAO (1.0 para linear, 5/3 para tanh, sqrt(2) para ReLU).

  No PyTorch: torch.nn.init.kaiming_normal_ e torch.nn.init.calculate_gain.
""")

# ---------------------------------------------------------------------------
# A alternativa moderna: nao brigar tanto pela inicializacao perfeita, e usar
# LayerNorm (Cap. 5) para RE-normalizar as ativacoes a cada bloco.
# ---------------------------------------------------------------------------
print("=== por que Transformers sofrem menos com isso ===")


def rodar_com_layernorm(gain, n_layers=N_LAYERS, dim=DIM):
    x = torch.randn(BATCH, dim)
    ln = torch.nn.LayerNorm(dim)
    h = x
    stds = []
    for _ in range(n_layers):
        W = torch.randn(dim, dim) * gain / dim ** 0.5
        h = torch.tanh(ln(h) @ W)      # LayerNorm ANTES, como no pre-norm do Cap. 5
        stds.append(h.std().item())
    return stds


for gain in (0.5, 1.0, 3.0):
    sem = rodar(gain)[0]
    com = rodar_com_layernorm(gain)
    print(f"  ganho {gain:4.2f}: sem LayerNorm c8={sem[-1]:.4f} | com LayerNorm c8={com[-1]:.4f}")

print("""
A LayerNorm re-normaliza as ativacoes a cada bloco, entao o valor exato da
inicializacao importa MENOS -- o erro nao se acumula ao longo da profundidade.
Isso nao dispensa inicializar bem (o inicio do treino ainda melhora), mas
explica por que o Transformer do Capitulo 5 treinou sem nenhum cuidado especial
de inicializacao: a LayerNorm estava fazendo esse trabalho.
""")
