"""
recompensa.py — as funcoes de recompensa deste capitulo.

Nao ha' preferencia humana aqui, e a escolha e' deliberada. Coletar preferencia
exige gente, e o Capitulo 11 ja' estabeleceu o criterio que vale para dados
gerados: eles ajudam quando existe um VERIFICADOR externo. Uma recompensa
calculavel por programa e' exatamente isso -- objetiva, reproduzivel, de graca.

Definimos DUAS, de proposito:

  COMPRIMENTO_ALVO  -- uma recompensa bem-comportada. Ela mede algo que o
                       modelo pode melhorar sem trapacear.

  MUITOS_PONTOS     -- uma recompensa MAL ESPECIFICADA, escolhida porque e'
                       facil de hackear. Ela existe para o modelo hackear.

A segunda e' o coracao do capitulo. Toda recompensa e' uma medida do que voce
quer, nunca o que voce quer -- e o RL otimiza a medida.

Run (a partir da pasta do capitulo):
    python recompensa.py
"""

import sys
from pathlib import Path

import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "11-datasets"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ALVO = 30           # tokens


def comprimento_alvo(sequencias, fins, fim_token):
    """Quanto mais perto de ALVO tokens, melhor. Recompensa em [0, 1].

    Bem-comportada: o unico jeito de maximizar e' produzir respostas do tamanho
    pedido. Nao ha' atalho.
    """
    comps = _comprimentos(sequencias, fim_token)
    return 1.0 - (comps - ALVO).abs().float() / ALVO


def muitos_pontos(sequencias, fins, fim_token):
    """Fracao de tokens da resposta que contem sinal de fim de frase.

    MAL ESPECIFICADA de proposito. A intencao por tras dela e' 'escreva frases
    completas, bem pontuadas'. O que ela MEDE e' 'use muitos pontos'.

    As duas coisas coincidem em texto normal -- e' por isso que a metrica parece
    razoavel. Elas deixam de coincidir assim que alguem otimiza a metrica.
    """
    B, T = sequencias.shape
    e_fim = torch.zeros(B, T, dtype=torch.bool)
    for t in fins:
        e_fim |= (sequencias == t)
    valido = _mascara_valida(sequencias, fim_token)
    n_validos = valido.sum(1).clamp(min=1)
    return (e_fim & valido).sum(1).float() / n_validos


def _mascara_valida(sequencias, fim_token):
    """True nas posicoes ate' o primeiro fim_token (exclusive)."""
    B, T = sequencias.shape
    e_fim = sequencias == fim_token
    # posicao do primeiro fim; T se nao houver
    primeiro = torch.where(e_fim.any(1), e_fim.float().argmax(1), torch.tensor(T))
    return torch.arange(T).unsqueeze(0) < primeiro.unsqueeze(1)


def _comprimentos(sequencias, fim_token):
    return _mascara_valida(sequencias, fim_token).sum(1)


RECOMPENSAS = {
    "comprimento": comprimento_alvo,
    "pontos": muitos_pontos,
}


if __name__ == "__main__":
    sys.path.insert(0, str(AQUI.parent / "14-sft"))
    from dataset import carregar_tokenizador
    from preparar_sft import FIM, tokens_de_fim_de_frase

    _, vocab = carregar_tokenizador()
    fins = tokens_de_fim_de_frase(vocab)

    print("=" * 74)
    print("As duas recompensas, em exemplos construidos a mao")
    print("=" * 74)
    ponto = next(iter(fins))
    letra = 100
    casos = {
        f"{ALVO} tokens, sem pontos": [letra] * ALVO + [FIM],
        "10 tokens, sem pontos": [letra] * 10 + [FIM],
        "60 tokens, sem pontos": [letra] * 60 + [FIM],
        f"{ALVO} tokens, metade pontos": ([letra, ponto] * (ALVO // 2)) + [FIM],
        f"{ALVO} tokens, SO' pontos": [ponto] * ALVO + [FIM],
    }
    largura = max(len(k) for k in casos)
    print(f"  {'sequencia':>{largura}s} {'comprimento':>12s} {'pontos':>9s}")
    for nome, seq in casos.items():
        s = torch.tensor([seq + [FIM] * (80 - len(seq))])
        r1 = comprimento_alvo(s, fins, FIM).item()
        r2 = muitos_pontos(s, fins, FIM).item()
        print(f"  {nome:>{largura}s} {r1:>12.2f} {r2:>9.2f}")
    print("""
  Olhe a ultima linha: uma resposta que e' SO' pontuacao tira nota maxima na
  recompensa 'pontos'. Nenhum humano chamaria aquilo de 'frases bem pontuadas'.

  A recompensa nao esta' com bug -- ela mede exatamente o que foi escrito. O
  problema e' que o que foi escrito nao e' o que se queria. E' assim que toda
  recompensa mal especificada se parece ANTES de alguem otimiza-la: razoavel.""")
