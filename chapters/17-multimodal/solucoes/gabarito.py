"""
Gabarito executavel do Capitulo 17 — multimodal.

Roda E2 (straight-through), E3 (as duas perdas) e E4 (codigos mortos).
O E1 e' conceitual; o E5 se responde lendo o codigo; E6 e E7 sao construcoes.

Run (a partir da pasta do capitulo):
    python solucoes/gabarito.py
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CAP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CAP))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import vqvae as V
from vqvae import VQVAE, carregar_mnist, treinar

PASSOS = 600          # menos que os 1500 da apostila: sao varios treinos

# ===========================================================================
print("=" * 74)
print("E2 — o straight-through estimator")
print("=" * 74)


class QuantSemST(V.Quantizador):
    """Sem a linha do straight-through: o gradiente morre no argmin."""

    def forward(self, z):
        B, d, H, W = z.shape
        plano = z.permute(0, 2, 3, 1).reshape(-1, d)
        dist = (plano.pow(2).sum(1, keepdim=True)
                - 2 * plano @ self.codebook.weight.t()
                + self.codebook.weight.pow(2).sum(1))
        indices = dist.argmin(1)
        z_q = self.codebook(indices).view(B, H, W, d).permute(0, 3, 1, 2)
        perda = (F.mse_loss(z_q, z.detach())
                 + V.BETA_COMP * F.mse_loss(z, z_q.detach()))
        return z_q, perda, indices.view(B, H, W)      # <- SEM o straight-through


def treinar_variante(classe_quant=None, beta=None, k=V.CODEBOOK, passos=PASSOS):
    beta_orig = V.BETA_COMP
    if beta is not None:
        V.BETA_COMP = beta
    torch.manual_seed(1337)
    tr, va = carregar_mnist()
    m = VQVAE(k)
    if classe_quant is not None:
        m.quant = classe_quant(k, V.D_LATENTE)
    opt = torch.optim.AdamW(m.parameters(), lr=V.LR)
    g = torch.Generator().manual_seed(1337)
    for _ in range(passos):
        ix = torch.randint(0, tr.shape[0], (V.BATCH,), generator=g)
        x = tr[ix]
        rec, perda_vq, _ = m(x)
        (F.mse_loss(rec, x) + perda_vq).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    m.eval()
    with torch.no_grad():
        rec, _, idx = m(va[:2000])
        erro = F.mse_loss(rec, va[:2000]).item()
        usados = len(torch.unique(idx))
    V.BETA_COMP = beta_orig
    return erro, usados


print(f"  {PASSOS} passos, codebook de {V.CODEBOOK}\n")
print(f"  {'variante':>28s} {'erro de reconstrucao':>22s} {'codigos usados':>16s}")
e_com, u_com = treinar_variante()
e_sem, u_sem = treinar_variante(classe_quant=QuantSemST)
print(f"  {'COM straight-through':>28s} {e_com:>22.4f} {u_com:>16d}")
print(f"  {'SEM straight-through':>28s} {e_sem:>22.4f} {u_sem:>16d}")
print(f"""
  Respostas:
  1. Sem o straight-through a reconstrucao fica MUITO pior ({e_sem:.4f} contra
     {e_com:.4f}). O decoder ainda aprende -- ele recebe gradiente normalmente --
     mas o ENCODER nao: o argmin corta a cadeia, e nada diz a ele que vetores
     produzir. Ele fica praticamente na inicializacao.
  2. z_q = z + (z_q - z).detach()
       forward : z + z_q - z  =  z_q          (o vetor quantizado, exato)
       backward: d/dz [z + constante]  =  1   (passa direto para o encoder)
     O .detach() congela o valor mas nao o numero -- e' um truque de contabilidade
     do grafo, nao de aritmetica.
  3. O gradiente e' aproximado: o encoder recebe o gradiente do decoder como se
     nao houvesse quantizacao. Funciona porque z e z_q sao PROXIMOS por
     construcao -- e o commitment loss existe justamente para mante-los proximos.
     A aproximacao e' boa exatamente na medida em que a outra perda faz o seu
     trabalho.""")

# ===========================================================================
print("=" * 74)
print("E3 — as duas perdas")
print("=" * 74)
print(f"  {'BETA_COMP':>12s} {'erro':>10s} {'codigos usados':>16s}")
for beta in (0.0, 0.25, 2.0):
    e, u = treinar_variante(beta=beta)
    print(f"  {beta:>12.2f} {e:>10.4f} {u:>16d}", flush=True)
print("""
  Respostas:
  1. Com BETA_COMP = 0 o resultado e' CATASTROFICO: erro 0,0755 e UM UNICO
     codigo usado. Sem nada que o prenda, o encoder foge para uma regiao que o
     codebook nao alcanca, e a quantizacao vira uma constante -- todo vetor cai
     no mesmo codigo. E' o mesmo colapso do E2, por outra causa.

  2. E AQUI EU ESTAVA ERRADO. Eu tinha escrito que BETA_COMP alto prenderia o
     encoder e pioraria a representacao. Medido, o 2,0 e' o MELHOR dos tres nos
     DOIS eixos: erro 0,0046 (contra 0,0069) e 105 codigos usados (contra 67).

     O valor 0,25 e' o do artigo original do VQ-VAE, e eu o tratei como se fosse
     um otimo demonstrado. Nesta configuracao -- MNIST, codebook 128, 600 passos
     -- nao e'. Um commitment mais forte mantem o encoder mais perto do
     codebook, o que faz mais codigos serem alcancaveis, o que melhora tudo.

     A licao nao e' "use 2,0". E' que um hiperparametro herdado de um artigo foi
     ajustado para OUTRA configuracao, e vale medir na sua. E' o mesmo padrao do
     Capitulo 11, onde a melhor learning rate mudou com o orcamento.

  3. O papel de cada uma:
       perda_codebook -- move o CODEBOOK em direcao ao encoder
       perda_commit   -- move o ENCODER em direcao ao codebook
     Sao os dois lados do mesmo encontro, e cada uma move um lado.

     Removendo a perda_codebook, os vetores do codebook so' se moveriam pelo
     gradiente que chega via decoder -- muito mais fraco. Na pratica o codebook
     ficaria quase congelado na inicializacao, e o encoder teria de se contorcer
     para caber num alfabeto aleatorio.""")

# ===========================================================================
print("=" * 74)
print("E4 — codigos mortos")
print("=" * 74)
print(f"  {'codebook':>10s} {'usados':>8s} {'fracao':>8s} {'erro':>9s}")
for k in (32, 64, 128, 256):
    e, u = treinar_variante(k=k)
    print(f"  {k:>10d} {u:>8d} {u/k:>7.0%} {e:>9.4f}", flush=True)
print("""
  Respostas:
  1. A FRACAO usada CAI conforme o codebook cresce, e o erro melhora pouco. Um
     codebook maior nao vira automaticamente mais capacidade -- vira mais codigos
     mortos.

     O mecanismo e' de realimentacao: para ser escolhido, um codigo precisa estar
     perto de algum vetor do encoder; para chegar perto, precisa ser escolhido e
     receber gradiente. Quem nasce longe nunca entra no jogo.

  2 e 3. A correcao (reinicializar codigos mortos sobre vetores do batch) e' o
     E4 item 2, e vale prestar atencao ao que ela melhora. Ela quase sempre
     aumenta MUITO a fracao usada -- e melhora POUCO o erro de reconstrucao.

     Isso nao a torna inutil, e sim mal-avaliada pela estatistica de uso. O ganho
     real aparece depois: um codebook bem aproveitado da' ao Transformer um
     vocabulario mais informativo, e e' la' que se deve medir.

     E' o mesmo tipo de armadilha do Capitulo 15: uma metrica que melhora sem que
     o sistema tenha melhorado no que importa.""")

# ===========================================================================
print("=" * 74)
print("E5 — o mesmo Transformer")
print("=" * 74)
print("""  Resposta 1: ZERO linhas do modelo. O gerar_imagens.py faz

      from modelo import GPT

  e instancia com outra config. A classe e' a mesma, byte por byte, que escreveu
  prosa nos capitulos 11 a 15.

  Resposta 2: a comparacao NAO e' justa, e vale entender por que.

      Capitulo 11 : perplexidade 51,7 entre 1.024 tokens
      Capitulo 17 : perplexidade  3,9 entre   128 tokens

  Perplexidade e' "entre quantas opcoes o modelo esta' efetivamente escolhendo".
  Comparar 3,9 com 51,7 ignora que os vocabularios diferem 8x. Normalizando pelo
  vocabulario: 3,9/128 = 3,0% contra 51,7/1024 = 5,0%.

  Mesmo essa conta e' fraca, porque as TAREFAS sao diferentes: prever o proximo
  pedaco de um digito manuscrito e' bem mais restrito que prever a proxima
  palavra de Machado. Numeros de perplexidade so' se comparam dentro da mesma
  tarefa e do mesmo tokenizador.

  Resposta 3: a ordem de varredura e' arbitraria para imagem, e cria um problema
  real -- o modelo gera a parte de cima sem saber o que vira embaixo, e nao pode
  voltar atras. Um traco mal comecado no topo condena o digito inteiro.

  Texto tem ordem natural (o tempo); imagem nao tem. Alternativas: gerar em
  ordem aleatoria com o modelo sabendo as posicoes, gerar em multiplas escalas
  (grosseiro primeiro, refinamento depois), ou abandonar a autorregressao -- que
  e' o que a difusao faz, refinando a imagem inteira de uma vez, varias vezes.""")
