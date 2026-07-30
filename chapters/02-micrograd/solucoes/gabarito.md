# Gabarito — Capítulo 02

> Respostas discursivas. Os **números** vêm de [`gabarito.py`](gabarito.py).

---

## E1 — Leitura de código

**1. Para `f = a * b` com `a = 3`, `b = 4`:** `df/da = b = 4` e `df/db = a = 3`.
A derivada em relação a um fator é o **outro** fator.

**2. Para `d = a*b + c`:** `dd/dc = 1`. A soma não altera o gradiente porque a derivada de
uma soma em relação a cada parcela é 1 — ela apenas **distribui** o gradiente que chega,
sem escalá-lo. É por isso que, no código, o `_backward` da soma é
`self.grad += 1.0 * out.grad`.

**3. Por que `tanh` saturada tem gradiente quase zero:** a derivada é `1 − tanh(x)²`. Para
`x` grande, `tanh(x) → ±1`, logo `tanh(x)² → 1` e a derivada `→ 0`. Nas regiões extremas a
curva é praticamente horizontal — mexer na entrada quase não muda a saída, e o gradiente
morre. É o mesmo efeito que reaparece no softmax do Capítulo 4 (daí o fator `1/√d`) e na
inicialização do Capítulo 7.

---

## E2 — Conferindo com diferença finita

| Variável | Autograd | Diferença finita | Erro relativo |
|----------|----------|------------------|---------------|
| `a` | −0,00402285 | −0,00402285 | 5,97e-10 |
| `b` | +0,00268190 | +0,00268190 | 5,97e-10 |
| `c` | +0,00134095 | +0,00134095 | 5,97e-10 |

Batem. Dois detalhes que valem a pena:

- Usamos a **diferença central**, `(f(x+h) − f(x−h)) / 2h`, e não a ingênua
  `(f(x+h) − f(x)) / h`. A central tem erro `O(h²)` em vez de `O(h)` — bem mais precisa
  pelo mesmo custo.
- Existe um `h` **ótimo**: grande demais e a aproximação da derivada é ruim; pequeno
  demais e o cancelamento em ponto flutuante domina (você subtrai dois números quase
  iguais). Por volta de `1e-6` costuma ser um bom meio-termo em `float64`.

Esta é a forma clássica de **testar** um autograd, e é o que bibliotecas de verdade usam
nos seus testes (`torch.autograd.gradcheck` faz exatamente isso).

---

## E3 — Implementando `log()`

**1.** A derivada local é `d(ln x)/dx = 1/x`.

**2.** Seguindo o padrão de sempre — derivada local vezes o gradiente que vem de fora:

```python
def log(self):
    out = Value(math.log(self.data), (self,), "log")
    def _backward():
        self.grad += (1.0 / self.data) * out.grad
    out._backward = _backward
    return out
```

**3.** Verificado: `log(3.0) = 1,098612` (bate com `math.log`), e `d(log)/dx = 0,333333`
contra `0,333333` da diferença finita — erro de `4,66e-11`.

> Cuidado que o exercício não pede mas vale saber: `log` só é definido para `x > 0`. Numa
> implementação séria você trataria `x <= 0` explicitamente, em vez de deixar o
> `math.log` levantar exceção no meio de um treino.

---

## E4 — Por que zerar o gradiente

Medido, 20 passos:

| Configuração | passo 0 | passo 5 | passo 19 |
|---|---------|---------|----------|
| **com** `zero_grad` | 3,0929 | 0,0006 | **0,0001** |
| **sem** `zero_grad` | 3,0929 | 0,7659 | **0,6158** |

**1 e 2. Leia os números, não a intuição.** Sem o `zero_grad` o treino **não explode neste
caso — ele estagna.** A loss cai até ~0,6 e trava, enquanto com `zero_grad` chega a 0,0001.

Por que estagna em vez de divergir? Os gradientes são acumulados com `+=`, então o
gradiente do passo N é a soma de todos os anteriores. Isso produz dois efeitos que se
combatem: o passo fica **grande demais** (empurra para divergir) mas também
**desatualizado** — carrega direções calculadas para pesos que já mudaram. Aqui o modelo
entra num vai-e-vem e não converge. Com learning rate maior, ou em outros problemas, o
mesmo bug realmente diverge.

**3.** É exatamente o `W.grad = None` do Capítulo 1, pelo mesmo motivo.

> **A lição que mais importa:** o resultado fica errado de um jeito que **não se parece com
> um erro**. Não há exceção, não há `NaN` — só um treino pior. Bugs assim são os mais caros
> de encontrar, e é por isso que se compara sempre contra uma referência.

---

## E5 — Bate com o PyTorch?

Solução em [`e5_check_vs_torch.py`](e5_check_vs_torch.py). Resultado: `ALL MATCH: True`,
com diferença máxima de **8,94e-08**.

**2.** Trocando `tanh` por `relu`, continuam batendo — o mecanismo é o mesmo, só muda a
derivada local registrada por aquela operação.

---

## E6 — Arquitetura e learning rate

| Arquitetura | Parâmetros | Loss final | Situação |
|-------------|-----------|-----------|----------|
| `[4, 4, 1]` | 41 | 0,000000 | convergiu |
| `[8, 1]` | 41 | 0,000000 | convergiu |
| `[16, 16, 1]` | 353 | **1,38e+147** | **DIVERGIU** |

| Learning rate | Loss final | Situação |
|---------------|-----------|----------|
| 0,5 | **1,80e+248** | **DIVERGIU** |
| 0,05 | 0,000000 | boa |
| 0,001 | 0,136863 | lenta demais |

**1. A primeira resposta contraria o senso comum.** `[4,4,1]` e `[8,1]` convergem — e, por
coincidência, têm exatamente os **mesmos 41 parâmetros**. Mas `[16,16,1]`, com 353
parâmetros, **diverge** com a mesma learning rate.

Rede maior não é automaticamente melhor. Com mais parâmetros, a soma dos gradientes fica
maior, e uma `lr` que servia para a rede pequena passa a ser grande demais. **A learning
rate adequada depende do tamanho do modelo** — que é justamente o que motiva a
inicialização escalada e o agendamento do Capítulo 7.

**2.** A curva em U clássica: `0,5` diverge, `0,05` converge, `0,001` quase não anda no
orçamento de 100 passos. O Capítulo 7 mede isso de forma sistemática (exercício E5 de lá).

**3.** Trocar `tanh` por `relu` neste problema faz pouca diferença: com 4 pontos de dados,
qualquer não-linearidade razoável resolve. A diferença entre ativações aparece em redes
profundas, e é o assunto da Seção 2 do Capítulo 7.

---

## E7 — Visualizando o grafo

Implementação em [`gabarito.py`](gabarito.py). Para `(a*b + c).tanh()`:

```
Value(data=0.9993, grad=1.0000) [tanh]
  +-- Value(data=4.0000, grad=0.0013) [+]
    +-- Value(data=10.0000, grad=0.0013)
    +-- Value(data=-6.0000, grad=0.0013) [*]
      +-- Value(data=2.0000, grad=-0.0040)
      +-- Value(data=-3.0000, grad=0.0027)
```

Leia de cima para baixo: a raiz é a saída, e cada nível mostra os "pais" que a produziram.
A `backward()` percorre exatamente esta estrutura, aplicando a regra da cadeia.

Repare em como o gradiente se comporta:

- O `tanh` na raiz tem `grad = 1.0` (é a semente, `d(saída)/d(saída) = 1`).
- A **soma** distribui `0,0013` igualmente para `c` e para o produto — sem alterar nada.
- A **multiplicação** escala: `a` recebe `0,0013 × (−3) = −0,0040` e `b` recebe
  `0,0013 × 2 = 0,0027`. Cada um multiplicado pelo valor do outro.

E note que todos os gradientes são pequenos (~1e-3): a `tanh` está saturada em `x = 4`, o
que confirma o que a resposta do E1.3 previu.
