# Exercícios — Capítulo 08 (Device)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> **Sem GPU?** Os exercícios E1, E3 e E7 funcionam na CPU. Os demais precisam de GPU —
> veja [`SETUP-GPU.md`](SETUP-GPU.md), ou use o Google Colab (GPU gratuita).

---

### E1 — Latência vs vazão (aquecimento)
Sem rodar código:
1. Por que a GPU **perde** da CPU numa matmul de 128×128?
2. Na tabela da Seção 2, a CPU fica em ~340 GFLOP/s em todos os tamanhos, enquanto a GPU
   sobe de 167 a 5.500. Explique a diferença em termos de "núcleos ocupados".
3. Uma tarefa que precisa de muitos `if` e desvios roda melhor em CPU ou GPU? Por quê?

---

### E2 — Encontre o SEU ponto de virada
Rode `python benchmark.py` na sua máquina.
1. A partir de qual tamanho de matriz a sua GPU passa a ganhar?
2. Qual o speedup máximo que você observa? Ele estabiliza?
3. Compare os GFLOP/s de pico da sua GPU com a especificação oficial da placa. Você chega
   perto? (Dificilmente — e a diferença é o que os capítulos 9 e 10 atacam.)

---

### E3 — A armadilha do `.item()`
No laço de treino do `train_device.py`, adicione uma leitura do valor da loss **a cada
passo**:

```python
historico.append(loss.item())     # força transferência + sincronização
```

1. Quanto o tempo por passo aumenta? Meça com o modelo **médio**, na GPU.
2. Por que essa linha é tão caro? (Duas coisas acontecem: transferência **e**
   sincronização.)
3. Como registrar a loss sem pagar esse preço a cada passo? (Dica: a cada quantos passos
   você realmente precisa dela?)

---

### E4 — Tamanho do batch na GPU
Com o modelo **médio** na GPU, varie o batch: 32, 128, 512, 2048.
1. O tempo por passo cresce proporcionalmente ao batch, ou mais devagar?
2. Calcule o **tempo por exemplo** (tempo do passo ÷ batch). Onde ele é menor?
3. Aumente o batch até dar erro de memória. Qual o limite da sua placa?
4. Cuidado conceitual: batch maior processa mais exemplos por segundo, mas **cada passo
   dá um passo só** de gradiente. Isso significa que o treino inteiro fica mais rápido?
   (Releia o E5 do Capítulo 7 sobre a relação entre batch e learning rate.)

---

### E5 — Meça errado de propósito
Na função `cronometrar` do `benchmark.py`, remova a chamada a `drenar()` (tanto a de
aquecimento quanto a final).
1. Que speedups a GPU passa a "apresentar"? Eles fazem sentido físico?
2. Por que a medição da CPU **não** é afetada por essa remoção?
3. Esse é um erro que se detecta olhando o quê? (Dica: o que a curva de speedup faz
   quando a medição está contaminada?)

---

### E6 — Operações que caem de volta na CPU (desafio)
Ao rodar `train_device.py` no DirectML, aparece um aviso: o `AdamW` usa `aten::lerp`, que
não é implementada, e essa parte executa na CPU.
1. Rode com `python -W always::UserWarning train_device.py` e liste **todas** as operações
   que caem de volta.
2. Troque o `torch.optim.AdamW` por `torch.optim.SGD` (que não usa `lerp`). O aviso
   desaparece? O speedup da GPU **melhora**?
3. Se melhorar: o que isso diz sobre os números da apostila? (Eles são um piso ou um
   teto do que a placa pode dar?)
4. Em CUDA esse problema não existe. Que lição geral você tira sobre depender de um
   backend menos maduro?

---

### E7 — Quando vale a pena mudar para GPU? (desafio)
Junte os dados que você mediu.
1. Usando o tempo por passo dos três tamanhos, estime a partir de **quantos parâmetros**
   (ou de que dimensão `n_embd`) a GPU passa a valer a pena na sua máquina.
2. Considere o custo total: se treinar na GPU exige instalar um ambiente separado e
   depurar operações não suportadas, a partir de que ponto o ganho justifica o trabalho?
3. Para o modelo **pequeno** deste curso, qual dispositivo você recomendaria? Justifique
   com números — e note que essa é a resposta contra-intuitiva do capítulo.
