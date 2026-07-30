# Exercícios — Capítulo 11 (Datasets)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Rode `python prepare_data.py` uma vez antes dos exercícios (ele baixa e prepara o
> corpus; os livros ficam em cache, então rodar de novo é rápido).

---

### E1 — Por que essa ordem? (aquecimento)
Sem rodar:
1. Por que o tokenizador é treinado **só** no texto de treino? O que exatamente vazaria
   se ele visse o texto de validação?
2. Por que dividimos **por obra** e não sorteando parágrafos? Dê um exemplo concreto do
   que o modelo "veria de graça" na divisão errada.
3. Por que gravamos os tokens como `uint16` e não `int64`? Quanto de espaço isso poupa?

---

### E2 — Faça a divisão errada de propósito (importante)
Modifique o `prepare_data.py` para juntar **todas as 5 obras**, embaralhar os parágrafos e
separar 15% aleatoriamente para validação.
1. Treine o modelo com essa divisão. A loss de validação fica **melhor** ou pior?
2. Ela ficou melhor? Isso significa que o modelo é melhor? Explique o que aconteceu.
3. Esse é um dos erros mais comuns em ML aplicado — e ele faz o modelo parecer bom no
   relatório e falhar na prática. Como você detectaria isso numa revisão de código?
> Solução de referência: [`solucoes/e2_vazamento.py`](solucoes/e2_vazamento.py).

---

### E3 — O tokenizador é do domínio
O BPE treinado neste corpus aprendeu tokens como `'José Dias '`, `'Capitú, '` e
`'minha mã'`.
1. Liste os 20 tokens mais longos aprendidos. Quantos são nomes de personagens?
2. Isso é bom ou ruim? (Pense: e se você usasse este tokenizador para tokenizar um texto
   de medicina?)
3. Treine um BPE no corpus de **nomes** do Capítulo 6 e tokenize um trecho de Machado com
   ele. Quantos tokens a mais são necessários?

---

### E4 — Tamanho do contexto
No `train_text.py`, varie o `block_size`: 32, 128 e 256.
1. Como muda a loss de validação? E o tempo por passo?
2. O custo da atenção cresce com T². Meça: dobrar o contexto quadruplica o tempo?
3. Para prosa, o contexto maior ajuda mais do que ajudava para nomes. Por quê?

---

### E5 — Prever em todas as posições
Este capítulo mudou o modelo para prever em **todas** as posições, não só na última.
1. Quantas previsões por batch o modelo faz agora, contra antes? (Use `batch_size` e
   `block_size`.)
2. Modifique o `forward` para usar só a última posição (como nos capítulos 5–7) e treine
   com o mesmo orçamento de passos. A loss final é pior? Muito?
3. O custo do forward mudou? Se não mudou, o que isso diz sobre "aproveitar o cálculo"?

---

### E6 — Dados sintéticos (desafio)
O syllabus deste capítulo inclui **geração de dados sintéticos**. Use o modelo treinado
para gerar 200 KB de texto, e depois treine um **segundo** modelo só com esse texto
gerado.
1. O segundo modelo chega perto do primeiro? Ou fica pior?
2. Esse fenômeno tem nome na literatura (*model collapse*). Explique com os seus números
   por que treinar em dados gerados pelo próprio modelo degrada a qualidade.
3. Em que situações dados sintéticos **ajudam** de verdade? (Dica: pense em tarefas onde
   é possível **verificar** a resposta, como matemática ou código.)

---

### E7 — Escalando o corpus (desafio)
O `prepare_data.py` usa 5 obras. A lista completa de Machado no Project Gutenberg tem 12.
1. Acrescente mais obras e refaça o preparo. Como muda a loss de validação?
2. Mantenha o modelo do mesmo tamanho. A partir de quanto texto o ganho satura?
3. Relacione com o Capítulo 3: lá, aumentar os dados de 155 para 64 mil nomes resolveu o
   *overfitting*. A mesma lógica vale aqui? Compare as losses de treino e validação.
