# Exercícios — Capítulo 15 (RL)

Faça na ordem. Soluções comentadas em [`solucoes/`](solucoes/) — **tente antes de olhar**.

> Este capítulo usa o modelo do Capítulo 14. Se `../14-sft/modelo_sft.pt` não existir, rode
> antes `cd ../14-sft && python preparar_sft.py && python sft.py`.
>
> ⏱️ Os experimentos de RL são **lentos**: cada configuração de 400 passos leva ~50 min,
> porque cada passo gera 48 tokens com dois modelos (política e referência). Planeje.

---

### E1 — SFT e RL, lado a lado (aquecimento)
Sem rodar:
1. O Capítulo 14 treinou com `cross_entropy` contra respostas prontas. Aqui a loss é
   `-(R - baseline) * log_prob`. Descreva, em uma frase cada, **de onde vem o sinal** nos
   dois casos.
2. Por que o RL precisa **gerar** durante o treino, e o SFT não? O que isso custa?
3. O baseline é a média da recompensa no batch. Ele não muda o gradiente esperado — só a
   variância. Explique por quê. (Dica: qual é o valor esperado de `R - média(R)`?)

---

### E2 — Escreva uma recompensa ruim (importante)
O capítulo usa duas: `comprimento` (bem especificada) e `pontos` (mal especificada).
1. Escreva uma **terceira** recompensa que pareça razoável e seja hackeável. Antes de
   rodar, descreva **como** você espera que o modelo a hackeie.
2. Rode com `beta=0.0` e 400 passos. O modelo hackeou do jeito que você previu, ou achou
   outro caminho?
3. Toda recompensa da Seção 5 da apostila tem a forma "medir X como proxy de Y". Para a
   sua, qual é o X e qual é o Y — e onde eles deixam de coincidir?

---

### E3 — O baseline importa mesmo? (importante)
O `reinforce.py` aceita `--sem-baseline`.
1. Rode a recompensa `comprimento` com e sem baseline, mesma semente. Compare a curva de
   recompensa e a variância dela entre passos.
2. Sem baseline, **toda** resposta com recompensa positiva é reforçada — inclusive as
   ruins. Explique por que isso não impede o aprendizado, só o torna mais lento.
3. Se a sua recompensa fosse sempre negativa (digamos, de −1 a 0), o que aconteceria sem
   baseline? E isso sugere o quê sobre normalizar recompensas?

---

### E4 — Onde fica a faixa útil do β
Rode [`solucoes/e4_dial_do_kl.py`](solucoes/e4_dial_do_kl.py), que varre o β.
1. Faça a tabela de β contra três colunas: recompensa final, KL e **custo em português**.
   Onde está a faixa em que a recompensa sobe *e* o custo é pequeno?
2. Entre β = 0,02 e β = 0,10 há um fator de cinco. O que muda no comportamento do modelo
   nessa faixa? Por que a transição é tão abrupta?
3. Com β grande o custo vai a zero. Isso é sucesso? Olhe também a coluna da recompensa
   antes de responder.

---

### E5 — A recompensa não é o objetivo
1. Para as três configurações do `experimento.py`, ordene os modelos por **recompensa** e
   depois por **loss em Machado**. As ordens coincidem?
2. Imagine que você só reportasse a recompensa num relatório. Que conclusão o leitor
   tiraria? E ela seria falsa em que sentido exatamente?
3. Proponha **duas** métricas que um projeto real de RLHF deveria acompanhar além da
   recompensa, e diga o que cada uma detectaria que a recompensa não detecta.

---

### E6 — Conserte a recompensa
O problema de `pontos` não é o β — é a especificação.
1. Reescreva `muitos_pontos` para capturar melhor a intenção original ("escreva frases
   completas e bem pontuadas"). Pense no que distingue pontuação **útil** de pontuação
   **repetida**.
2. Rode com `beta=0.0`. A sua versão resiste sem freio nenhum?
3. Se resistiu: existe alguma recompensa que resista a qualquer otimização? Se não
   resistiu: o que isso diz sobre a estratégia de "escrever uma recompensa melhor"?

---

### E7 — DPO (desafio)
O DPO troca o RL por uma loss supervisionada sobre **pares** de preferência.
1. Gere pares com o próprio modelo: para cada pedido, amostre duas respostas e rotule a de
   maior recompensa como preferida. Você acabou de construir um dataset de preferências
   sintético.
2. Implemente a loss do DPO e treine. Compare com o REINFORCE em recompensa, KL e custo.
3. **A pergunta que importa:** o DPO evita o reward hacking? Justifique pelo que ele
   otimiza — e note que os seus pares foram rotulados pela **mesma** recompensa ruim.
