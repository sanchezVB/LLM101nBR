# LLM101n-BR — Panorama do Estado Atual

> Relatório de progresso do projeto em **01/06/2026**.
> Curso prático e bilíngue para construir um LLM do zero, inspirado no LLM101n.

---

## 1. Resumo executivo

O projeto tem hoje uma **base sólida e publicada**: as **Fases I e II estão completas**
— o **Transformer**, o **tokenizador** e o **treino afinado** já estão construídos — e a
Fase III está na metade. São **9 capítulos** prontos, testados e no GitHub. Cada
capítulo entregue inclui apostila, código executável, exercícios com soluções e PDF — e
todo número citado no texto foi obtido rodando o código de verdade.

**O marco:** o curso já entrega um **GPT funcional** com **tokenizador próprio**. A
arquitetura do Capítulo 5 é, em estrutura, a mesma do GPT-2, e o BPE do Capítulo 6 é o
mesmo algoritmo que o GPT usa — ambos construídos peça por peça, do zero.

| Indicador | Valor |
|-----------|-------|
| Capítulos concluídos | **9 de 17** (53%) |
| Estado | Fases I e II completas; Fase III em 2/4 |
| Melhor modelo | Transformer + agendamento de lr, **1,776** (vs ~2,4 do bigrama) |
| Repositório | Publicado e versionado (11 commits) |
| Arquivos versionados | 65 |
| Linhas de código (didático) | ~4.200 (Python) |
| PDFs gerados | 9 capítulos + panorama + este relatório |
| Verificação | 100% do código roda; autograd, LayerNorm e AdamW batem com PyTorch |

---

## 2. Onde está o projeto

- **Repositório GitHub:** <https://github.com/sanchezVB/LLM101nBR> (público)
- **Cópia local:** `C:\Users\User\llm101n-curso`
- **Branch:** `main` — sincronizada com o GitHub

### Histórico de commits

| Commit | Data | Descrição |
|--------|------|-----------|
| `f091c85` | 01/06/2026 | Estrutura inicial do curso + Capítulo 01 |
| `58d67ed` | 01/06/2026 | Capítulo 02 — Micrograd |
| `5fcb0fe` | 01/06/2026 | Panorama do curso + suporte a `--file` no gerador de PDF |
| `a18b020` | 01/06/2026 | Panorama do estado atual |
| `0547d63` | 01/06/2026 | Capítulo 03 — N-gram model (MLP) + dataset do IBGE |
| `6f09fd3` | 01/06/2026 | Capítulo 04 — Attention + correções no gerador de PDF |
| `1d7c34a` | 01/06/2026 | Capítulo 05 — Transformer (arquitetura GPT-2) |
| `5f123b3` | 01/06/2026 | Capítulo 06 — Tokenization (BPE do zero) |
| `61269d0` | 01/06/2026 | Capítulo 07 — Optimization (fecha a Fase II) |
| `b1b0318` | 01/06/2026 | Capítulo 08 — Device (CPU/GPU), medido na Radeon |
| (atual) | 01/06/2026 | Capítulo 09 — Precision (fp16/bf16, loss scaling) |

---

## 3. Capítulos: o que está pronto e o que falta

### Fase I — Fundamentos

| # | Capítulo | Estado |
|---|----------|--------|
| 01 | Bigram Language Model | **Concluído** |
| 02 | Micrograd | **Concluído** |
| 03 | N-gram model (MLP) | **Concluído** |

### Fase II — O Transformer

| # | Capítulo | Estado |
|---|----------|--------|
| 04 | Attention | **Concluído** |
| 05 | Transformer | **Concluído** |
| 06 | Tokenization | **Concluído** |
| 07 | Optimization | **Concluído** |

### Fase III — Velocidade e escala

| # | Capítulo | Estado |
|---|----------|--------|
| 08 | Device (CPU/GPU) | **Concluído** |
| 09 | Precision | **Concluído** |
| 10 | Distributed | A fazer (próximo) |
| 11 | Datasets | A fazer |

### Fase IV — Inferência e refinamento

| # | Capítulo | Estado |
|---|----------|--------|
| 12 | Inference I: KV-cache | A fazer |
| 13 | Inference II: Quantization | A fazer |
| 14 | Finetuning I: SFT | A fazer |
| 15 | Finetuning II: RL | A fazer |

### Fase V — Produto e além

| # | Capítulo | Estado |
|---|----------|--------|
| 16 | Deployment | A fazer |
| 17 | Multimodal | A fazer |

**Progresso por fase:**

| Fase | Concluídos | % |
|------|-----------|---|
| I — Fundamentos | 3 / 3 | **100%** |
| II — Transformer | 4 / 4 | **100%** |
| III — Velocidade e escala | 2 / 4 | 50% |
| IV — Inferência e refinamento | 0 / 4 | 0% |
| V — Produto e além | 0 / 2 | 0% |
| **TOTAL** | **9 / 17** | **53%** |

---

## 4. Detalhe do que foi entregue

### Capítulo 01 — Bigram Language Model

Conteúdo (10 páginas no PDF):

- **Apostila** (`README.md`) — modelo de linguagem, bigrama, vocabulário,
  contagem → probabilidades, sampling, loss (NLL), a virada para rede neural,
  gradient descent, regularização.
- **`bigram.py`** (71 linhas) — versão por contagem. *Loss* medida ≈ **2,38**.
- **`bigram_nn.py`** (82 linhas) — versão rede neural. *Loss* converge a ≈ **2,2**.
- **`exercicios.md`** — 7 exercícios; solução E6 incluída.
- **`names.txt`** — dataset de 155 nomes.

**Lição central verificada:** contar é um caso particular de otimizar; a diferença de
loss vem só do grau de suavização (confirmado no exercício E5).

### Capítulo 02 — Micrograd

Conteúdo (9 páginas no PDF):

- **Apostila** (`README.md`) — derivada, regra da cadeia, grafo de computação,
  backpropagation, ordenação topológica, e a equivalência com o PyTorch.
- **`micrograd.py`** (131 linhas) — motor de autograd do zero (classe `Value`:
  `+`, `*`, `**`, `tanh`, `exp`, `relu`, `backward()`).
- **`nn.py`** (91 linhas) — `Neuron → Layer → MLP`; demo de treino a *loss* ≈ **0**.
- **`exercicios.md`** — 7 exercícios; solução E5 incluída.

**Validação rigorosa:** os gradientes do nosso motor **batem com os do PyTorch até a
6ª casa decimal** (`ALL MATCH: True`).

### Capítulo 03 — N-gram model (MLP)

Conteúdo (9 páginas no PDF):

- **Apostila** (`README.md`) — a maldição da dimensionalidade, embeddings, `view`,
  `matmul`, GELU, `cross_entropy`, mini-batches e **treino/validação/teste**.
- **`mlp.py`** (~150 linhas) — MLP char-level em PyTorch. Loss **1,97** nos três
  splits (praticamente idênticas = generaliza), **batendo o bigrama** (~2,4).
- **`prepare_data.py`** + **`names.txt`** — dataset real de **64 mil nomes** do IBGE.
- **`exercicios.md`** — 7 exercícios; solução E5 incluída.

**Decisão de projeto:** o dataset de 155 nomes dos capítulos anteriores causava
*overfitting* severo (treino 0,80 vs validação 6,51). Trocá-lo pela base do IBGE foi o
que tornou o capítulo honesto — e o contraste virou a lição central, demonstrada na
solução do E5.

### Capítulo 04 — Attention

Conteúdo (10 páginas no PDF):

- **Apostila** (`README.md`) — o mecanismo em 4 versões equivalentes (média em loop →
  matmul com `tril` → softmax mascarado → **query/key/value**), a escala `1/√d`, a
  máscara causal e o embedding posicional.
- **`attention.py`** — as 4 versões, com prova de equivalência entre elas.
- **`model.py`** — modelo de linguagem com atenção; constante `USE_FEEDFORWARD` para
  o experimento central do capítulo.
- **`exercicios.md`** — 7 exercícios; solução E6 (inspeção dos pesos) incluída.

**Descoberta honesta:** com parâmetros equiparados, a atenção **sozinha perde** do MLP
(2,099 vs 1,967) — porque ela **comunica** mas não **computa**. Adicionando o
feedforward, cai para **1,913**. Isso motiva diretamente o Capítulo 5: o Transformer é
a combinação das duas metades.

### Capítulo 05 — Transformer

Conteúdo (9 páginas no PDF):

- **Apostila** (`README.md`) — multi-head attention, conexões residuais (e por que
  viabilizam profundidade), LayerNorm, o bloco, pre-norm vs post-norm, e a montagem do
  modelo completo.
- **`layernorm.py`** — LayerNorm do zero, **verificada contra `nn.LayerNorm`**
  (diferença 4,8e-07) e demonstrando a independência do batch (por que não BatchNorm).
- **`transformer.py`** — o modelo completo: 3 blocos, 4 cabeças, `n_embd=64`,
  **153.499 parâmetros**. Treina em ~6 min na CPU.
- **`exercicios.md`** — 7 exercícios; solução E2 (ablação dos residuais) incluída.

**Resultado:** loss de validação **1,811** — o melhor do curso até então (superado pelo
Cap. 7, que chega a 1,776 só com otimização). Progressão completa:
2,4 (bigrama) → 1,967 (MLP) → 1,913 (atenção+ff) → **1,811 (Transformer)**. Treino
(1,791) e validação (1,811) próximos: sem *overfitting* relevante.

**Marco:** a arquitetura deste capítulo é, em estrutura, **a mesma do GPT-2**. Para
referência, o GPT-2 small tem 124 milhões de parâmetros, 12 blocos e 12 cabeças — mesmo
desenho, outra escala.

### Capítulo 06 — Tokenization (BPE)

Conteúdo (10 páginas no PDF). Capítulo em Python puro, sem PyTorch, roda em segundos:

- **Apostila** (`README.md`) — Unicode, UTF-8, por que começar pelos bytes, o algoritmo
  BPE, a importância da ordem das fusões, e o "imposto do português".
- **`unicode_utf8.py`** — caractere → code point → bytes, com o custo dos acentos.
- **`bpe.py`** — o tokenizador completo: `train`, `encode`, `decode`, compressão e
  verificação de **round-trip** (`decode(encode(x)) == x`).
- **`exercicios.md`** — 7 exercícios; solução E5 incluída.

**Resultados:** compressão **2,24x** no domínio de treino e **2,17x** em nomes novos (ele
aprendeu o padrão, não decorou). Round-trip passou em **todos** os casos, incluindo
acentos, japonês, emoji e string vazia — a garantia dos bytes funcionando.

**O que o algoritmo descobriu sozinho:** entre os tokens mais longos aprendidos estão
`'ilson'`, `'erson'`, `'ilton'`, `'iana'` — sufixos produtivos de nomes brasileiros.
Ninguém ensinou morfologia; é contagem de pares virando estrutura linguística.

**Achado com consequência prática:** treinando um segundo tokenizador do **mesmo
tamanho** em português com acentos (usando as apostilas do curso como corpus), a mesma
frase passa de **50 para 25 tokens** — **53% de economia**. Como APIs de LLM cobram por
token e o contexto é medido em tokens, escrever em português num tokenizador treinado em
inglês custa mais caro pelo mesmo conteúdo. O capítulo mede isso.

### Capítulo 07 — Optimization

Conteúdo (11 páginas no PDF). Fecha a Fase II:

- **Apostila** (`README.md`) — inicialização (de onde vem o `5/3` do Cap. 3), a escada
  SGD → momentum → Adam → AdamW, correção de viés, weight decay desacoplado, warmup +
  cosine decay, gradient clipping e como calibrá-lo.
- **`initialization.py`** — mede o desvio padrão das ativações em 8 camadas com
  diferentes ganhos: ganho errado faz o sinal morrer (0,003) ou saturar. Mostra também
  que a LayerNorm reduz muito essa sensibilidade.
- **`optimizers.py`** — SGD, SGD+momentum e **AdamW do zero**, este último **verificado
  contra `torch.optim.AdamW`** (diferença 8,9e-08).
- **`train_tuned.py`** — a **ablação** das quatro técnicas, uma por vez.
- **`exercicios.md`** — 8 exercícios; solução E5 (curva da learning rate) incluída.

**A ablação e seu resultado inesperado:**

| Configuração | Loss validação | vs baseline |
|--------------|----------------|-------------|
| baseline (Cap. 5) | 1,8114 | — |
| **só agendamento** | **1,7760** | **+0,0355** |
| só clipping | 1,8114 | 0,0000 |
| só weight decay 0,1 | 1,8646 | -0,0532 |
| só init escalada | 1,8187 | -0,0073 |
| tudo junto | 1,8110 | +0,0004 |

Das quatro técnicas, **só o agendamento ajudou** — e chegou a **1,776**, o melhor modelo
do curso, sem tocar na arquitetura. O clipping foi exatamente neutro (nenhum passo
cortado). O weight decay alto **piorou** mais do que qualquer outra coisa ajudou, porque
combate *overfitting* e este modelo não está decorando (treino 1,791 vs validação 1,811).
A init escalada, criada para modelos de 12+ blocos, atrapalha num de 3.

**Lição central:** "melhores práticas" não são aditivas nem universais. "Tudo junto" deu
quase zero porque ganhos e perdas se cancelaram. A única forma de saber é medir uma por
vez.

**Um erro meu, documentado na apostila:** a primeira versão usava `GRAD_CLIP = 1.0` e
cortava **99% dos passos** — deixando de ser clipping e virando normalização de todo
gradiente. A correção (medir a norma típica antes de escolher o limite) virou seção do
capítulo.

### Capítulo 08 — Device (CPU/GPU)

Conteúdo (14 páginas no PDF, incluindo o anexo de instalação). Abre a Fase III:

- **Apostila** (`README.md`) — latência vs vazão, o ponto de virada, custo de
  transferência, efeito do batch, código portátil e **como medir GPU sem se enganar**.
- **`SETUP-GPU.md`** — instalação nos três caminhos: NVIDIA (CUDA), AMD/Intel no Windows
  (DirectML) e sem GPU. Incluído no PDF para ele ser autossuficiente.
- **`device.py`** — detecção portátil de dispositivo (nunca `.cuda()` cravado).
- **`benchmark.py`** — matmul, transferência e batch, CPU vs GPU.
- **`train_device.py`** — treino nos dois dispositivos, em três tamanhos de modelo.
- **`exercicios.md`** — 7 exercícios; solução E6 incluída.

**Medido de verdade** numa **AMD Radeon RX 7600** via DirectML (a máquina não tem NVIDIA,
então não há CUDA; o Python 3.12 já instalado permitiu usar o `torch-directml`):

| Matmul | Speedup | GFLOP/s CPU | GFLOP/s GPU |
|--------|---------|-------------|-------------|
| 128×128 | **0,75x** | 223 | 167 |
| 1024×1024 | 15,26x | 343 | 5.240 |
| 4096×4096 | 15,72x | 349 | 5.493 |

A CPU **satura** em ~340 GFLOP/s; a GPU escala de 167 até ~5.500. E em matrizes pequenas
a GPU **perde**.

**O resultado mais útil do capítulo** — treinando o nosso Transformer:

| Modelo | Parâmetros | Speedup GPU |
|--------|-----------|-------------|
| pequeno (o do curso) | 153 mil | **0,30x** (3,3x mais lento!) |
| médio | 3,2 milhões | 2,65x |
| grande | 18,9 milhões | 6,82x |

**O modelo deste curso é mais lento na GPU.** As matmuls dele (64×64) estão na faixa onde
a GPU perde. A recomendação honesta é continuar na CPU — e agora sabemos medir a partir de
quando vale trocar.

**Dois achados adicionais:**

1. **Lacuna silenciosa de backend.** O AdamW usa `aten::lerp`, que o DirectML não
   implementa e executa na CPU a cada passo. Trocar para SGD (sem essa lacuna) **dobra** o
   ganho da GPU: 2,65x → 5,94x. A operação custava ~42 ms/passo, mais da metade do tempo.
   Não aparece como erro — só como lentidão.
2. **Não há reprodutibilidade bit-exata entre dispositivos.** A diferença de loss CPU↔GPU
   é 4,8e-07 no modelo pequeno mas **8,5e-03** no grande: o treino é caótico e amplifica
   o arredondamento ao longo dos passos. Compare estatísticas, não valores exatos.

**Erro meu, documentado na apostila:** a primeira medição deu speedups não-monotônicos
(0,62x em 512 e 18x em 1024) porque eu sincronizava a GPU de forma inadequada — medindo o
*enfileiramento* em vez da *execução*. A correção (drenar lendo o próprio resultado,
aquecer antes, usar o mínimo de várias rodadas) virou a Seção 6 do capítulo.

### Capítulo 09 — Precision

Conteúdo (12 páginas no PDF):

- **Apostila** (`README.md`) — anatomia de um float, alcance vs precisão, overflow e
  underflow, por que o bf16 venceu, loss scaling, precisão mista e pesos mestres.
- **`floats.py`** — os bits de fp32/fp16/bf16, limites de cada formato, sobrevivência de
  gradientes típicos e consumo de memória.
- **`precision_bench.py`** — velocidade por precisão (com repetições **adaptativas**, para
  o caso patológico não fazer o benchmark levar minutos) e a demonstração dos pesos mestres.
- **`loss_scaling.py`** — o problema medido e um **GradScaler dinâmico do zero**.
- **`exercicios.md`** — 7 exercícios (E1–E5 rodam sem GPU).

**O achado principal, contra o discurso comum:** neste hardware, precisão reduzida compra
**memória, não velocidade**.

| Dispositivo | fp32 | fp16 | Ganho |
|-------------|------|------|-------|
| CPU | 5,35 ms | 1.951 ms | **~0,003x** (centenas de vezes pior) |
| Radeon RX 7600 | 0,40 ms | 0,41 ms | **0,98x** (nenhum) |

A CPU não tem unidades de 16 bits para matmul (emula elemento por elemento); o DirectML não
despacha para os kernels de 16 bits da RDNA3. O ganho de velocidade exige hardware **e**
backend preparados — na prática, NVIDIA com tensor cores. O de memória é garantido (metade
dos bytes por parâmetro).

**bf16 nesta placa aborta o processo** — com erro fatal, não exceção capturável:
`[F] Invalid or unsupported data type BFloat16`. Por isso o script consulta uma lista de
suporte conhecido **antes** de tentar. Lição prática: em backend imaturo, "não suportado"
pode significar "seu programa morre".

**O que foi medido e vale em qualquer máquina:**

- **50,3% dos gradientes viram zero em fp16** (mediana 2,93e-08 vs mínimo normal 6,10e-05).
  Metade do modelo não aprende — e sem erro nenhum.
- **A janela do loss scaling:** escala 1 perde 50,3%; 1.024 a 65.536 funcionam; 2²² em
  diante causa overflow. A janela **se move** durante o treino → daí o scaler dinâmico.
- **Pesos mestres:** somando 1e-4 a um peso 1,0 cem vezes, o fp32 chega a 1,010002 e o
  fp16/bf16 ficam em **1,000000** — o peso não se moveu nem uma vez.

---

## 5. Infraestrutura montada

Além dos capítulos, o projeto já tem toda a "fundação" pronta — não precisa ser
refeita a cada novo capítulo:

| Item | Estado |
|------|--------|
| Estrutura de pastas e convenções | Pronta |
| `README.md` principal com syllabus | Pronto |
| `SETUP.md` (preparar ambiente) | Pronto |
| `requirements.txt` (torch, numpy, matplotlib) | Pronto |
| Licença (MIT + CC BY 4.0) | Pronta |
| `.gitignore` / `.gitattributes` | Prontos |
| **Gerador de PDF** (`tools/build_pdf.py`) | Pronto e reutilizável |
| Ambiente Python (3.14 + torch 2.12 CPU) | Instalado e testado |

O gerador de PDF é um pipeline **100% offline** (Markdown → HTML → PDF, sem LaTeX) e
serve qualquer capítulo, a apostila completa ou um documento avulso.

---

## 6. Qualidade e verificação

Práticas aplicadas em todo capítulo entregue:

- **Código sempre executado** antes de documentar — nenhum número é inventado.
- **Comparação com PyTorch** onde faz sentido (Cap. 2: gradientes idênticos).
- **Comparações justas** — modelos comparados com orçamento de parâmetros equivalente;
  quando não é possível, isso é declarado no texto (Cap. 4).
- **PDFs inspecionados** visualmente (acentuação, formatação, código).
- **Versionamento limpo** — arquivos temporários ignorados; commits descritivos.

**Defeitos encontrados e corrigidos nesta rodada** (achados durante a inspeção visual
dos PDFs, afetavam todos os capítulos):

1. **Código virava linha corrida** — o `xhtml2pdf` ignora `white-space: pre-wrap` e
   colapsava as quebras de linha. Corrigido convertendo quebras em `<br/>` e
   indentação em espaços rígidos.
2. **Setas `→` desapareciam** — eram removidas junto com os emojis, alterando o sentido
   do texto (`Neuron → Layer` virava `Neuron Layer`). Agora são mapeadas para `->`.
3. **Diagramas de árvore viravam quadrados pretos** — as fontes do PDF não cobrem
   caracteres de desenho de caixa. Agora convertidos para ASCII (`+--`).
4. **Rede de segurança adicionada** — o gerador agora **avisa** se encontrar qualquer
   caractere sem cobertura na fonte, em vez de falhar silenciosamente.

---

## 7. Próximo passo

**Capítulo 10 — Distributed.** Fecha a trilogia "Need for Speed": treinar em **vários
dispositivos** ao mesmo tempo, com o `all-reduce` sincronizando os gradientes. Como a
máquina tem uma GPU só, a **mecânica** será verificada com múltiplos processos na CPU
(backend `gloo`) — e o limite declarado, como no Capítulo 9.

### Sobre a Fase III (velocidade e escala) — situação de hardware **resolvida**

A máquina tem uma **AMD Radeon RX 7600** (sem NVIDIA, logo sem CUDA). O `torch-directml`
não tem distribuição para Python 3.14, mas **o Python 3.12 já estava instalado** em
`C:\Users\User\AppData\Local\Programs\Python\Python312` — então nada precisou ser
instalado no sistema. O ambiente de GPU vive isolado em **`C:\dml312`** (torch 2.4.1 +
torch-directml 0.2.5) e é usado apenas nos capítulos da Fase III.

> Para rodar os scripts de GPU: `C:\dml312\Scripts\python.exe benchmark.py`.
> Os demais capítulos seguem no ambiente principal (Python 3.14 + torch 2.12 CPU).

Com isso, o **Capítulo 8 foi verificado com medições reais** na placa. Restam duas
ressalvas conhecidas para os próximos:

- **Capítulo 9 (precisão):** o DirectML tem suporte parcial a `float16` e provavelmente
  nenhum a `bfloat16` (que depende de hardware/backend). Vou medir o que existe e declarar
  explicitamente o que não pôde ser testado.
- **Capítulo 10 (distribuído):** DDP de verdade exige **múltiplas** GPUs, o que esta
  máquina não tem. A *mecânica* do `all-reduce` é verificável com múltiplos processos na
  CPU (backend `gloo`), e é assim que o capítulo será construído — com o limite declarado.

### Sugestão de ritmo

Mantendo o padrão atual (1 capítulo = apostila + código testado + exercícios +
soluções + PDF + commit), o curso segue capítulo a capítulo na ordem do syllabus. Cada
capítulo termina publicado e consistente, então é seguro parar entre capítulos.
