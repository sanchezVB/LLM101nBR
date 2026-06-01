# LLM101n-BR — Panorama do Estado Atual

> Relatório de progresso do projeto em **01/06/2026**.
> Curso prático e bilíngue para construir um LLM do zero, inspirado no LLM101n.

---

## 1. Resumo executivo

O projeto saiu do zero e tem hoje uma **base sólida e publicada**: a infraestrutura do
curso está montada e os **2 primeiros capítulos estão completos, testados e no
GitHub**. Cada capítulo entregue inclui apostila, código executável, exercícios com
soluções e PDF — e todo número citado no texto foi obtido rodando o código de verdade.

| Indicador | Valor |
|-----------|-------|
| Capítulos concluídos | **2 de 17** (12%) |
| Estado | Infraestrutura pronta + Fase I iniciada |
| Repositório | Publicado e versionado (3 commits) |
| Arquivos versionados | 22 |
| Linhas de código (didático) | ~450 (Python) |
| PDFs gerados | 2 capítulos + este panorama |
| Verificação | 100% do código roda; gradientes do Cap. 2 batem com PyTorch |

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

---

## 3. Capítulos: o que está pronto e o que falta

### Fase I — Fundamentos

| # | Capítulo | Estado |
|---|----------|--------|
| 01 | Bigram Language Model | **Concluído** |
| 02 | Micrograd | **Concluído** |
| 03 | N-gram model | A fazer (próximo) |

### Fase II — O Transformer

| # | Capítulo | Estado |
|---|----------|--------|
| 04 | Attention | A fazer |
| 05 | Transformer | A fazer |
| 06 | Tokenization | A fazer |
| 07 | Optimization | A fazer |

### Fase III — Velocidade e escala

| # | Capítulo | Estado |
|---|----------|--------|
| 08 | Device (CPU/GPU) | A fazer |
| 09 | Precision | A fazer |
| 10 | Distributed | A fazer |
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
| I — Fundamentos | 2 / 3 | 67% |
| II — Transformer | 0 / 4 | 0% |
| III — Velocidade e escala | 0 / 4 | 0% |
| IV — Inferência e refinamento | 0 / 4 | 0% |
| V — Produto e além | 0 / 2 | 0% |
| **TOTAL** | **2 / 17** | **12%** |

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
- **PDFs inspecionados** visualmente (acentuação, formatação, código).
- **Versionamento limpo** — arquivos temporários ignorados; commits descritivos.

---

## 7. Próximo passo

**Capítulo 03 — N-gram model.** Voltar ao problema de gerar nomes do Capítulo 1, mas
trocando o bigrama por um **MLP** que olha vários caracteres de contexto, já em
**PyTorch de verdade** (embeddings, `matmul`, ativação GELU). É a ponte do modelo de
brinquedo para o modelo de verdade, e usa diretamente o que foi construído nos
capítulos 1 (modelo de linguagem) e 2 (como uma rede aprende).

### Sugestão de ritmo

Mantendo o padrão atual (1 capítulo = apostila + código testado + exercícios +
soluções + PDF + commit), a Fase I fica completa com o próximo capítulo, e o curso
segue capítulo a capítulo na ordem do syllabus.
