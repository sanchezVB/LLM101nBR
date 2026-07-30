# LLM101n-BR — Panorama do Estado Atual

> Relatório de progresso do projeto em **01/06/2026**.
> Curso prático e bilíngue para construir um LLM do zero, inspirado no LLM101n.

---

## 1. Resumo executivo

O projeto tem hoje uma **base sólida e publicada**: a **Fase I (Fundamentos) está
completa** e o Transformer já começou. São **4 capítulos** prontos, testados e no
GitHub. Cada capítulo entregue inclui apostila, código executável, exercícios com
soluções e PDF — e todo número citado no texto foi obtido rodando o código de verdade.

| Indicador | Valor |
|-----------|-------|
| Capítulos concluídos | **4 de 17** (24%) |
| Estado | Fase I completa; Fase II (Transformer) iniciada |
| Repositório | Publicado e versionado (6 commits) |
| Arquivos versionados | 30 |
| Linhas de código (didático) | ~1.340 (Python) |
| PDFs gerados | 4 capítulos + panorama + este relatório |
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
| `a18b020` | 01/06/2026 | Panorama do estado atual |
| `0547d63` | 01/06/2026 | Capítulo 03 — N-gram model (MLP) + dataset do IBGE |
| (atual) | 01/06/2026 | Capítulo 04 — Attention + correções no gerador de PDF |

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
| 05 | Transformer | A fazer (próximo) |
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
| I — Fundamentos | 3 / 3 | **100%** |
| II — Transformer | 1 / 4 | 25% |
| III — Velocidade e escala | 0 / 4 | 0% |
| IV — Inferência e refinamento | 0 / 4 | 0% |
| V — Produto e além | 0 / 2 | 0% |
| **TOTAL** | **4 / 17** | **24%** |

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

**Capítulo 05 — Transformer.** O Capítulo 4 terminou com as duas metades na mão:
**comunicação** (atenção) e **computação** (feedforward). O próximo capítulo junta as
duas num **bloco**, adiciona **múltiplas cabeças** de atenção, **conexões residuais** e
**layer normalization**, e empilha tudo em profundidade — chegando à arquitetura do
GPT-2.

### Sobre os capítulos 8–10 (velocidade e escala)

Um ponto a resolver quando chegarmos lá: a máquina tem uma **AMD Radeon RX 7600**, mas
o PyTorch instalado é a build de CPU e o caminho CUDA não se aplica a placas AMD. O
`torch-directml` (que roda em GPU AMD no Windows) **não tem versão para Python 3.14**.
Alternativa concreta: instalar um **Python 3.12 paralelo** só para esses capítulos.
O Capítulo 10 (treino distribuído) exige múltiplas GPUs para um teste real, mas a
*mecânica* do `all-reduce` pode ser verificada com múltiplos processos na CPU.

### Sugestão de ritmo

Mantendo o padrão atual (1 capítulo = apostila + código testado + exercícios +
soluções + PDF + commit), o curso segue capítulo a capítulo na ordem do syllabus. Cada
capítulo termina publicado e consistente, então é seguro parar entre capítulos.
