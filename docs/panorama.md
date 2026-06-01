# LLM101n-BR — Panorama Completo do Curso

> **Construindo um Modelo de Linguagem do Zero**
> Curso prático e bilíngue (texto em português; termos técnicos e código em inglês),
> inspirado no **LLM101n** de Andrej Karpathy / Eureka Labs.

Este documento é uma visão geral de todo o curso: a filosofia, o público, a estrutura,
o programa dos 17 capítulos, o que já está pronto e como usar o material. Serve como
mapa para quem está começando e como referência para acompanhar o progresso.

---

## 1. A ideia em uma frase

Você vai **construir um Large Language Model (LLM) do zero** — da peça mais simples
(um modelo que "conta letras") até um Transformer treinável, com finetuning e deploy.
A filosofia é uma só: **nada de caixa-preta**. Cada conceito é implementado por você,
porque entender de verdade é conseguir reconstruir.

Um LLM gigante (como os que respondem no ChatGPT) e um modelo de brinquedo de uma
linha fazem **a mesma tarefa**: prever o próximo *token*, dado o que veio antes. A
diferença é só de escala e refinamento. O curso percorre exatamente esse caminho, do
brinquedo ao modelo de verdade.

---

## 2. Para quem é

**Pré-requisitos (o mínimo necessário):**

- **Python básico** — variáveis, funções, loops, listas e dicionários.
- **Noção básica de programação** — rodar um script, ler um erro, instalar um pacote.
- **Matemática de ensino superior básica** — funções, um pouco de cálculo (a ideia de
  *derivada* como "taxa de variação") e álgebra (vetores e matrizes).

Tudo que for além disso é **revisado no próprio capítulo, quando necessário**. Você
**não** precisa de experiência prévia com machine learning, nem de GPU, nem de
matemática avançada.

---

## 3. Como o curso funciona

Cada capítulo é, ao mesmo tempo, **apostila e repositório de código**:

| Componente | O que é |
|------------|---------|
| `README.md` do capítulo | O **texto didático** (a "apostila"): teoria, intuição e a matemática na medida certa. |
| arquivos `.py` | O **código do zero**, executável e comentado em inglês. |
| `exercicios.md` | Exercícios para fixar, com pasta `solucoes/`. |
| `Capitulo-XX.pdf` | O capítulo inteiro em PDF, para ler offline ou imprimir. |

**Princípio de qualidade do curso:** todo número que aparece no texto (uma *loss*, um
resultado) é obtido **rodando o código de verdade** antes de ser escrito. Nada de
valores inventados.

A ordem importa: cada capítulo assume o anterior. A progressão é cumulativa.

---

## 4. O programa completo (syllabus)

Seguimos a espinha dorsal do LLM101n original: 17 capítulos, agrupados em fases.

### Fase I — Fundamentos (do brinquedo à rede neural)

| # | Capítulo | O que você aprende a fazer |
|---|----------|----------------------------|
| 01 | **Bigram Language Model** | Entender "prever o próximo token" e treinar o LLM mais simples possível, por contagem e como rede neural. |
| 02 | **Micrograd** | Implementar *autograd* (derivadas automáticas) do zero e entender como redes neurais realmente aprendem. |
| 03 | **N-gram model** | Subir de bigrama para um MLP (multi-layer perceptron) que olha mais contexto. |

### Fase II — O Transformer

| # | Capítulo | O que você aprende a fazer |
|---|----------|----------------------------|
| 04 | **Attention** | Implementar o mecanismo de atenção (*attention*), o coração do Transformer. |
| 05 | **Transformer** | Montar um Transformer completo no estilo GPT-2 (residual, layernorm). |
| 06 | **Tokenization** | Construir um tokenizador BPE (*byte pair encoding*) do zero. |
| 07 | **Optimization** | Inicializar pesos direito e treinar de forma estável com AdamW. |

### Fase III — Velocidade e escala (*Need for Speed*)

| # | Capítulo | O que você aprende a fazer |
|---|----------|----------------------------|
| 08 | **Device (CPU/GPU)** | Mover o treino para a GPU e entender o porquê. |
| 09 | **Precision** | Treinar mais rápido com precisão reduzida (fp16, bf16, fp8). |
| 10 | **Distributed** | Distribuir o treino em vários dispositivos (DDP, ZeRO). |
| 11 | **Datasets** | Carregar, preparar e gerar dados de treino (inclusive sintéticos). |

### Fase IV — Inferência e refinamento

| # | Capítulo | O que você aprende a fazer |
|---|----------|----------------------------|
| 12 | **Inference I: KV-cache** | Acelerar a geração de texto com cache. |
| 13 | **Inference II: Quantization** | Encolher o modelo para inferência. |
| 14 | **Finetuning I: SFT** | Transformar o modelo base em um assistente de chat (SFT, PEFT, LoRA). |
| 15 | **Finetuning II: RL** | Alinhar o modelo com feedback humano (RLHF, PPO, DPO). |

### Fase V — Produto e além

| # | Capítulo | O que você aprende a fazer |
|---|----------|----------------------------|
| 16 | **Deployment** | Servir o modelo num web app (API). |
| 17 | **Multimodal** | Estender para imagens e além de texto (VQVAE, diffusion transformer). |

### Apêndice — tópicos transversais

Trabalhados ao longo dos capítulos, conforme aparecem:

- **Linguagens:** Assembly, C, Python.
- **Tipos de dados:** Integer, Float, String (ASCII, Unicode, UTF-8).
- **Tensor:** shapes, views, strides, contiguous.
- **Frameworks:** PyTorch, JAX.
- **Arquiteturas:** GPT (1–4), Llama (RoPE, RMSNorm, GQA), MoE.
- **Multimodal:** imagem, áudio, vídeo, VQVAE, VQGAN, diffusion.

---

## 5. O que já está pronto

Status em **01/06/2026**:

| Capítulo | Estado |
|----------|--------|
| 01 — Bigram Language Model | **Completo** (apostila + código + exercícios + PDF) |
| 02 — Micrograd | **Completo** (apostila + código + exercícios + PDF) |
| 03 a 17 | Planejados (syllabus aprovado) |

O Capítulo 1 funciona como a "fatia vertical" de referência: define o padrão de
qualidade (profundidade do texto, estilo do código, formato dos exercícios) para todos
os capítulos seguintes.

### Capítulo 01 — Bigram Language Model (detalhe)

**Objetivo:** entender o que significa "prever o próximo token" e treinar o modelo de
linguagem mais simples possível — primeiro **por contagem**, depois como **rede
neural** — e perceber que as duas abordagens são, no fundo, a mesma coisa.

- **Tarefa concreta:** gerar nomes de pessoas plausíveis a partir de um dataset de
  nomes reais.
- **`bigram.py`** — versão por contagem: monta a matriz de contagens de pares de
  letras, normaliza para probabilidades e gera nomes. *Loss* ≈ **2,38**.
- **`bigram_nn.py`** — versão rede neural: uma matriz de pesos treinada por gradient
  descent que aprende **a mesma tabela** de probabilidades. *Loss* converge a ≈ **2,2**.
- **A grande lição:** contar é um caso particular de otimizar. A diferença residual de
  *loss* vem só do grau de suavização de cada lado (smoothing vs. regularização) — ao
  igualá-los, os dois valores se encontram em ~2,1 (exercício E5).
- **Conceitos introduzidos:** token, vocabulário, one-hot, logits, softmax, loss
  (negative log-likelihood), gradient descent, learning rate, regularização, sampling.

### Capítulo 02 — Micrograd (detalhe)

**Objetivo:** abrir a caixa-preta do `loss.backward()`. Construir, do zero, um motor de
**autograd** (diferenciação automática) e usá-lo para treinar uma rede neural de
verdade.

- **`micrograd.py`** — a classe `Value`: embrulha um número, lembra de onde veio
  (formando um grafo de computação) e sabe propagar o gradiente de volta. Implementa
  `+`, `*`, `**`, `tanh`, `exp`, `relu` e a `backward()` (via ordenação topológica do
  grafo e regra da cadeia).
- **`nn.py`** — uma mini biblioteca de redes neurais sobre o `Value`: `Neuron → Layer
  → MLP`, com uma demonstração de treino que atinge *loss* ≈ **0**.
- **Validação rigorosa:** os gradientes do nosso motor **batem com os do PyTorch até a
  6ª casa decimal** (verificado no exercício E5).
- **A grande lição:** o `backward()` do PyTorch não é mágica — é exatamente este
  mecanismo, só que vetorizado (operando sobre tensores na GPU em vez de um número por
  vez).
- **Conceitos introduzidos:** derivada, regra da cadeia (chain rule), grafo de
  computação, backpropagation, ordenação topológica, acumulação de gradientes (e por
  que zerá-los), neurônio, camada, MLP.

---

## 6. Como usar o material

### Preparar o ambiente

Você só precisa de **Python 3.10+** e **PyTorch**:

```bash
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### Rodar um capítulo

```bash
cd chapters/01-bigram-language-model
python bigram.py
```

### Gerar os PDFs (opcional)

Os PDFs já vêm prontos, mas são regeráveis a partir do Markdown (pipeline 100%
offline, sem LaTeX):

```bash
pip install -r tools/requirements-docs.txt
python tools/build_pdf.py --chapter 01    # um capítulo
python tools/build_pdf.py --all           # apostila completa em dist/
```

---

## 7. Onde está o curso

- **Repositório (GitHub):** <https://github.com/sanchezVB/LLM101nBR>
- **Cópia local:** `C:\Users\User\llm101n-curso`

```
LLM101nBR/
├── README.md            visão geral + syllabus
├── SETUP.md             preparar o ambiente
├── requirements.txt     dependências para rodar o código
├── LICENSE              MIT (código) + CC BY 4.0 (material)
├── docs/                este panorama
├── tools/               gerador de PDF (build_pdf.py)
└── chapters/
    ├── 01-bigram-language-model/   apostila + código + exercícios + PDF
    └── 02-micrograd/               apostila + código + exercícios + PDF
```

---

## 8. Próximos passos

O próximo capítulo a ser escrito é o **Capítulo 03 — N-gram model**: voltamos ao
problema de gerar nomes do Capítulo 1, mas trocamos o bigrama por um **MLP** que olha
vários caracteres de contexto — agora usando **PyTorch de verdade**, com embeddings,
`matmul` e a função de ativação **GELU**. É a ponte do modelo de brinquedo para o
modelo de verdade.

---

## Créditos

Baseado no **[LLM101n](https://github.com/karpathy/LLM101n)** de Andrej Karpathy /
Eureka Labs, e na linhagem *makemore* / *nanoGPT*. Material didático original em
português; não afiliado ao curso oficial.

*Licença: código sob MIT; material didático sob CC BY 4.0.*
