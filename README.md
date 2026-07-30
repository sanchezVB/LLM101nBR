# LLM101n-BR — Construindo um Modelo de Linguagem do Zero

> Curso prático, **bilíngue** (texto em português, termos técnicos e comentários de
> código em inglês), inspirado no **LLM101n** do Andrej Karpathy / Eureka Labs.
> Você vai construir um Large Language Model (LLM) **do zero** (*from scratch*),
> entendendo cada peça porque é você quem a implementa.

---

## Para quem é este curso

**Pré-requisitos (o mínimo que você precisa saber):**

- **Python básico** — variáveis, funções, loops, listas e dicionários. Não precisa
  ser avançado.
- **Noção básica de programação** — saber rodar um script, ler uma mensagem de erro,
  instalar um pacote.
- **Matemática de ensino superior básica** — o que se vê no começo de um curso de
  exatas: funções, um pouco de cálculo (a ideia de *derivada* como "taxa de
  variação") e álgebra (vetores e matrizes). **Tudo que for além disso é revisado no
  próprio capítulo, quando precisar.**

Você **não** precisa de: experiência prévia com machine learning, GPU, nem
matemática avançada. A filosofia é a mesma do LLM101n: nada de caixa-preta — a gente
abre cada caixa.

---

## Como o curso funciona

Cada capítulo é, ao mesmo tempo, **apostila e repositório**:

- `README.md` do capítulo → o **texto didático** (a "apostila"): explica a teoria,
  a intuição e a matemática na medida certa.
- arquivos `.py` → o **código do zero**, executável e comentado em inglês.
- `exercicios.md` → exercícios para fixar, com pasta `solucoes/`.
- `Capitulo-XX.pdf` → o capítulo inteiro (texto + exercícios) **em PDF**, para ler
  no tablet, imprimir ou estudar offline. É gerado a partir do Markdown — veja
  [como gerar os PDFs](#gerando-a-apostila-em-pdf).

A ordem importa: cada capítulo assume o anterior. A progressão vai de um modelo de
"contar letras" (Capítulo 1) até um Transformer treinável, com finetuning e deploy.

```
llm101n-curso/
├── README.md                      ← você está aqui
├── SETUP.md                       ← como preparar o ambiente (Python, venv, torch)
├── requirements.txt               ← dependências para RODAR o código (torch, ...)
├── tools/
│   ├── build_pdf.py               ← gera a apostila em PDF a partir do Markdown
│   └── requirements-docs.txt      ← dependências só para gerar PDF
└── chapters/
    ├── 01-bigram-language-model/
    │   ├── README.md              ← a apostila do capítulo
    │   ├── Capitulo-01.pdf        ← o capítulo em PDF (gerado)
    │   ├── names.txt              ← dataset de exemplo
    │   ├── bigram.py              ← versão por contagem
    │   ├── bigram_nn.py           ← versão rede neural
    │   ├── exercicios.md
    │   └── solucoes/
    ├── 02-micrograd/
    ├── 03-ngram-model/            ← MLP em PyTorch (embeddings, GELU, splits)
    ├── 04-attention/              ← self-attention, máscara causal, posicional
    ├── 05-transformer/            ← multi-head, residuais, LayerNorm, GPT-2
    ├── 06-tokenization/           ← Unicode/UTF-8 e BPE do zero
    ├── 07-optimization/           ← init, AdamW do zero, warmup+cosine, clipping
    ├── 08-device/                 ← CPU vs GPU, benchmark, código portátil
    ├── 09-precision/              ← fp16/bf16, loss scaling, precisão mista
    ├── 10-distributed/            ← all-reduce, DDP, ZeRO
    └── ...
```

Comece por [`SETUP.md`](SETUP.md) e depois vá para o
[Capítulo 01](chapters/01-bigram-language-model/README.md).

### Gerando a apostila em PDF

Os PDFs já vêm prontos no repositório, mas você pode regerá-los a qualquer momento
(o Markdown é a fonte da verdade). É um pipeline **100% offline**, sem LaTeX:

```bash
pip install -r tools/requirements-docs.txt

python tools/build_pdf.py --chapter 01    # gera chapters/01-.../Capitulo-01.pdf
python tools/build_pdf.py --all           # gera dist/ com a apostila completa
```

---

## Syllabus (programa do curso)

Seguimos a espinha dorsal do LLM101n original. Cada capítulo lista o **objetivo de
aprendizagem** (o que você sai sabendo fazer).

| # | Capítulo | Tópicos | Objetivo de aprendizagem |
|---|----------|---------|--------------------------|
| 01 | **Bigram Language Model** | language modeling | Entender o que é "prever o próximo token" e treinar o LLM mais simples possível, por contagem e como rede neural. |
| 02 | **Micrograd** | machine learning, backpropagation | Implementar *autograd* (derivadas automáticas) do zero e entender como redes neurais realmente aprendem. |
| 03 | **N-gram model** | MLP, matmul, GELU | Subir de bigrama para um MLP (multi-layer perceptron) que olha mais contexto. |
| 04 | **Attention** | attention, softmax, positional encoder | Implementar o mecanismo de atenção, o coração do Transformer. |
| 05 | **Transformer** | transformer, residual, layernorm, GPT-2 | Montar um Transformer completo no estilo GPT-2. |
| 06 | **Tokenization** | minBPE, byte pair encoding | Construir um tokenizador BPE do zero (texto → tokens). |
| 07 | **Optimization** | initialization, optimization, AdamW | Inicializar pesos direito e treinar de forma estável com AdamW. |
| 08 | **Need for Speed I: Device** | CPU, GPU | Mover o treino para a GPU e entender o porquê. |
| 09 | **Need for Speed II: Precision** | mixed precision, fp16/bf16/fp8 | Treinar mais rápido com precisão reduzida. |
| 10 | **Need for Speed III: Distributed** | DDP, ZeRO | Distribuir o treino em vários dispositivos. |
| 11 | **Datasets** | data loading, synthetic data | Carregar, preparar e gerar dados de treino. |
| 12 | **Inference I: KV-cache** | kv-cache | Acelerar a geração de texto com cache. |
| 13 | **Inference II: Quantization** | quantization | Encolher o modelo para inferência. |
| 14 | **Finetuning I: SFT** | SFT, PEFT, LoRA, chat | Transformar o modelo base em um assistente de chat. |
| 15 | **Finetuning II: RL** | RLHF, PPO, DPO | Alinhar o modelo com feedback (reinforcement learning). |
| 16 | **Deployment** | API, web app | Servir o modelo num web app. |
| 17 | **Multimodal** | VQVAE, diffusion transformer | Estender para imagens (e além de texto). |

### Apêndice — tópicos transversais

Trabalhados ao longo dos capítulos, conforme aparecem:

- **Linguagens**: Assembly, C, Python.
- **Tipos de dados**: Integer, Float, String (ASCII, Unicode, UTF-8).
- **Tensor**: shapes, views, strides, contiguous.
- **Frameworks**: PyTorch, JAX.
- **Arquiteturas**: GPT (1–4), Llama (RoPE, RMSNorm, GQA), MoE.
- **Multimodal**: imagem, áudio, vídeo, VQVAE, VQGAN, diffusion.

---

## Status

| Capítulo | Estado |
|----------|--------|
| 01 — Bigram Language Model | ✅ Completo (apostila + código + exercícios + PDF) |
| 02 — Micrograd | ✅ Completo (apostila + código + exercícios + PDF) |
| 03 — N-gram model (MLP) | ✅ Completo (apostila + código + exercícios + PDF) |
| 04 — Attention | ✅ Completo (apostila + código + exercícios + PDF) |
| 05 — Transformer | ✅ Completo (apostila + código + exercícios + PDF) |
| 06 — Tokenization | ✅ Completo (apostila + código + exercícios + PDF) |
| 07 — Optimization | ✅ Completo (apostila + código + exercícios + PDF) |
| 08 — Device (CPU/GPU) | ✅ Completo (apostila + código + exercícios + PDF) |
| 09 — Precision | ✅ Completo (apostila + código + exercícios + PDF) |
| 10 — Distributed | ✅ Completo (apostila + código + exercícios + PDF) |
| 11–17 | ⏳ Planejados (syllabus aprovado) |

> Este repositório é construído **em fases**. O Capítulo 1 é a "fatia vertical" de
> referência: define o padrão de qualidade (profundidade do texto, estilo do código,
> formato dos exercícios) para todos os capítulos seguintes.

---

## Créditos e inspiração

Baseado no **[LLM101n](https://github.com/karpathy/LLM101n)** de Andrej Karpathy /
Eureka Labs, e na linhagem *makemore* / *nanoGPT*. Este material é uma releitura
didática original em português; não é afiliado ao curso oficial.
