# SETUP — preparando o ambiente

Você só precisa de **Python 3.10+** e do **PyTorch**. Siga os passos abaixo.

## 1. Verifique o Python

```bash
python --version
```

Se aparecer algo como `Python 3.10` ou mais novo, ok. Se não, instale em
<https://www.python.org/downloads/>.

> No Windows, se `python` não funcionar, tente `py --version`.

## 2. Crie um ambiente virtual (venv)

Isolar as dependências do curso evita conflito com outros projetos.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Quando ativo, o prompt mostra `(.venv)` no começo da linha.

## 3. Instale as dependências

```bash
pip install -r requirements.txt
```

Isso instala **PyTorch** (a única dependência essencial dos primeiros capítulos),
além de `numpy` e `matplotlib` para visualizações.

> A instalação padrão do PyTorch via `pip` roda na **CPU**, que é suficiente até
> o Capítulo 8. Para usar **GPU**, siga as instruções específicas em
> <https://pytorch.org/get-started/locally/> (escolha sua versão de CUDA).

## 4. Teste

```bash
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

Se imprimir a versão do torch sem erro, está tudo pronto. `cuda: False` é normal e
esperado se você não tem GPU configurada.

## 5. Rode o Capítulo 1

```bash
cd chapters/01-bigram-language-model
python bigram.py
```

Você deve ver nomes inventados pelo modelo e um valor de *loss*. Bem-vindo. 🚀
