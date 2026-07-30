# Preparando a GPU — os três caminhos

O código do capítulo (`device.py`) detecta o dispositivo automaticamente, então os
scripts **rodam sem alteração** em qualquer um dos casos abaixo. O que muda é a
instalação.

Descubra primeiro que placa você tem:

```bash
# Windows
wmic path win32_VideoController get name
# Linux
lspci | grep -i vga
```

---

## Caminho 1 — NVIDIA (CUDA)

O mais simples e o melhor suportado. Instale a build de CUDA do PyTorch (a versão do
`cu###` depende do seu driver — consulte <https://pytorch.org/get-started/locally/>):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verifique:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve imprimir `True` e o nome da placa. Nada mais é necessário — `device.py` já
encontra.

---

## Caminho 2 — AMD ou Intel no Windows (DirectML)

Placas AMD e Intel **não** rodam CUDA. No Windows, a alternativa é o **DirectML**, que
funciona sobre qualquer GPU compatível com DirectX 12.

> **Atenção à versão do Python.** O `torch-directml` não tem distribuição para todas as
> versões. No momento em que este capítulo foi escrito, ele exigia **Python 3.10–3.12**
> (não funciona em 3.13+). Se o seu Python principal for mais novo, crie um ambiente
> separado só para isso.

```bash
# Windows PowerShell — use um Python 3.12 (ajuste o caminho)
py -3.12 -m venv C:\dml312
C:\dml312\Scripts\python.exe -m pip install torch-directml
```

> **Cuidado com caminho longo:** o Windows limita caminhos a 260 caracteres, e a
> instalação do numpy cria caminhos profundos. Se der `WinError 206`, crie o ambiente num
> diretório curto (`C:\dml312`) em vez de dentro de pastas aninhadas.

Verifique:

```bash
C:\dml312\Scripts\python.exe -c "import torch_directml; print(torch_directml.device_count(), torch_directml.device())"
```

Deve imprimir `1 privateuseone:0` (ou mais dispositivos).

**Limitações honestas do DirectML:**

- É mais lento que CUDA em hardware equivalente e recebe menos otimização.
- **A cobertura de operações é boa, mas não é total** — e as lacunas aparecem em lugares
  inesperados. Todas as operações do *modelo* funcionam (embedding, softmax, GELU,
  `masked_fill`, LayerNorm, `cross_entropy`, `multinomial` e o `backward`), mas o
  **AdamW** usa `aten::lerp`, que **não** é implementada: o PyTorch avisa e executa essa
  parte na CPU.

  ```
  UserWarning: The operator 'aten::lerp.Scalar_out' is not currently supported
  on the DML backend and will fall back to run on the CPU.
  ```

  O treino continua correto, mas parte do passo do otimizador atravessa a fronteira
  CPU↔GPU a cada iteração — o que come parte do ganho. Ou seja: **os speedups que
  medimos neste capítulo são um piso**, não o teto do que a placa poderia dar com um
  backend completo.

  > **Como detectar isso no seu caso:** rode com
  > `python -W always::UserWarning seu_script.py` e procure por "fall back to run on
  > the CPU". Um aviso desses dentro do laço de treino é um gargalo silencioso.

- Não expõe `torch.cuda.synchronize()`; para medir tempo é preciso forçar a
  sincronização de outra forma (ver a Seção 6 da apostila).

**Alternativa no Linux:** placas AMD suportam **ROCm**, que é bem mais completo que o
DirectML. Se você usa Linux com uma AMD recente, prefira ROCm:
<https://pytorch.org/get-started/locally/>

---

## Caminho 3 — sem GPU (CPU)

Nada a instalar: é a instalação padrão do curso. Os scripts detectam a ausência de GPU e
rodam na CPU.

Você perde as comparações de velocidade, mas **todo o conteúdo conceitual do capítulo
continua acessível** — inclusive porque uma das lições é justamente que, para modelos
pequenos, a CPU **ganha**.

Se quiser experimentar GPU sem ter uma, o **Google Colab** oferece GPU gratuita: suba os
arquivos do capítulo e rode lá.

---

## Como saber que funcionou

```bash
python device.py
```

Saída esperada (exemplo, no caminho DirectML):

```
dispositivo: DirectML -> privateuseone:0 (1 disponivel)

rotulo: DirectML
torch: 2.4.1+cpu
tensor de teste criado em privateuseone:0: soma = -1.2345
```

O `+cpu` na versão do torch é normal no DirectML: o pacote base é a build de CPU, e o
DirectML entra como um backend adicional.
