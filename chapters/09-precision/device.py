"""
device.py — detecta o melhor dispositivo disponivel, de forma portatil.

Este modulo e' importado pelos outros arquivos do capitulo. Ele cobre os tres
caminhos que existem na pratica:

    NVIDIA        -> CUDA          (torch.cuda)
    AMD/Intel/Win -> DirectML      (torch_directml)
    sem GPU       -> CPU

Escrever assim -- em vez de cravar `.cuda()` no codigo -- e' o padrao correto: o
mesmo script roda na sua maquina, na do colega e no servidor, sem alteracao.
"""

import torch


def pegar_device(verbose=True):
    """Devolve o melhor dispositivo disponivel e um rotulo legivel."""
    # 1. CUDA (NVIDIA) e' o caminho mais comum e mais bem suportado
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        nome = torch.cuda.get_device_name(0)
        if verbose:
            print(f"dispositivo: CUDA -> {nome}")
        return dev, f"CUDA ({nome})"

    # 2. DirectML: roda em qualquer GPU DirectX 12 (AMD, Intel, NVIDIA) no Windows
    try:
        import torch_directml
        if torch_directml.device_count() > 0:
            dev = torch_directml.device()
            if verbose:
                print(f"dispositivo: DirectML -> {dev} ({torch_directml.device_count()} disponivel)")
            return dev, "DirectML"
    except ImportError:
        pass

    # 3. CPU: sempre funciona
    if verbose:
        print("dispositivo: CPU (nenhuma GPU detectada)")
    return torch.device("cpu"), "CPU"


def sincronizar(dev):
    """Espera a GPU terminar o trabalho enfileirado.

    ISTO E' ESSENCIAL PARA MEDIR TEMPO. Chamadas para a GPU sao ASSINCRONAS: o
    Python devolve o controle imediatamente, antes de o calculo terminar. Sem
    sincronizar, voce mede o tempo de ENFILEIRAR a operacao (microssegundos), e
    nao o de execut^a-la -- e conclui, errado, que a GPU e' absurdamente rapida.
    """
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type != "cpu":
        # DirectML nao expoe synchronize(); copiar um valor de volta para a CPU
        # forca a fila a esvaziar, que e' o efeito que queremos.
        torch.zeros(1, device=dev).cpu()


if __name__ == "__main__":
    dev, rotulo = pegar_device()
    print(f"\nrotulo: {rotulo}")
    print(f"torch: {torch.__version__}")
    x = torch.randn(4, 4, device=dev)
    print(f"tensor de teste criado em {x.device}: soma = {x.sum().item():.4f}")
