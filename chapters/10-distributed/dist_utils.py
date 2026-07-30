"""
dist_utils.py — infraestrutura para rodar varios processos que treinam juntos.

Resolve dois problemas praticos que costumam travar quem tenta treino
distribuido pela primeira vez -- e ambos aparecem como um PROGRAMA QUE PENDURA,
sem mensagem de erro:

  1. RENDEZVOUS: como os processos se encontram. Usamos um arquivo em disco, que
     funciona em qualquer sistema e nao depende de portas TCP livres.

  2. INTERFACE DE REDE: mesmo rodando tudo numa maquina so', o gloo abre sockets
     TCP entre os processos. Ele escolhe a interface resolvendo o HOSTNAME -- e
     se o hostname da maquina resolver para um IP que nao e' local (acontece com
     VPN, adaptadores virtuais ou DNS corporativo), a conexao falha e o programa
     trava para sempre.

     Foi exatamente o que aconteceu na maquina onde este capitulo foi escrito: o
     hostname resolvia para um IP publico da AWS, e o init_process_group ficava
     preso indefinidamente. A funcao `escolher_interface()` abaixo detecta uma
     interface com IP privado e a informa ao gloo via GLOO_SOCKET_IFNAME.
"""

import os
import socket
import tempfile

import torch
import torch.distributed as dist

ARQUIVO_RENDEZVOUS = os.path.join(tempfile.gettempdir(), "llm101n_rdzv")


def _e_privado(ip: str) -> bool:
    """10.x, 192.168.x, 172.16-31.x ou 127.x -- enderecos de rede local."""
    partes = ip.split(".")
    if len(partes) != 4:
        return False
    a, b = int(partes[0]), int(partes[1])
    return a == 10 or a == 127 or (a == 192 and b == 168) or (a == 172 and 16 <= b <= 31)


def escolher_interface(verbose=True):
    """Define GLOO_SOCKET_IFNAME se necessario. Devolve o nome escolhido ou None.

    Em Linux normalmente nada disso e' preciso ('lo' resolve). Em Windows, ou em
    maquinas com VPN, e' o que evita o travamento silencioso.
    """
    if os.environ.get("GLOO_SOCKET_IFNAME"):
        return os.environ["GLOO_SOCKET_IFNAME"]      # respeita a escolha do usuario

    # o hostname resolve para um IP local? se sim, o padrao do gloo funciona
    try:
        ip_host = socket.gethostbyname(socket.gethostname())
        if _e_privado(ip_host):
            return None
        if verbose:
            print(f"  [rede] hostname resolve para {ip_host}, que NAO e' local; "
                  f"procurando interface adequada...")
    except socket.gaierror:
        if verbose:
            print("  [rede] hostname nao resolve; procurando interface adequada...")

    if os.name != "nt":
        # em Linux, forcar a loopback resolve o mesmo problema
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        return "lo"

    # Windows: procura um adaptador com IP privado usando o PowerShell
    import subprocess
    try:
        saida = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-NetIPAddress -AddressFamily IPv4 | "
             "ForEach-Object { \"$($_.IPAddress)|$($_.InterfaceAlias)\" }"],
            capture_output=True, text=True, timeout=20,
        )
        candidatos = []
        for linha in saida.stdout.splitlines():
            if "|" not in linha:
                continue
            ip, alias = linha.strip().split("|", 1)
            if _e_privado(ip) and ip != "127.0.0.1":
                candidatos.append((ip, alias))

        # prefere 10.x / 192.168.x (rede de verdade) a link-local 169.254.x
        candidatos.sort(key=lambda c: c[0].startswith("169.254"))
        if candidatos:
            ip, alias = candidatos[0]
            os.environ["GLOO_SOCKET_IFNAME"] = alias
            if verbose:
                print(f"  [rede] usando interface '{alias}' ({ip})")
            return alias
    except Exception as e:
        if verbose:
            print(f"  [rede] deteccao automatica falhou ({type(e).__name__}); "
                  f"defina GLOO_SOCKET_IFNAME manualmente")
    return None


def iniciar(rank, world_size, verbose=False):
    """Entra no grupo de processos. Todo rank precisa chamar isto."""
    if rank == 0 or verbose:
        escolher_interface(verbose=(rank == 0 and verbose))
    else:
        escolher_interface(verbose=False)

    dist.init_process_group(
        backend="gloo",                                    # CPU; use "nccl" em GPU NVIDIA
        init_method=f"file:///{ARQUIVO_RENDEZVOUS}",
        rank=rank,
        world_size=world_size,
    )


def encerrar():
    if dist.is_initialized():
        dist.destroy_process_group()


def limpar_rendezvous():
    """O arquivo de rendezvous precisa ser novo a cada execucao."""
    if os.path.exists(ARQUIVO_RENDEZVOUS):
        try:
            os.remove(ARQUIVO_RENDEZVOUS)
        except OSError:
            pass


def apenas_no_rank0(rank, *args, **kwargs):
    """Imprime so' no rank 0 -- senao a saida sai N vezes embaralhada."""
    if rank == 0:
        print(*args, **kwargs)


if __name__ == "__main__":
    print("=== diagnostico de rede para treino distribuido ===")
    print(f"  hostname: {socket.gethostname()}")
    try:
        ip = socket.gethostbyname(socket.gethostname())
        print(f"  resolve para: {ip}  ({'local' if _e_privado(ip) else 'NAO e local!'})")
    except socket.gaierror as e:
        print(f"  nao resolve: {e}")
    iface = escolher_interface()
    print(f"  GLOO_SOCKET_IFNAME = {iface or '(nao necessario)'}")
    print(f"  arquivo de rendezvous: {ARQUIVO_RENDEZVOUS}")
    print(f"  gloo disponivel: {dist.is_gloo_available()}")
    print(f"  nccl disponivel: {dist.is_nccl_available()} (so' faz sentido com GPU NVIDIA)")
    print(f"  nucleos de CPU: {os.cpu_count()}")
