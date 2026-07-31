"""
prepare_data.py — prepara o dataset de nomes a partir dos dados do IBGE.

Fonte: datasets-br/prenomes (frequencia de prenomes nos censos do IBGE).
Le o CSV bruto (_ibge.csv), normaliza cada nome (minusculo, sem acento, so a-z)
e escreve um nome por linha em names.txt.

Reproduzir:
    1. baixe o csv:
       curl -L -o _ibge.csv https://raw.githubusercontent.com/datasets-br/prenomes/master/data/nomes-censos-ibge.csv
    2. python prepare_data.py
"""

import csv
import unicodedata


def normalize(name: str) -> str:
    """maria-jose / JOSÉ -> 'mariajose' / 'jose' (minusculo, sem acento, so a-z)."""
    name = name.strip().lower()
    # remove acentos: 'é' -> 'e', 'ã' -> 'a'
    name = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    # mantem apenas letras a-z (descarta espacos, hifens, numeros)
    name = "".join(c for c in name if "a" <= c <= "z")
    return name


def main():
    names = []
    seen = set()
    from pathlib import Path
    # NAO chamar esta variavel de 'csv': o modulo csv e' usado logo abaixo
    caminho_csv = Path(__file__).resolve().parent / "_ibge.csv"
    if not caminho_csv.exists():
        raise SystemExit(
            f"Arquivo {caminho_csv.name} nao encontrado.\n"
            f"Baixe antes (uma vez so'):\n"
            f"    curl -L -o _ibge.csv https://raw.githubusercontent.com/"
            f"datasets-br/prenomes/master/data/nomes-censos-ibge.csv\n"
            f"O names.txt ja' pronto esta' versionado -- este script so' e' "
            f"preciso para regera-lo."
        )
    with open(caminho_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:                       # ja vem ordenado por frequencia
            n = normalize(row["Nome"])
            if 2 <= len(n) <= 12 and n not in seen:   # nomes "normais"
                seen.add(n)
                names.append(n)

    with open("names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    print(f"{len(names)} nomes escritos em names.txt")
    print("exemplos:", names[:10])


if __name__ == "__main__":
    main()
