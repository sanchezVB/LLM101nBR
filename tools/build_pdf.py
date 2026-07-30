"""
build_pdf.py — gera a apostila do curso em PDF a partir dos arquivos Markdown.

Pipeline 100% offline e sem dependencias de sistema (sem LaTeX/pandoc):
    Markdown  ->  HTML estilizado  ->  PDF (xhtml2pdf / reportlab)

Usamos as fontes nativas do PDF (Helvetica/Courier, da familia "base-14"). Elas
cobrem todo o portugues (Latin-1) e evitam o carregamento de arquivos .ttf, que
e' instavel no Windows com reportlab 4.x.

Uso:
    python tools/build_pdf.py --chapter 01      # PDF de um capitulo
    python tools/build_pdf.py --all             # apostila completa (todos)
    python tools/build_pdf.py --gabaritos       # PDF separado com as solucoes

Os PDFs de capitulo ficam dentro da pasta do proprio capitulo, e o de gabaritos
em docs/ -- os dois sao material de LEITURA e ficam versionados. A apostila
completa vai para dist/, que e' ignorado pelo git por ser so' uma juncao dos
capitulos, regeravel com --all.
"""

import argparse
import re
from datetime import date
from pathlib import Path

import markdown
from xhtml2pdf import pisa
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = ROOT / "chapters"
DIST_DIR = ROOT / "dist"

# --------------------------------------------------------------------------
# Emoji handling: as fontes base-14 nao tem glifos de emoji. Trocamos os
# conhecidos por texto e removemos qualquer outro para nao virar quadrado.
# --------------------------------------------------------------------------
# As fontes base-14 do PDF cobrem WinAnsi (cp1252). Tudo fora disso corre o risco
# de virar quadrado preto -- ou, pior, de ser apagado silenciosamente e mudar o
# sentido do texto. Por isso mapeamos explicitamente para equivalentes ASCII.
EMOJI_REPLACEMENTS = {
    # emojis e dingbats
    "✅": "[OK]",
    "❌": "[nao]",
    "❗": "(!)",
    "✔": "[OK]",
    "✖": "[nao]",
    "⏳": "[...]",
    "➡️": "->",
    "➡": "->",
    "►": ">",
    "🎯": "",
    "🚀": "",
    "🔑": "",
    "⚠️": "(!)",
    "⚠": "(!)",
    # setas (NUNCA apagar: elas carregam significado, ex.: "Neuron -> Layer")
    "→": "->",
    "←": "<-",
    "↔": "<->",
    "⇒": "=>",
    # desenho de caixa (arvores de diretorio) -> ASCII
    "─": "-",
    "│": "|",
    "├": "+",
    "└": "\\",
    "┌": "+",
    "┐": "+",
    "┘": "+",
    "┤": "+",
    "┬": "+",
    # sinais matematicos que parecem ASCII mas nao sao
    "−": "-",     # U+2212 MINUS SIGN (diferente do hifen ASCII!)
    "–": "-",     # en dash
    "≫": ">>",    # U+226B  (muito maior que)
    "≪": "<<",
    # sobrescritos
    "ᵀ": "T",
    "ᵗ": "t",
    "ⁿ": "n",
    "⁰": "0",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    # subscritos
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
}
# Rede de seguranca: remove o que sobrou de emoji (blocos pictograficos).
# NAO inclui a faixa de setas (2190-21FF) nem matematica -- essas sao mapeadas acima.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U0001F000-\U0001F0FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)


def clean_emoji(text: str) -> str:
    for emo, rep in EMOJI_REPLACEMENTS.items():
        text = text.replace(emo, rep)
    text = _EMOJI_RE.sub("", text)
    _avisar_chars_sem_cobertura(text)
    return text


# Caracteres fora do cp1252 que a experiencia mostra que o reportlab consegue
# renderizar mesmo assim (ele substitui de fontes simbolicas).
_TOLERADOS = set("≈√∂≤≥≠×÷∞Σαβγδθλμπσφω")


def _avisar_chars_sem_cobertura(text: str):
    """Avisa se sobrou caractere que pode virar quadrado preto no PDF.

    E' uma rede de seguranca para capitulos futuros: melhor um aviso no console do
    que descobrir um quadrado preto olhando o PDF depois de publicado.
    """
    ruins = {}
    for ch in text:
        if ord(ch) < 128 or ch in _TOLERADOS:
            continue
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            ruins[ch] = ruins.get(ch, 0) + 1
    if ruins:
        detalhe = ", ".join(f"U+{ord(c):04X} ({n}x)" for c, n in sorted(ruins.items()))
        print(f"  AVISO: caracteres sem cobertura na fonte: {detalhe}")
        print("         adicione um mapeamento em EMOJI_REPLACEMENTS (build_pdf.py)")


# --------------------------------------------------------------------------
# CSS da apostila. xhtml2pdf suporta um subconjunto de CSS.
# --------------------------------------------------------------------------
def build_css() -> str:
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    return (
        """
@page {
    size: a4 portrait;
    margin: 2.2cm 1.9cm 2.4cm 1.9cm;
    @frame footer {
        -pdf-frame-content: footerContent;
        bottom: 1.1cm; left: 1.9cm; right: 1.9cm; height: 1cm;
    }
}
body { font-family: helvetica; font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 21pt; color: #0b3d63; margin-top: 18pt; margin-bottom: 6pt;
     border-bottom: 2pt solid #0b3d63; padding-bottom: 3pt; }
h2 { font-size: 15pt; color: #0b3d63; margin-top: 16pt; margin-bottom: 5pt; }
h3 { font-size: 12.5pt; color: #14507e; margin-top: 12pt; margin-bottom: 4pt; }
p  { margin: 5pt 0; text-align: justify; }
a  { color: #14507e; text-decoration: none; }
strong { color: #0b2233; }
ul, ol { margin: 4pt 0 4pt 6pt; }
li { margin: 2pt 0; }
blockquote {
    background: #eef4fa; border-left: 3pt solid #2f7fbf;
    margin: 8pt 0; padding: 5pt 9pt; color: #233; font-size: 10pt;
}
code {
    font-family: courier; font-size: 9pt;
    background: #f0f2f4; color: #b3105a;
}
.codehilite {
    background: #f6f8fa; border: 0.6pt solid #d6dde3;
    padding: 6pt 8pt; margin: 8pt 0;
}
.codehilite pre {
    font-family: courier; font-size: 8.3pt; line-height: 1.35;
    margin: 0; white-space: pre-wrap; word-wrap: break-word; color: #1a1a1a;
}
.codehilite code { background: transparent; color: inherit; }
table { margin: 8pt 0; width: 100%; }
th { background: #0b3d63; color: #ffffff; font-size: 8.5pt;
     padding: 4pt 5pt; text-align: left; }
td { font-size: 8.5pt; padding: 4pt 5pt; border-bottom: 0.5pt solid #d6dde3;
     vertical-align: top; }
hr { border: none; border-top: 0.6pt solid #c4ccd3; margin: 12pt 0; }
#footerContent { font-family: helvetica; font-size: 8pt; color: #8a939b; text-align: center; }
.cover { text-align: center; }
.cover h1 { border: none; font-size: 30pt; margin-top: 5.5cm; color: #0b3d63; }
.cover .sub { font-size: 13pt; color: #44525c; margin-top: 6pt; }
.cover .meta { font-size: 10pt; color: #7a838b; margin-top: 4cm; }
.cover .tag { font-size: 10pt; color: #2f7fbf; margin-top: 10pt; }
"""
        + pygments_css
    )


MD_EXTENSIONS = ["fenced_code", "codehilite", "tables", "toc", "sane_lists", "attr_list"]
MD_CONFIGS = {"codehilite": {"guess_lang": False, "noclasses": False}}


def _fix_pre_blocks(html: str) -> str:
    """Preserva a formatacao dentro de <pre>.

    O xhtml2pdf ignora `white-space: pre-wrap` e colapsa os espacos em branco como
    HTML comum -- o que destroi o codigo (tudo vira uma linha corrida). A correcao
    e' explicita: cada quebra de linha vira <br/> e a indentacao vira &nbsp;.
    """

    def convert(inner: str) -> str:
        inner = inner.strip("\n")
        out = []
        for line in inner.split("\n"):
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            out.append("&nbsp;" * indent + stripped)
        return "<br/>".join(out)

    return re.sub(
        r"(<pre[^>]*>)(.*?)(</pre>)",
        lambda m: m.group(1) + convert(m.group(2)) + m.group(3),
        html,
        flags=re.S,
    )


def md_to_html(md_text: str) -> str:
    md_text = clean_emoji(md_text)
    html = markdown.markdown(md_text, extensions=MD_EXTENSIONS, extension_configs=MD_CONFIGS)
    return _fix_pre_blocks(html)


def cover_html(title: str, subtitle: str) -> str:
    today = date.today().strftime("%d/%m/%Y")
    return (
        '<div class="cover">'
        f"<h1>{title}</h1>"
        f'<div class="sub">{subtitle}</div>'
        '<div class="tag">Construindo um Modelo de Linguagem do Zero</div>'
        '<div class="tag">Curso bilingue inspirado no LLM101n</div>'
        f'<div class="meta">Gerado em {today}</div>'
        "</div>"
        '<div style="page-break-after: always;"></div>'
    )


def render_pdf(inner_html: str, out_path: Path, title: str):
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{build_css()}</style></head>
<body>
<div id="footerContent">{title} &nbsp;|&nbsp; pag. <pdf:pagenumber> de <pdf:pagecount></div>
{inner_html}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if result.err:
        raise RuntimeError(f"Falha ao gerar PDF: {out_path}")
    try:
        shown = out_path.relative_to(ROOT)
    except ValueError:
        shown = out_path  # destino fora do repo (ex.: area de trabalho)
    print(f"OK  ->  {shown}")


def find_chapter_dir(num: str) -> Path:
    matches = sorted(CHAPTERS_DIR.glob(f"{num}-*"))
    if not matches:
        raise SystemExit(f"Capitulo {num} nao encontrado em {CHAPTERS_DIR}")
    return matches[0]


def nice_title(ch_dir: Path) -> str:
    # "01-bigram-language-model" -> "Bigram Language Model"
    raw = ch_dir.name.split("-", 1)[1] if "-" in ch_dir.name else ch_dir.name
    return raw.replace("-", " ").title()


def _markdowns_do_capitulo(ch_dir: Path):
    """Ordem de montagem do PDF: apostila, anexos (SETUP-*), exercicios.

    Qualquer .md extra no capitulo entra como anexo, para que o PDF seja
    autossuficiente -- quem le' so' o PDF nao pode ficar sem as instrucoes.
    """
    readme = ch_dir / "README.md"
    exer = ch_dir / "exercicios.md"
    anexos = sorted(
        p for p in ch_dir.glob("*.md") if p.name not in {"README.md", "exercicios.md"}
    )
    return [p for p in [readme, *anexos, exer] if p.exists()]


def build_chapter(num: str):
    ch_dir = find_chapter_dir(num)
    parts = []
    for i, md in enumerate(_markdowns_do_capitulo(ch_dir)):
        if i:
            parts.append('<div style="page-break-before: always;"></div>')
        parts.append(md_to_html(md.read_text(encoding="utf-8")))
    title = f"Capitulo {num}"
    cover = cover_html(f"Capitulo {num}", nice_title(ch_dir))
    render_pdf(cover + "".join(parts), ch_dir / f"Capitulo-{num}.pdf", title)


def build_all():
    parts = [md_to_html((ROOT / "README.md").read_text(encoding="utf-8"))]
    for ch_dir in sorted(CHAPTERS_DIR.glob("[0-9][0-9]-*")):
        parts.append('<div style="page-break-before: always;"></div>')
        readme = ch_dir / "README.md"
        if readme.exists():
            parts.append(md_to_html(readme.read_text(encoding="utf-8")))
        exer = ch_dir / "exercicios.md"
        if exer.exists():
            parts.append('<div style="page-break-before: always;"></div>')
            parts.append(md_to_html(exer.read_text(encoding="utf-8")))
    cover = cover_html("LLM101n-BR", "Apostila completa")
    render_pdf(cover + "".join(parts), DIST_DIR / "LLM101n-BR-Apostila-completa.pdf", "LLM101n-BR")


def build_gabaritos():
    """PDF SEPARADO com todos os gabaritos.

    Separado de proposito. Se as respostas fossem para o PDF do capitulo, elas
    ficariam na pagina seguinte a' dos exercicios -- e um exercicio cuja resposta
    esta' a um rolar de distancia nao e' um exercicio.

    Aqui o leitor precisa abrir OUTRO arquivo, o que ja' e' uma decisao
    consciente.
    """
    parts = []
    for ch_dir in sorted(CHAPTERS_DIR.glob("[0-9][0-9]-*")):
        gab = ch_dir / "solucoes" / "gabarito.md"
        if not gab.exists():
            continue
        if parts:
            parts.append('<div style="page-break-before: always;"></div>')
        parts.append(md_to_html(gab.read_text(encoding="utf-8")))
    if not parts:
        raise SystemExit("Nenhum solucoes/gabarito.md encontrado.")
    cover = cover_html("LLM101n-BR", "Gabaritos comentados")
    saida = ROOT / "docs"
    saida.mkdir(parents=True, exist_ok=True)
    # vai para docs/ (versionado), nao para dist/ (ignorado): este PDF e' material
    # de leitura, como os PDFs de capitulo -- nao um artefato de build
    render_pdf(cover + "".join(parts), saida / "LLM101n-BR-Gabaritos.pdf",
               "LLM101n-BR — Gabaritos")


def build_file(md_path: str, out_path: str | None, title: str | None):
    """Gera PDF de um arquivo Markdown qualquer (ex.: docs/panorama.md)."""
    src = Path(md_path)
    if not src.is_absolute():
        src = (ROOT / md_path).resolve()
    if not src.exists():
        raise SystemExit(f"Arquivo nao encontrado: {src}")
    out = Path(out_path) if out_path else src.with_suffix(".pdf")
    doc_title = title or src.stem.replace("-", " ").title()
    html = md_to_html(src.read_text(encoding="utf-8"))
    render_pdf(html, out, doc_title)


def main():
    ap = argparse.ArgumentParser(description="Gera a apostila do curso em PDF.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", help="numero do capitulo, ex.: 01")
    g.add_argument("--all", action="store_true", help="apostila completa")
    g.add_argument("--file", help="caminho de um .md avulso para virar PDF")
    g.add_argument("--gabaritos", action="store_true",
                   help="PDF separado com todos os gabaritos")
    ap.add_argument("--out", help="caminho de saida do PDF (use com --file)")
    ap.add_argument("--title", help="titulo no rodape (use com --file)")
    args = ap.parse_args()
    if args.all:
        build_all()
    elif args.gabaritos:
        build_gabaritos()
    elif args.file:
        build_file(args.file, args.out, args.title)
    else:
        build_chapter(args.chapter)


if __name__ == "__main__":
    main()
