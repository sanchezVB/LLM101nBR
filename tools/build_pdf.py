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

Os PDFs de capitulo ficam dentro da pasta do proprio capitulo;
a apostila completa vai para dist/.
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
EMOJI_REPLACEMENTS = {
    "✅": "[OK]",
    "⏳": "[...]",
    "➡️": "->",
    "➡": "->",
    "🎯": "",
    "🚀": "",
    "🔑": "",
    "⚠️": "(!)",
    "⚠": "(!)",
}
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F0FF"
    "\U0000FE00-\U0000FE0F"
    "\U00002190-\U000021FF"
    "]+",
    flags=re.UNICODE,
)


def clean_emoji(text: str) -> str:
    for emo, rep in EMOJI_REPLACEMENTS.items():
        text = text.replace(emo, rep)
    return _EMOJI_RE.sub("", text)


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


def md_to_html(md_text: str) -> str:
    md_text = clean_emoji(md_text)
    return markdown.markdown(md_text, extensions=MD_EXTENSIONS, extension_configs=MD_CONFIGS)


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
    print(f"OK  ->  {out_path.relative_to(ROOT)}")


def find_chapter_dir(num: str) -> Path:
    matches = sorted(CHAPTERS_DIR.glob(f"{num}-*"))
    if not matches:
        raise SystemExit(f"Capitulo {num} nao encontrado em {CHAPTERS_DIR}")
    return matches[0]


def nice_title(ch_dir: Path) -> str:
    # "01-bigram-language-model" -> "Bigram Language Model"
    raw = ch_dir.name.split("-", 1)[1] if "-" in ch_dir.name else ch_dir.name
    return raw.replace("-", " ").title()


def build_chapter(num: str):
    ch_dir = find_chapter_dir(num)
    parts = []
    readme = ch_dir / "README.md"
    if readme.exists():
        parts.append(md_to_html(readme.read_text(encoding="utf-8")))
    exer = ch_dir / "exercicios.md"
    if exer.exists():
        parts.append('<div style="page-break-before: always;"></div>')
        parts.append(md_to_html(exer.read_text(encoding="utf-8")))
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


def main():
    ap = argparse.ArgumentParser(description="Gera a apostila do curso em PDF.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", help="numero do capitulo, ex.: 01")
    g.add_argument("--all", action="store_true", help="apostila completa")
    args = ap.parse_args()
    if args.all:
        build_all()
    else:
        build_chapter(args.chapter)


if __name__ == "__main__":
    main()
