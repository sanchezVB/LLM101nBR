"""
build_academico.py — a apostila completa em formatacao academica (ABNT).

Este e' um documento DIFERENTE do que o build_pdf.py produz, e nao uma variacao
de estilo. O build_pdf.py gera material didatico: cores, titulos grandes, blocos
destacados, gabaritos em arquivo separado. Aqui a norma manda, e ela pede o
oposto -- Times 12, entrelinha 1,5, tudo monocromatico, secoes numeradas
progressivamente, e um unico volume com apendices.

    python tools/build_academico.py

Normas aplicadas:
    NBR 14724  estrutura do trabalho, margens, fonte, entrelinha, paginacao
    NBR 6024   numeracao progressiva das secoes
    NBR 6027   sumario
    NBR 6023   referencias

ESTRUTURA DO VOLUME
    capa                         nao contada, sem numero
    folha de rosto               contada como 1, sem numero impresso
    sumario                      contado, sem numero impresso
    1..17  capitulos             numeracao impressa a partir daqui
    REFERENCIAS
    APENDICE A  gabaritos
    APENDICE B  fontes por capitulo

POR QUE TRES PDFs E UMA COLAGEM
    A ABNT nao conta a capa. O <pdf:pagenumber> do xhtml2pdf e' absoluto e nao
    aceita deslocamento, entao um documento unico erraria a numeracao em 1 --
    justamente na folha que a norma manda nao contar. A saida e' renderizar capa,
    pre-textual e miolo em separado, juntar com o pypdf e carimbar os numeros com
    o reportlab, que da' controle exato de quais folhas recebem numero e de qual
    numero cada uma recebe.

POR QUE O SUMARIO PRECISA DE DUAS PASSADAS
    Numero de pagina no sumario exige saber em que pagina cada secao caiu, o que
    so' se sabe depois de renderizar. O xhtml2pdf tem <pdf:toc>, mas ele sai
    vazio nesta versao -- foi testado. O caminho que funciona e' ler o OUTLINE do
    PDF ja' gerado: o pypdf devolve a pagina de destino de cada marcador.

    O miolo e' renderizado UMA vez (e' o caro). So' o pre-textual entra no laco,
    porque so' ele muda -- e ele muda pouco: o numero de LINHAS do sumario nao
    depende dos digitos, entao a contagem de folhas converge quase sempre na
    primeira tentativa. O laco existe para o caso em que nao converge, e ele
    verifica em vez de supor.
"""

import io
import re
import sys
from datetime import date
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from xhtml2pdf import pisa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pdf import ROOT, md_to_html                  # noqa: E402

CHAPTERS_DIR = ROOT / "chapters"
DOCS = ROOT / "docs"
# Vai para docs/ (versionado), e nao para dist/ (ignorado), pela mesma razao que
# o PDF de gabaritos: e' documento final de LEITURA, nao artefato de build. Quem
# clona o repositorio deve receber o volume pronto, sem precisar instalar a
# cadeia de geracao de PDF so' para te'-lo.
SAIDA = ROOT / "docs" / "LLM101n-BR-Apostila-Academica.pdf"

AUTOR = "Vinicius Brasil Sanchez"
CIDADE = "Santo André"


# ==========================================================================
# CSS -- a norma, traduzida
# ==========================================================================
# NBR 14724: fonte tamanho 12 no texto, entrelinha 1,5, margens esquerda e
# superior de 3 cm, direita e inferior de 2 cm, texto justificado.
#
# "Times New Roman" e' pedida pelo nome. Usamos `times`, que o xhtml2pdf resolve
# para a Times-Roman das fontes base-14 do PDF -- metricamente identica a Times
# New Roman e presente em qualquer leitor, sem carregar .ttf (que e' instavel no
# Windows com reportlab 4.x, e por isso o resto do projeto tambem evita).
CSS = """
@page {
    size: a4 portrait;
    margin: 3cm 2cm 2cm 3cm;
}
body { font-family: times; font-size: 12pt; line-height: 1.5; color: #000000; }

/* O xhtml2pdf NAO herda font-family de forma confiavel: sem declarar elemento a
   elemento, ul/li/blockquote/td caem na Helvetica padrao. Descobri isso lendo os
   recursos de fonte do PDF gerado, e nao olhando a pagina -- a Helvetica e a
   Times sao parecidas o bastante para passar despercebidas num PDF de 400
   folhas, e a norma pede uma delas por nome. */
div, span, p, ul, ol, li, blockquote, table, tr, td, th, em, strong, i, b {
    font-family: times;
}

/* NBR 6024 pede distincao tipografica entre os niveis de secao. Todos ficam no
   corpo 12: o que muda e' caixa e peso, nao tamanho.
   A CAIXA ALTA e' aplicada em Python, nao aqui: text-transform nao e' suportado
   pelo xhtml2pdf -- ele aceita a propriedade em silencio e nao faz nada. */
h1 { font-family: times; font-size: 12pt; font-weight: bold;
     text-align: left; margin: 0 0 18pt 0; -pdf-outline: true; -pdf-outline-level: 0;
     -pdf-keep-with-next: true; }
h2 { font-family: times; font-size: 12pt; font-weight: bold;
     margin: 18pt 0 8pt 0; -pdf-outline: true; -pdf-outline-level: 1;
     -pdf-keep-with-next: true; }
h3 { font-family: times; font-size: 12pt; font-weight: normal;
     margin: 14pt 0 6pt 0; -pdf-outline: true; -pdf-outline-level: 2;
     -pdf-keep-with-next: true; }
h4 { font-family: times; font-size: 12pt; font-style: italic; margin: 12pt 0 5pt 0; }

p { text-align: justify; text-indent: 1.25cm; margin: 0 0 6pt 0; }

/* NBR 10520: citacao com mais de tres linhas recua 4 cm, cai para corpo menor e
   volta a espacamento simples. As citacoes em bloco do original viram isso. */
blockquote { margin: 10pt 0 10pt 4cm; font-size: 10pt; line-height: 1.0;
             text-align: justify; }
blockquote p { text-indent: 0; margin: 0 0 4pt 0; font-size: 10pt; }

ul, ol { margin: 6pt 0 6pt 1.25cm; }
li { margin: 0 0 3pt 0; text-align: justify; }

a { color: #000000; text-decoration: none; }
strong { font-weight: bold; }

/* Codigo nao e' previsto pela norma. Mantemos monoespacado, corpo 10 e
   espacamento simples: recuar ou justificar destruiria a indentacao, que aqui
   carrega significado. */
code { font-family: courier; font-size: 10pt; }
.codehilite { border: 0.5pt solid #999999; padding: 5pt 7pt; margin: 8pt 0 8pt 0; }
.codehilite pre { font-family: courier; font-size: 9pt; line-height: 1.0; margin: 0;
                  color: #000000; }
.codehilite code { font-size: 9pt; }

/* Padrao IBGE, adotado pela ABNT: laterais abertas, sem grade interna. */
table { margin: 10pt 0; width: 100%; border-top: 1pt solid #000000;
        border-bottom: 1pt solid #000000; }
th { font-family: times; font-size: 10pt; font-weight: bold; padding: 4pt 5pt;
     text-align: left; border-bottom: 0.5pt solid #000000; }
td { font-family: times; font-size: 10pt; padding: 4pt 5pt; vertical-align: top; }

hr { border: none; border-top: 0.5pt solid #000000; margin: 12pt 0; }

/* ---- folhas especiais ---- */
.capa { text-align: center; }
.capa .inst { font-size: 14pt; font-weight: bold; margin-top: 0; text-indent: 0; }
.capa .autor { font-size: 12pt; margin-top: 3.5cm; text-indent: 0; }
.capa .titulo { font-size: 16pt; font-weight: bold;
                margin-top: 5cm; text-indent: 0; }
.capa .sub { font-size: 13pt; margin-top: 10pt; text-indent: 0; }
.capa .local { font-size: 12pt; margin-top: 6.5cm; text-indent: 0; }

.rosto { text-align: center; }
.rosto .autor { font-size: 12pt; margin-top: 0; text-indent: 0; }
.rosto .titulo { font-size: 16pt; font-weight: bold;
                 margin-top: 4.5cm; text-indent: 0; }
.rosto .sub { font-size: 13pt; margin-top: 10pt; text-indent: 0; }
/* NBR 14724: a natureza do trabalho fica recuada a partir do meio da mancha. */
.rosto .natureza { font-size: 10pt; line-height: 1.0; text-align: justify;
                   margin: 3.5cm 0 0 8cm; text-indent: 0; }
.rosto .local { font-size: 12pt; margin-top: 3cm; text-indent: 0; }

.titulo-sem-numero { font-family: times; font-size: 12pt; font-weight: bold;
 text-align: center;
                     margin: 0 0 18pt 0; }
/* NBR 14724: titulos sem indicativo numerico -- REFERENCIAS, APENDICE -- centrados. */
h1.centrado { text-align: center; }

/* Sumario: titulo a esquerda, folha a direita, sem grade. */
table.sumario { border: none; margin: 0; }
table.sumario td { font-size: 12pt; padding: 2pt 0; border: none; }
td.sum-pag { text-align: right; width: 1.6cm; }
td.sum-n1 { font-weight: bold; }
td.sum-n2 { padding-left: 1cm; }

/* Referencias: NBR 6023 pede alinhamento a esquerda e espacamento simples entre
   linhas da mesma entrada, com linha em branco entre entradas. */
.referencias p { text-align: left; text-indent: 0; line-height: 1.0;
                 margin: 0 0 12pt 0; }
"""


def _pygments_neutro() -> str:
    """Realce de sintaxe em PRETO -- a norma nao admite cor no corpo do texto.

    O build_pdf.py usa o tema 'friendly', colorido. Aqui as classes do pygments
    recebem so' peso e italico, que sobrevivem a uma impressao monocromatica.
    """
    return """
.codehilite .k, .codehilite .kn, .codehilite .kd, .codehilite .kc { font-weight: bold; }
.codehilite .c, .codehilite .c1, .codehilite .cm { font-style: italic; color: #555555; }
.codehilite .s, .codehilite .s1, .codehilite .s2, .codehilite .sd { color: #333333; }
.codehilite .nf, .codehilite .nc { font-weight: bold; }
"""


# ==========================================================================
# Numeracao progressiva das secoes (NBR 6024)
# ==========================================================================
# Os originais trazem "# Capitulo 09 -- Precision", "## 3. Alcance e precisao" e
# "### Overflow: numeros grandes demais". A norma quer "9 PRECISION", "9.3
# Alcance e precisao" e "9.3.1 Overflow". Entao: descartar a numeracao que ja'
# existe no h2 e reconstruir tudo a partir do numero do capitulo.
RE_H = re.compile(r"^(#{1,4})\s+(.*)$")
RE_CAP = re.compile(r"^Cap[ií]tulo\s+\d+\s*[—–-]\s*(.*)$", re.I)
RE_NUM_INICIAL = re.compile(r"^\d+\.\s*")


def renumerar(md: str, prefixo: str, *, titulo_h1=None, h1_literal=None,
              n2_inicial=0, rebaixar=False):
    """Reescreve os titulos na numeracao progressiva. Devolve (texto, ultimo_n2).

    `prefixo` e' o que antecede a numeracao: "9" para o capitulo 9, "A.9" para o
    gabarito dele no apendice. Sem isso, o gabarito do capitulo 9 abriria uma
    secao "9" ja' ocupada pelo proprio capitulo -- duas secoes com o mesmo
    numero, que e' exatamente o que a NBR 6024 existe para impedir.

    `rebaixar` desce o documento inteiro um nivel. Serve para os arquivos que sao
    CONTINUACAO de um capitulo, e nao capitulos: os exercicios do capitulo 9 sao
    a secao 9.10, nao uma secao 10. O h1 da origem vira h2, e tudo abaixo dele
    vira h3 -- numa unica sequencia, porque exercicios.md usa h3 direto, sem h2.

    `n2_inicial` continua a contagem de onde o arquivo anterior parou.
    """
    saida = []
    n2, n3 = n2_inicial, 0
    dentro_de_codigo = False
    for linha in md.split("\n"):
        # '#' dentro de bloco de codigo e' comentario Python, nao titulo.
        if linha.lstrip().startswith("```"):
            dentro_de_codigo = not dentro_de_codigo
            saida.append(linha)
            continue
        m = None if dentro_de_codigo else RE_H.match(linha)
        if not m:
            saida.append(linha)
            continue
        nivel, texto = len(m.group(1)), m.group(2).strip()
        texto = RE_NUM_INICIAL.sub("", texto)

        if rebaixar:
            if nivel == 1:
                n2 += 1
                n3 = 0
                saida.append(f"## {prefixo}.{n2} {titulo_h1 or texto}")
            else:
                n3 += 1
                saida.append(f"### {prefixo}.{n2}.{n3} {texto}")
            continue

        if nivel == 1:
            # CAIXA ALTA aqui, e nao no CSS: o xhtml2pdf ignora text-transform.
            # Fazendo no markdown, o titulo tambem entra em caixa alta no OUTLINE
            # -- e e' do outline que o sumario e' montado, entao os dois batem
            # sem nenhum tratamento extra.
            if h1_literal:
                saida.append(f"# {h1_literal.upper()}")
            else:
                mc = RE_CAP.match(texto)
                nome = titulo_h1 or (mc.group(1) if mc else texto)
                saida.append(f"# {prefixo} {nome.upper()}")
            n2, n3 = n2_inicial, 0
        elif nivel == 2:
            n2 += 1
            n3 = 0
            saida.append(f"## {prefixo}.{n2} {texto}")
        elif nivel == 3:
            # Um h3 antes de qualquer h2 nao tem pai: viraria 'x.0.1'. Promovemos.
            if n2 == n2_inicial:
                n2 += 1
                saida.append(f"## {prefixo}.{n2} {texto}")
            else:
                n3 += 1
                saida.append(f"### {prefixo}.{n2}.{n3} {texto}")
        else:
            saida.append(f"#### {texto}")
    return "\n".join(saida), n2


# Markdown -> HTML e' o MESMO do build_pdf.py, de proposito: emoji, blocos de
# codigo e a correcao das listas coladas valem para os dois documentos. Duplicar
# aqui garantiria que uma correcao futura entrasse so' num deles.
md_html = md_to_html


def quebra() -> str:
    return '<div style="page-break-before: always;"></div>'


# ==========================================================================
# Folhas pre-textuais
# ==========================================================================
def capa_html() -> str:
    ano = date.today().year
    return f"""<div class="capa">
<p class="inst">LLM101n-BR</p>
<p class="autor">{AUTOR}</p>
<p class="titulo">CONSTRUINDO UM MODELO DE LINGUAGEM DO ZERO</p>
<p class="sub">Do bigrama ao Transformer multimodal, com medicao de cada afirmacao</p>
<p class="local">{CIDADE}<br/>{ano}</p>
</div>"""


def folha_rosto_html() -> str:
    ano = date.today().year
    return f"""<div class="rosto">
<p class="autor">{AUTOR}</p>
<p class="titulo">CONSTRUINDO UM MODELO DE LINGUAGEM DO ZERO</p>
<p class="sub">Do bigrama ao Transformer multimodal, com medicao de cada afirmacao</p>
<p class="natureza">Material didatico em dezessete capitulos, com programas,
exercicios e gabaritos, sobre a construcao integral de um modelo de linguagem
autorregressivo sem o uso de bibliotecas de alto nivel. Organizado segundo o
programa do curso LLM101n, de Andrej Karpathy e Eureka Labs.</p>
<p class="local">{CIDADE}<br/>{ano}</p>
</div>"""


def _partir_apresentacao():
    """docs/apresentacao.md guarda duas pecas que a norma separa.

    O RESUMO e' pre-textual (vem antes do sumario); a APRESENTACAO abre o texto.
    Ficam no mesmo arquivo porque sao escritos juntos e se referenciam.
    """
    txt = (DOCS / "apresentacao.md").read_text(encoding="utf-8")
    marca = "\n# APRESENTAÇÃO"
    if marca not in txt:
        raise SystemExit("docs/apresentacao.md perdeu o titulo '# APRESENTAÇÃO'")
    resumo, apres = txt.split(marca, 1)
    return resumo.rstrip().rstrip("-").rstrip(), "# APRESENTAÇÃO" + apres


def sumario_html(entradas) -> str:
    """entradas: lista de (nivel, texto, folha) -- folha None na passada cega."""
    linhas = []
    for nivel, texto, folha in entradas:
        classe = "sum-n1" if nivel == 0 else "sum-n2"
        pag = "&nbsp;" if folha is None else str(folha)
        linhas.append(f'<tr><td class="{classe}">{texto}</td>'
                      f'<td class="sum-pag">{pag}</td></tr>')
    return ('<p class="titulo-sem-numero">SUMARIO</p>'
            '<table class="sumario">' + "".join(linhas) + "</table>")


# ==========================================================================
# Montagem do miolo
# ==========================================================================
def _titulo_do_capitulo(ch_dir: Path) -> str:
    """Le' o titulo do proprio README, que e' a fonte da verdade."""
    txt = (ch_dir / "README.md").read_text(encoding="utf-8")
    primeira = txt.split("\n", 1)[0].lstrip("# ").strip()
    m = RE_CAP.match(primeira)
    return m.group(1) if m else primeira


def _centrar_h1(html: str) -> str:
    """Titulos SEM indicativo numerico (REFERENCIAS, APENDICE) vao centrados.

    NBR 14724. Continuam sendo <h1> para que entrem no outline -- e' de la' que o
    sumario tira as folhas.
    """
    return re.sub(r"<h1>(.*?)</h1>",
                  lambda m: f'<h1 class="centrado">{m.group(1).upper()}</h1>',
                  html, count=1)


def montar_miolo() -> str:
    caps = sorted(CHAPTERS_DIR.glob("[0-9][0-9]-*"))

    # ---------------- APRESENTACAO (abre o texto, sem numero) ----------------
    # (sem quebra() aqui: o laco dos capitulos ja' insere uma antes do cap. 1)
    _, apres = _partir_apresentacao()
    partes = [_centrar_h1(md_html(apres))]

    # ---------------- textual: os 17 capitulos ----------------
    for ch_dir in caps:
        n = int(ch_dir.name[:2])
        if partes:
            partes.append(quebra())
        html, n2 = renumerar((ch_dir / "README.md").read_text(encoding="utf-8"), str(n))
        partes.append(md_html(html))

        # Anexos (ex.: SETUP-GPU.md) e exercicios sao CONTINUACAO do capitulo:
        # entram rebaixados, seguindo a contagem de secoes de onde ela parou.
        extras = sorted(p for p in ch_dir.glob("*.md")
                        if p.name not in {"README.md", "exercicios.md"})
        for extra in extras:
            html, n2 = renumerar(extra.read_text(encoding="utf-8"), str(n),
                                 titulo_h1=extra.stem.replace("-", " ").title(),
                                 n2_inicial=n2, rebaixar=True)
            partes.append(md_html(html))

        exer = ch_dir / "exercicios.md"
        if exer.exists():
            html, n2 = renumerar(exer.read_text(encoding="utf-8"), str(n),
                                 titulo_h1="Exercicios", n2_inicial=n2,
                                 rebaixar=True)
            partes.append(md_html(html))

    # ---------------- REFERENCIAS ----------------
    # Pos-textual, e ANTES dos apendices: e' a ordem da NBR 14724.
    partes.append(quebra())
    ref = _centrar_h1(md_html((DOCS / "referencias.md").read_text(encoding="utf-8")))
    partes.append(f'<div class="referencias">{ref}</div>')

    # ---------------- APENDICE A: gabaritos ----------------
    partes.append(quebra())
    partes.append('<h1 class="centrado">APENDICE A — GABARITOS COMENTADOS</h1>')
    partes.append(
        "<p>Os gabaritos foram reunidos em apendice, e nao ao final de cada "
        "capitulo, por uma razao pedagogica: um exercicio cuja resposta esta' na "
        "pagina seguinte nao e' um exercicio. Consultar este apendice deve ser "
        "uma decisao, nao um acidente de leitura.</p>"
        "<p>Todos os numeros aqui apresentados vem de execucao. Onde a medicao "
        "contrariou a hipotese inicial do autor, o texto registra as duas coisas "
        "— a previsao e o resultado —, porque o erro documentado ensina mais que "
        "o acerto silencioso.</p>")
    for ch_dir in caps:
        gab = ch_dir / "solucoes" / "gabarito.md"
        if not gab.exists():
            continue
        n = int(ch_dir.name[:2])
        partes.append(quebra())
        html, _ = renumerar(
            gab.read_text(encoding="utf-8"), f"A.{n}",
            titulo_h1=f"Gabarito do capitulo {n} — {_titulo_do_capitulo(ch_dir)}")
        partes.append(md_html(html))

    # ---------------- APENDICE B: fontes por capitulo ----------------
    partes.append(quebra())
    html, _ = renumerar((DOCS / "fontes-por-capitulo.md").read_text(encoding="utf-8"),
                        "B", h1_literal="APENDICE B — Fontes primarias por capitulo")
    partes.append(_centrar_h1(md_html(html)))

    return "".join(partes)


# ==========================================================================
# Renderizacao
# ==========================================================================
def render(inner_html: str, destino: Path):
    html = (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f"<style>{CSS}{_pygments_neutro()}</style></head>"
            f"<body>{inner_html}</body></html>")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "wb") as f:
        r = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if r.err:
        raise RuntimeError(f"falha ao renderizar {destino}")


def entradas_do_outline(pdf: Path, deslocamento: int):
    """Le' o outline do PDF e devolve (nivel, titulo, folha_abnt).

    `deslocamento` converte a pagina interna do miolo na folha do volume colado.
    So' os niveis 0 e 1 entram no sumario -- o nivel 3 existiria em ~400 linhas.
    """
    leitor = PdfReader(str(pdf))
    fora = []

    def andar(itens, nivel=0):
        for it in itens:
            if isinstance(it, list):
                andar(it, nivel + 1)
            elif nivel <= 1:
                pag = leitor.get_destination_page_number(it)
                fora.append((nivel, str(it.title), pag + deslocamento))

    andar(leitor.outline)
    return fora


def carimbar(entrada: Path, saida: Path, primeira_com_numero: int):
    """Imprime o numero da folha no canto superior direito, a 2 cm das bordas.

    NBR 14724. A capa nao e' contada; a contagem comeca na folha de rosto, que
    e' o indice 1 (base zero) do volume colado -- por isso o numero impresso e'
    o proprio indice.
    """
    leitor = PdfReader(str(entrada))
    escritor = PdfWriter()
    largura, altura = A4
    for i, pagina in enumerate(leitor.pages):
        if i >= primeira_com_numero:
            buf = io.BytesIO()
            c = canvas.Canvas(buf, pagesize=A4)
            c.setFont("Times-Roman", 10)
            c.drawRightString(largura - 2 * cm, altura - 2 * cm, str(i))
            c.save()
            buf.seek(0)
            pagina.merge_page(PdfReader(buf).pages[0])
        escritor.add_page(pagina)
    with open(saida, "wb") as f:
        escritor.write(f)


def main():
    tmp = ROOT / "dist" / "_academico"
    tmp.mkdir(parents=True, exist_ok=True)
    p_capa, p_pre, p_miolo = tmp / "capa.pdf", tmp / "pre.pdf", tmp / "miolo.pdf"

    print("  [1/5] capa e folha de rosto")
    render(capa_html(), p_capa)

    print("  [2/5] miolo (17 capitulos, referencias, 2 apendices) -- demora")
    render(montar_miolo(), p_miolo)
    n_miolo = len(PdfReader(str(p_miolo)).pages)
    print(f"        {n_miolo} folhas")

    print("  [3/5] resumo e sumario, ate' as folhas convergirem")
    resumo, _ = _partir_apresentacao()
    # Passada cega: o numero de LINHAS do sumario ja' e' o definitivo, entao a
    # contagem de folhas do pre-textual sai certa mesmo sem os numeros.
    esqueleto = entradas_do_outline(p_miolo, 0)
    n_pre = None
    for tentativa in range(1, 5):
        desloc = 1 + (n_pre if n_pre else 1)
        entradas = [(n, t, p + desloc) for n, t, p in esqueleto] if n_pre else \
                   [(n, t, None) for n, t, _ in esqueleto]
        # NBR 14724, ordem do pre-textual: folha de rosto, resumo, sumario.
        render(folha_rosto_html() + quebra()
               + _centrar_h1(md_html(resumo)) + quebra()
               + sumario_html(entradas), p_pre)
        novo = len(PdfReader(str(p_pre)).pages)
        if novo == n_pre:
            print(f"        convergiu em {tentativa} passada(s): "
                  f"{n_pre} folhas pre-textuais")
            break
        n_pre = novo
    else:
        raise RuntimeError("o sumario nao convergiu em 4 passadas")

    print("  [4/5] colando o volume")
    juntos = tmp / "juntos.pdf"
    w = PdfWriter()
    for p in (p_capa, p_pre, p_miolo):
        for pag in PdfReader(str(p)).pages:
            w.add_page(pag)
    with open(juntos, "wb") as f:
        w.write(f)

    print("  [5/5] numerando as folhas (NBR 14724)")
    carimbar(juntos, SAIDA, primeira_com_numero=1 + n_pre)

    total = len(PdfReader(str(SAIDA)).pages)
    print(f"\nOK  ->  {SAIDA.relative_to(ROOT)}  ({total} folhas)")
    print(f"        capa 1 | pre-textual {n_pre} | miolo {n_miolo}")

    # As tres pecas intermediarias so' existem para serem coladas. Deixa-las no
    # disco convida alguem a abrir a errada -- a capa avulsa e o miolo sem
    # numeracao sao parecidos o bastante com o volume real para confundir.
    for p in (p_capa, p_pre, p_miolo, juntos):
        p.unlink(missing_ok=True)
    tmp.rmdir()


if __name__ == "__main__":
    main()
