"""Tests de l'extraction : XHTML des actes européens et PDF."""

from pathlib import Path

import pymupdf

from rag_rgpd.processing.html import extract_eu_act
from rag_rgpd.processing.pdf import extract_pdf
from rag_rgpd.processing.sections import clean_text

FIXTURES = Path(__file__).parent / "fixtures"


def test_clean_text_normalise_espaces_et_caracteres_invisibles() -> None:
    brut = "Article 6 ;  données­personnelles\n\n  suite "
    assert clean_text(brut) == "Article 6 ; donnéespersonnelles•\nsuite"


def test_acte_ue_considerants_puis_articles() -> None:
    sections = extract_eu_act((FIXTURES / "acte-ue.html").read_bytes())

    assert [s.label for s in sections] == [
        "Considérant 1",
        "Considérant 2",
        "Article 1",
        "Article 6",
        "Article 12",
    ]


def test_acte_ue_considerant_sans_espace_parasite() -> None:
    sections = extract_eu_act((FIXTURES / "acte-ue.html").read_bytes())

    # La balise <span class="italic"> ne doit pas ajouter d'espace avant la virgule.
    assert "physiques devraient, quelle" in sections[1].text


def test_acte_ue_article_titre_contexte_et_paragraphes() -> None:
    article_6 = extract_eu_act((FIXTURES / "acte-ue.html").read_bytes())[3]

    assert article_6.title == "Licéité du traitement"
    assert article_6.context == ("Chapitre II : Principes",)
    assert [p.number for p in article_6.paragraphs] == ["1", "2"]
    # Les points a) et b) restent rattachés au paragraphe 1, une ligne chacun.
    assert article_6.paragraphs[0].text.splitlines() == [
        "1. Le traitement n'est licite que si, et dans la mesure où, au moins une "
        "des conditions suivantes est remplie:",
        "a) la personne concernée a consenti au traitement;",
        "b) le traitement est nécessaire à l'exécution d'un contrat;",
    ]


def test_acte_ue_section_dans_chapitre_et_fin_du_texte() -> None:
    article_12 = extract_eu_act((FIXTURES / "acte-ue.html").read_bytes())[-1]

    assert article_12.context[1] == "Section 1 : Transparence et modalités"
    # Ni la signature ni les notes de bas de page ne sont reprises.
    assert article_12.text == "Le responsable du traitement prend des mesures appropriées."


def test_acte_ue_format_recent_avec_prefixe_oj() -> None:
    html = (
        '<html><body><p class="oj-ti-art">Article 5</p>'
        '<p class="oj-sti-art">Principes</p>'
        '<p class="oj-normal">1. Les données doivent être traitées de manière licite.</p>'
        "</body></html>"
    ).encode()

    [article] = extract_eu_act(html)

    assert (article.label, article.title) == ("Article 5", "Principes")
    assert article.paragraphs[0].number == "1"


def _pdf(pages: list[list[str]]) -> bytes:
    """Fabrique un petit PDF : une liste de lignes de texte par page."""
    document = pymupdf.open()
    for lines in pages:
        page = document.new_page()
        for i, line in enumerate(lines):
            # Lignes espacées de 100 points : PyMuPDF en fait des blocs séparés.
            page.insert_text((72, 72 + 100 * i), line)
    content = document.tobytes()
    document.close()
    return content


def test_pdf_une_section_par_page_et_entetes_retires() -> None:
    contenus = ["Le RGPD.", "La CNPD.", "Le responsable.", "Le sous-traitant.", "Les droits."]
    pages = [["Brochure CNPD", contenu, str(n)] for n, contenu in enumerate(contenus, start=1)]

    sections = extract_pdf(_pdf(pages))

    assert [s.label for s in sections] == [f"Page {n}" for n in range(1, 6)]
    assert sections[2].page == 3
    assert sections[2].mergeable
    # En-tête répété et numéro de page retirés, seul le contenu reste.
    assert sections[2].text == "Le responsable."


def test_pdf_page_vide_ignoree() -> None:
    sections = extract_pdf(_pdf([["Bonjour."], [], ["Au revoir."]]))

    assert [s.label for s in sections] == ["Page 1", "Page 3"]
