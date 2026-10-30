"""Tests du découpage hybride."""

import pytest

from rag_rgpd.processing.chunking import chunk_sections
from rag_rgpd.processing.sections import Paragraph, Section


def _article(*paragraphs: str, numbered: bool = True) -> Section:
    return Section(
        label="Article 6",
        title="Licéité du traitement",
        context=("Chapitre II : Principes",),
        paragraphs=tuple(
            Paragraph(text=f"{i}. {t}" if numbered else t, number=str(i) if numbered else None)
            for i, t in enumerate(paragraphs, start=1)
        ),
    )


def _page(number: int, text: str) -> Section:
    return Section(
        label=f"Page {number}", title="", context=(), paragraphs=(Paragraph(text),),
        page=number, mergeable=True,
    )


def test_section_courte_un_seul_morceau() -> None:
    [piece] = chunk_sections([_article("Court.", "Aussi court.")], max_chars=200)

    assert piece.label == "Article 6"
    assert piece.text == "1. Court.\n\n2. Aussi court."
    assert piece.context == ("Chapitre II : Principes",)


def test_section_longue_redecoupee_par_paragraphe() -> None:
    article = _article("a" * 80, "b" * 80, "c" * 80)

    pieces = chunk_sections([article], max_chars=200)

    assert [p.label for p in pieces] == [
        "Article 6, paragraphes 1 à 2",
        "Article 6, paragraphe 3",
    ]
    assert all(len(p.text) <= 200 for p in pieces)
    assert all(p.title == "Licéité du traitement" for p in pieces)


def test_paragraphe_trop_long_coupe_aux_lignes_puis_suite() -> None:
    liste = "Introduction:\n" + "\n".join(f"{x}) " + "mot " * 20 for x in "abcdef")
    pieces = chunk_sections([_article(liste)], max_chars=200)

    assert pieces[0].label == "Article 6, paragraphe 1"
    assert pieces[1].label == "Article 6, paragraphe 1 (suite)"
    assert all(len(p.text) <= 200 for p in pieces)
    # Aucune ligne de liste n'est coupée en deux.
    lines = [line for p in pieces for line in p.text.split("\n")]
    assert sum(line.startswith(("a)", "b)", "c)", "d)", "e)", "f)")) for line in lines) == 6


def test_texte_sans_paragraphe_numerote_coupe_en_parties() -> None:
    phrases = " ".join(f"Phrase numéro {i} assez longue pour remplir." for i in range(20))

    pieces = chunk_sections([_article(phrases, numbered=False)], max_chars=200)

    assert pieces[0].label == f"Article 6 (partie 1/{len(pieces)})"
    assert all(len(p.text) <= 200 for p in pieces)
    assert all(p.text.endswith("remplir.") for p in pieces)


def test_pages_courtes_fusionnees_mais_pas_les_articles() -> None:
    sections = [_page(1, "Titre."), _page(2, "Sommaire."), _page(3, "x" * 150), _article("Court.")]

    pieces = chunk_sections(sections, max_chars=200, min_chars=50)

    assert [p.label for p in pieces] == ["Pages 1 à 3", "Article 6"]
    assert (pieces[0].page_start, pieces[0].page_end) == (1, 3)
    assert pieces[1].page_start is None


def test_page_longue_pas_fusionnee() -> None:
    pieces = chunk_sections([_page(1, "x" * 150), _page(2, "y")], max_chars=200, min_chars=50)

    assert [p.label for p in pieces] == ["Page 1", "Page 2"]
    assert pieces[1].page_start == pieces[1].page_end == 2


def test_plafond_trop_petit_refuse() -> None:
    with pytest.raises(ValueError):
        chunk_sections([], max_chars=10)
