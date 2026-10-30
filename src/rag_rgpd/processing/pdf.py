"""Extraction du texte des PDF (guides et brochures de la CNPD) avec PyMuPDF.

Un PDF ne contient pas de « paragraphes » : seulement des morceaux de texte
placés à des coordonnées sur la page. PyMuPDF les regroupe en *blocs*
(à peu près des paragraphes) ; l'option ``sort=True`` les remet dans l'ordre
de lecture (de haut en bas, puis de gauche à droite), ce qui compte pour les
PDF mis en page comme des diapositives.
"""

from __future__ import annotations

import re
from collections import Counter

import pymupdf

from rag_rgpd.processing.sections import Paragraph, Section, clean_text

# Mot coupé en fin de ligne : « traite-\nment » doit redevenir « traitement ».
# On ne recolle que si la ligne suivante commence par une minuscule, pour ne
# pas abîmer les vrais mots composés (« sous-\nTraitant » est rare, mais
# « Grand-\nDuché » existe).
_HYPHENATION = re.compile(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])")
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*/\s*\d+)?$", re.IGNORECASE)

# Seuils pour reconnaître les en-têtes et pieds de page répétés.
_MIN_PAGES_FOR_REPEATS = 4
_REPEAT_RATIO = 0.5


def _block_text(raw: str) -> str:
    """Recolle les mots coupés et met les lignes d'un bloc bout à bout.

    À l'intérieur d'un bloc, un retour à la ligne vient de la largeur de la
    colonne, pas du sens : on le remplace par une espace.
    """
    return clean_text(_HYPHENATION.sub("", raw).replace("\n", " "))


def _signature(text: str) -> str:
    """Forme simplifiée d'un bloc pour repérer les répétitions.

    Les chiffres sont neutralisés : « Brochure CNPD, page 3 » et
    « Brochure CNPD, page 4 » sont le même pied de page.
    """
    return re.sub(r"\d+", "#", text.lower())


def _repeated_blocks(pages: list[list[str]]) -> set[str]:
    """Repère les blocs présents sur au moins la moitié des pages.

    Ce sont les en-têtes et pieds de page (titre de la brochure, logo,
    mention de copyright) : répétés partout, ils pollueraient chaque
    morceau et fausseraient la recherche.

    Args:
        pages: blocs de texte de chaque page.

    Returns:
        Les signatures (voir ``_signature``) des blocs à retirer.
    """
    if len(pages) < _MIN_PAGES_FOR_REPEATS:
        return set()
    counts = Counter(sig for blocks in pages for sig in {_signature(b) for b in blocks})
    threshold = max(3, _REPEAT_RATIO * len(pages))
    return {sig for sig, n in counts.items() if n >= threshold}


def extract_pdf(content: bytes) -> list[Section]:
    """Extrait le texte d'un PDF, page par page.

    Chaque page devient une section (« Page 4 ») marquée fusionnable : le
    découpage pourra regrouper des pages voisines trop courtes. Les images
    sont ignorées ; un PDF scanné (sans texte) renverra donc une liste vide,
    il faudrait alors de l'OCR (reconnaissance de caractères).

    Args:
        content: fichier PDF brut.

    Returns:
        Une section par page contenant du texte utile.
    """
    with pymupdf.open(stream=content, filetype="pdf") as document:
        pages = [
            [
                text
                # Un bloc est un tuple (x0, y0, x1, y1, texte, numéro, type),
                # type 0 = texte, 1 = image.
                for *_, raw, _number, kind in page.get_text("blocks", sort=True)
                if kind == 0 and (text := _block_text(raw))
            ]
            for page in document
        ]

    repeated = _repeated_blocks(pages)
    sections = []
    for number, blocks in enumerate(pages, start=1):
        kept = [
            b for b in blocks if _signature(b) not in repeated and not _PAGE_NUMBER.match(b)
        ]
        if kept:
            sections.append(
                Section(
                    label=f"Page {number}",
                    title="",
                    context=(),
                    paragraphs=tuple(Paragraph(text=b) for b in kept),
                    page=number,
                    mergeable=True,
                )
            )
    return sections
