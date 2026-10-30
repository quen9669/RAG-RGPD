"""Découpage hybride des sections en morceaux (chunks).

Pourquoi découper ? Plus tard, chaque morceau sera transformé en embedding
(un vecteur qui résume son sens) puis comparé à la question. Un morceau trop
long mélange plusieurs idées et son vecteur devient flou ; un morceau trop
court perd son contexte. Le découpage hybride (décision 0006) :

1. part de la structure : un article, un considérant ou une page ;
2. fusionne les pages de PDF voisines trop courtes (jamais les articles) ;
3. redécoupe une section trop longue par paragraphe, puis par ligne, puis
   par phrase si un paragraphe dépasse encore le plafond.

La taille est mesurée en caractères, simple à comprendre et indépendante du
modèle d'embedding qui sera choisi plus tard (un « token » de modèle vaut à
peu près 4 caractères en français).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_rgpd.processing.sections import Paragraph, Section

DEFAULT_MAX_CHARS = 1500
"""Plafond par défaut : environ 375 tokens, soit une demi-page. Assez pour
un paragraphe d'article avec ses points a) à f), assez court pour rester
centré sur une idée."""

DEFAULT_MIN_CHARS = 300
"""En dessous, une page de PDF est fusionnée avec la suivante."""

_SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class Piece:
    """Un morceau de texte et l'endroit précis du document d'où il vient.

    Attributes:
        label: désignation à citer (« Article 6, paragraphe 1 »).
        title: intitulé de la section d'origine.
        context: titres englobants (chapitre, section).
        text: texte du morceau.
        page_start: première page couverte (PDF), sinon ``None``.
        page_end: dernière page couverte (PDF), sinon ``None``.
    """

    label: str
    title: str
    context: tuple[str, ...]
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class _Unit:
    """Fragment indivisible pour le regroupement.

    ``part`` vaut 0 pour un paragraphe entier ou le début d'un paragraphe
    coupé, 1, 2... pour ses suites.
    """

    text: str
    number: str | None
    part: int = 0


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Coupe un texte trop long en respectant au mieux lignes et phrases.

    Ordre de préférence : retours à la ligne (éléments de liste), fins de
    phrase, puis espaces. On ne coupe au milieu d'un mot qu'en tout dernier
    recours (texte sans espace plus long que le plafond).

    Args:
        text: texte à couper.
        max_chars: taille maximale de chaque partie.

    Returns:
        Les parties, chacune d'au plus ``max_chars`` caractères.
    """
    if len(text) <= max_chars:
        return [text]
    for separator, pieces in (
        ("\n", text.split("\n")),
        (" ", _SENTENCE_END.split(text)),
        (" ", text.split(" ")),
    ):
        if len(pieces) > 1:
            return _pack_strings(pieces, separator, max_chars)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _pack_strings(pieces: list[str], separator: str, max_chars: int) -> list[str]:
    """Regroupe des fragments consécutifs sans dépasser ``max_chars``."""
    result: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = f"{buffer}{separator}{piece}" if buffer else piece
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            result.append(buffer)
        if len(piece) > max_chars:
            # Fragment encore trop long : on descend d'un niveau (ligne ->
            # phrase -> mot), sauf la dernière partie qui peut accueillir
            # la suite.
            *full, buffer = _hard_split(piece, max_chars)
            result.extend(full)
        else:
            buffer = piece
    if buffer:
        result.append(buffer)
    return result


def _units(paragraphs: tuple[Paragraph, ...], max_chars: int) -> list[_Unit]:
    """Transforme les paragraphes en fragments d'au plus ``max_chars``."""
    units = []
    for paragraph in paragraphs:
        for part, text in enumerate(_hard_split(paragraph.text, max_chars)):
            units.append(_Unit(text=text, number=paragraph.number, part=part))
    return units


def _group_label(base: str, group: list[_Unit], index: int, total: int) -> str:
    """Construit la désignation d'un morceau issu d'une section coupée.

    Exemples : « Article 6, paragraphe 1 », « Article 6, paragraphes 2 à 4 »,
    « Article 6, paragraphe 1 (suite) », « Page 3 (partie 2/3) ».
    """
    numbers = list(dict.fromkeys(u.number for u in group if u.number))
    if not numbers:
        return f"{base} (partie {index}/{total})"
    if len(numbers) == 1:
        label = f"{base}, paragraphe {numbers[0]}"
    else:
        label = f"{base}, paragraphes {numbers[0]} à {numbers[-1]}"
    if group[0].part > 0:
        label += " (suite)"
    return label


def _split_section(section: Section, max_chars: int) -> list[Piece]:
    """Découpe une section en morceaux d'au plus ``max_chars`` caractères."""
    pages = {"page_start": section.page, "page_end": section.last_page or section.page}
    if len(section.text) <= max_chars:
        return [Piece(section.label, section.title, section.context, section.text, **pages)]

    groups: list[list[_Unit]] = []
    size = 0
    for unit in _units(section.paragraphs, max_chars):
        added = len(unit.text) + (2 if groups and groups[-1] else 0)
        if groups and size + added <= max_chars:
            groups[-1].append(unit)
            size += added
        else:
            groups.append([unit])
            size = len(unit.text)

    return [
        Piece(
            label=_group_label(section.label, group, i, len(groups)),
            title=section.title,
            context=section.context,
            text="\n\n".join(u.text for u in group),
            **pages,
        )
        for i, group in enumerate(groups, start=1)
    ]


def _merge_short_pages(sections: list[Section], min_chars: int, max_chars: int) -> list[Section]:
    """Fusionne les pages voisines trop courtes (sections ``mergeable``).

    Une diapositive contient souvent une vingtaine de mots : seule, elle
    donnerait un morceau sans contexte. On l'accole à la suivante tant que
    le total reste sous le plafond. Le libellé devient « Pages 3 à 5 ».
    """
    merged: list[Section] = []
    for section in sections:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and previous.mergeable
            and section.mergeable
            and len(previous.text) < min_chars
            and len(previous.text) + 2 + len(section.text) <= max_chars
        ):
            last_page = section.last_page or section.page
            merged[-1] = Section(
                label=f"Pages {previous.page} à {last_page}",
                title=previous.title or section.title,
                context=previous.context,
                paragraphs=previous.paragraphs + section.paragraphs,
                page=previous.page,
                last_page=last_page,
                mergeable=True,
            )
        else:
            merged.append(section)
    return merged


def chunk_sections(
    sections: list[Section],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Piece]:
    """Découpe les sections d'un document en morceaux prêts à indexer.

    Args:
        sections: sections extraites du document, dans l'ordre.
        max_chars: taille maximale d'un morceau, en caractères.
        min_chars: en dessous de cette taille, une page de PDF est
            fusionnée avec la suivante.

    Returns:
        Les morceaux, dans l'ordre du document.

    Raises:
        ValueError: si ``max_chars`` est trop petit pour être raisonnable.
    """
    if max_chars < 100:
        raise ValueError("max_chars doit valoir au moins 100 caractères")
    return [
        piece
        for section in _merge_short_pages(sections, min_chars, max_chars)
        for piece in _split_section(section, max_chars)
    ]
