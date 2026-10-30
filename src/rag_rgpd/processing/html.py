"""Extraction du RGPD depuis le XHTML de l'Office des publications (Cellar).

Le XHTML des actes de l'Union européenne repère chaque élément par une classe
CSS. Ce sont ces classes qu'on lit, plutôt que la mise en forme :

=================  ==============================================
Classe             Rôle
=================  ==============================================
``ti-section-1``   « CHAPITRE II » ou « Section 1 »
``ti-section-2``   intitulé du chapitre ou de la section
``ti-art``         « Article 6 »
``sti-art``        intitulé de l'article (« Licéité du traitement »)
``normal``         texte courant
``note``           note de bas de page (ignorée)
=================  ==============================================

Les versions récentes du format ajoutent le préfixe ``oj-`` (``oj-ti-art``) :
il est retiré pour traiter les deux formats de la même façon. Les listes
(« a) », « b) ») et les considérants (« (1) », « (2) ») sont des tableaux à
deux colonnes : chaque ligne de tableau devient une ligne de texte.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from rag_rgpd.processing.sections import Paragraph, Section, clean_text

_PARAGRAPH_NUMBER = re.compile(r"^(\d+)\.\s")  # « 1. Le présent règlement... »
_RECITAL = re.compile(r"^\((\d+)\)\s*(.+)", re.DOTALL)  # « (1) La protection... »
# Le texte s'arrête à la formule de signature ; ce qui suit (signatures,
# notes de bas de page) n'apporte rien aux réponses.
_END_OF_TEXT = re.compile(r"^Fait à \w+")


def _classes(tag: Tag) -> set[str]:
    """Classes CSS de la balise, sans le préfixe ``oj-`` des formats récents."""
    return {c.removeprefix("oj-") for c in tag.get("class") or []}


def _flat(text: str) -> str:
    """Remplace les retours à la ligne du fichier source par des espaces.

    Dans le XHTML, un retour à la ligne à l'intérieur d'un ``<p>`` ne sert
    qu'à la présentation du fichier : il ne doit pas couper la ligne de
    texte (sinon « a) » se retrouve seul sur sa ligne).
    """
    return text.replace("\r", " ").replace("\n", " ")


def _blocks(soup: BeautifulSoup) -> Iterator[tuple[set[str], str]]:
    """Parcourt le document dans l'ordre et renvoie ses lignes de texte.

    Une ligne est soit un paragraphe ``<p>`` hors tableau, soit une ligne de
    tableau ``<tr>`` dont les cellules sont mises bout à bout (« a) » + le
    texte de l'élément de liste). Les tableaux imbriqués (sous-listes) sont
    lus avec la ligne qui les contient, pour ne pas compter deux fois le
    même texte.

    Args:
        soup: document analysé.

    Yields:
        Les classes CSS de la ligne et son texte nettoyé (non vide).
    """
    for tag in soup.find_all(["p", "tr"]):
        if tag.find_parent("tr") is not None:
            continue
        if tag.name == "tr":
            cells = tag.find_all(["td", "th"], recursive=False)
            # Séparateur vide à l'intérieur d'une cellule : les balises de
            # mise en forme (<span class="italic">) ne doivent pas ajouter
            # d'espace avant une virgule.
            text = " ".join(_flat(cell.get_text("")) for cell in cells)
            classes = set().union(*(_classes(p) for p in tag.find_all("p")))
        else:
            text = _flat(tag.get_text(""))
            classes = _classes(tag)
        text = clean_text(text)
        if text:
            yield classes, text


def _article_label(text: str) -> str:
    """« Article premier » devient « Article 1 », pour des citations uniformes."""
    return "Article 1" if text.lower() == "article premier" else text


@dataclass
class _Builder:
    """Section en cours de lecture (version modifiable de ``Section``)."""

    label: str
    context: tuple[str, ...]
    title: str = ""
    paragraphs: list[Paragraph] = field(default_factory=list)

    def add_line(self, text: str) -> None:
        """Ajoute une ligne : nouveau paragraphe numéroté, ou suite du précédent.

        Les éléments de liste (« a) ... ») suivent le paragraphe qui les
        annonce : on les y rattache pour que « article 6, paragraphe 1 »
        contienne bien ses points a) à f).
        """
        match = _PARAGRAPH_NUMBER.match(text)
        if match:
            self.paragraphs.append(Paragraph(text=text, number=match.group(1)))
        elif self.paragraphs:
            last = self.paragraphs[-1]
            self.paragraphs[-1] = Paragraph(text=f"{last.text}\n{text}", number=last.number)
        else:
            self.paragraphs.append(Paragraph(text=text))

    def build(self) -> Section:
        """Fige la section."""
        return Section(
            label=self.label,
            title=self.title,
            context=self.context,
            paragraphs=tuple(self.paragraphs),
        )


def extract_eu_act(content: bytes) -> list[Section]:
    """Découpe le XHTML d'un acte européen en considérants et articles.

    Le préambule (« LE PARLEMENT EUROPÉEN ET LE CONSEIL... », « vu le
    traité... ») est ignoré : il ne contient pas de règle. Les considérants,
    eux, sont gardés car ils expliquent comment interpréter les articles.

    Args:
        content: fichier XHTML brut (l'encodage est lu dans le fichier).

    Returns:
        Les considérants puis les articles, dans l'ordre du texte.
    """
    soup = BeautifulSoup(content, "html.parser")
    sections: list[Section] = []
    chapter = subsection = ""
    waiting_for_title: str | None = None  # "chapter" ou "section"
    current: _Builder | None = None

    def flush() -> None:
        nonlocal current
        if current is not None and current.paragraphs:
            sections.append(current.build())
        current = None

    for classes, text in _blocks(soup):
        if "note" in classes:
            continue
        if _END_OF_TEXT.match(text):
            break

        if "ti-section-1" in classes:
            flush()
            if text.upper().startswith("CHAPITRE"):
                # « CHAPITRE II » -> « Chapitre II » (le numéro romain reste en capitales).
                chapter = "Chapitre" + text[len("CHAPITRE") :]
                subsection, waiting_for_title = "", "chapter"
            else:
                subsection, waiting_for_title = text, "section"
            continue
        if "ti-section-2" in classes and waiting_for_title:
            if waiting_for_title == "chapter":
                chapter = f"{chapter} : {text}"
            else:
                subsection = f"{subsection} : {text}"
            waiting_for_title = None
            continue

        context = tuple(c for c in (chapter, subsection) if c)
        if "ti-art" in classes:
            flush()
            current = _Builder(label=_article_label(text), context=context)
            continue
        if current is not None:
            if "sti-art" in classes and not current.title and not current.paragraphs:
                current.title = text
            else:
                current.add_line(text)
            continue

        # Avant le premier article : seuls les considérants nous intéressent.
        recital = _RECITAL.match(text)
        if recital:
            sections.append(
                Section(
                    label=f"Considérant {recital.group(1)}",
                    title="",
                    context=(),
                    paragraphs=(Paragraph(text=recital.group(2)),),
                )
            )

    flush()
    return sections
