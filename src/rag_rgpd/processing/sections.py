"""Structures communes à l'extraction et au découpage, et nettoyage du texte."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Espaces « exotiques » fréquents dans les documents officiels : espace
# insécable, espace fine insécable (avant « ; » et « : » en français), etc.
# On les remplace par une espace simple pour que la recherche ne les voie
# pas comme des caractères différents.
_SPECIAL_SPACES = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000\t]")
_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\u2060\ufeff\u00ad]")
# Les puces des PDF sont souvent des glyphes de polices « symboles », rangés
# dans la zone Unicode à usage privé : illisibles une fois extraits.
_PRIVATE_USE = re.compile(r"[\ue000-\uf8ff]")
# Puces de liste des PDF (•, ▪, ▫, ◦, ‣...) : on les ramène toutes à « • »,
# une seule fois, suivie d'une espace (« ▫ ▫ nom » et « •Nom » deviennent
# « • nom » et « • Nom »).
_BULLETS = re.compile(r"(?:[•▪▫◦‣∙●○■□➢➤►][ \t\u00a0]*)+")
# Apostrophes et guillemets typographiques : une seule forme d'apostrophe,
# pour que « l'article » et « l’article » soient le même mot pour la
# recherche ; pas d'espace juste à l'intérieur des guillemets anglais.
_APOSTROPHES = re.compile(r"[’‘ʼ]")
_QUOTE_INNER_SPACES = re.compile(r"“\s+|\s+”")


def clean_text(text: str) -> str:
    """Nettoie un fragment de texte extrait d'un document.

    Étapes :

    1. normalisation Unicode NFC (un « é » a toujours le même code, qu'il
       vienne du HTML ou du PDF) ;
    2. suppression des caractères invisibles (espace de largeur nulle,
       trait d'union conditionnel) ;
    3. puces (y compris celles des polices symboles) ramenées à « • »,
       apostrophes typographiques ramenées à « ' », espaces parasites des
       guillemets anglais retirées ;
    4. espaces spéciales converties et espaces multiples réduites à une
       seule, sauf les retours à la ligne qui portent la structure (listes).

    Args:
        text: texte brut extrait.

    Returns:
        Le texte nettoyé, sans espaces en début ni en fin.
    """
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _PRIVATE_USE.sub("•", text)
    text = _BULLETS.sub("• ", text)
    # Une puce au milieu d'une ligne (diapositive dont les lignes ont été
    # mises bout à bout) commence en fait un nouvel élément de liste.
    text = re.sub(r"(?<=\S) • ", "\n• ", text)
    text = _APOSTROPHES.sub("'", text)
    text = _QUOTE_INNER_SPACES.sub(lambda m: m.group().strip(), text)
    text = _SPECIAL_SPACES.sub(" ", text)
    lines = (re.sub(r" {2,}", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


@dataclass(frozen=True)
class Paragraph:
    """Un paragraphe d'une section.

    Attributes:
        text: texte nettoyé ; les lignes d'une liste (« a) », « b) »...)
            restent séparées par des retours à la ligne.
        number: numéro officiel du paragraphe s'il existe (« 1 » pour
            l'article 6, paragraphe 1 du RGPD), sinon ``None``.
    """

    text: str
    number: str | None = None


@dataclass(frozen=True)
class Section:
    """Une unité de structure du document : article, considérant ou page.

    C'est l'unité de citation la plus naturelle (« article 6 du RGPD »,
    « page 4 de la brochure »). Le découpage part de là.

    Attributes:
        label: désignation à citer (« Article 6 », « Considérant 47 »,
            « Page 4 »).
        title: intitulé officiel (« Licéité du traitement »), ou ``""``.
        context: titres englobants, du plus large au plus précis
            (« Chapitre II : Principes », « Section 1 : ... »).
        paragraphs: contenu, dans l'ordre du document.
        page: première page couverte (PDF uniquement), en partant de 1.
        last_page: dernière page couverte quand la section en regroupe
            plusieurs, sinon ``None`` (même page que ``page``).
        mergeable: ``True`` si la section peut être fusionnée avec ses
            voisines quand elle est trop courte. Vrai pour les pages de PDF
            (une diapositive seule contient souvent quelques mots), faux pour
            les articles, qu'on ne mélange jamais pour garder des citations
            exactes.
    """

    label: str
    title: str
    context: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]
    page: int | None = None
    last_page: int | None = None
    mergeable: bool = False

    @property
    def text(self) -> str:
        """Texte complet de la section, paragraphes séparés par une ligne vide."""
        return "\n\n".join(p.text for p in self.paragraphs)
