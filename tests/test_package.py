"""Premiers tests : vérifient que le paquet s'installe et s'importe."""

import re

import rag_rgpd


def test_version_suit_le_format_semver() -> None:
    """La version doit être de la forme MAJEUR.MINEUR.CORRECTIF (ex. 0.1.0).

    Pourquoi : la version servira à tracer quelle version du système a
    produit chaque réponse (fiche du système, journal des réponses).
    """
    assert re.fullmatch(r"\d+\.\d+\.\d+", rag_rgpd.__version__)
