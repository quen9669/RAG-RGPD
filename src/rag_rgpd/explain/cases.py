"""Cas de test « de comportement » : une liste de questions dans un fichier YAML.

Chaque cas dit ce qu'on attend de l'assistant, sans fixer le texte exact de
la réponse (qui varie d'un appel à l'autre) :

.. code-block:: yaml

    - id: effacement
      question: Puis-je demander l'effacement de mes données ?
      sources: [Article 17]   # au moins un passage de l'une de ces sources
      rang_max: 8             # ... parmi les 8 premiers (facultatif)
      refus: false            # l'assistant doit répondre (true = doit refuser)
      defaut_connu: ""        # texte non vide = échec attendu (xfail)

Une source s'écrit comme l'étiquette d'un morceau (« Article 17 » couvre
aussi « Article 17, paragraphe 1 »), ou comme l'identifiant d'un document
(« cnpd-brochure-responsable-sous-traitant »).

Ces cas servent aux tests pytest, aux essais en lot (``rag-batch``) et
deviendront la base du jeu d'évaluation de l'étape 5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CASES_PATH = Path("tests/cases/questions.yaml")


@dataclass(frozen=True)
class Case:
    """Une question et le comportement attendu.

    Attributes:
        id: nom court, qui devient le nom du test.
        question: question posée.
        sources: sources dont au moins une doit être retrouvée (vide =
            pas de vérification, par exemple pour un refus attendu).
        refused: ``True`` si l'assistant doit répondre « je ne sais pas ».
        max_rank: rang maximal du premier passage attendu (``None`` = parmi
            tous les passages envoyés).
        known_issue: description d'un défaut connu ; non vide, le test est
            marqué « échec attendu » au lieu de faire échouer la suite.
    """

    id: str
    question: str
    sources: tuple[str, ...] = ()
    refused: bool = False
    max_rank: int | None = None
    known_issue: str = ""


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[Case]:
    """Lit les cas d'un fichier YAML.

    Raises:
        ValueError: si un cas n'a pas d'``id`` ou de ``question``, ou si deux
            cas portent le même ``id``.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = []
    for entry in raw:
        if not entry.get("id") or not entry.get("question"):
            raise ValueError(f"cas incomplet dans {path} : {entry!r}")
        cases.append(Case(
            id=str(entry["id"]),
            question=str(entry["question"]),
            sources=tuple(entry.get("sources") or ()),
            refused=bool(entry.get("refus", False)),
            max_rank=entry.get("rang_max"),
            known_issue=str(entry.get("defaut_connu") or ""),
        ))
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"identifiants en double dans {path} : {sorted(duplicates)}")
    return cases


def matches(source: str, label: str, document_id: str) -> bool:
    """Le passage (``label``, ``document_id``) correspond-il à ``source`` ?

    « Article 17 » correspond à « Article 17 » et « Article 17, paragraphe
    1 », mais pas à « Article 170 ».
    """
    return (
        source == document_id
        or label == source
        or label.startswith(source + ",")
        or label.startswith(source + " ")
    )


def first_match_rank(case: Case, passages: list[tuple[int, str, str]]) -> int | None:
    """Rang du premier passage qui correspond à une source attendue.

    Args:
        case: le cas.
        passages: (rang, étiquette, document) de chaque passage, dans l'ordre.

    Returns:
        Le rang, ou ``None`` si aucun passage ne correspond.
    """
    for rank, label, document_id in passages:
        if any(matches(s, label, document_id) for s in case.sources):
            return rank
    return None
