"""Agent 3, le vérificateur : bloque toute affirmation sans appui.

La barrière 3 de l'étape 4 contrôle la forme (une citation existe et se
retrouve dans le passage). Le vérificateur contrôle le fond : chaque
affirmation de la réponse est-elle **dite** par un des passages envoyés ?
Il lit avec Haiku 5.5 par défaut (décision 0014), Sonnet 5.5 en option.

La décision est prise par du code, pas par le modèle : la réponse est
acceptée seulement si aucune affirmation n'est jugée sans appui. Le modèle
ne fait que lister les affirmations et dire si chacune est appuyée.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_rgpd.agents.llm import JsonAgent
from rag_rgpd.answering.generator import Answer

DEFAULT_VERIFIER_MODEL = "claude-haiku-5-5"

VERIFIER_PROMPT = """\
Tu vérifies la réponse d'un assistant sur la protection des données (RGPD, \
documents de la CNPD). L'assistant n'a le droit d'utiliser que les passages \
numérotés qu'il a reçus ; les marqueurs [n] de sa réponse renvoient à ces \
passages.

Vérifie uniquement à partir des passages fournis, pas de tes connaissances : \
une affirmation vraie en droit mais absente des passages n'est PAS appuyée.

Découpe la réponse en affirmations (une idée vérifiable chacune). Ne compte \
PAS comme affirmation, donc ne les liste pas : \
- les phrases d'introduction ou de conclusion sans contenu ; \
- les remarques sur ce que les passages ne couvrent pas ; \
- l'avertissement « ne remplace pas un avis juridique » ; \
- les explications de vocabulaire ou de sigle (ce qu'est un considérant, \
un responsable du traitement, la CNPD, un niveau d'amende) et les remarques \
sur la nature d'un texte (« le considérant éclaire l'article ») : ce sont \
des repères pour le lecteur, pas des règles ; \
- les simples reformulations en langage courant d'une idée qui figure dans \
un passage, même si les mots changent.

Pour chaque affirmation retenue, « appuyee » est vrai si un des passages \
fournis dit bien cela, ou le dit autrement, ou en découle directement sans \
rien ajouter. Elle est fausse seulement si elle ajoute un fait, un chiffre, \
un délai ou une condition absents des passages, ou si elle transpose à un \
droit ce qui est dit d'un autre. En cas de doute sur une simple \
reformulation, considère-la comme appuyée.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "affirmations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "texte": {"type": "string"},
                    "appuyee": {"type": "boolean"},
                },
                "required": ["texte", "appuyee"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["affirmations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verdict:
    """Décision du vérificateur sur une réponse.

    Attributes:
        ok: ``True`` si toutes les affirmations sont appuyées.
        unsupported: affirmations jugées sans appui.
        claims: nombre d'affirmations examinées.
        step: ligne de suivi (modèle, tokens, coût).
    """

    ok: bool
    unsupported: list[str] = field(default_factory=list)
    claims: int = 0
    step: dict[str, Any] = field(default_factory=dict)


def make_verifier(client: Any = None, *, model: str = DEFAULT_VERIFIER_MODEL,
                  cache: Any = None) -> JsonAgent:
    """Construit l'agent qui relit les réponses."""
    return JsonAgent("vérificateur", VERIFIER_PROMPT, SCHEMA, model=model,
                     client=client, cache=cache)


def verifier_message(answer: Answer) -> str:
    """Texte soumis au vérificateur : passages, question, réponse."""
    parts = ["Passages reçus par l'assistant :"]
    for number, hit in enumerate(answer.hits, start=1):
        parts.append(f"[{number}] {hit.chunk.heading}\n{hit.chunk.text}")
    parts.append(f"Question : {answer.question}")
    parts.append(f"Réponse de l'assistant :\n{answer.text}")
    return "\n\n".join(parts)


def verify(agent: JsonAgent, answer: Answer) -> Verdict:
    """Fait relire une réponse non refusée.

    Si le vérificateur échoue (refus du modèle, JSON illisible), la réponse
    est **bloquée** : mieux vaut un refus qu'une réponse non vérifiée.
    """
    try:
        call = agent.call(verifier_message(answer))
    except ValueError as error:
        return Verdict(False, [f"vérification impossible : {error}"],
                       step={"agent": agent.name, "error": str(error), "cost_usd": 0.0})
    claims = call.data.get("affirmations", [])
    bad = [c["texte"] for c in claims if not c["appuyee"]]
    return Verdict(not bad, bad, len(claims), call.step | {"affirmations": len(claims),
                                                             "sans_appui": bad})
