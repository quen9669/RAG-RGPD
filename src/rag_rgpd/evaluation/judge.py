"""Correcteur des réponses : un second LLM note chaque réponse (« LLM juge »).

Certaines erreurs se mesurent avec du code (bonne source retrouvée, citation
vérifiée, refus attendu). D'autres demandent de lire : une affirmation
est-elle vraiment appuyée par le passage cité ? La réponse contient-elle
les points clés attendus ? A-t-elle évité le piège connu ? On confie cette
lecture à un modèle plus fort que celui qui rédige (Claude Sonnet 5.5 par
défaut, contre Haiku 5.5 pour les réponses), avec une grille précise et une
réponse au format JSON imposé : chaque note est expliquée et relisible.

Limite à garder en tête : un correcteur LLM se trompe aussi. Ses notes
servent à comparer deux versions entre elles, sur les mêmes questions, et
à repérer les réponses à relire soi-même ; pas à certifier une réponse.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic.types.beta import BetaMessage

from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import FALLBACK_MODELS, usage_dict
from rag_rgpd.explain.cases import Case
from rag_rgpd.explain.trace import Trace

DEFAULT_JUDGE_MODEL = "claude-sonnet-5-5"

JUDGE_PROMPT = """\
Tu corriges les réponses d'un assistant sur la protection des données \
(RGPD, documents de la CNPD). L'assistant n'a le droit d'utiliser que les \
passages numérotés qu'il a reçus ; les marqueurs [n] de sa réponse renvoient \
à ces passages.

Corrige uniquement à partir des passages fournis, pas de tes connaissances.

1. Découpe la réponse en affirmations (une idée vérifiable chacune). Ne \
compte pas : les phrases d'introduction ou de conclusion sans contenu, les \
remarques sur ce que les passages ne couvrent pas (« les passages ne \
précisent pas... »), le développement d'un sigle, et l'avertissement « ne \
remplace pas un avis juridique ». Pour chacune :
   - « appuyee » : vrai si un des passages fournis dit bien cela (pas \
seulement un sujet voisin) ;
   - « citee » : vrai si l'affirmation porte un marqueur [n] (dans sa phrase \
ou juste après) qui renvoie à un passage qui l'appuie.
2. Pour chaque point clé attendu, « couvert » : vrai si la réponse l'exprime, \
même avec d'autres mots.
3. « piege_evite » : vrai si la réponse ne commet pas l'erreur décrite dans \
« Piège » (vrai s'il n'y a pas de piège).
4. « commentaire » : une ou deux phrases sur le principal défaut, ou « RAS ».
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
                    "citee": {"type": "boolean"},
                },
                "required": ["texte", "appuyee", "citee"],
                "additionalProperties": False,
            },
        },
        "points_cles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "point": {"type": "string"},
                    "couvert": {"type": "boolean"},
                },
                "required": ["point", "couvert"],
                "additionalProperties": False,
            },
        },
        "piege_evite": {"type": "boolean"},
        "commentaire": {"type": "string"},
    },
    "required": ["affirmations", "points_cles", "piege_evite", "commentaire"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Judgement:
    """Note d'une réponse par le correcteur.

    Attributes:
        claims: affirmations relevées, avec ``appuyee`` et ``citee``.
        key_points: points clés attendus, avec ``couvert``.
        trap_avoided: le piège décrit dans le cas a été évité.
        comment: principal défaut, en une ou deux phrases.
        model: modèle correcteur.
        usage: tokens consommés par la correction.
        from_cache: correction relue dans le cache local.
    """

    claims: list[dict[str, Any]] = field(default_factory=list)
    key_points: list[dict[str, Any]] = field(default_factory=list)
    trap_avoided: bool = True
    comment: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    from_cache: bool = False

    @property
    def unsupported(self) -> list[str]:
        """Affirmations sans appui dans les passages."""
        return [c["texte"] for c in self.claims if not c["appuyee"]]

    @property
    def faithful(self) -> bool:
        """Aucune affirmation sans appui dans les passages."""
        return not self.unsupported

    @property
    def support_rate(self) -> float | None:
        """Part des affirmations appuyées par les passages (``None`` si aucune).

        Une part plutôt qu'un tout-ou-rien : une affirmation douteuse sur
        dix ne rend pas la réponse entière « infidèle ».
        """
        if not self.claims:
            return None
        return sum(1 for c in self.claims if c["appuyee"]) / len(self.claims)

    @property
    def citation_rate(self) -> float | None:
        """Part des affirmations qui portent une citation (``None`` si aucune)."""
        if not self.claims:
            return None
        return sum(1 for c in self.claims if c["citee"]) / len(self.claims)

    @property
    def completeness(self) -> float | None:
        """Part des points clés couverts (``None`` s'il n'y en a pas)."""
        if not self.key_points:
            return None
        return sum(1 for p in self.key_points if p["couvert"]) / len(self.key_points)


def judge_message(case: Case, trace: Trace) -> str:
    """Texte soumis au correcteur : passages envoyés, question, réponse, attentes."""
    parts = ["Passages reçus par l'assistant :"]
    for number, passage in enumerate(trace.sent_passages, start=1):
        parts.append(f"[{number}] {passage.heading}\n{passage.text}")
    parts.append(f"Question : {case.question}")
    parts.append(f"Réponse de l'assistant :\n{trace.answer}")
    points = "\n".join(f"- {p}" for p in case.key_points) or "(aucun)"
    parts.append(f"Points clés attendus :\n{points}")
    parts.append(f"Piège : {case.trap or '(aucun)'}")
    return "\n\n".join(parts)


class ClaudeJudge:
    """Correcteur fondé sur Claude, avec réponse JSON imposée par un schéma."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = DEFAULT_JUDGE_MODEL,
        effort: str = "medium",
        cache: ResponseCache | None = None,
    ) -> None:
        """Prépare le client.

        Args:
            client: client ``anthropic.Anthropic`` (un faux dans les tests).
            model: modèle correcteur.
            effort: profondeur de réflexion ; ``medium`` suffit pour comparer
                des phrases à des passages courts.
            cache: cache local ; une même réponse n'est corrigée (et payée)
                qu'une fois.
        """
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.effort = effort
        self.cache = cache

    def judge(self, case: Case, trace: Trace) -> Judgement:
        """Corrige la réponse d'une trace.

        Raises:
            ValueError: si le correcteur décline la demande ou ne renvoie pas
                de JSON valide.
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": JUDGE_PROMPT,
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            "messages": [{"role": "user", "content": judge_message(case, trace)}],
        }
        if self.model in FALLBACK_MODELS:
            request |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        cached = None
        key = ""
        if self.cache is not None:
            key = self.cache.key(request)
            cached = self.cache.get(key)
        if cached is not None:
            response: Any = BetaMessage.model_validate(cached)
        else:
            response = self._client.beta.messages.create(**request)
            if self.cache is not None:
                self.cache.put(key, response.model_dump(mode="json"))

        if response.stop_reason == "refusal":
            raise ValueError(f"le correcteur a décliné la correction du cas {case.id}")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"correction illisible pour le cas {case.id} : {error}") from error
        return Judgement(
            claims=data["affirmations"],
            key_points=data["points_cles"],
            trap_avoided=bool(data["piege_evite"]) if case.trap else True,
            comment=data["commentaire"],
            model=getattr(response, "model", self.model),
            usage=usage_dict(response),
            from_cache=cached is not None,
        )


_QUOTED = re.compile(r"«[^»]*»|\"[^\"]*\"|“[^”]*”")
_WORD = re.compile(r"\w+")


def verbatim_copies(answer: str, passages: list[str], *, min_words: int = 20) -> list[str]:
    """Phrases recopiées mot pour mot d'un passage sans guillemets.

    Contrôle fait par le code, sans LLM : on cherche dans la réponse (hors
    passages entre guillemets) toute suite d'au moins ``min_words`` mots
    identique à une suite d'un passage. Vingt mots, c'est plus que les
    formules qui reviennent dans tout le RGPD (« le responsable du traitement
    et le sous-traitant mettent en œuvre les mesures techniques... ») : une
    phrase d'article reprise presque entière.

    Args:
        answer: réponse de l'assistant.
        passages: textes des passages envoyés.
        min_words: longueur minimale d'une recopie.

    Returns:
        Les extraits recopiés (un par recopie, fusionnés s'ils se suivent).
    """
    def words(text: str) -> list[str]:
        return [w.lower() for w in _WORD.findall(text)]

    grams: set[tuple[str, ...]] = set()
    for passage in passages:
        tokens = words(passage)
        grams.update(tuple(tokens[i:i + min_words]) for i in range(len(tokens) - min_words + 1))

    copies: list[str] = []
    for segment in _QUOTED.split(re.sub(r"\[\d+\]", " ", answer)):
        tokens = words(segment)
        start = None
        end = -1
        for i in range(len(tokens) - min_words + 1):
            if tuple(tokens[i:i + min_words]) in grams:
                if start is None or i > end:
                    if start is not None:
                        copies.append(" ".join(tokens[start:end]))
                    start = i
                end = i + min_words
        if start is not None:
            copies.append(" ".join(tokens[start:end]))
    return copies
