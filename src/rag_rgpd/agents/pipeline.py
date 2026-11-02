"""Agent 2, le rédacteur, et l'orchestrateur des trois agents.

Le rédacteur est le ``ClaudeGenerator`` de l'étape 4 (mêmes consignes, mêmes
citations de l'API). Ce module le fait travailler avec le vérificateur :

1. le rédacteur écrit une réponse citée ;
2. le vérificateur la relit ;
3. si une affirmation n'est pas appuyée, ses remarques sont renvoyées au
   rédacteur pour une seconde version (``max_rounds`` tours au total) ;
4. si la dernière version est encore refusée par le vérificateur, la
   réponse est remplacée par « Je ne sais pas. » (règle de refus).

L'orchestrateur reste du code Python ordinaire, sans cadriciel : l'ordre des
agents est fixe et chaque décision est visible dans la fiche de décision
(clé ``agents`` des paramètres).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rag_rgpd.agents.llm import JsonAgent
from rag_rgpd.agents.researcher import AgentSearcher, make_researcher
from rag_rgpd.agents.verifier import make_verifier, verify
from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import Answer, AnswerGenerator, refuse
from rag_rgpd.retrieval.index import DEFAULT_MIN_SIMILARITY, Hit

VERIFICATION_FAILED = "verification_echouee"
DEFAULT_ROUNDS = 2

FEEDBACK_TEMPLATE = """

Un vérificateur a relu ta réponse précédente et jugé que ces affirmations ne \
sont dites par aucun des passages :
{claims}
Réécris la réponse en retirant ou en corrigeant ces affirmations, en \
t'appuyant uniquement sur les passages. S'il ne reste rien d'appuyé, réponds \
exactement « Je ne sais pas. »"""


def _sum_usage(usages: list[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            total[key] = total.get(key, 0) + value
    return total


class AgentGenerator:
    """Rédacteur + vérificateur, avec la même interface qu'un ``AnswerGenerator``."""

    def __init__(self, writer: AnswerGenerator, verifier: JsonAgent, *,
                 max_rounds: int = DEFAULT_ROUNDS) -> None:
        """Assemble les deux agents.

        Args:
            writer: rédacteur (``ClaudeGenerator`` ou un faux dans les tests).
            verifier: agent vérificateur (voir ``make_verifier``).
            max_rounds: nombre maximal de rédactions (2 = une reprise).
        """
        self.writer = writer
        self.verifier = verifier
        self.max_rounds = max(1, max_rounds)
        self.model = getattr(writer, "model", "")
        self.effort = getattr(writer, "effort", "")

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        """Rédige, fait vérifier, reprend si besoin, ou refuse."""
        steps: list[dict[str, Any]] = []
        drafts: list[Answer] = []
        feedback = ""
        verdict = None
        for round_number in range(1, self.max_rounds + 1):
            draft = self.writer.generate(question + feedback, hits)
            draft = replace(draft, question=question)
            drafts.append(draft)
            steps.append({"agent": "rédacteur", "tour": round_number, "model": draft.model,
                          "usage": draft.usage, "from_cache": draft.from_cache,
                          "cost_usd": 0.0})  # déjà compté dans le coût de la réponse
            if draft.refused:
                break
            verdict = verify(self.verifier, draft)
            steps.append(verdict.step | {"tour": round_number, "ok": verdict.ok})
            if verdict.ok:
                break
            feedback = FEEDBACK_TEMPLATE.format(
                claims="\n".join(f"- {c}" for c in verdict.unsupported))

        last = drafts[-1]
        common = {
            "model": last.model, "raw_text": last.raw_text, "request": last.request,
            "usage": _sum_usage([d.usage for d in drafts]), "steps": steps,
            "from_cache": all(d.from_cache for d in drafts),
        }
        if last.refused or verdict is None or verdict.ok:  # sinon : bloquée
            return replace(last, **common)
        claims = "; ".join(verdict.unsupported)
        return refuse(question, VERIFICATION_FAILED,
                      f"le vérificateur n'a pas trouvé d'appui pour : {claims}", hits,
                      citations=last.citations, **common)


def add_agent_options(parser: Any) -> None:
    """Options communes à ``rag-ask`` et ``rag-eval``."""
    parser.add_argument("--agents", action="store_true",
                        help="étape 6 : chercheur + rédacteur + vérificateur")
    parser.add_argument("--researcher-model", default="claude-haiku-5-5",
                        help="modèle du chercheur (avec --agents)")
    parser.add_argument("--verifier-model", default="claude-sonnet-5-5",
                        help="modèle du vérificateur (avec --agents)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS,
                        help="rédactions au maximum (avec --agents)")


def with_agents(args: Any, searcher: Any, generator: Any, cache: ResponseCache | None,
                *, min_similarity: float = DEFAULT_MIN_SIMILARITY) -> tuple[Any, Any]:
    """Enveloppe la recherche et le rédacteur si ``--agents`` est demandé.

    Returns:
        (moteur de recherche, rédacteur) à passer à ``explain_question``.
    """
    if not args.agents:
        return searcher, generator
    researcher = None if generator is None else make_researcher(
        model=args.researcher_model, cache=cache)
    searcher = AgentSearcher(searcher, researcher, min_similarity=min_similarity)
    if generator is not None:
        generator = AgentGenerator(generator, make_verifier(model=args.verifier_model,
                                                            cache=cache),
                                   max_rounds=args.rounds)
    return searcher, generator
