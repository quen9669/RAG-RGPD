"""Moteur du chat : une question en entrée, une réponse présentable en sortie.

Aucune dépendance à Streamlit ici, pour pouvoir le tester sans navigateur.
Il réutilise exactement le traitement de ``rag-ask`` (``explain_question``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from rag_rgpd.agents.pipeline import DEFAULT_ROUNDS, with_agents, with_router
from rag_rgpd.answering.generator import Answer, AnswerGenerator
from rag_rgpd.explain.trace import DEFAULT_EXTRA, Trace, explain_question, write_trace
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY


@dataclass(frozen=True)
class Source:
    """Une source citée dans la réponse."""

    number: int
    heading: str
    pages: str
    url: str
    quotes: list[str] = field(default_factory=list)
    text: str = ""


@dataclass(frozen=True)
class Reply:
    """Réponse prête à afficher."""

    text: str
    refused: bool
    reason: str
    sources: list[Source]
    cost_usd: float | None
    trace_id: str
    trace: Trace | None = None


def _pages(start: int | None, end: int | None) -> str:
    if start is None:
        return ""
    return f"p. {start}-{end}" if end and end != start else f"p. {start}"


def to_reply(answer: Answer, cost_usd: float | None, trace_id: str,
             trace: Trace | None = None) -> Reply:
    """Transforme une ``Answer`` en ``Reply`` (sources citées seulement)."""
    sources = []
    if not answer.refused:
        for number in answer.cited_numbers:
            chunk = answer.hits[number - 1].chunk
            quotes = [c.cited_text for c in answer.citations
                      if c.source_number == number and c.cited_text]
            sources.append(Source(number, chunk.heading,
                                  _pages(chunk.page_start, chunk.page_end),
                                  chunk.url, quotes, chunk.text))
    return Reply(answer.text, answer.refused, answer.reason, sources, cost_usd, trace_id, trace)


class ChatEngine:
    """Pose des questions à l'assistant, avec ou sans agents.

    Args:
        searcher: moteur de recherche (index ouvert).
        make_generator: fabrique ``modèle -> rédacteur``.
        trace_path: où écrire les fiches de décision (``None`` = nulle part).
        route: un petit modèle choisit les domaines avant la recherche
            (décision 0016).
    """

    def __init__(self, searcher: Any, make_generator: Callable[[str], AnswerGenerator],
                 trace_path: Any = None, route: bool = False) -> None:
        self.route = route
        self.searcher = searcher
        self.make_generator = make_generator
        self.trace_path = trace_path

    def ask(self, question: str, *, model: str, agents: bool = False,
            k: int = DEFAULT_K, min_similarity: float = DEFAULT_MIN_SIMILARITY,
            researcher_model: str = "claude-haiku-5-5",
            verifier_model: str = "claude-sonnet-5-5") -> Reply:
        """Répond à une question."""
        args = SimpleNamespace(agents=agents, researcher_model=researcher_model,
                               verifier_model=verifier_model, rounds=DEFAULT_ROUNDS)
        generator = self.make_generator(model)
        routed = with_router(SimpleNamespace(route=self.route, domain=None), self.searcher,
                             generator, None)
        searcher, generator = with_agents(args, routed, generator,
                                          None, min_similarity=min_similarity)
        trace, answer = explain_question(question, searcher, generator, k=k,
                                         min_similarity=min_similarity, extra=DEFAULT_EXTRA)
        if self.trace_path is not None:
            write_trace(trace, self.trace_path)
        assert answer is not None  # un générateur est toujours fourni
        return to_reply(answer, trace.cost_usd, trace.trace_id, trace)
