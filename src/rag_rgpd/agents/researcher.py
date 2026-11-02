"""Agent 1, le chercheur : trouve les passages pertinents.

La recherche de l'étape 4 ne lit la question qu'une fois, telle qu'elle est
écrite. Or une question en langage courant (« puis-je filmer mes salariés ? »)
n'emploie pas les mots du texte juridique (« surveillance », « intérêt
légitime », « article 6 »). Le chercheur demande donc à un petit modèle
d'écrire jusqu'à deux formulations voisines, lance la recherche pour chacune,
puis fusionne les résultats.

Garde-fous :

- on ne reformule que si la question d'origine a déjà franchi la barrière de
  similarité : une question hors sujet reste refusée, une reformulation ne
  peut pas la « rapprocher » du corpus ;
- le classement final est refait **avec la question d'origine** (reclassement
  si disponible), pas avec les reformulations ;
- les articles placés en tête parce que la question les cite le restent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from rag_rgpd.agents.llm import JsonAgent
from rag_rgpd.answering.generator import best_similarity
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY, Hit, embedding_text

DEFAULT_RESEARCHER_MODEL = "claude-haiku-5-5"
MAX_REWRITES = 2
RRF_CONSTANT = 60

RESEARCHER_PROMPT = """\
Tu aides un moteur de recherche à retrouver des passages du RGPD (règlement \
européen sur la protection des données) et de documents de la CNPD \
(Commission nationale pour la protection des données, Luxembourg).

On te donne la question d'un utilisateur. Écris au plus deux autres \
formulations de recherche, courtes, qui reprennent le vocabulaire des textes \
juridiques (par exemple « intérêt légitime », « base de licéité », \
« personne concernée »). Chaque formulation doit viser un aspect différent de \
la question.

Règles :
- Ne réponds pas à la question.
- Cite un numéro d'article (« article 6 ») seulement si tu en es sûr.
- Reste sur le sujet de la question : n'élargis pas à d'autres thèmes.
- Si la question est déjà précise, renvoie une liste vide.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"requetes": {"type": "array", "items": {"type": "string"}}},
    "required": ["requetes"],
    "additionalProperties": False,
}


def make_researcher(client: Any = None, *, model: str = DEFAULT_RESEARCHER_MODEL,
                    cache: Any = None) -> JsonAgent:
    """Construit le petit agent qui reformule les questions."""
    return JsonAgent("chercheur", RESEARCHER_PROMPT, SCHEMA, model=model,
                     client=client, cache=cache)


def merge_hits(question: str, lists: list[list[Hit]], reranker: Any = None) -> list[Hit]:
    """Fusionne plusieurs listes de passages, sans doublon.

    Args:
        question: question d'origine (c'est elle qui fixe l'ordre final).
        lists: résultats de chaque recherche ; la première est celle de la
            question d'origine et passe avant les autres à égalité.
        reranker: modèle de reclassement ; sans lui, fusion par rang (RRF).

    Returns:
        Les passages épinglés de la première liste, puis les autres dans
        l'ordre fixé par le reclassement (ou par la fusion des rangs).
    """
    first: dict[str, Hit] = {}
    for hits in lists:
        for hit in hits:
            first.setdefault(hit.chunk.chunk_id, hit)
    pinned = [h for h in lists[0] if h.pinned]
    pinned_ids = {h.chunk.chunk_id for h in pinned}
    rest = [h for cid, h in first.items() if cid not in pinned_ids]

    if reranker is not None and rest:
        notes = reranker.score(question, [embedding_text(h.chunk) for h in rest])
        rest = [replace(h, rerank_score=n) for h, n in zip(rest, notes, strict=True)]
        rest.sort(key=lambda h: -(h.rerank_score or 0.0))
    else:
        fused: dict[str, float] = {}
        for hits in lists:
            for rank, hit in enumerate(hits, start=1):
                fused[hit.chunk.chunk_id] = fused.get(hit.chunk.chunk_id, 0.0) + 1 / (
                    RRF_CONSTANT + rank)
        rest.sort(key=lambda h: -fused[h.chunk.chunk_id])
    return pinned + rest


class AgentSearcher:
    """Moteur de recherche qui passe par le chercheur (même interface que ``Searcher``)."""

    def __init__(self, inner: Any, researcher: JsonAgent | None, *,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY) -> None:
        """Enveloppe un ``Searcher``.

        Args:
            inner: moteur de recherche de l'étape 4/5.
            researcher: agent de reformulation ; ``None`` = recherche simple.
            min_similarity: seuil de la barrière 1, sous lequel on ne
                reformule pas.
        """
        self._inner = inner
        self._researcher = researcher
        self._min_similarity = min_similarity
        self._steps: list[dict[str, Any]] = []
        self.meta = inner.meta

    @property
    def settings(self) -> dict[str, object]:
        """Réglages de la recherche, plus ceux du chercheur."""
        return {**getattr(self._inner, "settings", {}),
                "researcher_model": self._researcher.model if self._researcher else ""}

    @property
    def reranker(self) -> Any:
        """Modèle de reclassement du moteur sous-jacent."""
        return getattr(self._inner, "reranker", None)

    def take_steps(self) -> list[dict[str, Any]]:
        """Renvoie (et vide) le suivi des appels du chercheur depuis le dernier appel."""
        steps, self._steps = self._steps, []
        return steps

    def ranks(self, question: str, *, depth: int = 50) -> dict[str, tuple[int | None, int | None]]:
        """Rangs de la question d'origine (explication de la fiche de décision)."""
        return self._inner.ranks(question, depth=depth)

    def search(self, question: str, *, k: int = DEFAULT_K) -> list[Hit]:
        """Recherche la question d'origine, puis ses reformulations, et fusionne."""
        base = self._inner.search(question, k=k)
        if self._researcher is None or best_similarity(base) < self._min_similarity:
            return base
        try:
            call = self._researcher.call(f"Question : {question}")
        except ValueError as error:
            self._steps.append({"agent": "chercheur", "error": str(error), "cost_usd": 0.0})
            return base
        self._steps.append({**call.step, "requetes": []})
        rewrites = [q.strip() for q in call.data.get("requetes", []) if q.strip()]
        rewrites = [q for q in dict.fromkeys(rewrites) if q != question][:MAX_REWRITES]
        self._steps[-1]["requetes"] = rewrites
        if not rewrites:
            return base
        lists = [base] + [self._inner.search(q, k=k) for q in rewrites]
        return merge_hits(question, lists, self.reranker)[:k]
