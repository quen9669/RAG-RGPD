"""Fiche de décision (trace) : tout ce qui explique une réponse.

Pour chaque question, la trace garde la réponse à six questions :

1. Quels passages ont été trouvés ? (rang, similarité, score, source)
2. Pourquoi ceux-là ? (paramètres, rang dans la recherche par le sens et par
   mots-clés, passages écartés juste après la coupure)
3. Qu'a reçu le modèle ? (la requête exacte : consigne, passages, question)
4. Pourquoi cette réponse ? (texte brut, chaque citation reliée à son
   passage, et vérifiée)
5. Pourquoi un refus ? (code et raison)
6. Combien ça a coûté ? (temps, tokens, coût estimé)

Les traces sont ajoutées, une par ligne, à ``data/traces/traces.jsonl``
(format JSON Lines, UTF-8). Elles contiennent des extraits des documents
(dont la CNPD) : elles restent locales, hors de Git, comme ``data/raw/``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from rag_rgpd.answering.generator import (
    Answer,
    AnswerGenerator,
    answer_question,
    best_similarity,
)
from rag_rgpd.explain.cost import estimate_cost
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY, Hit, IndexMeta

DEFAULT_TRACE_PATH = Path("data/traces/traces.jsonl")

DEFAULT_EXTRA = 4
"""Passages regardés au-delà des ``k`` envoyés au modèle. Ils ne lui sont
pas montrés, mais la fiche les affiche : voir ce qui a été écarté de peu
aide à régler ``k``."""

RANK_DEPTH = 50
"""Profondeur des deux recherches séparées (sens, mots-clés) utilisées pour
expliquer le classement."""

# Décisions finales possibles.
ANSWERED = "réponse"
REFUSED = "refus"
SEARCH_ONLY = "recherche seule"


class SearchEngine(Protocol):
    """Ce dont la trace a besoin côté recherche (``Searcher`` en pratique)."""

    meta: IndexMeta

    def search(self, question: str, *, k: int = DEFAULT_K) -> list[Hit]:
        """Passages les plus pertinents, du meilleur au moins bon."""
        ...

    def ranks(self, question: str, *, depth: int = 50) -> dict[str, tuple[int | None, int | None]]:
        """Rangs par le sens et par mots-clés, avant fusion."""
        ...


@dataclass
class Passage:
    """Un passage retrouvé, vu par la fiche de décision.

    Attributes:
        rank: rang après fusion (1 = le plus pertinent).
        chunk_id: identifiant du morceau.
        document_id: document d'origine.
        label: endroit précis (« Article 17, paragraphe 1 », « Page 3 »).
        heading: chemin complet affiché dans les citations.
        pages: pages du PDF (« 3-4 »), vide pour une page web.
        url: adresse d'origine.
        similarity: similarité cosinus avec la question (-1 à 1).
        score: score de fusion RRF, qui fixe l'ordre.
        meaning_rank: rang dans la recherche par le sens seule.
        words_rank: rang dans la recherche par mots-clés seule.
        sent: ``True`` si le passage a été envoyé au modèle, ``False`` s'il
            a été écarté juste après la coupure ``k``.
        cited_as: numéros de source sous lesquels la réponse le cite.
        text: texte du morceau.
    """

    rank: int
    chunk_id: str
    document_id: str
    label: str
    heading: str
    pages: str
    url: str
    similarity: float
    score: float
    meaning_rank: int | None
    words_rank: int | None
    sent: bool
    cited_as: list[int] = field(default_factory=list)
    text: str = ""


@dataclass
class Trace:
    """Fiche de décision d'une question.

    Attributes:
        trace_id: identifiant court (date + empreinte), pour la retrouver.
        at: date et heure (UTC, ISO 8601).
        question: question posée.
        decision: ``ANSWERED``, ``REFUSED`` ou ``SEARCH_ONLY``.
        answer: réponse affichée (avec marqueurs ``[n]``) ou « Je ne sais pas. ».
        refusal_code: code du refus, vide sinon.
        reason: raison du refus en français, vide sinon.
        params: paramètres utilisés (k, seuil, modèles, effort).
        index: fiche de l'index (modèle d'embedding, empreinte des morceaux).
        best_similarity: similarité du meilleur passage envoyé.
        passages: passages envoyés puis écartés.
        request: requête exacte envoyée au LLM (``{}`` s'il n'a pas été appelé).
        raw_text: texte brut renvoyé par le LLM.
        llm_model: modèle qui a réellement répondu.
        citations: chaque citation, reliée à son passage et vérifiée.
        usage: tokens comptés par l'API.
        cost_usd: coût estimé en dollars (``None`` si prix inconnu).
        from_cache: réponse relue dans le cache local (rien payé).
        timings: durées en secondes (``search``, ``llm``, ``total``).
    """

    trace_id: str
    at: str
    question: str
    decision: str
    answer: str
    refusal_code: str
    reason: str
    params: dict[str, Any]
    index: dict[str, Any]
    best_similarity: float
    passages: list[Passage]
    request: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""
    llm_model: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    cost_usd: float | None = 0.0
    from_cache: bool = False
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def sent_passages(self) -> list[Passage]:
        """Passages réellement montrés au modèle."""
        return [p for p in self.passages if p.sent]

    @property
    def held_back(self) -> list[Passage]:
        """Passages écartés juste après la coupure ``k``."""
        return [p for p in self.passages if not p.sent]

    def to_dict(self) -> dict[str, Any]:
        """Version JSON de la trace."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        """Relit une trace écrite par ``to_dict`` (champs inconnus ignorés)."""
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        values["passages"] = [Passage(**p) for p in data.get("passages", [])]
        return cls(**values)


def _pages(hit: Hit) -> str:
    start, end = hit.chunk.page_start, hit.chunk.page_end
    if start is None:
        return ""
    return f"{start}-{end}" if end and end != start else str(start)


def _trace_id(at: datetime, question: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps([question, params, at.isoformat()], default=str).encode("utf-8")
    ).hexdigest()
    return f"{at:%Y%m%d-%H%M%S}-{digest[:6]}"


def explain_question(
    question: str,
    searcher: SearchEngine,
    generator: AnswerGenerator | None,
    *,
    k: int = DEFAULT_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    extra: int = DEFAULT_EXTRA,
    params: dict[str, Any] | None = None,
) -> tuple[Trace, Answer | None]:
    """Pose une question et remplit la fiche de décision à chaque étape.

    C'est le même traitement que ``rag-ask`` (recherche, barrière de
    similarité, LLM, contrôle des citations), avec les mesures en plus.

    Args:
        question: question posée.
        searcher: moteur de recherche (index ouvert).
        generator: LLM ; ``None`` pour s'arrêter après la recherche
            (gratuit, sans clé d'API).
        k: passages envoyés au modèle.
        min_similarity: seuil de la barrière 1.
        extra: passages regardés en plus, montrés comme « écartés ».
        params: autres paramètres à tracer (modèle d'embedding...).

    Returns:
        La trace, et la réponse (``None`` en recherche seule).
    """
    started = time.perf_counter()
    candidates = searcher.search(question, k=k + extra)
    hits = candidates[:k]
    searched = time.perf_counter()
    ranks = searcher.ranks(question, depth=max(RANK_DEPTH, k + extra))
    ranked = time.perf_counter()

    answer: Answer | None = None
    if generator is not None:
        answer = answer_question(
            question, hits, generator, min_similarity=min_similarity,
            params={**(params or {}), "k": k, "extra": extra},
        )
    finished = time.perf_counter()

    used = answer.params if answer else {
        **(params or {}), "k": k, "extra": extra, "min_similarity": min_similarity,
    }
    passages = []
    for rank, hit in enumerate(candidates, start=1):
        meaning, words = ranks.get(hit.chunk.chunk_id, (None, None))
        passages.append(Passage(
            rank=rank, chunk_id=hit.chunk.chunk_id, document_id=hit.chunk.document_id,
            label=hit.chunk.label, heading=hit.chunk.heading, pages=_pages(hit),
            url=hit.chunk.url, similarity=round(hit.similarity, 4),
            score=round(hit.score, 6), meaning_rank=meaning, words_rank=words,
            sent=rank <= k, text=hit.chunk.text,
        ))

    citations = []
    if answer is not None:
        for c in answer.citations:
            citations.append({
                "source_number": c.source_number, "chunk_id": c.chunk_id,
                "label": passages[c.source_number - 1].label if c.chunk_id else "",
                "cited_text": c.cited_text, "verified": c.verified,
            })
            if c.chunk_id and c.source_number not in passages[c.source_number - 1].cited_as:
                passages[c.source_number - 1].cited_as.append(c.source_number)

    now = datetime.now(UTC)
    if answer is None:
        decision = SEARCH_ONLY
    else:
        decision = REFUSED if answer.refused else ANSWERED
    usage = answer.usage if answer else {}
    llm_model = answer.model if answer else ""
    trace = Trace(
        trace_id=_trace_id(now, question, used),
        at=now.isoformat(timespec="seconds"),
        question=question,
        decision=decision,
        answer=answer.text if answer else "",
        refusal_code=answer.refusal_code if answer else "",
        reason=answer.reason if answer else "",
        params=used,
        index=asdict(searcher.meta),
        best_similarity=round(best_similarity(hits), 4),
        passages=passages,
        request=answer.request if answer else {},
        raw_text=answer.raw_text if answer else "",
        llm_model=llm_model,
        citations=citations,
        usage=usage,
        cost_usd=0.0 if answer is None or answer.from_cache else estimate_cost(llm_model, usage),
        from_cache=answer.from_cache if answer else False,
        timings={
            "search": round(searched - started, 3),
            "ranks": round(ranked - searched, 3),
            "llm": round(finished - ranked, 3),
            "total": round(finished - started, 3),
        },
    )
    return trace, answer


def write_trace(trace: Trace, path: Path = DEFAULT_TRACE_PATH) -> None:
    """Ajoute la trace à la fin du fichier (dossiers créés au besoin)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")


def read_traces(path: Path = DEFAULT_TRACE_PATH) -> list[Trace]:
    """Relit toutes les traces, de la plus ancienne à la plus récente."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]


def find_trace(traces: list[Trace], wanted: str) -> Trace:
    """Retrouve une trace par son identifiant (ou son début), ou « last ».

    Raises:
        LookupError: si aucune trace ne correspond.
    """
    if not traces:
        raise LookupError("aucune trace : poser d'abord une question avec `rag-ask`")
    if wanted == "last":
        return traces[-1]
    matches = [t for t in traces if t.trace_id.startswith(wanted)]
    if not matches:
        raise LookupError(f"aucune trace ne commence par {wanted!r}")
    return matches[-1]
