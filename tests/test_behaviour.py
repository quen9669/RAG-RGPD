"""Tests de comportement : chaque cas de tests/cases/questions.yaml devient un test.

Deux niveaux, sur les vraies données :

- ``test_recherche`` (marqueur ``index``) : la vraie recherche, sans LLM.
  Gratuit, mais le premier test attend environ une minute que BGE-M3 se
  charge : lancé seulement avec ``uv run pytest --index``. Sauté si
  ``data/index/`` n'existe pas (lancer ``rag-index``).
- ``test_reponse`` (marqueur ``llm``) : la chaîne complète avec Claude.
  Payant (environ 0,001 $ par cas avec Haiku) : lancé seulement avec
  ``uv run pytest --llm``.

Lancer un seul niveau : ``uv run pytest -m index --index`` ou
``uv run pytest -m llm --llm``. Un seul cas : ajouter ``-k effacement``.
"""

import os
from pathlib import Path

import pytest

from rag_rgpd.explain.cases import Case, first_match_rank, load_cases
from rag_rgpd.explain.render import format_trace
from rag_rgpd.explain.trace import ANSWERED, REFUSED, Trace, explain_question
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY

ROOT = Path(__file__).resolve().parent.parent
CASES = load_cases(ROOT / "tests" / "cases" / "questions.yaml")
INDEX_DIR = Path(os.environ.get("RAG_INDEX_DIR", ROOT / "data" / "index"))


def _params(case: Case, *marks: pytest.MarkDecorator) -> pytest.param:
    """Un cas -> un paramètre pytest, marqué « échec attendu » si défaut connu."""
    all_marks = list(marks)
    if case.known_issue:
        all_marks.append(pytest.mark.xfail(reason=case.known_issue, strict=False))
    return pytest.param(case, id=case.id, marks=all_marks)


@pytest.fixture(scope="session")
def searcher():
    """Le vrai moteur de recherche, chargé une seule fois pour toute la session."""
    if not (INDEX_DIR / "index_meta.json").exists():
        pytest.skip(f"pas d'index dans {INDEX_DIR} : lancer `uv run rag-index`")
    from rag_rgpd.retrieval.embeddings import SentenceTransformerEmbedder
    from rag_rgpd.retrieval.index import Searcher, read_meta
    from rag_rgpd.retrieval.rerank import CrossEncoderReranker

    # Même chaîne que rag-ask : recherche hybride, reclassement, articles cités.
    return Searcher(INDEX_DIR, SentenceTransformerEmbedder(read_meta(INDEX_DIR).embedding_model),
                    reranker=CrossEncoderReranker())


def _check_sources(case: Case, trace: Trace) -> None:
    if not case.sources:
        return
    sent = [(p.rank, p.label, p.document_id) for p in trace.sent_passages]
    rank = first_match_rank(case, sent)
    limit = case.max_rank or DEFAULT_K
    # En cas d'échec, pytest affiche la fiche complète : on voit tout de suite
    # quels passages ont été retrouvés à la place.
    assert rank is not None and rank <= limit, (
        f"attendu {list(case.sources)} parmi les {limit} premiers, trouvé au rang {rank}\n"
        + format_trace(trace)
    )


@pytest.mark.index
@pytest.mark.parametrize("case", [_params(c) for c in CASES])
def test_recherche(case: Case, searcher) -> None:
    trace, _ = explain_question(case.question, searcher, None)

    _check_sources(case, trace)
    refused = trace.best_similarity < DEFAULT_MIN_SIMILARITY
    assert refused == case.refused, (
        f"refus attendu : {case.refused}, meilleure similarité {trace.best_similarity:.2f}\n"
        + format_trace(trace)
    )


@pytest.mark.index
@pytest.mark.llm
@pytest.mark.parametrize("case", [_params(c) for c in CASES])
def test_reponse(case: Case, searcher) -> None:
    from dotenv import load_dotenv

    from rag_rgpd.answering.generator import ClaudeGenerator

    load_dotenv(ROOT / ".env", override=False, encoding="utf-8")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY absente du fichier .env")
    trace, _ = explain_question(case.question, searcher, ClaudeGenerator())

    expected = REFUSED if case.refused else ANSWERED
    assert trace.decision == expected, format_trace(trace)
    assert all(c["verified"] for c in trace.citations), format_trace(trace)
    if case.sources and not case.refused:
        cited = [(c["source_number"], c["label"], trace.passages[c["source_number"] - 1].document_id)
                 for c in trace.citations if c["chunk_id"]]
        assert first_match_rank(case, cited) is not None, (
            f"aucune citation de {list(case.sources)}\n" + format_trace(trace)
        )
