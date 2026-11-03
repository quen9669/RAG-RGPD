"""Tests rapides du moteur du chat (sans navigateur, sans réseau, sans coût)."""

from pathlib import Path

import pytest
from fakes import CHUNKS, FakeEmbedder, FakeGenerator

from rag_rgpd.chat.engine import ChatEngine
from rag_rgpd.explain.trace import read_traces
from rag_rgpd.retrieval.index import Searcher, build_index


@pytest.fixture
def engine(tmp_path: Path) -> ChatEngine:
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index")
    return ChatEngine(Searcher(tmp_path / "index", FakeEmbedder()),
                      lambda model: FakeGenerator(), trace_path=tmp_path / "traces.jsonl")


def test_reponse_avec_sources(engine: ChatEngine, tmp_path: Path) -> None:
    reply = engine.ask("droit à l'effacement de mes données", model="x", min_similarity=0.0)

    assert not reply.refused
    assert reply.sources
    assert reply.sources[0].url and reply.sources[0].heading
    assert reply.cost_usd is not None
    assert read_traces(tmp_path / "traces.jsonl")[-1].trace_id == reply.trace_id


def test_refus_sans_source(engine: ChatEngine) -> None:
    reply = engine.ask("effacement", model="x", min_similarity=0.99)

    assert reply.refused
    assert reply.sources == []
    assert reply.reason


def test_coulisses_reponse(engine: ChatEngine) -> None:
    from rag_rgpd.chat.coulisses import passage_rows, stages, timing_text

    reply = engine.ask("droit à l'effacement de mes données", model="x", min_similarity=0.0)
    steps = stages(reply.trace)

    assert [s.status for s in steps].count("stop") == 0
    assert steps[-1].summary.startswith("réponse")
    assert passage_rows(reply.trace)[0]["Rang"] == 1
    assert "total" in timing_text(reply.trace)


def test_coulisses_refus_barriere_1(engine: ChatEngine) -> None:
    from rag_rgpd.chat.coulisses import stages

    reply = engine.ask("effacement", model="x", min_similarity=0.99)
    steps = stages(reply.trace)

    assert steps[-1].status == "stop"
    assert "Barrière 1" in steps[-1].title
