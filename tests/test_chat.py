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
