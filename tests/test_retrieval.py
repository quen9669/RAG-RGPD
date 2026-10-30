"""Tests de l'index LanceDB et de la recherche hybride, avec un faux modèle."""

import hashlib
import json
from pathlib import Path

import pytest

from rag_rgpd.retrieval.index import (
    IndexMismatchError,
    Searcher,
    build_index,
    embedding_text,
    read_meta,
)

from fakes import CHUNKS, FakeEmbedder


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_bytes(b"{}\n")  # octets exacts : write_text écrirait \r\n sous Windows
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index", chunks_path=chunks_path)
    return tmp_path / "index"


def test_le_texte_encode_porte_le_titre_complet() -> None:
    text = embedding_text(CHUNKS[0])

    assert text.startswith("RGPD > Chapitre II : Principes > Article 6 : Licéité")
    assert text.endswith("données à caractère personnel.")


def test_fiche_de_l_index_tracee(index_dir: Path) -> None:
    meta = read_meta(index_dir)

    assert meta.embedding_model == "faux-modele"
    assert meta.chunk_count == 4
    assert meta.dimension == 128
    assert meta.chunks_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))


def test_recherche_retrouve_le_bon_article(index_dir: Path) -> None:
    hits = Searcher(index_dir, FakeEmbedder()).search("droit à l'effacement de mes données", k=2)

    assert hits[0].chunk.label == "Article 17"
    assert hits[0].similarity > hits[1].similarity


def test_recherche_par_mot_cle_sans_accent(index_dir: Path) -> None:
    hits = Searcher(index_dir, FakeEmbedder()).search("delegue", k=1)

    assert hits[0].chunk.label == "Article 37"


def test_metadonnees_de_citation_conservees(index_dir: Path) -> None:
    hits = Searcher(index_dir, FakeEmbedder()).search("brochure citoyen", k=4)

    page = next(h.chunk for h in hits if h.chunk.document_id == "brochure")
    assert page == CHUNKS[3]
    article = next(h.chunk for h in hits if h.chunk.label == "Article 6")
    assert article.context == ("Chapitre II : Principes",)
    assert article.page_start is None


def test_refuse_un_index_d_un_autre_modele(index_dir: Path) -> None:
    with pytest.raises(IndexMismatchError):
        Searcher(index_dir, FakeEmbedder(model_name="autre-modele"))


def test_index_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="rag-index"):
        Searcher(tmp_path, FakeEmbedder())


def test_rien_a_indexer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rag-chunk"):
        build_index([], FakeEmbedder(), tmp_path)
