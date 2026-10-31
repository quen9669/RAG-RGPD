"""Tests de l'index LanceDB et de la recherche hybride, avec un faux modèle."""

import hashlib
import json
from pathlib import Path

import pytest

from rag_rgpd.retrieval.index import (
    IndexMismatchError,
    Searcher,
    build_index,
    cited_labels,
    embedding_text,
    read_meta,
)

from fakes import CHUNKS, FakeEmbedder, FakeReranker


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


# --- Étape 5 : articles cités et reclassement --------------------------------


@pytest.mark.parametrize(("question", "expected"), [
    ("article 17", ["Article 17"]),
    ("Que dit l'Article 6 ?", ["Article 6"]),
    ("art. 9 et considérant 51", ["Article 9", "Considérant 51"]),
    ("articles 15 puis article 15", ["Article 15"]),
    ("Puis-je demander l'effacement ?", []),
])
def test_articles_cites_reconnus(question: str, expected: list[str]) -> None:
    assert cited_labels(question) == expected


def test_article_cite_place_en_tete(index_dir: Path) -> None:
    # « délégué » attire l'article 37 ; la question nomme pourtant l'article 17.
    hits = Searcher(index_dir, FakeEmbedder()).search("article 17 délégué", k=2)

    assert hits[0].chunk.label == "Article 17"
    assert hits[0].pinned
    assert not hits[1].pinned


def test_article_cite_peut_etre_ignore(index_dir: Path) -> None:
    searcher = Searcher(index_dir, FakeEmbedder(), pin_cited_articles=False)

    assert not any(h.pinned for h in searcher.search("article 17 délégué", k=4))


def test_reclassement_fixe_l_ordre(index_dir: Path) -> None:
    reranker = FakeReranker(favourite="Brochure")
    searcher = Searcher(index_dir, FakeEmbedder(), reranker=reranker, pool=4)

    hits = searcher.search("droit à l'effacement de mes données", k=2)

    assert hits[0].chunk.document_id == "brochure"
    assert hits[0].rerank_score == 1.0
    assert reranker.calls == [("droit à l'effacement de mes données", 4)]
    assert searcher.settings == {"reranker": "faux-reclasseur", "pool": 4,
                                 "pin_cited_articles": True}
