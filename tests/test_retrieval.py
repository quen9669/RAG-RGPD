"""Tests de l'index LanceDB et de la recherche hybride, avec un faux modèle."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from rag_rgpd.retrieval import index as index_module
from rag_rgpd.retrieval.index import (
    IndexMismatchError,
    Searcher,
    build_index,
    cited_labels,
    embedding_text,
    min_similarity_for,
    read_meta,
)

from fakes import CHUNKS, FakeEmbedder, FakeReranker, make_chunk


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
                                 "pin_cited_articles": True, "domains": None}


# --- Étape 8 : domaines -------------------------------------------------

RAIL = make_chunk(
    9, "Article 5", "Droit au transport de vélos",
    "Les entreprises ferroviaires permettent aux voyageurs d'embarquer leur vélo dans le train.",
    document_id="voyageurs-ue", domain="ferroviaire",
)


@pytest.fixture
def mixed_index(tmp_path: Path) -> Path:
    build_index([*CHUNKS, RAIL], FakeEmbedder(), tmp_path / "index")
    return tmp_path / "index"


def test_le_filtre_de_domaine_ecarte_les_autres_sujets(mixed_index: Path) -> None:
    hits = Searcher(mixed_index, FakeEmbedder(), domains=["ferroviaire"]).search("vélo train", k=5)

    assert [h.chunk.document_id for h in hits] == ["voyageurs-ue"]


def test_sans_filtre_tous_les_domaines_sont_cherches(mixed_index: Path) -> None:
    hits = Searcher(mixed_index, FakeEmbedder()).search("vélo train", k=5)

    assert {h.chunk.domain for h in hits} == {"rgpd", "ferroviaire"}


def test_le_filtre_vaut_aussi_pour_les_articles_cites(mixed_index: Path) -> None:
    # « Article 5 » existe seulement dans le domaine ferroviaire ici ; le
    # filtre rgpd ne doit pas le remonter.
    hits = Searcher(mixed_index, FakeEmbedder(), domains=["rgpd"]).search("article 5 vélo", k=5)

    assert all(h.chunk.domain == "rgpd" for h in hits)


def test_les_rangs_respectent_le_filtre(mixed_index: Path) -> None:
    searcher = Searcher(mixed_index, FakeEmbedder(), domains=["rgpd"])

    assert RAIL.chunk_id not in searcher.ranks("vélo train")


def test_les_domaines_sont_traces(mixed_index: Path) -> None:
    assert Searcher(mixed_index, FakeEmbedder(), domains=["rgpd"]).settings["domains"] == ["rgpd"]
    assert Searcher(mixed_index, FakeEmbedder()).settings["domains"] is None


def test_domaine_invalide_refuse(mixed_index: Path) -> None:
    with pytest.raises(ValueError):
        Searcher(mixed_index, FakeEmbedder(), domains=["x' OR '1'='1"])


def test_seuil_par_domaine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(index_module.DOMAIN_MIN_SIMILARITY, "ferroviaire", 0.5)

    assert min_similarity_for(None) == index_module.DEFAULT_MIN_SIMILARITY
    assert min_similarity_for(["rgpd"]) == index_module.DEFAULT_MIN_SIMILARITY
    assert min_similarity_for(["ferroviaire"]) == 0.5
    assert min_similarity_for(["rgpd", "ferroviaire"]) == 0.5  # le plus strict


def test_ancien_morceau_sans_domaine_reste_rgpd() -> None:
    assert replace(CHUNKS[0]).domain == "rgpd"
