"""Tests de la chaîne manifeste -> morceaux et de la commande rag-chunk."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_rgpd.ingestion.storage import RawStore
from rag_rgpd.processing.cli import main
from rag_rgpd.processing.pipeline import (
    IntegrityError,
    chunk_document,
    latest_documents,
    read_chunks,
    write_chunks,
)

FIXTURES = Path(__file__).parent / "fixtures"
JOUR = datetime(2026, 10, 30, 8, 0, tzinfo=UTC)


def _store_with_rgpd(root: Path, *, empty_after: bool = False) -> RawStore:
    store = RawStore(root)
    store.save(
        document_id="rgpd-fr", source="eur-lex", url="https://exemple.eu/rgpd",
        content=(FIXTURES / "acte-ue.html").read_bytes(),
        content_type="application/xhtml+xml;charset=UTF-8",
        filename="rgpd-fr.html", licence="CC BY 4.0", retrieved_at=JOUR,
    )
    store.save(
        document_id="catalogue", source="data-public-lu", url="https://exemple.lu/api",
        content=b"[]", content_type="application/json", filename="catalogue.json",
        licence="CC0", retrieved_at=JOUR,
    )
    if empty_after:
        store.save(
            document_id="rgpd-fr", source="eur-lex", url="https://exemple.eu/rgpd",
            content=b"", content_type="text/html", filename="rgpd-fr.html",
            licence="CC BY 4.0", retrieved_at=JOUR,
        )
    return store


def test_version_vide_ne_masque_pas_la_bonne(tmp_path: Path) -> None:
    store = _store_with_rgpd(tmp_path, empty_after=True)

    rgpd = next(e for e in latest_documents(store) if e.document_id == "rgpd-fr")

    assert rgpd.size_bytes > 0


def test_morceaux_tracables_et_citables(tmp_path: Path) -> None:
    store = _store_with_rgpd(tmp_path)
    entry = store.latest("rgpd-fr")

    chunks = chunk_document(entry, store, title="RGPD")

    article_6 = next(c for c in chunks if c.label == "Article 6")
    assert article_6.heading == "RGPD > Chapitre II : Principes > Article 6 : Licéité du traitement"
    assert article_6.source_sha256 == entry.sha256
    assert article_6.raw_path == entry.path
    assert article_6.licence == "CC BY 4.0"
    assert chunks[0].chunk_id == f"rgpd-fr-{entry.sha256[:8]}-0001"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_fichier_modifie_detecte(tmp_path: Path) -> None:
    store = _store_with_rgpd(tmp_path)
    entry = store.latest("rgpd-fr")
    (tmp_path / entry.path).write_bytes(b"<html>autre</html>")

    with pytest.raises(IntegrityError):
        chunk_document(entry, store, title="RGPD")


def test_ecriture_relecture_utf8(tmp_path: Path) -> None:
    store = _store_with_rgpd(tmp_path)
    chunks = chunk_document(store.latest("rgpd-fr"), store, title="RGPD")
    output = tmp_path / "out" / "chunks.jsonl"

    write_chunks(chunks, output)

    assert read_chunks(output) == chunks
    # Les accents sont écrits tels quels en UTF-8, pas en séquences é.
    assert "Licéité" in output.read_text(encoding="utf-8")


def test_commande_rag_chunk(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _store_with_rgpd(tmp_path)
    sources = tmp_path / "sources.toml"
    sources.write_text(
        '[[documents]]\nid = "rgpd-fr"\nsource = "eur-lex"\ntitle = "RGPD"\n'
        'url = "https://exemple.eu/rgpd"\nlicence = "CC BY 4.0"\n',
        encoding="utf-8",
    )

    main(["--data-dir", str(tmp_path), "--sources", str(sources)])

    out = capsys.readouterr().out
    assert "catalogue: ignoré" in out
    assert "rgpd-fr: 5 morceaux" in out
    lines = (tmp_path / "processed" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["document_title"] == "RGPD"
