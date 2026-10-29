"""Tests du stockage versionné et du manifeste."""

from datetime import UTC, datetime
from pathlib import Path

from rag_rgpd.ingestion.storage import RawStore, sha256_hex

JOUR_1 = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
JOUR_2 = datetime(2026, 11, 5, 12, 0, tzinfo=UTC)


def _save(store: RawStore, content: bytes, when: datetime):
    return store.save(
        document_id="doc",
        source="test",
        url="https://exemple.lu/doc",
        content=content,
        content_type="text/html",
        filename="doc.html",
        licence="CC0",
        retrieved_at=when,
    )


def test_enregistre_fichier_et_ligne_de_manifeste(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    entry = _save(store, b"v1", JOUR_1)

    assert entry is not None
    assert entry.path == "raw/test/2026-10-29/doc.html"
    assert (tmp_path / entry.path).read_bytes() == b"v1"
    assert entry.sha256 == sha256_hex(b"v1")
    assert store.entries() == [entry]


def test_contenu_identique_pas_de_doublon(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    _save(store, b"v1", JOUR_1)

    assert _save(store, b"v1", JOUR_2) is None
    assert len(store.entries()) == 1


def test_contenu_modifie_nouvelle_version_datee(tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    _save(store, b"v1", JOUR_1)
    entry = _save(store, b"v2", JOUR_2)

    assert entry is not None
    assert entry.path == "raw/test/2026-11-05/doc.html"
    # L'ancienne version reste sur disque et dans le manifeste (historique).
    assert (tmp_path / "raw/test/2026-10-29/doc.html").read_bytes() == b"v1"
    assert [e.sha256 for e in store.entries()] == [sha256_hex(b"v1"), sha256_hex(b"v2")]
    assert store.latest("doc") == entry
