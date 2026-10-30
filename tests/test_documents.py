"""Tests de l'ingestion des documents listés dans sources.toml."""

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from rag_rgpd.ingestion.documents import (
    extension_for,
    ingest_documents,
    load_sources,
)
from rag_rgpd.ingestion.storage import RawStore

RACINE = Path(__file__).resolve().parents[1]


def test_le_fichier_sources_du_projet_est_valide() -> None:
    documents = load_sources(RACINE / "config" / "sources.toml")
    assert documents, "config/sources.toml ne liste aucun document"
    assert all(d.licence for d in documents), "chaque source doit indiquer sa licence"


def test_identifiants_en_double_refuses(tmp_path: Path) -> None:
    entree = 'id = "a"\nsource = "s"\ntitle = "t"\nurl = "https://x.lu"\nlicence = "l"\n'
    fichier = tmp_path / "sources.toml"
    fichier.write_text(f"[[documents]]\n{entree}\n[[documents]]\n{entree}", encoding="utf-8")
    with pytest.raises(ValueError, match="double"):
        load_sources(fichier)


@pytest.mark.parametrize(
    ("content_type", "extension"),
    [("text/html; charset=UTF-8", ".html"), ("application/pdf", ".pdf"), ("x/y", ".bin")],
)
def test_extension_selon_le_type(content_type: str, extension: str) -> None:
    assert extension_for(content_type) == extension


def test_ingestion_de_bout_en_bout(make_client, tmp_path: Path) -> None:
    sources = tmp_path / "sources.toml"
    sources.write_text(
        '[[documents]]\nid = "guide"\nsource = "cnpd"\ntitle = "Guide"\n'
        'url = "https://exemple.lu/guide"\nlicence = "privé"\n',
        # Encodage explicite : sous Windows, le défaut (cp1252) casse le « é ».
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.7", headers={"content-type": "application/pdf"})

    store = RawStore(tmp_path / "data")
    documents = load_sources(sources)
    now = lambda: datetime(2026, 10, 29, tzinfo=UTC)  # noqa: E731

    [entry] = ingest_documents(documents, make_client(handler), store, now=now)
    assert entry is not None
    assert entry.path == "raw/cnpd/2026-10-29/guide.pdf"
    assert entry.licence == "privé"

    # Deuxième passage : même contenu, donc rien de nouveau.
    assert ingest_documents(documents, make_client(handler), store, now=now) == [None]
