"""Tests de l'ingestion du catalogue data.public.lu (API paginée)."""

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rag_rgpd.ingestion.catalogue import fetch_catalogue, ingest_catalogue
from rag_rgpd.ingestion.storage import RawStore


def _faux_api(request: httpx.Request) -> httpx.Response:
    """Simule deux pages de l'API udata, la seconde sans page suivante."""
    if request.url.path == "/robots.txt":
        return httpx.Response(404)
    if request.url.params.get("page") == "2":
        return httpx.Response(200, json={"data": [{"id": "a"}], "next_page": None})
    return httpx.Response(
        200,
        json={
            "data": [{"id": "c"}, {"id": "b"}],
            "next_page": "https://data.public.lu/api/1/datasets/?page=2&page_size=2",
        },
    )


def test_suit_toutes_les_pages(make_client) -> None:
    datasets = fetch_catalogue(make_client(_faux_api), page_size=2)
    assert [d["id"] for d in datasets] == ["c", "b", "a"]


def test_max_pages_limite_la_recuperation(make_client) -> None:
    assert len(fetch_catalogue(make_client(_faux_api), page_size=2, max_pages=1)) == 2


def test_instantane_trie_et_trace(make_client, tmp_path: Path) -> None:
    store = RawStore(tmp_path)
    entry = ingest_catalogue(
        make_client(_faux_api), store, page_size=2, now=lambda: datetime(2026, 10, 29, tzinfo=UTC)
    )
    assert entry is not None
    assert entry.path == "raw/data-public-lu/2026-10-29/catalogue.json"
    contenu = json.loads((tmp_path / entry.path).read_text())
    assert [d["id"] for d in contenu] == ["a", "b", "c"]
