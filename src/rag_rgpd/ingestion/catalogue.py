"""Ingestion par API du catalogue de data.public.lu.

data.public.lu repose sur le logiciel libre udata. Son API renvoie les jeux
de données page par page ; chaque page indique l'adresse de la suivante
(``next_page``), jusqu'à ce qu'elle vaille ``null``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from rag_rgpd.ingestion.http import PoliteClient
from rag_rgpd.ingestion.storage import ManifestEntry, RawStore

API_URL = "https://data.public.lu/api/1/datasets/"
LICENCE = "CC0 1.0 (contenu du site data.public.lu, sauf mention contraire)"


def fetch_catalogue(
    client: PoliteClient, page_size: int = 100, max_pages: int | None = None
) -> list[dict[str, Any]]:
    """Récupère la description de tous les jeux de données publiés.

    Args:
        client: client HTTP poli (il espace déjà les appels).
        page_size: nombre de jeux de données par page demandée.
        max_pages: limite de pages, utile pour un essai rapide ; ``None``
            pour tout récupérer.

    Returns:
        La liste des jeux de données, tels que renvoyés par l'API.
    """
    datasets: list[dict[str, Any]] = []
    url: str | None = API_URL
    params: dict[str, str | int] | None = {"page_size": page_size}
    pages = 0
    while url is not None and (max_pages is None or pages < max_pages):
        payload = client.get(url, params=params).json()
        datasets.extend(payload["data"])
        url = payload.get("next_page")
        params = None  # l'URL "next_page" contient déjà les paramètres
        pages += 1
    return datasets


def ingest_catalogue(
    client: PoliteClient,
    store: RawStore,
    page_size: int = 100,
    max_pages: int | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> ManifestEntry | None:
    """Récupère le catalogue et l'enregistre comme un instantané daté.

    Le catalogue entier est stocké dans un seul fichier JSON, trié pour que
    deux catalogues identiques donnent la même empreinte.

    Args:
        client: client HTTP poli.
        store: stockage des fichiers bruts et du manifeste.
        page_size: nombre de jeux de données par page demandée.
        max_pages: limite de pages (``None`` = tout le catalogue).
        now: horloge, remplaçable dans les tests.

    Returns:
        La nouvelle entrée du manifeste, ou ``None`` si rien n'a changé.
    """
    retrieved_at = now()
    datasets = fetch_catalogue(client, page_size=page_size, max_pages=max_pages)
    datasets.sort(key=lambda d: d["id"])
    content = json.dumps(datasets, ensure_ascii=False, sort_keys=True, indent=1).encode()
    return store.save(
        document_id="data-public-lu-catalogue",
        source="data-public-lu",
        url=API_URL,
        content=content,
        content_type="application/json",
        filename="catalogue.json",
        licence=LICENCE,
        retrieved_at=retrieved_at,
    )
