"""Téléchargement des documents listés dans ``config/sources.toml``."""

from __future__ import annotations

import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rag_rgpd.ingestion.http import PoliteClient
from rag_rgpd.ingestion.storage import ManifestEntry, RawStore

# Extension de fichier selon le type annoncé par le serveur. On s'en sert
# plutôt que de l'URL, car les URL d'EUR-Lex ne finissent pas par ".html".
_EXTENSIONS = {
    "text/html": ".html",
    "application/pdf": ".pdf",
    "application/json": ".json",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


@dataclass(frozen=True)
class DocumentSource:
    """Un document à récupérer, tel que décrit dans ``config/sources.toml``.

    Attributes:
        id: identifiant stable, réutilisé dans le manifeste et les citations.
        source: organisme d'origine (``"eur-lex"``, ``"cnpd"``...).
        title: titre lisible du document.
        url: adresse de téléchargement.
        licence: licence ou conditions d'usage relevées (voir docs/sources/).
    """

    id: str
    source: str
    title: str
    url: str
    licence: str


def load_sources(path: Path) -> list[DocumentSource]:
    """Lit la liste des documents à télécharger.

    Args:
        path: chemin du fichier TOML (section ``[[documents]]`` répétée).

    Returns:
        Les documents, dans l'ordre du fichier.

    Raises:
        ValueError: si deux documents ont le même identifiant (le
            manifeste ne pourrait plus les distinguer).
    """
    with path.open("rb") as f:
        raw = tomllib.load(f)
    documents = [DocumentSource(**item) for item in raw.get("documents", [])]
    ids = [d.id for d in documents]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Identifiants en double dans {path} : {sorted(duplicates)}")
    return documents


def extension_for(content_type: str) -> str:
    """Choisit l'extension de fichier à partir du type MIME.

    Args:
        content_type: en-tête ``Content-Type`` (ex. ``"text/html; charset=UTF-8"``).

    Returns:
        L'extension (``".html"``, ``".pdf"``...) ou ``".bin"`` si inconnue.
    """
    mime = content_type.split(";")[0].strip().lower()
    return _EXTENSIONS.get(mime, ".bin")


def ingest_documents(
    documents: list[DocumentSource],
    client: PoliteClient,
    store: RawStore,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> list[ManifestEntry | None]:
    """Télécharge chaque document et l'enregistre s'il a changé.

    Args:
        documents: documents à récupérer.
        client: client HTTP poli.
        store: stockage des fichiers bruts et du manifeste.
        now: horloge, remplaçable dans les tests.

    Returns:
        Pour chaque document, la nouvelle entrée du manifeste, ou ``None``
        s'il est identique à la version déjà stockée.
    """
    results: list[ManifestEntry | None] = []
    for doc in documents:
        response = client.get(doc.url)
        content_type = response.headers.get("content-type", "application/octet-stream")
        results.append(
            store.save(
                document_id=doc.id,
                source=doc.source,
                url=str(response.url),  # URL finale, après redirections éventuelles
                content=response.content,
                content_type=content_type,
                filename=doc.id + extension_for(content_type),
                licence=doc.licence,
                retrieved_at=now(),
            )
        )
    return results
