"""Commande ``rag-ingest`` : lance les téléchargements depuis le terminal.

Exemples ::

    uv run rag-ingest documents          # documents de config/sources.toml
    uv run rag-ingest catalogue          # catalogue complet de data.public.lu
    uv run rag-ingest catalogue --max-pages 1   # essai rapide
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from rag_rgpd.ingestion.catalogue import ingest_catalogue
from rag_rgpd.ingestion.documents import ingest_documents, load_sources
from rag_rgpd.ingestion.http import EmptyResponseError, PoliteClient, RobotsDisallowedError
from rag_rgpd.ingestion.storage import RawStore


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée de la commande ``rag-ingest``.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    parser = argparse.ArgumentParser(prog="rag-ingest", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--delay", type=float, default=2.0, help="pause entre requêtes (s)")
    sub = parser.add_subparsers(dest="command", required=True)

    docs = sub.add_parser("documents", help="télécharge les documents listés")
    docs.add_argument("--sources", type=Path, default=Path("config/sources.toml"))

    cat = sub.add_parser("catalogue", help="récupère le catalogue data.public.lu")
    cat.add_argument("--max-pages", type=int, default=None)

    args = parser.parse_args(argv)
    store = RawStore(args.data_dir)

    with PoliteClient(min_delay=args.delay) as client:
        if args.command == "documents":
            # Un document à la fois : l'échec d'une source ne bloque pas les autres.
            for doc in load_sources(args.sources):
                try:
                    [entry] = ingest_documents([doc], client, store)
                except (RobotsDisallowedError, EmptyResponseError, httpx.HTTPError) as error:
                    print(f"{doc.id}: ÉCHEC, {error}")
                    continue
                status = f"nouveau -> {entry.path}" if entry else "inchangé"
                print(f"{doc.id}: {status}")
        else:
            entry = ingest_catalogue(client, store, max_pages=args.max_pages)
            print(f"catalogue: {f'nouveau -> {entry.path}' if entry else 'inchangé'}")


if __name__ == "__main__":
    main()
