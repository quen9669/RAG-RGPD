"""Commande ``rag-chunk`` : extrait et découpe les documents téléchargés.

Exemples ::

    uv run rag-chunk                     # tous les documents du manifeste
    uv run rag-chunk --max-chars 1000    # morceaux plus petits
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_rgpd.ingestion.documents import load_sources
from rag_rgpd.ingestion.storage import RawStore
from rag_rgpd.processing.chunking import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS
from rag_rgpd.processing.pipeline import (
    EXTRACTORS,
    Chunk,
    IntegrityError,
    chunk_document,
    latest_documents,
    media_type,
    write_chunks,
)


def main(argv: list[str] | None = None) -> None:
    """Point d'entrée de la commande ``rag-chunk``.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    parser = argparse.ArgumentParser(prog="rag-chunk", description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("config/sources.toml"))
    parser.add_argument("--output", type=Path, default=None,
                        help="défaut : <data-dir>/processed/chunks.jsonl")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    args = parser.parse_args(argv)

    store = RawStore(args.data_dir)
    titles = {doc.id: doc.title for doc in load_sources(args.sources)}
    output = args.output or args.data_dir / "processed" / "chunks.jsonl"

    chunks: list[Chunk] = []
    for entry in latest_documents(store):
        if media_type(entry.content_type) not in EXTRACTORS:
            print(f"{entry.document_id}: ignoré (type {entry.content_type})")
            continue
        try:
            doc_chunks = chunk_document(
                entry,
                store,
                title=titles.get(entry.document_id, entry.document_id),
                max_chars=args.max_chars,
                min_chars=args.min_chars,
            )
        except (FileNotFoundError, IntegrityError) as error:
            # Un document en échec ne bloque pas les autres.
            print(f"{entry.document_id}: ÉCHEC, {error}")
            continue
        sizes = [c.char_count for c in doc_chunks]
        average = sum(sizes) // len(sizes) if sizes else 0
        print(
            f"{entry.document_id}: {len(doc_chunks)} morceaux "
            f"(taille moyenne {average}, min {min(sizes, default=0)}, "
            f"max {max(sizes, default=0)} caractères)"
        )
        chunks.extend(doc_chunks)

    write_chunks(chunks, output)
    print(f"total: {len(chunks)} morceaux -> {output.as_posix()}")


if __name__ == "__main__":
    main()
