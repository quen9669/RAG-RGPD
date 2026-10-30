"""Commandes ``rag-index`` et ``rag-ask``.

Exemples ::

    uv run rag-index                               # construit data/index/
    uv run rag-ask "Qu'est-ce qu'une donnée sensible ?"
    uv run rag-ask --search-only "article 6"       # passages seuls, sans LLM
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from rag_rgpd.answering.generator import (
    DEFAULT_MODEL,
    Answer,
    AnswerGenerator,
    ClaudeGenerator,
    answer_question,
)
from rag_rgpd.answering.journal import log_answer
from rag_rgpd.processing.pipeline import Chunk, read_chunks
from rag_rgpd.retrieval.embeddings import DEFAULT_MODEL as DEFAULT_EMBEDDING_MODEL
from rag_rgpd.retrieval.embeddings import Embedder, SentenceTransformerEmbedder
from rag_rgpd.retrieval.index import (
    DEFAULT_K,
    DEFAULT_MIN_SIMILARITY,
    Hit,
    Searcher,
    build_index,
)

# Fabriques remplaçables dans les tests, pour ne charger ni PyTorch ni l'API.
make_embedder: Callable[[str], Embedder] = SentenceTransformerEmbedder
make_generator: Callable[[str], AnswerGenerator] = lambda model: ClaudeGenerator(model=model)


def _utf8_output() -> None:
    """Force l'UTF-8 en sortie.

    Sous Windows, quand la sortie est redirigée vers un fichier ou un autre
    programme, Python utilise l'encodage local (cp1252) : les guillemets
    « » et certains caractères des textes feraient planter l'affichage.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _pages(chunk: Chunk) -> str:
    if chunk.page_start is None:
        return ""
    if chunk.page_end and chunk.page_end != chunk.page_start:
        return f", p. {chunk.page_start}-{chunk.page_end}"
    return f", p. {chunk.page_start}"


def format_source(number: int, hit: Hit) -> str:
    """« [1] RGPD > ... > Article 6 : Licéité du traitement, p. 3 — url »."""
    return f"[{number}] {hit.chunk.heading}{_pages(hit.chunk)}\n    {hit.chunk.url}"


def format_answer(answer: Answer) -> str:
    """Réponse lisible : texte, puis uniquement les sources citées."""
    lines = [answer.text]
    if answer.refused:
        lines.append(f"\n(Raison : {answer.reason}.)")
        return "\n".join(lines)
    lines.append("\nSources :")
    for number in answer.cited_numbers:
        lines.append(format_source(number, answer.hits[number - 1]))
    lines.append(
        "\nCet assistant aide à l'analyse et ne remplace pas un avis juridique."
    )
    return "\n".join(lines)


def index_main(argv: list[str] | None = None) -> None:
    """Point d'entrée de ``rag-index`` : encode les morceaux et crée l'index.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    _utf8_output()
    parser = argparse.ArgumentParser(prog="rag-index", description="Construit l'index de recherche.")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args(argv)

    chunks = read_chunks(args.chunks)
    print(f"{len(chunks)} morceaux à encoder avec {args.embedding_model}...")
    meta = build_index(chunks, make_embedder(args.embedding_model), args.index_dir,
                       chunks_path=args.chunks)
    print(f"index prêt : {meta.chunk_count} morceaux, vecteurs de dimension "
          f"{meta.dimension} -> {args.index_dir.as_posix()}")


def ask_main(argv: list[str] | None = None) -> None:
    """Point d'entrée de ``rag-ask`` : recherche, puis réponse citée.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    _utf8_output()
    parser = argparse.ArgumentParser(prog="rag-ask", description="Pose une question.")
    parser.add_argument("question")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="passages retrouvés")
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle Claude")
    parser.add_argument("--search-only", action="store_true",
                        help="affiche les passages retrouvés, sans appeler le LLM")
    parser.add_argument("--log", type=Path, default=Path("data/logs/answers.jsonl"))
    args = parser.parse_args(argv)

    searcher = Searcher(args.index_dir, make_embedder(args.embedding_model))
    hits = searcher.search(args.question, k=args.k)

    if args.search_only:
        for number, hit in enumerate(hits, start=1):
            print(f"{format_source(number, hit)}\n    similarité {hit.similarity:.2f}, "
                  f"score {hit.score:.4f}\n    {hit.chunk.text[:200]}...\n")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Clé manquante : définir ANTHROPIC_API_KEY (voir README), "
                 "ou utiliser --search-only.")
    answer = answer_question(
        args.question, hits, make_generator(args.model),
        min_similarity=args.min_similarity,
        params={"k": args.k, "embedding_model": args.embedding_model},
    )
    log_answer(answer, searcher.meta, args.log)
    print(format_answer(answer))


if __name__ == "__main__":
    ask_main()
