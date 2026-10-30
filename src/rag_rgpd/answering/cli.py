"""Commandes ``rag-index`` et ``rag-ask``.

Exemples ::

    uv run rag-index                               # construit data/index/
    uv run rag-ask "Qu'est-ce qu'une donnée sensible ?"
    uv run rag-ask --search-only "article 6"       # passages seuls, sans LLM
    uv run rag-ask --explain "..."                 # réponse + fiche de décision
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from rag_rgpd.answering.cache import DEFAULT_CACHE_DIR, ResponseCache
from rag_rgpd.answering.generator import (
    DEFAULT_MODEL,
    Answer,
    AnswerGenerator,
    ClaudeGenerator,
)
from rag_rgpd.answering.journal import log_answer
from rag_rgpd.explain.render import format_trace
from rag_rgpd.explain.trace import DEFAULT_EXTRA, DEFAULT_TRACE_PATH, explain_question, write_trace
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
make_generator: Callable[[str, ResponseCache | None], AnswerGenerator] = (
    lambda model, cache=None: ClaudeGenerator(model=model, cache=cache)
)


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


def load_env(env_file: Path) -> None:
    """Charge la clé d'API depuis ``.env``.

    La clé vit dans ce fichier, propre au projet et exclu de Git. Une
    variable déjà définie dans le terminal reste prioritaire.
    """
    load_dotenv(env_file, override=False, encoding="utf-8")


def require_api_key() -> None:
    """Arrête le programme avec un message clair si la clé manque."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Clé manquante : ajouter ANTHROPIC_API_KEY=... dans le fichier .env "
                 "(voir README), ou utiliser --search-only.")


def _api_error_message(error: anthropic.APIError) -> str:
    """Message court et compréhensible pour une erreur de l'API."""
    if isinstance(error, anthropic.AuthenticationError):
        return "clé refusée, vérifier ANTHROPIC_API_KEY dans .env."
    if isinstance(error, anthropic.APIConnectionError):
        return "connexion impossible, vérifier l'accès à Internet."
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        detail = body.get("error", {}).get("message")
        if detail:
            return detail
    return error.message


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
    parser.add_argument("--explain", action="store_true",
                        help="affiche la fiche de décision complète")
    parser.add_argument("--show-prompt", action="store_true",
                        help="avec --explain : affiche le texte exact envoyé au modèle")
    parser.add_argument("--extra", type=int, default=DEFAULT_EXTRA,
                        help="passages regardés au-delà de k, montrés comme écartés")
    parser.add_argument("--cache", action="store_true",
                        help=f"réutilise une réponse déjà payée ({DEFAULT_CACHE_DIR.as_posix()})")
    parser.add_argument("--log", type=Path, default=Path("data/logs/answers.jsonl"))
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH,
                        help="fichier des fiches de décision")
    parser.add_argument("--env-file", type=Path, default=Path(".env"),
                        help="fichier local contenant ANTHROPIC_API_KEY")
    args = parser.parse_args(argv)
    load_env(args.env_file)

    searcher = Searcher(args.index_dir, make_embedder(args.embedding_model))

    generator = None
    if not args.search_only:
        require_api_key()
        cache = ResponseCache() if args.cache else None
        generator = make_generator(args.model, cache)
    try:
        trace, answer = explain_question(
            args.question, searcher, generator, k=args.k,
            min_similarity=args.min_similarity, extra=args.extra,
            params={"embedding_model": args.embedding_model},
        )
    except anthropic.APIError as error:
        # Clé refusée, crédit épuisé, réseau coupé... : une phrase lisible
        # plutôt qu'une trace Python de quarante lignes.
        sys.exit(f"Erreur de l'API Anthropic : {_api_error_message(error)}")
    write_trace(trace, args.trace)

    if answer is None:
        if args.explain:
            print(format_trace(trace))
            return
        for p in trace.sent_passages:
            pages = f", p. {p.pages}" if p.pages else ""
            print(f"[{p.rank}] {p.heading}{pages}\n    {p.url}\n    similarité "
                  f"{p.similarity:.2f}, score {p.score:.4f}\n    {p.text[:200]}...\n")
        return

    log_answer(answer, searcher.meta, args.log)
    if args.explain:
        print(format_trace(trace, show_prompt=args.show_prompt))
        print()
    print(format_answer(answer))
    print(f"\n(fiche de décision : uv run rag-trace {trace.trace_id})")


if __name__ == "__main__":
    ask_main()
