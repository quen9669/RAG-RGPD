"""Commandes ``rag-trace`` (relire une fiche) et ``rag-batch`` (essais en lot).

Exemples ::

    uv run rag-trace                       # fiche de la dernière question
    uv run rag-trace --list                # les 20 dernières questions
    uv run rag-trace 20261030-101500       # une fiche précise (début d'identifiant)
    uv run rag-batch --search-only --k 4 8 12          # gratuit, sans LLM
    uv run rag-batch --k 8 --min-similarity 0.40 0.45  # avec LLM (cache activé)
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from rag_rgpd.answering import cli as ask_cli
from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import DEFAULT_MODEL
from rag_rgpd.explain.batch import make_variants, run_batch, summarize, write_csv
from rag_rgpd.explain.cases import DEFAULT_CASES_PATH, load_cases
from rag_rgpd.explain.render import format_trace, format_trace_list
from rag_rgpd.explain.trace import DEFAULT_TRACE_PATH, find_trace, read_traces
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY


def trace_main(argv: list[str] | None = None) -> None:
    """Point d'entrée de ``rag-trace`` : réaffiche une fiche de décision.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    ask_cli._utf8_output()
    parser = argparse.ArgumentParser(prog="rag-trace", description="Relit une fiche de décision.")
    parser.add_argument("trace_id", nargs="?", default="last",
                        help="identifiant (ou son début) ; par défaut la dernière")
    parser.add_argument("--list", action="store_true", help="liste les dernières questions")
    parser.add_argument("--show-prompt", action="store_true",
                        help="affiche le texte exact envoyé au modèle")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    args = parser.parse_args(argv)

    traces = read_traces(args.trace)
    if args.list:
        print(format_trace_list(traces))
        return
    try:
        trace = find_trace(traces, args.trace_id)
    except LookupError as error:
        sys.exit(str(error))
    print(format_trace(trace, show_prompt=args.show_prompt))


def batch_main(argv: list[str] | None = None) -> None:
    """Point d'entrée de ``rag-batch`` : cas de test × réglages -> tableau.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    ask_cli._utf8_output()
    parser = argparse.ArgumentParser(prog="rag-batch", description="Lance des essais en lot.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH,
                        help="fichier YAML des questions")
    parser.add_argument("--only", nargs="+", default=None,
                        help="identifiants des cas à lancer (par défaut tous)")
    parser.add_argument("--k", type=int, nargs="+", default=[DEFAULT_K])
    parser.add_argument("--min-similarity", type=float, nargs="+",
                        default=[DEFAULT_MIN_SIMILARITY])
    parser.add_argument("--model", nargs="+", default=[DEFAULT_MODEL])
    parser.add_argument("--search-only", action="store_true",
                        help="recherche seule : gratuit, sans clé d'API")
    parser.add_argument("--no-cache", action="store_true",
                        help="rappelle le LLM même si la réponse est déjà en cache")
    ask_cli.add_search_options(parser)
    parser.add_argument("--out", type=Path, default=None,
                        help="fichier CSV (par défaut data/batches/<date UTC>.csv)")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    ask_cli.load_env(args.env_file)

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
        if not cases:
            sys.exit(f"aucun cas ne porte ces identifiants : {args.only}")
    models = [""] if args.search_only else args.model
    if not args.search_only:
        ask_cli.require_api_key()
    variants = make_variants(args.k, args.min_similarity, models)
    print(f"{len(cases)} questions × {len(variants)} réglages = "
          f"{len(cases) * len(variants)} essais")

    cache = None if args.no_cache else ResponseCache()
    searcher = ask_cli.open_searcher(args)
    try:
        rows = run_batch(
            cases, variants, searcher, lambda model: ask_cli.make_generator(model, cache),
            trace_path=args.trace, params={"embedding_model": args.embedding_model},
            progress=print,
        )
    except anthropic.APIError as error:
        sys.exit(f"Erreur de l'API Anthropic : {ask_cli._api_error_message(error)}")

    out = args.out or Path("data/batches") / f"{datetime.now(UTC):%Y%m%d-%H%M%S}.csv"
    write_csv(rows, out)
    print()
    print(summarize(rows))
    print(f"\ntableau complet : {out.as_posix()} (s'ouvre dans Excel ou VS Code)")
    failed = [r for r in rows if r["source_ok"] is False or not r["refus_ok"]]
    if failed:
        print("\nÀ regarder de près (fiche : uv run rag-trace <id>) :")
        for r in failed:
            print(f"  {r['trace_id']}  {r['cas']:<28} {r['reglage']}")


if __name__ == "__main__":
    # Lancement en module, utilisé par le débogueur de VS Code (launch.json) :
    # `python -m rag_rgpd.explain.cli --batch ...` pour les lots, sinon rag-trace.
    if sys.argv[1:2] == ["--batch"]:
        batch_main(sys.argv[2:])
    else:
        trace_main()
