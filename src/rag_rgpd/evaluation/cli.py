"""Commande ``rag-eval`` : mesure la qualité sur le jeu de questions.

Exemples ::

    uv run rag-eval --search-only               # recherche seule, gratuit
    uv run rag-eval                             # chaîne complète + correcteur
    uv run rag-eval --no-rerank --no-pin        # comme à l'étape 4, pour comparer
    uv run rag-eval --save-reference            # la version testée devient la référence

Sans ``--save-reference``, la commande compare le résultat à la référence
(``evals/reference.json``) et se termine en erreur (code 1) si un
indicateur recule : c'est la barrière de qualité.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anthropic

from rag_rgpd.answering import cli as ask_cli
from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import DEFAULT_MODEL
from rag_rgpd.evaluation.judge import DEFAULT_JUDGE_MODEL, ClaudeJudge, Judgement
from rag_rgpd.evaluation.metrics import compare, format_summary, score_case, summarize
from rag_rgpd.explain.cases import DEFAULT_CASES_PATH, Case, load_cases
from rag_rgpd.explain.cost import estimate_cost
from rag_rgpd.explain.trace import ANSWERED, DEFAULT_TRACE_PATH, explain_question, write_trace
from rag_rgpd.retrieval.index import DEFAULT_K, DEFAULT_MIN_SIMILARITY

DEFAULT_REFERENCE = Path("evals/reference.json")

# Colonnes gardées dans la référence versionnée : uniquement des rangs et des
# vrai/faux, aucun texte des documents (dépôt public, licence CNPD).
REFERENCE_COLUMNS = [
    "cas", "decision", "code_refus", "rang", "source_k", "source_top3", "refus_ok",
    "citations_ok", "source_citee", "fidele", "toutes_citees", "completude", "piege_ok",
    "sans_recopie",
]


def run_eval(
    cases: list[Case],
    searcher: Any,
    generator: Any,
    judge: ClaudeJudge | None,
    *,
    k: int = DEFAULT_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    trace_path: Path | None = DEFAULT_TRACE_PATH,
    params: dict[str, Any] | None = None,
    progress: Any = None,
) -> list[dict[str, Any]]:
    """Pose chaque question, note la réponse, renvoie une ligne par cas.

    Args:
        cases: jeu de questions.
        searcher: moteur de recherche.
        generator: LLM rédacteur ; ``None`` pour la recherche seule.
        judge: correcteur ; ``None`` pour s'en passer (indicateurs du
            correcteur laissés vides).
        k: passages envoyés au modèle.
        min_similarity: seuil de la barrière 1.
        trace_path: fichier des fiches de décision ; ``None`` pour ne rien écrire.
        params: autres paramètres à tracer.
        progress: fonction appelée avec une ligne de suivi par cas.
    """
    rows = []
    for case in cases:
        trace, _ = explain_question(case.question, searcher, generator, k=k,
                                    min_similarity=min_similarity, params=params)
        if trace_path is not None:
            write_trace(trace, trace_path)
        judgement: Judgement | None = None
        if judge is not None and trace.decision == ANSWERED:
            judgement = judge.judge(case, trace)
        row = score_case(case, trace, judgement, min_similarity=min_similarity)
        if judgement is not None and not judgement.from_cache:
            row["cout_usd"] += estimate_cost(judgement.model, judgement.usage) or 0.0
        rows.append(row)
        if progress:
            rank = "-" if row["rang"] is None else row["rang"]
            progress(f"{case.id:<32} {trace.decision:<15} rang {rank}")
    return rows


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def eval_main(argv: list[str] | None = None) -> None:
    """Point d'entrée de ``rag-eval``.

    Args:
        argv: arguments de la ligne de commande (``None`` = ceux du terminal).
    """
    ask_cli._utf8_output()
    parser = argparse.ArgumentParser(prog="rag-eval", description="Mesure la qualité.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--only", nargs="+", default=None, help="identifiants des cas à lancer")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--min-similarity", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="modèle qui rédige")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="modèle correcteur")
    parser.add_argument("--search-only", action="store_true",
                        help="indicateurs de recherche seulement : gratuit, sans clé")
    parser.add_argument("--no-judge", action="store_true",
                        help="sans correcteur (moins cher, indicateurs de lecture vides)")
    parser.add_argument("--no-cache", action="store_true",
                        help="rappelle les LLM même si la réponse est en cache")
    ask_cli.add_search_options(parser)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--save-reference", action="store_true",
                        help="enregistre ce résultat comme nouvelle référence")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="recul accepté par la barrière, en points (0.02 = 2 points)")
    parser.add_argument("--out", type=Path, default=None,
                        help="rapport complet (par défaut data/evals/<date UTC>.json)")
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE_PATH)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)
    ask_cli.load_env(args.env_file)

    cases = load_cases(args.cases)
    if args.only:
        cases = [c for c in cases if c.id in set(args.only)]
        if not cases:
            sys.exit(f"aucun cas ne porte ces identifiants : {args.only}")

    cache = None if args.no_cache else ResponseCache()
    generator = judge = None
    if not args.search_only:
        ask_cli.require_api_key()
        generator = ask_cli.make_generator(args.model, cache)
        if not args.no_judge:
            judge = ClaudeJudge(model=args.judge_model, cache=cache)
    searcher = ask_cli.open_searcher(args)
    settings = {
        "k": args.k, "min_similarity": args.min_similarity,
        "embedding_model": args.embedding_model, **searcher.settings,
        "llm_model": "" if args.search_only else args.model,
        "judge_model": args.judge_model if judge else "",
    }
    print(f"{len(cases)} questions, réglages : {settings}")
    try:
        rows = run_eval(cases, searcher, generator, judge, k=args.k,
                        min_similarity=args.min_similarity, trace_path=args.trace,
                        params={"embedding_model": args.embedding_model}, progress=print)
    except anthropic.APIError as error:
        sys.exit(f"Erreur de l'API Anthropic : {ask_cli._api_error_message(error)}")
    summary = summarize(rows)

    at = datetime.now(UTC)
    out = args.out or Path("data/evals") / f"{at:%Y%m%d-%H%M%S}.json"
    report = {"at": at.isoformat(timespec="seconds"), "settings": settings,
              "summary": summary, "cases": rows}
    _write_json(report, out)

    reference = None
    if args.reference.exists() and not args.save_reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
    print()
    print(format_summary(summary, reference["summary"] if reference else None))
    print(f"\nrapport complet : {out.as_posix()}")

    to_read = [r for r in rows if r["source_k"] is False or not r["refus_ok"]
               or r["non_appuyees"] or r["piege_ok"] is False or r["recopies"]]
    if to_read:
        print("\nÀ relire (fiche : uv run rag-trace <id>) :")
        for r in to_read:
            note = (r["commentaire"] or (r["non_appuyees"] or [""])[0]
                    or ("recopie sans guillemets" if r["recopies"] else ""))
            print(f"  {r['trace_id']}  {r['cas']:<30} {note[:80]}")

    if args.save_reference:
        light = [{key: r[key] for key in REFERENCE_COLUMNS} for r in rows]
        _write_json({**report, "cases": light}, args.reference)
        print(f"\nnouvelle référence : {args.reference.as_posix()}")
        return
    if reference is None:
        print("\n(pas encore de référence : relancer avec --save-reference pour en créer une)")
        return
    regressions = compare(summary, reference["summary"], tolerance=args.tolerance)
    if regressions:
        print("\nBARRIÈRE DE QUALITÉ : recul par rapport à la référence")
        for line in regressions:
            print(f"  - {line}")
        sys.exit(1)
    print("\nBarrière de qualité franchie : aucun indicateur ne recule.")


if __name__ == "__main__":
    eval_main()
