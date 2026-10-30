"""Essais en lot : N questions × plusieurs réglages -> un tableau comparatif.

Exemple : comparer ``k = 4, 8, 12`` et deux seuils sur les cas de
``tests/cases/questions.yaml``, et voir pour chaque combinaison quels
passages sont retrouvés, s'il y a refus, et si les citations tiennent.

Chaque essai écrit aussi sa fiche de décision complète dans
``data/traces/traces.jsonl`` : une ligne surprenante du tableau se relit en
détail avec ``rag-trace <id>``.
"""

from __future__ import annotations

import csv
import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_rgpd.answering.generator import AnswerGenerator
from rag_rgpd.explain.cases import Case, first_match_rank
from rag_rgpd.explain.trace import (
    DEFAULT_TRACE_PATH,
    REFUSED,
    SEARCH_ONLY,
    SearchEngine,
    explain_question,
    write_trace,
)


@dataclass(frozen=True)
class Variant:
    """Un réglage à essayer.

    Attributes:
        k: passages envoyés au modèle.
        min_similarity: seuil de la barrière 1.
        model: modèle Claude, ou ``""`` pour la recherche seule.
    """

    k: int
    min_similarity: float
    model: str = ""

    @property
    def name(self) -> str:
        """Nom court, par exemple ``k=8 seuil=0.45 claude-haiku-5-5``."""
        base = f"k={self.k} seuil={self.min_similarity:.2f}"
        return f"{base} {self.model}" if self.model else base


def make_variants(ks: Iterable[int], thresholds: Iterable[float],
                  models: Iterable[str]) -> list[Variant]:
    """Toutes les combinaisons des réglages donnés."""
    return [Variant(k, t, m) for k, t, m in itertools.product(ks, thresholds, models)]


COLUMNS = [
    "cas", "reglage", "decision", "code_refus", "meilleure_sim", "rang_attendu",
    "source_ok", "refus_ok", "cites", "citations_ok", "cout_usd", "cache", "trace_id",
]


def run_batch(
    cases: list[Case],
    variants: list[Variant],
    searcher: SearchEngine,
    make_generator: Callable[[str], AnswerGenerator],
    *,
    trace_path: Path | None = DEFAULT_TRACE_PATH,
    params: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Lance chaque cas avec chaque réglage.

    Args:
        cases: questions et comportements attendus.
        variants: réglages à comparer.
        searcher: moteur de recherche (chargé une seule fois pour tout le lot).
        make_generator: fabrique un LLM pour un nom de modèle (appelée une
            fois par modèle).
        trace_path: fichier des traces ; ``None`` pour ne rien écrire.
        params: autres paramètres à tracer (modèle d'embedding...).
        progress: fonction appelée avec une ligne de suivi après chaque essai.

    Returns:
        Une ligne par essai, avec les colonnes de ``COLUMNS``.
    """
    generators: dict[str, AnswerGenerator] = {}
    rows = []
    for variant, case in itertools.product(variants, cases):
        generator = None
        if variant.model:
            if variant.model not in generators:
                generators[variant.model] = make_generator(variant.model)
            generator = generators[variant.model]
        trace, _ = explain_question(
            case.question, searcher, generator, k=variant.k,
            min_similarity=variant.min_similarity, params=params,
        )
        if trace_path is not None:
            write_trace(trace, trace_path)

        sent = [(p.rank, p.label, p.document_id) for p in trace.sent_passages]
        rank = first_match_rank(case, sent) if case.sources else None
        source_ok = None
        if case.sources:
            source_ok = rank is not None and rank <= (case.max_rank or variant.k)
        if trace.decision == SEARCH_ONLY:
            # Sans LLM, seule la barrière 1 peut décider d'un refus.
            refused = trace.best_similarity < variant.min_similarity
        else:
            refused = trace.decision == REFUSED
        cited = sorted({c["label"] for c in trace.citations if c["label"]})
        row = {
            "cas": case.id,
            "reglage": variant.name,
            "decision": trace.decision,
            "code_refus": trace.refusal_code,
            "meilleure_sim": trace.best_similarity,
            "rang_attendu": rank,
            "source_ok": source_ok,
            "refus_ok": refused == case.refused,
            "cites": " | ".join(cited),
            "citations_ok": all(c["verified"] for c in trace.citations),
            "cout_usd": trace.cost_usd,
            "cache": trace.from_cache,
            "trace_id": trace.trace_id,
        }
        rows.append(row)
        if progress:
            progress(f"{variant.name:<36} {case.id:<28} {trace.decision}")
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Écrit le tableau en CSV.

    Encodage ``utf-8-sig`` et séparateur ``;`` : Excel en français l'ouvre
    directement avec les accents et les colonnes au bon endroit.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> str:
    """Résumé par réglage : sources trouvées, refus corrects, coût total."""
    lines = [f"{'réglage':<38} {'source ok':>10} {'refus ok':>9} {'citations ok':>13} {'coût $':>9}"]
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["reglage"], []).append(row)
    for name, group in by_variant.items():
        with_source = [r for r in group if r["source_ok"] is not None]
        source_ok = sum(1 for r in with_source if r["source_ok"])
        refus_ok = sum(1 for r in group if r["refus_ok"])
        citations_ok = sum(1 for r in group if r["citations_ok"])
        cost = sum(r["cout_usd"] or 0 for r in group)
        lines.append(
            f"{name:<38} {source_ok:>4}/{len(with_source):<5} {refus_ok:>4}/{len(group):<4} "
            f"{citations_ok:>6}/{len(group):<6} {cost:>9.4f}"
        )
    return "\n".join(lines)
