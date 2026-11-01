"""Indicateurs de qualité et barrière de qualité.

Pour chaque question du jeu, ``score_case`` produit une ligne de résultats ;
``summarize`` en tire une dizaine d'indicateurs, tous entre 0 et 1 (plus
haut = mieux), et ``compare`` dit si une nouvelle version fait au moins
aussi bien que la version de référence (barrière de qualité de la
gouvernance).

Indicateurs de la recherche (gratuits, sans LLM) :

- **source dans les k** (rappel@k, « recall@k ») : part des questions dont la
  bonne source figure parmi les passages envoyés au modèle ;
- **source dans le top 3** : idem parmi les 3 premiers ;
- **MRR** (Mean Reciprocal Rank, rang réciproque moyen) : moyenne de 1/rang
  de la première bonne source (1 au rang 1, 0,5 au rang 2, 0 si absente) ;
- **refus corrects** : l'assistant refuse exactement quand il le faut.

Indicateurs de la réponse (avec le LLM et le correcteur) :

- **citations vérifiées** : toutes les citations se retrouvent dans leur passage ;
- **source attendue citée** : la réponse cite au moins une bonne source ;
- **fidélité** : part des affirmations appuyées par les passages (l'inverse
  du taux d'hallucination), moyenne sur les questions ;
- **affirmations citées** : part des affirmations qui portent un numéro de
  source ;
- **complétude** : part moyenne des points clés couverts ;
- **pièges évités** : l'erreur connue n'est pas commise ;
- **sans recopie** : aucune phrase recopiée sans guillemets.
"""

from __future__ import annotations

from typing import Any

from rag_rgpd.explain.cases import Case, first_match_rank
from rag_rgpd.explain.trace import ANSWERED, REFUSED, SEARCH_ONLY, Trace
from rag_rgpd.evaluation.judge import Judgement, verbatim_copies

METRICS: list[tuple[str, str]] = [
    # (clé, libellé affiché)
    ("source_k", "source dans les k"),
    ("source_top3", "source dans le top 3"),
    ("mrr", "MRR (rang réciproque moyen)"),
    ("refus_ok", "refus corrects"),
    ("citations_ok", "citations vérifiées"),
    ("source_citee", "source attendue citée"),
    ("fidele", "fidélité (affirmations appuyées)"),
    ("toutes_citees", "affirmations citées"),
    ("completude", "complétude (points clés)"),
    ("piege_ok", "pièges évités"),
    ("sans_recopie", "sans recopie sans guillemets"),
]


def score_case(
    case: Case, trace: Trace, judgement: Judgement | None, *, min_similarity: float
) -> dict[str, Any]:
    """Résultats d'une question.

    Args:
        case: la question et ce qu'on attend.
        trace: fiche de décision de la réponse.
        judgement: note du correcteur, ``None`` s'il n'a pas été appelé
            (recherche seule ou refus).
        min_similarity: seuil de la barrière 1, pour juger le refus en
            recherche seule.

    Returns:
        Une ligne : ``None`` dans une colonne = sans objet pour ce cas.
    """
    sent = [(p.rank, p.label, p.document_id) for p in trace.sent_passages]
    rank = first_match_rank(case, sent) if case.sources else None
    if trace.decision == SEARCH_ONLY:
        refused = trace.best_similarity < min_similarity
    else:
        refused = trace.decision == REFUSED
    answered = trace.decision == ANSWERED

    row: dict[str, Any] = {
        "cas": case.id,
        "decision": trace.decision,
        "code_refus": trace.refusal_code,
        "rang": rank,
        "source_k": (rank is not None) if case.sources else None,
        "source_top3": (rank is not None and rank <= 3) if case.sources else None,
        "mrr": (1 / rank if rank else 0.0) if case.sources else None,
        "refus_ok": refused == case.refused,
        "citations_ok": all(c["verified"] for c in trace.citations) if answered else None,
        "source_citee": None,
        "fidele": None,
        "toutes_citees": None,
        "completude": None,
        "piege_ok": None,
        "sans_recopie": None,
        "recopies": [],
        "non_appuyees": [],
        "commentaire": "",
        "cout_usd": trace.cost_usd or 0.0,
        "trace_id": trace.trace_id,
    }
    if answered:
        if case.sources:
            cited = [(c["source_number"], c["label"],
                      trace.passages[c["source_number"] - 1].document_id)
                     for c in trace.citations if c["chunk_id"]]
            row["source_citee"] = first_match_rank(case, cited) is not None
        copies = verbatim_copies(trace.answer, [p.text for p in trace.sent_passages])
        row["recopies"] = copies
        row["sans_recopie"] = not copies
    if answered and judgement is not None:
        row["fidele"] = judgement.support_rate
        row["toutes_citees"] = judgement.citation_rate
        row["non_appuyees"] = judgement.unsupported
        row["completude"] = judgement.completeness
        row["piege_ok"] = judgement.trap_avoided if case.trap else None
        row["commentaire"] = judgement.comment
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Moyenne de chaque indicateur sur les cas où il a un sens.

    Returns:
        ``{clé: valeur entre 0 et 1, ou None si aucun cas ne la mesure}``,
        plus ``questions`` et ``cout_usd``.
    """
    summary: dict[str, float | None] = {}
    for key, _ in METRICS:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        summary[key] = round(sum(values) / len(values), 4) if values else None
    summary["questions"] = len(rows)
    summary["cout_usd"] = round(sum(r.get("cout_usd") or 0.0 for r in rows), 4)
    return summary


def compare(
    current: dict[str, float | None],
    reference: dict[str, float | None],
    *,
    tolerance: float = 0.0,
) -> list[str]:
    """Barrière de qualité : indicateurs en recul par rapport à la référence.

    Seuls les indicateurs mesurés des deux côtés sont comparés (une
    référence en recherche seule ne dit rien de la fidélité).

    Args:
        current: résumé de la version testée.
        reference: résumé de la version de référence.
        tolerance: recul accepté, en points (0,02 = 2 points). Les notes du
            correcteur varient un peu d'un passage à l'autre : une petite
            tolérance évite de bloquer sur du bruit.

    Returns:
        Une phrase par indicateur en recul ; liste vide = barrière franchie.
    """
    regressions = []
    for key, label in METRICS:
        now, before = current.get(key), reference.get(key)
        if now is None or before is None:
            continue
        if now < before - tolerance:
            regressions.append(f"{label} : {before:.2f} -> {now:.2f}")
    return regressions


def format_summary(
    current: dict[str, float | None], reference: dict[str, float | None] | None = None
) -> str:
    """Tableau lisible des indicateurs, avec la référence si elle existe."""
    lines = [f"{'indicateur':<32} {'valeur':>8}" + (f" {'référence':>10}" if reference else "")]
    for key, label in METRICS:
        value = current.get(key)
        shown = "-" if value is None else f"{value:.2f}"
        line = f"{label:<32} {shown:>8}"
        if reference:
            before = reference.get(key)
            if before is not None and value is not None:
                arrow = "↑" if value > before else "↓" if value < before else "="
                line += f" {before:>9.2f} {arrow}"
            else:
                line += f" {'-':>9}"
        lines.append(line)
    lines.append(f"{'questions':<32} {current.get('questions', 0):>8}")
    lines.append(f"{'coût estimé ($)':<32} {current.get('cout_usd') or 0:>8.4f}")
    return "\n".join(lines)
