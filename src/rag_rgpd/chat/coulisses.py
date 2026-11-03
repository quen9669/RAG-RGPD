"""Données du panneau « Coulisses » : comment une réponse a été fabriquée.

Fonctions pures (aucune dépendance à Streamlit), construites à partir de la
fiche de décision (``Trace``) : la page n'a plus qu'à les afficher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_rgpd.explain.trace import REFUSED, Trace


@dataclass(frozen=True)
class Stage:
    """Une étape du parcours d'une question.

    Attributes:
        title: nom de l'étape.
        status: ``ok``, ``stop`` (arrêt ici) ou ``skip`` (étape non jouée).
        summary: une phrase qui dit ce qui s'est passé.
    """

    title: str
    status: str
    summary: str


def stages(trace: Trace) -> list[Stage]:
    """Parcours de la question, étape par étape, avec son verdict."""
    threshold = float(trace.params.get("min_similarity", 0))
    sent = trace.sent_passages
    out = [Stage("1. Question", "ok", f"« {trace.question} »")]

    agents = trace.params.get("agents") or []
    researcher = [s for s in agents if s.get("agent") == "chercheur"]
    if researcher:
        out.append(Stage("2. Chercheur (agent)", "ok",
                         f"{len(researcher)} appel(s) pour reformuler la recherche"))

    rerank = " puis reclassés" if trace.params.get("reranker") else ""
    out.append(Stage(
        f"{len(out) + 1}. Recherche", "ok",
        f"{len(trace.passages)} passages comparés par le sens et par mots-clés"
        f"{rerank} ; les {len(sent)} meilleurs sont gardés"))

    passed = trace.best_similarity >= threshold
    barrier = Stage(
        f"{len(out) + 1}. Barrière 1 : assez proche ?", "ok" if passed else "stop",
        f"meilleure similarité {trace.best_similarity:.2f} "
        f"{'≥' if passed else '<'} seuil {threshold:.2f}"
        + ("" if passed else " : refus sans appeler le modèle"))
    out.append(barrier)
    if not passed:
        return out

    out.append(Stage(f"{len(out) + 1}. Rédaction", "ok",
                     f"{trace.llm_model or 'modèle'} lit les {len(sent)} passages et rédige"
                     + (" (réponse relue dans le cache)" if trace.from_cache else "")))

    verifier = [s for s in agents if s.get("agent") == "vérificateur"]
    if verifier:
        failed = [s for s in verifier if not s.get("ok", True)]
        out.append(Stage(f"{len(out) + 1}. Vérificateur (agent)",
                         "stop" if trace.refusal_code == "verification_echouee" else "ok",
                         f"{len(verifier)} contrôle(s), {len(failed)} réponse(s) refusée(s)"))

    bad = [c for c in trace.citations if not c["verified"]]
    if trace.citations:
        out.append(Stage(f"{len(out) + 1}. Barrière 3 : citations vraies ?",
                         "stop" if bad else "ok",
                         f"{len(trace.citations) - len(bad)}/{len(trace.citations)} "
                         "citations retrouvées mot pour mot dans le passage"))
    out.append(Stage(
        f"{len(out) + 1}. Décision", "stop" if trace.decision == REFUSED else "ok",
        f"refus ({trace.refusal_code}) : {trace.reason}" if trace.decision == REFUSED
        else "réponse affichée avec ses sources"))
    return out


def passage_rows(trace: Trace) -> list[dict[str, Any]]:
    """Lignes du tableau des passages retrouvés (gardés puis écartés)."""
    threshold = float(trace.params.get("min_similarity", 0))
    rows = []
    for p in trace.passages:
        if p.pinned:
            note = "épinglé (article cité)"
        elif p.rerank_score is not None:
            note = f"{p.rerank_score:.3f}"
        else:
            note = ""
        rows.append({
            "Rang": p.rank,
            "Passage": p.heading.split(" > ")[-1] + (f" (p. {p.pages})" if p.pages else ""),
            "Document": p.document_id,
            "Similarité": p.similarity,
            "Note du reclasseur": note,
            "Envoyé au modèle": "oui" if p.sent else "non (écarté)",
            "Cité": ", ".join(f"[{n}]" for n in p.cited_as),
            "Sous le seuil": p.similarity < threshold,
        })
    return rows


def agent_rows(trace: Trace) -> list[dict[str, Any]]:
    """Appels des agents (étape 6), vide sans l'option agents."""
    return [{"Agent": s.get("agent", ""), "Modèle": s.get("model", ""),
             "Tour": s.get("tour", ""), "OK": s.get("ok", ""),
             "Coût ($)": round(s.get("cost_usd") or 0.0, 5)}
            for s in trace.params.get("agents") or []]


def timing_text(trace: Trace) -> str:
    """Durées et tokens en une phrase."""
    t, u = trace.timings, trace.usage
    text = (f"recherche {t.get('search', 0):.1f} s, modèle {t.get('llm', 0):.1f} s, "
            f"total {t.get('total', 0):.1f} s")
    if u:
        text += f" · tokens : {u.get('input_tokens', 0)} en entrée, {u.get('output_tokens', 0)} en sortie"
    return text
