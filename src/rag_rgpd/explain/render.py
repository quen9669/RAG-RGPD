"""Affichage lisible d'une fiche de décision dans le terminal de VS Code."""

from __future__ import annotations

import json

from rag_rgpd.answering.generator import (
    MODEL_INSUFFICIENT,
    NO_CITATION,
    NO_RELEVANT_PASSAGE,
    SAFETY_REFUSAL,
    UNVERIFIED_CITATION,
)
from rag_rgpd.explain.trace import ANSWERED, REFUSED, SEARCH_ONLY, Passage, Trace

WIDTH = 100

REFUSAL_EXPLANATIONS = {
    NO_RELEVANT_PASSAGE: "barrière 1 (avant le LLM) : aucun passage assez proche, "
                         "le modèle n'a pas été appelé",
    MODEL_INSUFFICIENT: "barrière 2 (consigne) : le modèle a jugé les passages "
                        "insuffisants et répondu « Je ne sais pas. »",
    NO_CITATION: "barrière 3 (contrôle) : le modèle a répondu sans citer aucun passage",
    UNVERIFIED_CITATION: "barrière 3 (contrôle) : une citation ne se retrouve pas dans "
                         "le passage qu'elle désigne",
    SAFETY_REFUSAL: "filtre de sécurité du modèle : il a décliné la demande",
}


def _rank(value: int | None) -> str:
    return "-" if value is None else str(value)


def _short(text: str, size: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= size else text[: size - 1] + "…"


def _passage_line(p: Passage, min_similarity: float) -> str:
    """Une ligne du tableau des passages."""
    below = "↓" if p.similarity < min_similarity else " "
    cited = ",".join(f"[{n}]" for n in p.cited_as) or ""
    where = p.label + (f" p.{p.pages}" if p.pages and not p.label.startswith("Page") else "")
    if p.pinned:
        rerank = "épinglé"
    else:
        rerank = "-" if p.rerank_score is None else f"{p.rerank_score:.3f}"
    return (f"{p.rank:>3}  {p.similarity:5.2f}{below} {p.score:7.4f} {rerank:>7}  "
            f"{_rank(p.meaning_rank):>4} {_rank(p.words_rank):>5}  "
            f"{_short(p.document_id, 22):<22} {_short(where, 30):<30} {cited}")


def _section(title: str) -> str:
    return f"\n── {title} " + "─" * max(0, WIDTH - len(title) - 4)


def format_prompt(trace: Trace) -> str:
    """Texte exact envoyé au modèle, en clair (consigne, passages, question)."""
    if not trace.request:
        return "(le modèle n'a pas été appelé)"
    lines = ["[CONSIGNE SYSTÈME]", trace.request.get("system", "")]
    for message in trace.request.get("messages", []):
        for number, block in enumerate(message.get("content", []), start=1):
            if block.get("type") == "search_result":
                lines.append(f"[PASSAGE {number}] source={block['source']}")
                lines.append(f"titre : {block['title']}")
                lines.extend(c["text"] for c in block["content"])
                lines.append("")
            elif block.get("type") == "text":
                lines.append(f"[MESSAGE] {block['text']}")
    options = {k: v for k, v in trace.request.items() if k not in ("system", "messages")}
    lines.append(f"[OPTIONS] {json.dumps(options, ensure_ascii=False)}")
    return "\n".join(lines)


def format_trace(trace: Trace, *, show_prompt: bool = False) -> str:
    """Fiche de décision complète, prête à afficher.

    Args:
        trace: la trace à présenter.
        show_prompt: ``True`` pour afficher le texte complet envoyé au
            modèle (long : tous les passages en entier).
    """
    params = trace.params
    threshold = float(params.get("min_similarity", 0))
    out = [
        "═" * WIDTH,
        f"FICHE DE DÉCISION {trace.trace_id}   ({trace.at})",
        f"Question : {trace.question}",
        "═" * WIDTH,
    ]

    out.append(_section("1. Paramètres"))
    out.append(
        f"k = {params.get('k')} passages envoyés (+{params.get('extra', 0)} regardés en plus), "
        f"seuil de similarité = {threshold:.2f}"
    )
    out.append(
        f"embedding : {trace.index.get('embedding_model', '?')} "
        f"(index de {trace.index.get('chunk_count', '?')} morceaux, "
        f"construit le {trace.index.get('built_at', '?')})"
    )
    if params.get("reranker"):
        out.append(f"reclassement : {params['reranker']} sur les {params.get('pool')} premiers")
    if params.get("pin_cited_articles"):
        out.append("articles cités par leur numéro dans la question : placés en tête")
    if trace.decision != SEARCH_ONLY:
        out.append(f"LLM : {params.get('llm_model', '?')}, effort {params.get('effort', '?')}")

    order = "reclassement" if params.get("reranker") else "fusion RRF"
    out.append(_section(f"2. Passages retrouvés (ordre du {order})"))
    out.append("  #   sim.    RRF  reclas.  sens  mots  document               endroit"
               "                        cité")
    for p in trace.sent_passages:
        out.append(_passage_line(p, threshold))
    if trace.held_back:
        out.append("  ··· écartés juste après la coupure k (non envoyés au modèle) ···")
        for p in trace.held_back:
            out.append(_passage_line(p, threshold))
    out.append("  sim. = similarité cosinus avec la question (↓ = sous le seuil) ; reclas. = "
               "note du modèle de")
    out.append("  reclassement (épinglé = article cité dans la question) ; "
               "sens / mots = rang dans")
    out.append("  chaque recherche avant fusion (- = absent des 50 premiers ; pour « mots », "
               "souvent aucun mot commun).")

    out.append(_section("3. Barrière 1 : le meilleur passage est-il assez proche ?"))
    passed = trace.best_similarity >= threshold
    verdict = "oui, on peut interroger le modèle" if passed else "non, refus sans appeler le modèle"
    sign = "≥" if passed else "<"
    out.append(f"meilleure similarité {trace.best_similarity:.2f} {sign} seuil "
               f"{threshold:.2f} : {verdict}")

    if trace.decision == SEARCH_ONLY:
        out.append("\n(Recherche seule : le modèle n'a pas été appelé.)")
    elif trace.request:
        out.append(_section("4. Texte envoyé au modèle"))
        if show_prompt:
            out.append(format_prompt(trace))
        else:
            system = trace.request.get("system", "")
            out.append(f"consigne système : {len(system)} caractères "
                       f"(« {_short(system, 70)} »)")
            out.append(f"{len(trace.sent_passages)} passages en blocs search_result "
                       "(citations activées), puis la question.")
            out.append("texte complet : relancer avec --show-prompt")

        out.append(_section("5. Réponse brute du modèle"))
        out.append(f"modèle : {trace.llm_model}" + ("  (relu dans le cache)" if trace.from_cache else ""))
        # Texte tel quel : un retour à la ligne automatique casserait les
        # paragraphes et le gras Markdown (« **...** ») renvoyés par le modèle.
        out.append(trace.raw_text or "(vide)")

        out.append(_section("6. Citations"))
        if not trace.citations:
            out.append("aucune citation")
        for c in trace.citations:
            mark = "✓ vérifiée" if c["verified"] else "✗ NON VÉRIFIÉE"
            target = f"{c['label']} ({c['chunk_id']})" if c["chunk_id"] else "passage inexistant"
            out.append(f"[{c['source_number']}] → {target}  {mark}")
            out.append(f"     « {_short(c['cited_text'], WIDTH - 10)} »")

    out.append(_section("7. Décision"))
    if trace.decision == ANSWERED:
        out.append("RÉPONSE")
        out.append(trace.answer)
    elif trace.decision == REFUSED:
        out.append(f"REFUS ({trace.refusal_code}) : {trace.reason}")
        if trace.refusal_code in REFUSAL_EXPLANATIONS:
            out.append(f"qui a décidé : {REFUSAL_EXPLANATIONS[trace.refusal_code]}")
    else:
        out.append("aucune (recherche seule)")

    out.append(_section("8. Coût"))
    t = trace.timings
    out.append(f"temps : recherche {t.get('search', 0):.2f} s, explication du classement "
               f"{t.get('ranks', 0):.2f} s, LLM {t.get('llm', 0):.2f} s, "
               f"total {t.get('total', 0):.2f} s")
    if trace.usage:
        u = trace.usage
        out.append(f"tokens : entrée {u.get('input_tokens', 0)}, sortie {u.get('output_tokens', 0)}, "
                   f"cache lu {u.get('cache_read_input_tokens', 0)}, "
                   f"cache écrit {u.get('cache_creation_input_tokens', 0)}")
    if trace.cost_usd is None:
        out.append("coût estimé : inconnu (prix du modèle absent de explain/cost.py)")
    else:
        note = " (réponse relue dans le cache, rien payé)" if trace.from_cache else ""
        out.append(f"coût estimé : {trace.cost_usd:.5f} $" + note)
    out.append("═" * WIDTH)
    return "\n".join(out)


def format_trace_list(traces: list[Trace], limit: int = 20) -> str:
    """Liste courte des dernières traces (une ligne chacune)."""
    lines = []
    for t in traces[-limit:]:
        status = t.decision if t.decision != REFUSED else f"refus ({t.refusal_code})"
        lines.append(f"{t.trace_id}  {_short(status, 32):<32} {_short(t.question, 55)}")
    return "\n".join(lines) or "(aucune trace)"
