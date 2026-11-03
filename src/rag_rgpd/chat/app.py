"""Page Streamlit du chat. Lancée par ``uv run rag-chat``."""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import streamlit as st

from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.cli import (
    _api_error_message,
    load_env,
    make_embedder,
    make_generator,
    make_reranker,
)
from rag_rgpd.answering.generator import DEFAULT_MODEL
from rag_rgpd.chat.coulisses import agent_rows, passage_rows, stages, timing_text
from rag_rgpd.chat.engine import ChatEngine, Reply
from rag_rgpd.explain.trace import DEFAULT_TRACE_PATH
from rag_rgpd.retrieval.embeddings import DEFAULT_MODEL as EMBEDDING_MODEL
from rag_rgpd.retrieval.index import Searcher
from rag_rgpd.retrieval.rerank import DEFAULT_MODEL as RERANK_MODEL

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]
DISCLAIMER = "Cet assistant aide à l'analyse et ne remplace pas un avis juridique."


@st.cache_resource(show_spinner="Chargement de l'index et des modèles (1re fois : ~1 min)…")
def get_engine() -> ChatEngine:
    load_env(Path(".env"))
    searcher = Searcher(Path("data/index"), make_embedder(EMBEDDING_MODEL),
                        reranker=make_reranker(RERANK_MODEL))
    cache = ResponseCache()
    return ChatEngine(searcher, lambda model: make_generator(model, cache),
                      trace_path=DEFAULT_TRACE_PATH)


def render_reply(reply: Reply) -> None:
    st.markdown(reply.text)
    if reply.refused and reply.reason:
        st.caption(f"Raison du refus : {reply.reason}.")
    if reply.sources:
        with st.expander(f"Sources ({len(reply.sources)})"):
            for s in reply.sources:
                title = f"**[{s.number}] {s.heading}**" + (f", {s.pages}" if s.pages else "")
                st.markdown(f"{title}  \n[{s.url}]({s.url})")
                for quote in s.quotes:
                    st.markdown(f"> {quote}")
    cost = "" if reply.cost_usd is None else f" · coût ≈ {reply.cost_usd:.3f} $"
    st.caption(f"{DISCLAIMER}{cost} · fiche : `{reply.trace_id}`")


ICONS = {"ok": "✅", "stop": "⛔", "skip": "⏭️"}


def _label(reply: Reply, index: int) -> str:
    trace = getattr(reply, "trace", None)
    return trace.question[:60] if trace else f"Réponse {index + 1}"


def render_coulisses(reply: Reply) -> None:
    """Panneau « Coulisses » : comment cette réponse a été fabriquée."""
    # getattr : une réponse gardée en mémoire avant une mise à jour n'a pas de fiche.
    trace = getattr(reply, "trace", None)
    if trace is None:
        st.info("Pas de fiche pour cette réponse.")
        return
    st.subheader("Étapes")
    for stage in stages(trace):
        st.markdown(f"{ICONS[stage.status]} **{stage.title}**  \n{stage.summary}")
    st.subheader("Passages retrouvés")
    st.caption("Classés par pertinence. Seuls ceux marqués « oui » sont lus par le modèle.")
    rows = passage_rows(trace)
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.bar_chart({"Similarité": {f"{r['Rang']}": r["Similarité"] for r in rows}},
                 y_label="Similarité avec la question", x_label="Rang")
    agents = agent_rows(trace)
    if agents:
        st.subheader("Agents")
        st.dataframe(agents, hide_index=True, use_container_width=True)
    if trace.request:
        with st.expander("Texte envoyé au modèle"):
            st.code(trace.request.get("system", ""), language=None, wrap_lines=True)
            st.caption(f"+ {len(trace.sent_passages)} passages et la question.")
    if trace.raw_text:
        with st.expander("Réponse brute du modèle"):
            st.code(trace.raw_text, language=None, wrap_lines=True)
    if trace.citations:
        with st.expander("Citations vérifiées"):
            for c in trace.citations:
                mark = "✅" if c["verified"] else "❌"
                st.markdown(f"{mark} [{c['source_number']}] {c['label']}  \n> {c['cited_text']}")
    st.caption(timing_text(trace) + f" · fiche `{trace.trace_id}`")


def main() -> None:
    st.set_page_config(page_title="Assistant RGPD", page_icon="⚖️", layout="wide")
    st.title("Assistant RGPD")
    st.caption("Pose une question sur la protection des données : l'assistant "
               "répond avec ses sources, ou dit « je ne sais pas ».")

    with st.sidebar:
        st.header("Réglages")
        model = st.selectbox("Modèle", MODELS, index=MODELS.index(DEFAULT_MODEL)
                             if DEFAULT_MODEL in MODELS else 0,
                             help="Haiku est le moins cher, Opus le plus cher.")
        agents = st.toggle("Agents (plus fiable, plus cher)", value=False,
                           help="Chercheur + rédacteur + vérificateur : plusieurs appels par question.")
        behind = st.toggle("Afficher les coulisses", value=True,
                           help="Montre à côté du chat comment chaque réponse est fabriquée.")
        if st.button("Nouvelle conversation"):
            st.session_state.history = []
            st.session_state.total = 0.0
            st.rerun()
        st.metric("Coût de la session", f"{st.session_state.get('total', 0.0):.3f} $")

    history: list[tuple[str, str | Reply]] = st.session_state.setdefault("history", [])
    st.session_state.setdefault("total", 0.0)

    question = st.chat_input("Ta question sur le RGPD…")
    if question:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            load_env(Path(".env"))
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("Clé manquante : ajouter ANTHROPIC_API_KEY=... dans le fichier .env.")
        else:
            with st.spinner("Recherche et rédaction…"):
                try:
                    reply = get_engine().ask(question, model=model, agents=agents)
                except anthropic.APIError as error:  # clé refusée, réseau, crédit...
                    st.error(f"Erreur de l'API Anthropic : {_api_error_message(error)}")
                else:
                    history.append(("user", question))
                    history.append(("assistant", reply))
                    st.session_state.total += reply.cost_usd or 0.0
                    st.session_state.pop("picked", None)
                    st.rerun()

    left, right = st.columns([3, 2]) if behind else (st.container(), None)
    with left:
        for role, content in history:
            with st.chat_message(role):
                if isinstance(content, Reply):
                    render_reply(content)
                else:
                    st.markdown(content)
    if right is not None:
        with right:
            st.header("Coulisses")
            replies = [c for _, c in history if isinstance(c, Reply)]
            if not replies:
                st.caption("Pose une question : ici s'affichera, étape par étape, "
                           "comment la réponse est fabriquée.")
            else:
                last = len(replies) - 1
                picked = st.selectbox(
                    "Réponse à expliquer", range(len(replies)), index=last,
                    key="picked", format_func=lambda i: _label(replies[i], i))
                render_coulisses(replies[picked])


main()
