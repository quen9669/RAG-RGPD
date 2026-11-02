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


def main() -> None:
    st.set_page_config(page_title="Assistant RGPD", page_icon="⚖️")
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
        if st.button("Nouvelle conversation"):
            st.session_state.history = []
            st.session_state.total = 0.0
            st.rerun()
        st.metric("Coût de la session", f"{st.session_state.get('total', 0.0):.3f} $")

    history: list[tuple[str, str | Reply]] = st.session_state.setdefault("history", [])
    st.session_state.setdefault("total", 0.0)
    for role, content in history:
        with st.chat_message(role):
            if isinstance(content, Reply):
                render_reply(content)
            else:
                st.markdown(content)

    question = st.chat_input("Ta question sur le RGPD…")
    if not question:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        load_env(Path(".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("Clé manquante : ajouter ANTHROPIC_API_KEY=... dans le fichier .env.")
        return
    history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche et rédaction…"):
            try:
                reply = get_engine().ask(question, model=model, agents=agents)
            except anthropic.APIError as error:  # clé refusée, réseau, crédit...
                st.error(f"Erreur de l'API Anthropic : {_api_error_message(error)}")
                history.pop()
                return
        render_reply(reply)
    history.append(("assistant", reply))
    st.session_state.total += reply.cost_usd or 0.0
    st.rerun()


main()
