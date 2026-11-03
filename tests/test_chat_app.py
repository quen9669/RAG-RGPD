"""Test de fumée de la page Streamlit (sans navigateur ni réseau)."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from rag_rgpd.chat import app as chat_app


def test_page_vide_affiche_les_coulisses() -> None:
    at = AppTest.from_file(chat_app.__file__, default_timeout=60).run()

    assert not at.exception
    assert any("Coulisses" in h.value for h in at.header)


def test_panneau_coulisses_rend_une_reponse(tmp_path) -> None:
    def script() -> None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(TESTS)))
        from fakes import CHUNKS, FakeEmbedder, FakeGenerator

        import rag_rgpd.chat.app as app_module
        from rag_rgpd.chat.engine import ChatEngine
        from rag_rgpd.retrieval.index import Searcher, build_index
        build_index(CHUNKS, FakeEmbedder(), Path(TMP) / "index")
        engine = ChatEngine(Searcher(Path(TMP) / "index", FakeEmbedder()),
                            lambda model: FakeGenerator())
        reply = engine.ask("droit à l'effacement", model="x", min_similarity=0.0)
        app_module.render_coulisses(reply)

    import inspect
    from pathlib import Path
    source = inspect.getsource(script).replace("TESTS", repr(str(Path(__file__).parent))) \
        .replace("TMP", repr(str(tmp_path)))
    at = AppTest.from_string("\n".join(line[4:] for line in source.splitlines()[1:]),
                             default_timeout=60).run()

    assert not at.exception
    assert any("Étapes" in s.value for s in at.subheader)


def test_ancienne_reponse_sans_fiche_ne_plante_pas() -> None:
    """Après une mise à jour, la mémoire de la page peut contenir d'anciennes réponses."""
    script = '''
import streamlit as st
from types import SimpleNamespace
from rag_rgpd.chat.app import _label, render_coulisses
old = SimpleNamespace(text="x")
st.write(_label(old, 0))
render_coulisses(old)
'''
    at = AppTest.from_string(script, default_timeout=60).run()

    assert not at.exception
