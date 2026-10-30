"""Outils partagés par les tests : un faux serveur web, sans réseau."""

from collections.abc import Callable

import httpx
import pytest

from rag_rgpd.ingestion.http import PoliteClient

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def make_client() -> Callable[[Handler], PoliteClient]:
    """Fabrique un ``PoliteClient`` branché sur un faux serveur.

    Pourquoi : les tests ne doivent dépendre ni d'Internet ni de l'état des
    sites réels ; ``httpx.MockTransport`` répond à la place du serveur.
    Le délai est à zéro pour que les tests restent rapides.
    """

    def factory(handler: Handler) -> PoliteClient:
        return PoliteClient(
            client=httpx.Client(transport=httpx.MockTransport(handler)), min_delay=0
        )

    return factory


# --- Niveaux de tests (étape 4 bis) -----------------------------------------
#
# 1. Rapides (par défaut) : faux modèle d'embedding, faux LLM. Quelques
#    secondes, gratuits, lancés à chaque modification.
# 2. « index » : le vrai index data/index/ et BGE-M3, sans LLM. Gratuits mais
#    lents (environ une minute pour charger BGE-M3) : lancés avec `--index`.
# 3. « llm » : le vrai Claude. Payants : lancés seulement avec `--llm`.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--index", action="store_true", default=False,
        help="lance aussi les tests sur le vrai index (lents, gratuits)",
    )
    parser.addoption(
        "--llm", action="store_true", default=False,
        help="lance aussi les tests qui appellent vraiment Claude (coûte de l'argent)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_llm = config.getoption("--llm")
    run_index = config.getoption("--index")
    skip_llm = pytest.mark.skip(reason="appelle Claude (payant) : lancer avec `pytest --llm`")
    skip_index = pytest.mark.skip(reason="vrai index, lent : lancer avec `pytest --index`")
    for item in items:
        if "llm" in item.keywords:
            if not run_llm:
                item.add_marker(skip_llm)
        elif "index" in item.keywords and not run_index:
            item.add_marker(skip_index)
