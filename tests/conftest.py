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
