"""Client HTTP « poli » : respecte robots.txt et espace les requêtes."""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = (
    "rag-rgpd/0.1 (projet d'apprentissage non commercial; "
    "+https://github.com/quen9669/RAG-RGPD)"
)
"""Identifie le programme auprès des serveurs, pour qu'ils sachent qui les
interroge et pourquoi (bonne pratique de moissonnage)."""


class RobotsDisallowedError(RuntimeError):
    """Levée quand le robots.txt du site interdit l'URL demandée."""


class PoliteClient:
    """Enveloppe autour de ``httpx.Client`` qui moissonne poliment.

    Pour chaque site (hôte), le client :

    1. lit une seule fois son ``robots.txt`` et refuse les URL interdites ;
    2. attend au moins ``min_delay`` secondes entre deux requêtes.

    Args:
        client: client httpx à utiliser. Les tests en passent un branché
            sur un faux serveur (``httpx.MockTransport``).
        min_delay: pause minimale, en secondes, entre deux requêtes vers
            le même hôte.
        sleep: fonction d'attente, remplaçable dans les tests pour ne pas
            ralentir la suite de tests.
        clock: horloge monotone, remplaçable dans les tests.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        min_delay: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        self._min_delay = min_delay
        self._sleep = sleep
        self._clock = clock
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    def get(self, url: str, params: dict[str, str | int] | None = None) -> httpx.Response:
        """Télécharge ``url`` en respectant robots.txt et le délai par hôte.

        Args:
            url: adresse à télécharger.
            params: paramètres de requête (``?cle=valeur``) éventuels.

        Returns:
            La réponse HTTP, dont le statut a été vérifié (2xx).

        Raises:
            RobotsDisallowedError: si robots.txt interdit l'URL.
            httpx.HTTPStatusError: si le serveur répond par une erreur.
        """
        if not self._allowed(url):
            raise RobotsDisallowedError(f"robots.txt interdit : {url}")
        response = self._throttled_get(url, params=params)
        response.raise_for_status()
        return response

    def close(self) -> None:
        """Ferme les connexions réseau ouvertes."""
        self._client.close()

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _allowed(self, url: str) -> bool:
        """Indique si robots.txt autorise notre programme à lire ``url``."""
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._robots:
            parser = RobotFileParser()
            response = self._throttled_get(f"{host}/robots.txt")
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                # Pas de robots.txt (404...) : la convention est que tout est permis.
                parser.parse([])
            self._robots[host] = parser
        return self._robots[host].can_fetch(USER_AGENT, url)

    def _throttled_get(
        self, url: str, params: dict[str, str | int] | None = None
    ) -> httpx.Response:
        """Fait la requête après avoir attendu le délai minimal pour cet hôte."""
        host = urlsplit(url).netloc
        last = self._last_request.get(host)
        if last is not None:
            remaining = self._min_delay - (self._clock() - last)
            if remaining > 0:
                self._sleep(remaining)
        try:
            return self._client.get(url, params=params)
        finally:
            self._last_request[host] = self._clock()
