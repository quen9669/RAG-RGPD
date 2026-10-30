"""Tests du client poli : robots.txt et délai entre requêtes."""

import httpx
import pytest

from rag_rgpd.ingestion.http import EmptyResponseError, PoliteClient, RobotsDisallowedError


def test_refuse_une_url_interdite_par_robots(make_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /prive/\n")
        return httpx.Response(200, text="ok")

    client = make_client(handler)
    assert client.get("https://exemple.lu/public/doc").text == "ok"
    with pytest.raises(RobotsDisallowedError):
        client.get("https://exemple.lu/prive/doc")


def test_sans_robots_txt_tout_est_permis(make_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="ok")

    assert make_client(handler).get("https://exemple.lu/doc").text == "ok"


def test_attend_entre_deux_requetes_vers_le_meme_hote() -> None:
    pauses: list[float] = []
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    client = PoliteClient(
        client=httpx.Client(transport=transport),
        min_delay=2.0,
        sleep=pauses.append,
        clock=lambda: 100.0,  # horloge figée : aucun temps ne s'écoule
    )
    # robots.txt (1re requête), puis le document : une pause de 2 s attendue.
    with pytest.raises(httpx.HTTPStatusError):
        client.get("https://exemple.lu/doc")
    assert pauses == [2.0]


@pytest.mark.parametrize(("status", "body"), [(202, b""), (200, b""), (202, b"<html>")])
def test_refuse_une_reponse_vide_ou_non_200(make_client, status: int, body: bytes) -> None:
    """Cas réel : EUR-Lex répond 202 sans contenu aux programmes (anti-robots)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(status, content=body)

    with pytest.raises(EmptyResponseError):
        make_client(handler).get("https://exemple.lu/doc")


def test_transmet_les_en_tetes_demandes(make_client) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text=request.headers.get("accept-language", ""))

    client = make_client(handler)
    assert client.get("https://exemple.lu/doc", headers={"Accept-Language": "fra"}).text == "fra"
