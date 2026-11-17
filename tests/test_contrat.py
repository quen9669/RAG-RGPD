"""Contrat commun : /health et /vitrine.json (RAG-08)."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from rag_rgpd import contrat


def test_health_degrade_sans_donnees(tmp_path) -> None:
    out = contrat.health(tmp_path / "absent.jsonl", tmp_path / "absent")
    assert out["etat"] == "degrade" and out["code"] == "EXP-03"
    assert {c["nom"] for c in out["controles"]} == {"manifeste", "index"}


def test_health_en_service(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("{}\n")
    (tmp_path / "index").mkdir()
    assert contrat.health(manifest, tmp_path / "index")["etat"] == "en_service"


def test_vitrine_compte_les_documents_et_la_fidelite(tmp_path) -> None:
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"a":1}\n{"a":2}\n')
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"summary": {"questions": 52, "fidele": 0.9688}}))
    out = contrat.vitrine(manifest, tmp_path / "pas-d-index", tmp_path / "x.yaml", ref)
    valeurs = {c["libelle"]: c["valeur"] for c in out["chiffres"]}
    assert valeurs == {"Documents sources": 2, "Fidélité (52 questions)": 0.9688}
    assert out["ligne"] == "ai"


def test_serveur_repond_et_refuse_le_reste() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), contrat.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200 and json.loads(resp.read())["app"] == "rag-rgpd"
        conn.request("GET", "/autre")
        assert conn.getresponse().status == 404
    finally:
        server.shutdown()


def _post(port: int, corps: bytes) -> tuple[int, bytes]:
    conn = HTTPConnection("127.0.0.1", port)
    conn.request("POST", "/api/ask", body=corps, headers={"Content-Length": str(len(corps))})
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_api_ask_repond_avec_sources_et_version(monkeypatch) -> None:
    from types import SimpleNamespace

    source = SimpleNamespace(number=1, heading="Article 17", pages="", url="https://x/rgpd",
                             quotes=["droit à l'effacement"])
    reply = SimpleNamespace(text="Oui [1].", refused=False, reason="", sources=[source],
                            cost_usd=0.0, trace_id="t1")
    monkeypatch.setattr(contrat, "_engine", SimpleNamespace(ask=lambda q, model: reply))
    monkeypatch.setattr(contrat, "versions", lambda: {"https://x/rgpd": "2026-10-30"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), contrat.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, corps = _post(server.server_port, b'{"question": "Qu\\u2019est-ce que l\\u2019effacement ?"}')
        data = json.loads(corps)
        assert status == 200 and data["texte"] == "Oui [1]."
        assert data["sources"][0]["version"] == "2026-10-30" and data["sources"][0]["verifie"]
        assert _post(server.server_port, b"pas du json")[0] == 400
        assert _post(server.server_port, b'{"question": "  "}')[0] == 400
    finally:
        server.shutdown()
