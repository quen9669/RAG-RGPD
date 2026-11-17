"""Contrat commun des apps du référentiel : ``/health`` et ``/vitrine.json``.

Serveur minimal (bibliothèque standard) : ``/health`` et ``/vitrine.json``
répondent sans modèle ni clé, même si le reste de l'app est arrêté. Il sert
aussi ``POST /api/ask`` (question en JSON, réponse citée) pour le front React :
le moteur ne se charge qu'à la première question. Port 8000 par convention
(architecture §6). Lancé par ``uv run rag-contrat``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from rag_rgpd import __version__
from rag_rgpd.retrieval.index import TABLE_NAME

APP, CODE = "rag-rgpd", "EXP-03"
MANIFEST = Path("data/manifest.jsonl")
INDEX = Path("data/index")
EXPEDITION = Path("expedition.yaml")
REFERENCE = Path("evals/reference.json")


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_chunks(index: Path = INDEX) -> int | None:
    """Nombre de morceaux de l'index, ou ``None`` s'il est absent."""
    if not index.is_dir():
        return None
    import lancedb  # import tardif : lourd, inutile si l'index est absent

    return lancedb.connect(index).open_table(TABLE_NAME).count_rows()


def health(manifest: Path = MANIFEST, index: Path = INDEX) -> dict:
    """État de l'app : ``en_service`` si le manifeste et l'index sont lisibles."""
    controles = []
    for nom, ok in (("manifeste", manifest.is_file()), ("index", index.is_dir())):
        controles.append({"nom": nom, "etat": "en_service" if ok else "arrete",
                          **({} if ok else {"detail": "absent"})})
    tout = all(c["etat"] == "en_service" for c in controles)
    return {"app": APP, "code": CODE, "etat": "en_service" if tout else "degrade",
            "version": __version__, "verifie_le": _now(), "controles": controles}


def vitrine(manifest: Path = MANIFEST, index: Path = INDEX,
            expedition: Path = EXPEDITION, reference: Path = REFERENCE) -> dict:
    """Chiffres clés publics : agrégats seulement, aucune donnée personnelle."""
    exp = yaml.safe_load(expedition.read_text(encoding="utf-8")) if expedition.is_file() else {}
    chiffres = []
    if manifest.is_file():
        docs = sum(1 for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip())
        chiffres.append({"libelle": "Documents sources", "valeur": docs})
    chunks = _count_chunks(index)
    if chunks is not None:
        chiffres.append({"libelle": "Morceaux indexés", "valeur": chunks})
    if reference.is_file():
        resume = json.loads(reference.read_text(encoding="utf-8"))["summary"]
        chiffres.append({"libelle": f"Fidélité ({resume['questions']} questions)",
                         "valeur": resume["fidele"]})
    return {"app": APP, "code": CODE, "ligne": exp.get("ligne", "ai"),
            "etape": exp.get("etape"), "etape_libelle": exp.get("etape_libelle"),
            "chiffres": chiffres, "maj": _now()}


ROUTES = {"/health": health, "/vitrine.json": vitrine}
MAX_BODY = 8_192  # une question tient dans quelques centaines d'octets
_engine = None


def versions(manifest: Path = MANIFEST) -> dict[str, str]:
    """Date de version (récupération) de chaque texte, par URL."""
    if not manifest.is_file():
        return {}
    lignes = (json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip())
    return {d["url"]: d["retrieved_at"][:10] for d in lignes if "url" in d and "retrieved_at" in d}


def reply_json(reply, dates: dict[str, str]) -> dict:
    """Réponse du moteur sous forme JSON : sources citées, version, vérification."""
    return {"texte": reply.text, "refus": reply.refused, "raison": reply.reason,
            "sources": [{"numero": s.number, "titre": s.heading, "pages": s.pages, "url": s.url,
                         "extraits": s.quotes, "verifie": bool(s.quotes),
                         "version": dates.get(s.url)} for s in reply.sources],
            "cout_usd": reply.cost_usd, "fiche": reply.trace_id}


def ask(question: str) -> dict:
    """Pose la question au moteur (chargé à la première demande)."""
    global _engine
    if _engine is None:
        from rag_rgpd.chat.engine import build_engine  # import tardif : modèles lourds

        _engine = build_engine()
    from rag_rgpd.answering.generator import DEFAULT_MODEL

    return reply_json(_engine.ask(question, model=DEFAULT_MODEL), versions())


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        route = ROUTES.get(self.path)
        if route is None:
            self.send_error(404)
            return
        body = json.dumps(route(), ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self.send_error(404)
            return
        try:
            taille = int(self.headers.get("Content-Length", ""))
            question = json.loads(self.rfile.read(min(taille, MAX_BODY)))["question"]
            if not isinstance(question, str) or not question.strip() or taille > MAX_BODY:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            self.send_error(400, "corps attendu : {\"question\": \"...\"}")
            return
        try:
            body = json.dumps(ask(question.strip()), ensure_ascii=False).encode()
        except Exception as erreur:  # noqa: BLE001 clé refusée, réseau, index absent : 502 lisible
            self.send_error(502, type(erreur).__name__)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:  # silence : pas de journal d'accès
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Sert /health et /vitrine.json.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
