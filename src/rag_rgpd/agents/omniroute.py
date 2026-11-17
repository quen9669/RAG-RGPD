"""Appel du LLM par OmniRoute, la seule porte d'entrée de l'IA (règles R-05, R-06).

OmniRoute expose une API au format OpenAI (``/chat/completions``). Le code
(rédacteur, chercheur, vérificateur, correcteur de ``rag-eval``) parle à un
client qui ressemble à celui d'Anthropic (``client.beta.messages.create(...)``) :
cette classe traduit la requête et renvoie un vrai ``BetaMessage``, si bien que
le reste du code (cache local, coût, lecture du JSON) ne change pas.

Configuration (variables d'environnement, jamais de secret dans le dépôt) :

- ``OMNIROUTE_BASE_URL`` : adresse de l'API (``http://omniroute:20128/v1`` sur
  le réseau ``llm-net``) ;
- ``OMNIROUTE_API_KEY`` : clé ``rag-rgpd``, injectée par le coffre.

Différences à connaître :

- On ne peut pas imposer un schéma JSON, seulement « du JSON valide »
  (``json_object``) : le schéma est donc recopié dans la consigne, et le code
  appelant vérifie les champs reçus.
- Les citations natives d'Anthropic n'existent pas ici : le rédacteur renvoie
  lui-même, pour chaque phrase, le passage et l'extrait cités, et le code les
  vérifie mot pour mot (voir ``answering/generator.py``).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import httpx
from anthropic.types.beta import BetaMessage

BASE_URL = "http://omniroute:20128/v1"
BASE_URL_VARIABLE = "OMNIROUTE_BASE_URL"
KEY_VARIABLE = "OMNIROUTE_API_KEY"
RETRIES = 2


class OmniRouteError(ValueError):
    """Échec de l'appel (clé, réseau, réponse vide).

    Sous-classe de ``ValueError`` : les agents savent déjà la traiter (le
    vérificateur bloque la réponse, le chercheur retombe sur la recherche
    simple), sans jamais inventer un résultat.
    """


class OmniRouteClient:
    """Faux « client Anthropic » branché sur OmniRoute."""

    def __init__(self, *, api_key: str | None = None, http: httpx.Client | None = None,
                 base_url: str | None = None) -> None:
        """Prépare le client.

        Args:
            api_key: clé ; par défaut la variable d'environnement
                ``OMNIROUTE_API_KEY``.
            http: client HTTP (un faux transport dans les tests).
            base_url: adresse de l'API ; par défaut ``OMNIROUTE_BASE_URL``.
        """
        self._key = api_key
        self._http = http or httpx.Client(timeout=120)
        self._base_url = base_url
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> BetaMessage:
        key = self._key or os.environ.get(KEY_VARIABLE, "")
        if not key:
            raise OmniRouteError(f"clé manquante : la variable {KEY_VARIABLE} n'est pas définie")
        base_url = (self._base_url or os.environ.get(BASE_URL_VARIABLE) or BASE_URL).rstrip("/")
        system = request.get("system", "")
        schema = (request.get("output_config") or {}).get("format", {}).get("schema")
        if schema is not None:
            system += (
                "\n\nRéponds uniquement par un objet JSON valide (mot-clé json) qui respecte "
                "ce schéma, sans texte autour :\n" + json.dumps(schema, ensure_ascii=False)
            )
        body = {
            "model": request["model"],
            "messages": [{"role": "system", "content": system}, *request["messages"]],
            "response_format": {"type": "json_object"},
            "max_tokens": min(int(request.get("max_tokens", 8000)), 8000),
            "temperature": 0,
        }
        for _ in range(RETRIES):
            try:
                reply = self._http.post(
                    f"{base_url}/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {key}"})
                reply.raise_for_status()
            except httpx.HTTPError as error:
                raise OmniRouteError(f"appel à OmniRoute impossible : {error}") from error
            data = reply.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text:  # certains modèles renvoient du blanc quand le JSON est invalide
                return _to_message(data, text, request["model"])
        raise OmniRouteError("OmniRoute : réponse vide")


def _to_message(data: dict[str, Any], text: str, model: str) -> BetaMessage:
    """Convertit la réponse OmniRoute en ``BetaMessage`` (tokens compris)."""
    usage = data.get("usage") or {}
    return BetaMessage.model_validate({
        "id": data.get("id", "omniroute"), "type": "message", "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": int(usage.get("prompt_tokens") or 0),
                  "output_tokens": int(usage.get("completion_tokens") or 0)},
    })


def make_client(model: str = "") -> OmniRouteClient:
    """Client OmniRoute (le nom du modèle est choisi par la requête)."""
    return OmniRouteClient()
