"""Appel de DeepSeek pour les petits agents à sortie JSON.

DeepSeek propose une API au format OpenAI (``/chat/completions``). Les agents
de l'étape 6 (chercheur, vérificateur) et le correcteur de ``rag-eval``
parlent à un client qui ressemble à celui d'Anthropic
(``client.beta.messages.create(...)``) : cette classe traduit la requête et
renvoie un vrai ``BetaMessage``, si bien que le reste du code (cache local,
coût, lecture du JSON) ne change pas.

Différences à connaître :

- DeepSeek ne sait pas imposer un schéma JSON, seulement « du JSON valide »
  (``json_object``) : le schéma est donc recopié dans la consigne, et le code
  des agents vérifie les champs reçus.
- Les citations natives d'Anthropic n'existent pas ici : DeepSeek ne sert pas
  à rédiger les réponses citées (le rédacteur reste Claude).
- Les serveurs de DeepSeek sont hors de l'Union européenne : les passages et
  la réponse relue lui sont envoyés (usage privé, voir la décision 0013).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import httpx
from anthropic.types.beta import BetaMessage

BASE_URL = "https://api.deepseek.com"
KEY_VARIABLE = "DEEPSEEK_API_KEY"
RETRIES = 2


class DeepSeekError(ValueError):
    """Échec de l'appel (clé, réseau, réponse vide).

    Sous-classe de ``ValueError`` : les agents savent déjà la traiter (le
    vérificateur bloque la réponse, le chercheur retombe sur la recherche
    simple), sans jamais inventer un résultat.
    """


def is_deepseek(model: str) -> bool:
    """``True`` si ce nom de modèle est servi par DeepSeek."""
    return model.startswith("deepseek")


class DeepSeekClient:
    """Faux « client Anthropic » branché sur l'API de DeepSeek."""

    def __init__(self, *, api_key: str | None = None, http: httpx.Client | None = None,
                 base_url: str = BASE_URL) -> None:
        """Prépare le client.

        Args:
            api_key: clé ; par défaut la variable d'environnement
                ``DEEPSEEK_API_KEY`` (fichier ``.env``).
            http: client HTTP (un faux transport dans les tests).
            base_url: adresse de l'API.
        """
        self._key = api_key
        self._http = http or httpx.Client(timeout=120)
        self._base_url = base_url
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> BetaMessage:
        key = self._key or os.environ.get(KEY_VARIABLE, "")
        if not key:
            raise DeepSeekError(f"clé manquante : ajouter {KEY_VARIABLE}=... dans le fichier .env")
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
        last = "réponse vide"
        for _ in range(RETRIES):
            try:
                reply = self._http.post(
                    f"{self._base_url}/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {key}"})
                reply.raise_for_status()
            except httpx.HTTPError as error:
                raise DeepSeekError(f"appel à DeepSeek impossible : {error}") from error
            data = reply.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if text:  # DeepSeek renvoie parfois du blanc quand le JSON est invalide
                return _to_message(data, text, request["model"])
        raise DeepSeekError(f"DeepSeek : {last}")


def _to_message(data: dict[str, Any], text: str, model: str) -> BetaMessage:
    """Convertit la réponse DeepSeek en ``BetaMessage`` (tokens compris)."""
    usage = data.get("usage") or {}
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = int(usage.get("prompt_tokens") or 0) - hit
    return BetaMessage.model_validate({
        "id": data.get("id", "deepseek"), "type": "message", "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": int(miss), "output_tokens": int(usage.get("completion_tokens") or 0),
                  "cache_read_input_tokens": hit},
    })


def make_client(model: str) -> Any:
    """Client adapté au modèle : DeepSeek pour ``deepseek-*``, Anthropic sinon."""
    if is_deepseek(model):
        return DeepSeekClient()
    import anthropic

    return anthropic.Anthropic()
