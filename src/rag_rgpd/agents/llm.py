"""Appel d'un agent à sortie JSON imposée, avec cache et mesure du coût.

Le chercheur et le vérificateur sont de petits agents : une consigne, un
message, une réponse au format JSON défini par un schéma. Ce module met en
commun ce qu'ils ont en commun (client, cache local, tokens, coût) pour que
chaque agent ne contienne que sa consigne et sa logique.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from anthropic.types.beta import BetaMessage

from rag_rgpd.agents.deepseek import make_client
from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import FALLBACK_MODELS, usage_dict
from rag_rgpd.explain.cost import estimate_cost


@dataclass(frozen=True)
class AgentCall:
    """Résultat d'un appel d'agent.

    Attributes:
        data: JSON renvoyé par le modèle, déjà décodé.
        step: ligne de suivi pour la fiche de décision (agent, modèle,
            tokens, coût estimé ; 0 $ si la réponse vient du cache local).
    """

    data: dict[str, Any]
    step: dict[str, Any]


class JsonAgent:
    """Un agent = un modèle + une consigne + un schéma de sortie."""

    def __init__(
        self,
        name: str,
        system: str,
        schema: dict[str, Any],
        *,
        model: str,
        client: Any = None,
        effort: str = "medium",
        cache: ResponseCache | None = None,
    ) -> None:
        """Prépare l'agent.

        Args:
            name: nom affiché dans la fiche de décision (« chercheur »...).
            system: consigne de l'agent.
            schema: schéma JSON de sa réponse.
            model: modèle Claude utilisé.
            client: client ``anthropic.Anthropic`` (un faux dans les tests).
            effort: profondeur de réflexion.
            cache: cache local des réponses (évite de repayer un même appel).
        """
        if client is None:
            client = make_client(model)
        self.name = name
        self.system = system
        self.schema = schema
        self.model = model
        self.effort = effort
        self._client = client
        self._cache = cache

    def call(self, message: str) -> AgentCall:
        """Envoie ``message`` à l'agent.

        Raises:
            ValueError: si le modèle décline la demande ou renvoie un JSON
                illisible. L'appelant décide quoi faire (jamais de réponse
                inventée à sa place).
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": self.system,
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self.schema},
            },
            "messages": [{"role": "user", "content": message}],
        }
        if self.model in FALLBACK_MODELS:
            request |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        cached = key = None
        if self._cache is not None:
            key = self._cache.key(request)
            cached = self._cache.get(key)
        if cached is not None:
            response: Any = BetaMessage.model_validate(cached)
        else:
            response = self._client.beta.messages.create(**request)
            if self._cache is not None:
                self._cache.put(key, response.model_dump(mode="json"))

        if response.stop_reason == "refusal":
            raise ValueError(f"l'agent {self.name} a décliné la demande")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"réponse illisible de l'agent {self.name} : {error}") from error
        usage = usage_dict(response)
        model = getattr(response, "model", self.model)
        cost = 0.0 if cached is not None else (estimate_cost(model, usage) or 0.0)
        step = {"agent": self.name, "model": model, "usage": usage,
                "cost_usd": cost, "from_cache": cached is not None}
        return AgentCall(data, step)
