"""Cache local des réponses du LLM, pour ne pas payer deux fois le même essai.

Pendant les essais en lot (étape 4 bis), on relance souvent les mêmes
questions avec des paramètres qui ne changent rien à la requête envoyée au
modèle (par exemple un seuil plus bas qui laisse passer les mêmes passages).
Si la requête est identique au caractère près, la réponse déjà reçue est
relue sur le disque au lieu d'appeler l'API.

La clé est l'empreinte SHA-256 de la requête complète (modèle, consigne,
passages, question, options) : le moindre changement de consigne, de
passage ou de modèle donne une autre clé, donc un vrai appel.

Les réponses contiennent des extraits des documents (dont la CNPD) : le
dossier reste local, comme ``data/processed/`` (voir ``.gitignore``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path("data/cache/llm")


class ResponseCache:
    """Réponses brutes de l'API rangées par empreinte de requête."""

    def __init__(self, directory: Path = DEFAULT_CACHE_DIR) -> None:
        """Prépare le cache.

        Args:
            directory: dossier des réponses (créé à la première écriture).
        """
        self.directory = directory

    @staticmethod
    def key(request: dict[str, Any]) -> str:
        """Empreinte stable d'une requête.

        Les clés JSON sont triées : deux dictionnaires égaux donnent
        toujours la même empreinte, quel que soit l'ordre de construction.
        """
        payload = json.dumps(request, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """Réponse enregistrée pour cette clé, ou ``None``."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, response: dict[str, Any]) -> None:
        """Enregistre une réponse (format JSON de l'API)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(
            json.dumps(response, ensure_ascii=False, indent=1), encoding="utf-8"
        )
