"""Stockage versionné des fichiers bruts et manifeste de lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


def sha256_hex(content: bytes) -> str:
    """Calcule l'empreinte SHA-256 d'un contenu.

    L'empreinte change dès qu'un seul octet change : elle permet de savoir si
    une source a été modifiée depuis le dernier téléchargement, et de prouver
    plus tard sur quelle version exacte une réponse s'appuie.

    Args:
        content: contenu brut du fichier.

    Returns:
        L'empreinte en hexadécimal (64 caractères).
    """
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    """Une ligne du manifeste : la trace d'un fichier brut téléchargé.

    Attributes:
        document_id: identifiant stable du document (ex. ``"rgpd-fr"``).
        source: organisme d'origine (ex. ``"eur-lex"``, ``"cnpd"``).
        url: adresse exacte de téléchargement.
        retrieved_at: date et heure du téléchargement, en UTC, format ISO 8601.
        sha256: empreinte du contenu.
        size_bytes: taille du fichier.
        content_type: type annoncé par le serveur (ex. ``"application/pdf"``).
        path: chemin du fichier brut, relatif à la racine des données.
        licence: licence ou conditions d'usage relevées pour cette source.
    """

    document_id: str
    source: str
    url: str
    retrieved_at: str
    sha256: str
    size_bytes: int
    content_type: str
    path: str
    licence: str


class RawStore:
    """Range les fichiers bruts et tient le manifeste à jour.

    Organisation sur disque (``root`` vaut ``data/`` par défaut) ::

        data/
          manifest.jsonl                  # versionné dans Git
          raw/<source>/<AAAA-MM-JJ>/<fichier>   # NON versionné (licences)

    Le manifeste est au format JSON Lines (un objet JSON par ligne) : on ne
    fait qu'ajouter des lignes, jamais en modifier, ce qui garde l'historique
    complet de toutes les versions récupérées.

    Args:
        root: dossier racine des données.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.manifest_path = root / "manifest.jsonl"

    def entries(self) -> list[ManifestEntry]:
        """Relit toutes les lignes du manifeste, de la plus ancienne à la plus récente."""
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open(encoding="utf-8") as f:
            return [ManifestEntry(**json.loads(line)) for line in f if line.strip()]

    def latest(self, document_id: str) -> ManifestEntry | None:
        """Renvoie la dernière version connue d'un document, ou ``None``."""
        matches = [e for e in self.entries() if e.document_id == document_id]
        return matches[-1] if matches else None

    def save(
        self,
        *,
        document_id: str,
        source: str,
        url: str,
        content: bytes,
        content_type: str,
        filename: str,
        licence: str,
        retrieved_at: datetime,
    ) -> ManifestEntry | None:
        """Enregistre un fichier brut s'il est nouveau ou a changé.

        Si le contenu est identique (même empreinte) à la dernière version du
        document, rien n'est écrit : on évite les doublons tout en sachant que
        la source n'a pas bougé.

        Args:
            document_id: identifiant stable du document.
            source: organisme d'origine, utilisé comme sous-dossier.
            url: adresse de téléchargement.
            content: contenu brut reçu.
            content_type: type annoncé par le serveur.
            filename: nom du fichier à créer.
            licence: licence ou conditions d'usage.
            retrieved_at: moment du téléchargement (avec fuseau horaire).

        Returns:
            La nouvelle entrée du manifeste, ou ``None`` si rien n'a changé.
        """
        digest = sha256_hex(content)
        previous = self.latest(document_id)
        if previous is not None and previous.sha256 == digest:
            return None

        relative = Path("raw") / source / retrieved_at.date().isoformat() / filename
        if (self.root / relative).exists():
            # Deuxième version différente le même jour : on ne doit pas écraser
            # la première, sinon une ligne du manifeste pointerait vers un
            # fichier qui n'a plus la bonne empreinte. On suffixe le nom.
            relative = relative.with_stem(f"{relative.stem}-{digest[:8]}")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

        entry = ManifestEntry(
            document_id=document_id,
            source=source,
            url=url,
            retrieved_at=retrieved_at.isoformat(timespec="seconds"),
            sha256=digest,
            size_bytes=len(content),
            content_type=content_type,
            path=relative.as_posix(),
            licence=licence,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        return entry
