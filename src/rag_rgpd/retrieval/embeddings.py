"""Transformation des textes en embeddings (vecteurs qui résument le sens).

Deux textes au sens proche donnent des vecteurs proches, même s'ils
n'emploient pas les mêmes mots (« effacement » et « droit à l'oubli »). On
compare deux vecteurs par la similarité cosinus : 1 = même direction, 0 =
sans rapport. Les vecteurs sont normalisés (longueur 1), ce qui ramène le
cosinus à un simple produit scalaire.

Le modèle retenu (décision 0007) est BGE-M3, multilingue, exécuté en local :
aucun texte ne quitte la machine à cette étape et il n'y a rien à payer.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"

MAX_SEQ_LENGTH = 1024
"""Longueur maximale lue par le modèle, en tokens. BGE-M3 accepte 8 192
tokens, mais nos morceaux font au plus 1 500 caractères (environ 400 tokens
avec leur titre) : limiter la longueur réduit la mémoire utilisée sans rien
couper."""


class Embedder(Protocol):
    """Ce que la recherche attend d'un modèle d'embedding.

    Un protocole plutôt qu'une classe concrète : les tests utilisent un faux
    modèle instantané, et changer de modèle ne touche pas au reste du code.
    """

    model_name: str
    """Nom du modèle, enregistré avec l'index pour détecter un mélange."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Calcule un vecteur normalisé par texte.

        Args:
            texts: textes à encoder.

        Returns:
            Tableau ``(len(texts), dimension)`` de ``float32``, chaque ligne
            de longueur 1.
        """
        ...


class SentenceTransformerEmbedder:
    """Modèle d'embedding local chargé avec sentence-transformers.

    Le premier usage télécharge le modèle (environ 2 Go pour BGE-M3) dans
    le cache de Hugging Face ; les suivants le relisent depuis le disque.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 8) -> None:
        """Charge le modèle.

        Args:
            model_name: identifiant du modèle sur Hugging Face.
            batch_size: textes encodés à la fois ; petit pour tenir dans la
                mémoire d'un PC sans carte graphique.
        """
        # Import tardif : PyTorch met plusieurs secondes à se charger, inutile
        # de le payer pour les commandes qui n'en ont pas besoin (et les tests).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._model.max_seq_length = MAX_SEQ_LENGTH

    def embed(self, texts: list[str]) -> np.ndarray:
        """Calcule un vecteur normalisé par texte (voir ``Embedder.embed``)."""
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self._batch_size,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)
