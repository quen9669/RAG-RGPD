"""Reclassement des passages (« reranking ») par un second modèle.

La recherche hybride (décision 0008) compare la question et chaque passage
**séparément** : chacun est résumé en un vecteur, puis on mesure la distance
entre les deux. C'est rapide, mais grossier : pour « Comment savoir quelles
données une entreprise détient sur moi ? », l'article 15 (droit d'accès)
n'arrivait qu'au 11e rang, derrière des considérants qui emploient les mêmes
mots.

Un modèle de reclassement (« cross-encoder ») lit la question **et** le
passage ensemble, mot à mot, et donne une note de pertinence. Trop lent pour
les 465 morceaux, il ne relit que les 30 premiers de la recherche hybride
(décision 0011). Modèle retenu : BGE-reranker-v2-m3, de la même famille que
BGE-M3, multilingue, exécuté en local (rien à payer, rien ne quitte la
machine).
"""

from __future__ import annotations

from typing import Protocol

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

MAX_LENGTH = 1024
"""Longueur lue pour la paire question + passage, en tokens : un morceau
fait au plus environ 400 tokens avec son titre, rien n'est coupé."""


class Reranker(Protocol):
    """Ce que la recherche attend d'un modèle de reclassement.

    Un protocole, comme pour ``Embedder`` : les tests utilisent un faux
    modèle instantané, sans PyTorch.
    """

    model_name: str
    """Nom du modèle, tracé dans les paramètres de chaque réponse."""

    def score(self, question: str, texts: list[str]) -> list[float]:
        """Note la pertinence de chaque texte pour la question.

        Args:
            question: question posée.
            texts: passages à noter.

        Returns:
            Une note par texte (plus haut = plus pertinent), entre 0 et 1.
        """
        ...


class CrossEncoderReranker:
    """Modèle de reclassement local chargé avec sentence-transformers.

    Le premier usage télécharge le modèle (environ 2 Go) dans le cache de
    Hugging Face, comme BGE-M3.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 8) -> None:
        """Charge le modèle.

        Args:
            model_name: identifiant du modèle sur Hugging Face.
            batch_size: paires notées à la fois ; petit pour un PC sans
                carte graphique.
        """
        # Import tardif, pour la même raison que dans embeddings.py.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._batch_size = batch_size
        self._model = CrossEncoder(model_name, max_length=MAX_LENGTH)

    def score(self, question: str, texts: list[str]) -> list[float]:
        """Note chaque passage (voir ``Reranker.score``)."""
        if not texts:
            return []
        # Le modèle sort une note brute ; la sigmoïde (activation par défaut
        # de ce modèle) la ramène entre 0 et 1, plus lisible dans la fiche.
        scores = self._model.predict(
            [(question, text) for text in texts],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]
