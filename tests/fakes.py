"""Faux objets partagés par les tests de recherche et de réponse."""

import hashlib
import re
import unicodedata

import numpy as np

from rag_rgpd.processing.pipeline import Chunk


class FakeEmbedder:
    """Faux modèle d'embedding : un sac de mots haché, instantané.

    Deux textes qui partagent des mots ont des vecteurs proches : assez pour
    tester la mécanique sans télécharger BGE-M3 (2 Go) ni charger PyTorch.
    """

    def __init__(self, model_name: str = "faux-modele", dimension: int = 128) -> None:
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode()
            for word in re.findall(r"\w{3,}", folded):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vectors[row, bucket] += 1.0
            norm = np.linalg.norm(vectors[row])
            if norm:
                vectors[row] /= norm
        return vectors


def make_chunk(rank: int, label: str, title: str, text: str, **extra: object) -> Chunk:
    values = {
        "chunk_id": f"rgpd-fr-abcdef12-{rank:04d}",
        "document_id": "rgpd-fr",
        "document_title": "RGPD",
        "source": "eur-lex",
        "label": label,
        "title": title,
        "context": ("Chapitre II : Principes",),
        "heading": f"RGPD > Chapitre II : Principes > {label} : {title}",
        "text": text,
        "char_count": len(text),
        "page_start": None,
        "page_end": None,
        "url": "https://exemple.eu/rgpd",
        "licence": "CC BY 4.0",
        "source_sha256": "abcdef12" * 8,
        "raw_path": "raw/eur-lex/rgpd-fr.html",
    }
    values.update(extra)
    return Chunk(**values)


CHUNKS = [
    make_chunk(1, "Article 6", "Licéité du traitement",
               "Le traitement n'est licite que si la personne concernée a consenti "
               "au traitement de ses données à caractère personnel."),
    make_chunk(2, "Article 17", "Droit à l'effacement",
               "La personne concernée a le droit d'obtenir l'effacement de ses données "
               "dans les meilleurs délais."),
    make_chunk(3, "Article 37", "Désignation du délégué à la protection des données",
               "Le responsable du traitement désigne un délégué à la protection des données."),
    make_chunk(4, "Page 3", "", "Brochure : vos droits en tant que citoyen.",
               document_id="brochure", page_start=3, page_end=4),
]
