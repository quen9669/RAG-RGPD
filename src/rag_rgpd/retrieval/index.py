"""Index LanceDB des morceaux et recherche hybride.

L'index est un dossier local (``data/index/``) au format Lance, construit sur
Apache Arrow : une table en colonnes, comme un fichier Parquet, avec en plus
une colonne de vecteurs et un index plein texte.

La recherche est **hybride** (décision 0008) :

- la recherche par le sens (vecteurs) retrouve « droit à l'oubli » pour une
  question sur l'effacement ;
- la recherche par mots-clés (BM25, l'algorithme classique des moteurs de
  recherche) retrouve « article 6 » ou « DPO », que les vecteurs captent mal ;
- LanceDB fusionne les deux classements par RRF (Reciprocal Rank Fusion :
  un passage bien classé dans l'une ou l'autre liste remonte).

Le score RRF ne dit rien de la qualité absolue d'un passage. Pour la règle
« je ne sais pas », on garde donc aussi la similarité cosinus de chaque
passage avec la question : si même le meilleur passage est loin, on refuse
de répondre sans appeler le LLM.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from rag_rgpd.processing.pipeline import Chunk
from rag_rgpd.retrieval.embeddings import Embedder

TABLE_NAME = "chunks"
META_FILE = "index_meta.json"

DEFAULT_K = 8
"""Passages envoyés au LLM : assez pour couvrir un article et ses
considérants, assez peu pour que chaque citation reste vérifiable."""

DEFAULT_MIN_SIMILARITY = 0.45
"""Similarité cosinus minimale du meilleur passage pour tenter une réponse.
Valeur de départ prudente pour BGE-M3 ; elle sera calibrée à l'étape 5 avec
le jeu de questions (trop haute, on refuse à tort ; trop basse, on laisse
passer des questions hors sujet)."""


class IndexMismatchError(RuntimeError):
    """Levée quand l'index a été construit avec un autre modèle d'embedding.

    Des vecteurs de deux modèles différents ne sont pas comparables : la
    recherche renverrait des résultats sans aucun sens, sans erreur visible.
    """


@dataclass(frozen=True)
class IndexMeta:
    """Fiche d'identité de l'index (lineage).

    Attributes:
        embedding_model: modèle qui a calculé les vecteurs.
        dimension: taille des vecteurs.
        chunk_count: nombre de morceaux indexés.
        chunks_sha256: empreinte du fichier ``chunks.jsonl`` d'origine.
        built_at: date de construction (UTC, ISO 8601).
    """

    embedding_model: str
    dimension: int
    chunk_count: int
    chunks_sha256: str
    built_at: str


@dataclass(frozen=True)
class Hit:
    """Un passage retrouvé pour une question.

    Attributes:
        chunk: le morceau, avec tout ce qu'il faut pour le citer.
        similarity: similarité cosinus avec la question (-1 à 1).
        score: score de fusion RRF, qui fixe l'ordre des résultats.
    """

    chunk: Chunk
    similarity: float
    score: float


def embedding_text(chunk: Chunk) -> str:
    """Texte réellement encodé et indexé pour un morceau.

    On préfixe le titre complet (« RGPD > Chapitre II > Article 6 :
    Licéité du traitement ») : un paragraphe isolé comme « 2. Le paragraphe
    1 ne s'applique pas... » n'a de sens qu'avec son contexte, et le mot
    « article 6 » devient trouvable par mots-clés.
    """
    return f"{chunk.heading}\n\n{chunk.text}"


def _schema(dimension: int) -> pa.Schema:
    """Schéma explicite de la table.

    Sans lui, Arrow devinerait les types à partir des données : une colonne
    de pages entièrement vide (aucun PDF) deviendrait de type « null ».
    """
    return pa.schema(
        [
            ("chunk_id", pa.string()),
            ("document_id", pa.string()),
            ("document_title", pa.string()),
            ("source", pa.string()),
            ("label", pa.string()),
            ("title", pa.string()),
            ("context", pa.list_(pa.string())),
            ("heading", pa.string()),
            ("text", pa.string()),
            ("char_count", pa.int32()),
            ("page_start", pa.int32()),
            ("page_end", pa.int32()),
            ("url", pa.string()),
            ("licence", pa.string()),
            ("source_sha256", pa.string()),
            ("raw_path", pa.string()),
            ("search_text", pa.string()),
            ("vector", pa.list_(pa.float32(), dimension)),
        ]
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_index(
    chunks: list[Chunk],
    embedder: Embedder,
    index_dir: Path,
    *,
    chunks_path: Path | None = None,
) -> IndexMeta:
    """Calcule les vecteurs et (re)crée l'index de recherche.

    L'index est entièrement reconstruit à chaque fois : 465 morceaux se
    réencodent en quelques minutes, et un index toujours aligné sur
    ``chunks.jsonl`` vaut mieux qu'une mise à jour partielle à surveiller.

    Args:
        chunks: morceaux à indexer.
        embedder: modèle d'embedding.
        index_dir: dossier de l'index (créé au besoin).
        chunks_path: fichier d'où viennent les morceaux, pour tracer son
            empreinte dans la fiche de l'index.

    Returns:
        La fiche de l'index, aussi écrite dans ``index_dir/index_meta.json``.

    Raises:
        ValueError: s'il n'y a aucun morceau à indexer.
    """
    if not chunks:
        raise ValueError("aucun morceau à indexer : lancer d'abord `rag-chunk`")
    texts = [embedding_text(c) for c in chunks]
    vectors = embedder.embed(texts)
    dimension = int(vectors.shape[1])

    rows = []
    for chunk, text, vector in zip(chunks, texts, vectors, strict=True):
        row = asdict(chunk)
        row["context"] = list(chunk.context)
        row["search_text"] = text
        row["vector"] = vector.tolist()
        rows.append(row)
    table_data = pa.Table.from_pylist(rows, schema=_schema(dimension))

    index_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(index_dir)
    table = db.create_table(TABLE_NAME, table_data, mode="overwrite")
    # Index plein texte en français : minuscules, accents retirés (« liceite »
    # trouve « licéité »), racinisation (« traitements » trouve
    # « traitement ») et mots vides ignorés (« le », « de »).
    table.create_index(
        "search_text",
        config=FTS(language="French", stem=True, remove_stop_words=True, ascii_folding=True),
        replace=True,
    )

    meta = IndexMeta(
        embedding_model=embedder.model_name,
        dimension=dimension,
        chunk_count=len(chunks),
        chunks_sha256=_file_sha256(chunks_path) if chunks_path else "",
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    (index_dir / META_FILE).write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def read_meta(index_dir: Path) -> IndexMeta:
    """Relit la fiche de l'index.

    Raises:
        FileNotFoundError: si l'index n'a pas encore été construit.
    """
    path = index_dir / META_FILE
    if not path.exists():
        raise FileNotFoundError(f"pas d'index dans {index_dir} : lancer d'abord `rag-index`")
    return IndexMeta(**json.loads(path.read_text(encoding="utf-8")))


_CHUNK_FIELDS = [f.name for f in fields(Chunk)]


def _row_to_chunk(row: dict) -> Chunk:
    values = {name: row[name] for name in _CHUNK_FIELDS}
    values["context"] = tuple(values["context"] or ())
    return Chunk(**values)


class Searcher:
    """Recherche hybride dans un index construit par ``build_index``."""

    def __init__(self, index_dir: Path, embedder: Embedder) -> None:
        """Ouvre l'index.

        Args:
            index_dir: dossier de l'index.
            embedder: modèle d'embedding, le même que pour la construction.

        Raises:
            FileNotFoundError: si l'index n'existe pas.
            IndexMismatchError: si l'index vient d'un autre modèle.
        """
        self.meta = read_meta(index_dir)
        if self.meta.embedding_model != embedder.model_name:
            raise IndexMismatchError(
                f"index construit avec {self.meta.embedding_model}, "
                f"recherche avec {embedder.model_name} : relancer `rag-index`"
            )
        self._embedder = embedder
        self._table = lancedb.connect(index_dir).open_table(TABLE_NAME)

    def search(self, question: str, *, k: int = DEFAULT_K) -> list[Hit]:
        """Retrouve les ``k`` passages les plus pertinents.

        Args:
            question: question en langage naturel.
            k: nombre de passages à renvoyer.

        Returns:
            Les passages, du plus pertinent au moins pertinent.
        """
        query_vector = self._embedder.embed([question])[0]
        rows = (
            self._table.search(query_type="hybrid")
            .vector(query_vector.tolist())
            .text(question)
            .limit(k)
            .to_list()
        )
        return [
            Hit(
                chunk=_row_to_chunk(row),
                # Vecteurs normalisés : le produit scalaire est le cosinus.
                similarity=float(np.dot(query_vector, np.asarray(row["vector"]))),
                score=float(row["_relevance_score"]),
            )
            for row in rows
        ]
