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
import re
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import FTS

from rag_rgpd.processing.pipeline import Chunk
from rag_rgpd.retrieval.embeddings import Embedder
from rag_rgpd.retrieval.rerank import Reranker

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

DEFAULT_POOL = 30
"""Passages relus par le modèle de reclassement : assez pour rattraper un
bon article classé 11e ou 20e par la recherche hybride, assez peu pour
rester autour d'une seconde sur un PC sans carte graphique."""

_REFERENCE = re.compile(
    r"\b(?P<kind>articles?|art\.|considérants?)\s*(?P<number>\d{1,3})\b",
    re.IGNORECASE,
)


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
        score: score de fusion RRF de la recherche hybride.
        rerank_score: note du modèle de reclassement (0 à 1), qui fixe
            l'ordre quand il est utilisé ; ``None`` sinon.
        pinned: ``True`` si le passage a été placé en tête parce que la
            question cite son article (« article 17 »).
    """

    chunk: Chunk
    similarity: float
    score: float
    rerank_score: float | None = None
    pinned: bool = False


def embedding_text(chunk: Chunk) -> str:
    """Texte réellement encodé et indexé pour un morceau.

    On préfixe le titre complet (« RGPD > Chapitre II > Article 6 :
    Licéité du traitement ») : un paragraphe isolé comme « 2. Le paragraphe
    1 ne s'applique pas... » n'a de sens qu'avec son contexte, et le mot
    « article 6 » devient trouvable par mots-clés.
    """
    return f"{chunk.heading}\n\n{chunk.text}"


def cited_labels(question: str) -> list[str]:
    """Articles et considérants cités par leur numéro dans la question.

    « Que dit l'article 17 ? » -> ``["Article 17"]`` ; « art. 6 et
    considérant 47 » -> ``["Article 6", "Considérant 47"]``. Ces étiquettes
    sont celles des morceaux du RGPD (voir ``processing/sections.py``).

    Pourquoi : la recherche par le sens ne sait pas qu'« article 17 » désigne
    un passage précis, et la recherche par mots-clés trouve « article » et
    « 17 » un peu partout (« l'article 17, paragraphe 3 » cité dans d'autres
    articles). Quand l'utilisateur nomme un article, on va le chercher
    directement.
    """
    labels = []
    for match in _REFERENCE.finditer(question):
        kind = "Considérant" if match["kind"].lower().startswith("consid") else "Article"
        label = f"{kind} {int(match['number'])}"
        if label not in labels:
            labels.append(label)
    return labels


def _label_filter(labels: list[str]) -> str:
    """Filtre SQL (syntaxe LanceDB) : « Article 17 » et ses paragraphes."""
    parts = []
    for label in labels:
        # Les étiquettes viennent de notre expression régulière (lettres,
        # espace, chiffres) : aucune apostrophe à échapper.
        parts.append(f"label = '{label}' OR label LIKE '{label},%' OR label LIKE '{label} %'")
    return " OR ".join(f"({p})" for p in parts)


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


def _label_position(label: str, labels: list[str]) -> int:
    for position, wanted in enumerate(labels):
        if label == wanted or label.startswith((wanted + ",", wanted + " ")):
            return position
    return len(labels)


def _row_to_chunk(row: dict) -> Chunk:
    values = {name: row[name] for name in _CHUNK_FIELDS}
    values["context"] = tuple(values["context"] or ())
    return Chunk(**values)


class Searcher:
    """Recherche hybride dans un index construit par ``build_index``.

    Trois étapes (décision 0011) :

    1. recherche hybride (sens + mots-clés, fusion RRF) ;
    2. si un modèle de reclassement est fourni, il relit les ``pool``
       premiers passages et fixe l'ordre ;
    3. si la question cite un article par son numéro, les morceaux de cet
       article passent en tête.
    """

    def __init__(
        self,
        index_dir: Path,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        pool: int = DEFAULT_POOL,
        pin_cited_articles: bool = True,
    ) -> None:
        """Ouvre l'index.

        Args:
            index_dir: dossier de l'index.
            embedder: modèle d'embedding, le même que pour la construction.
            reranker: modèle de reclassement ; ``None`` pour garder l'ordre
                de la recherche hybride (comportement de l'étape 4).
            pool: passages relus par le modèle de reclassement.
            pin_cited_articles: placer en tête les articles cités par leur
                numéro dans la question.

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
        self.reranker = reranker
        self.pool = pool
        self.pin_cited_articles = pin_cited_articles

    @property
    def settings(self) -> dict[str, object]:
        """Réglages de la recherche, à tracer avec chaque réponse."""
        return {
            "reranker": self.reranker.model_name if self.reranker else "",
            "pool": self.pool if self.reranker else None,
            "pin_cited_articles": self.pin_cited_articles,
        }

    def _hit(self, row: dict, query_vector: np.ndarray, **extra: object) -> Hit:
        return Hit(
            chunk=_row_to_chunk(row),
            # Vecteurs normalisés : le produit scalaire est le cosinus.
            similarity=float(np.dot(query_vector, np.asarray(row["vector"]))),
            score=float(row.get("_relevance_score", 0.0)),
            **extra,
        )

    def search(self, question: str, *, k: int = DEFAULT_K) -> list[Hit]:
        """Retrouve les ``k`` passages les plus pertinents.

        Args:
            question: question en langage naturel.
            k: nombre de passages à renvoyer.

        Returns:
            Les passages, du plus pertinent au moins pertinent.
        """
        query_vector = self._embedder.embed([question])[0]
        limit = max(k, self.pool) if self.reranker else k
        rows = (
            self._table.search(query_type="hybrid")
            .vector(query_vector.tolist())
            .text(question)
            .limit(limit)
            .to_list()
        )
        hits = [self._hit(row, query_vector) for row in rows]

        if self.reranker is not None and hits:
            notes = self.reranker.score(question, [embedding_text(h.chunk) for h in hits])
            hits = [replace(h, rerank_score=n) for h, n in zip(hits, notes, strict=True)]
            # Tri stable : à note égale, l'ordre de la recherche hybride reste.
            hits.sort(key=lambda h: -(h.rerank_score or 0.0))

        labels = cited_labels(question) if self.pin_cited_articles else []
        if labels:
            pinned_rows = (
                self._table.search()
                .where(_label_filter(labels), prefilter=True)
                .limit(max(k, 50))
                .to_list()
            )
            # Ordre des articles cités, puis ordre du texte (l'identifiant du
            # morceau finit par son numéro d'ordre dans le document).
            pinned_rows.sort(key=lambda r: (_label_position(r["label"], labels), r["chunk_id"]))
            pinned = [self._hit(row, query_vector, pinned=True) for row in pinned_rows]
            ids = {h.chunk.chunk_id for h in pinned}
            hits = pinned + [h for h in hits if h.chunk.chunk_id not in ids]
        return hits[:k]

    def ranks(self, question: str, *, depth: int = 50) -> dict[str, tuple[int | None, int | None]]:
        """Rang de chaque passage dans chacune des deux recherches, avant fusion.

        Sert à expliquer un classement : un passage peut remonter parce qu'il
        est proche par le sens (rang « sens »), parce qu'il contient les mots
        de la question (rang « mots »), ou les deux. Ces deux recherches
        supplémentaires ne servent qu'à l'explication, elles ne changent pas
        le résultat de ``search``.

        Args:
            question: question en langage naturel.
            depth: profondeur examinée dans chaque liste ; au-delà, le rang
                vaut ``None``.

        Returns:
            Pour chaque ``chunk_id`` vu : (rang par le sens, rang par mots-clés),
            à partir de 1.
        """
        query_vector = self._embedder.embed([question])[0]
        by_meaning = self._table.search(query_vector.tolist()).limit(depth).to_list()
        try:
            by_words = self._table.search(question, query_type="fts").limit(depth).to_list()
        except ValueError:
            # Question sans aucun mot indexable (que des mots vides) : la
            # recherche plein texte n'a rien à chercher.
            by_words = []
        result: dict[str, tuple[int | None, int | None]] = {}
        for rank, row in enumerate(by_meaning, start=1):
            result[row["chunk_id"]] = (rank, None)
        for rank, row in enumerate(by_words, start=1):
            meaning, _ = result.get(row["chunk_id"], (None, None))
            result[row["chunk_id"]] = (meaning, rank)
        return result
