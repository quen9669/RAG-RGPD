"""Chaîne complète : manifeste -> fichiers bruts -> sections -> morceaux."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_rgpd.ingestion.storage import ManifestEntry, RawStore, sha256_hex
from rag_rgpd.processing.chunking import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, chunk_sections
from rag_rgpd.processing.html import extract_eu_act
from rag_rgpd.processing.pdf import extract_pdf
from rag_rgpd.processing.sections import Section

Extractor = Callable[[bytes], list[Section]]

# Extracteur selon le type annoncé par le serveur au téléchargement. Le
# catalogue data.public.lu (JSON) n'est pas un texte à découper : il n'a pas
# d'extracteur et il est ignoré ici.
EXTRACTORS: dict[str, Extractor] = {
    "text/html": extract_eu_act,
    "application/xhtml+xml": extract_eu_act,
    "application/pdf": extract_pdf,
}


class IntegrityError(RuntimeError):
    """Levée quand un fichier brut ne correspond plus à son empreinte.

    Le fichier a été modifié ou remplacé depuis son téléchargement : les
    morceaux produits ne correspondraient plus à la version tracée dans le
    manifeste, et les citations deviendraient fausses.
    """


@dataclass(frozen=True)
class Chunk:
    """Un morceau prêt à être indexé, avec tout ce qu'il faut pour le citer.

    Attributes:
        chunk_id: identifiant stable (document, 8 premiers caractères de
            l'empreinte du fichier brut, rang). Il change si la source change,
            ce qui évite de citer un morceau d'une version périmée.
        document_id: identifiant du document (``"rgpd-fr"``).
        document_title: titre lisible (repris de ``config/sources.toml``).
        source: organisme d'origine (``"eur-lex"``, ``"cnpd"``).
        label: endroit précis à citer (« Article 6, paragraphe 1 »).
        title: intitulé de l'article, ou ``""``.
        context: titres englobants (chapitre, section).
        heading: tout le chemin en une ligne, pour l'affichage des citations
            et, plus tard, pour donner du contexte à l'embedding.
        text: texte du morceau.
        char_count: longueur du texte, en caractères.
        page_start: première page (PDF), sinon ``None``.
        page_end: dernière page (PDF), sinon ``None``.
        url: adresse d'origine du document.
        licence: licence ou conditions d'usage.
        source_sha256: empreinte du fichier brut dont vient le morceau
            (lineage : on peut toujours remonter à la version exacte).
        raw_path: chemin du fichier brut, relatif au dossier des données.
    """

    chunk_id: str
    document_id: str
    document_title: str
    source: str
    label: str
    title: str
    context: tuple[str, ...]
    heading: str
    text: str
    char_count: int
    page_start: int | None
    page_end: int | None
    url: str
    licence: str
    source_sha256: str
    raw_path: str


def latest_documents(store: RawStore) -> list[ManifestEntry]:
    """Dernière version utilisable de chaque document du manifeste.

    La dernière ligne d'un document fait référence. Les versions vides
    (premier essai bloqué par EUR-Lex, par exemple) sont écartées : une
    ligne vide ne doit pas masquer une version précédente correcte.

    Args:
        store: stockage des fichiers bruts.

    Returns:
        Une entrée par document, dans l'ordre de première apparition.
    """
    latest: dict[str, ManifestEntry] = {}
    for entry in store.entries():
        if entry.size_bytes > 0:
            latest[entry.document_id] = entry
    return list(latest.values())


def media_type(content_type: str) -> str:
    """« text/html; charset=UTF-8 » devient « text/html »."""
    return content_type.split(";")[0].strip().lower()


def _heading(title: str, context: tuple[str, ...], label: str, section_title: str) -> str:
    """Assemble « Document > Chapitre > Article 6 : Licéité du traitement »."""
    last = f"{label} : {section_title}" if section_title else label
    return " > ".join([title, *context, last])


def chunk_document(
    entry: ManifestEntry,
    store: RawStore,
    *,
    title: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Extrait et découpe un document du manifeste.

    Args:
        entry: ligne du manifeste décrivant le fichier brut.
        store: stockage des fichiers bruts.
        title: titre lisible du document.
        max_chars: taille maximale d'un morceau.
        min_chars: taille sous laquelle des pages voisines sont fusionnées.

    Returns:
        Les morceaux du document, dans l'ordre.

    Raises:
        KeyError: si aucun extracteur ne gère ce type de fichier.
        FileNotFoundError: si le fichier brut est absent.
        IntegrityError: si le fichier ne correspond plus à son empreinte.
    """
    extractor = EXTRACTORS[media_type(entry.content_type)]
    content = (store.root / entry.path).read_bytes()
    if sha256_hex(content) != entry.sha256:
        raise IntegrityError(f"{entry.path} a changé depuis son téléchargement")

    pieces = chunk_sections(extractor(content), max_chars=max_chars, min_chars=min_chars)
    return [
        Chunk(
            chunk_id=f"{entry.document_id}-{entry.sha256[:8]}-{rank:04d}",
            document_id=entry.document_id,
            document_title=title,
            source=entry.source,
            label=piece.label,
            title=piece.title,
            context=piece.context,
            heading=_heading(title, piece.context, piece.label, piece.title),
            text=piece.text,
            char_count=len(piece.text),
            page_start=piece.page_start,
            page_end=piece.page_end,
            url=entry.url,
            licence=entry.licence,
            source_sha256=entry.sha256,
            raw_path=entry.path,
        )
        for rank, piece in enumerate(pieces, start=1)
    ]


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Écrit les morceaux au format JSON Lines (un morceau par ligne).

    Le fichier est entièrement réécrit à chaque exécution : c'est un
    résultat dérivé, toujours reconstructible à partir des fichiers bruts.
    L'encodage UTF-8 est imposé, car sous Windows Python utiliserait sinon
    l'encodage local (cp1252) et abîmerait les caractères accentués.

    Args:
        chunks: morceaux à écrire.
        path: fichier de sortie (dossiers créés au besoin).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    """Relit un fichier écrit par ``write_chunks``."""
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Chunk(**{**row, "context": tuple(row["context"])}) for row in rows]
