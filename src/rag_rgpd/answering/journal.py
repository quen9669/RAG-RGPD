"""Journal des réponses (gouvernance IA).

Chaque question posée est tracée : réponse ou refus, raison, modèles
utilisés, passages montrés et passages cités. Ce journal sert à repérer les
ratés (étape 7) et à pouvoir expliquer après coup pourquoi l'assistant a
répondu ce qu'il a répondu.

Il reprend des extraits des documents (dont la CNPD) : il reste local, comme
``data/processed/`` (voir ``.gitignore``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag_rgpd.answering.generator import Answer
from rag_rgpd.retrieval.index import IndexMeta


def log_answer(answer: Answer, meta: IndexMeta, path: Path) -> None:
    """Ajoute une ligne au journal (format JSON Lines, UTF-8).

    Args:
        answer: réponse à tracer.
        meta: fiche de l'index utilisé (modèle d'embedding, empreinte).
        path: fichier du journal (dossiers créés au besoin).
    """
    record = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": answer.question,
        "refused": answer.refused,
        "reason": answer.reason,
        "answer": answer.text,
        "llm_model": answer.model,
        "embedding_model": meta.embedding_model,
        "chunks_sha256": meta.chunks_sha256,
        "retrieved": [
            {"chunk_id": h.chunk.chunk_id, "similarity": round(h.similarity, 4)}
            for h in answer.hits
        ],
        "cited": [answer.hits[n - 1].chunk.chunk_id for n in answer.cited_numbers],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
