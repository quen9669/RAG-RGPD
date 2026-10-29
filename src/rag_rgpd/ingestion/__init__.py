"""Ingestion : téléchargement des sources brutes et suivi de leur origine.

Principes (voir docs/decisions/0003 et 0004) :

- Les fichiers bruts sont stockés tels quels, jamais modifiés, dans
  ``data/raw/<source>/<date>/`` (dossier non versionné dans Git).
- Chaque téléchargement est tracé dans ``data/manifest.jsonl`` (versionné) :
  URL, date, licence, empreinte SHA-256. C'est la base du lineage.
- Les serveurs sont interrogés poliment (robots.txt respecté, pause entre
  deux requêtes, identification claire du programme).
"""
