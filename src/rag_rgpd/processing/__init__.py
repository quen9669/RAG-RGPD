"""Traitement : extraction du texte, nettoyage et découpage en morceaux.

Principes (voir docs/decisions/0005 et 0006) :

- On part toujours des fichiers bruts listés dans ``data/manifest.jsonl``
  (dernière version de chaque document), sans jamais les modifier.
- L'extraction produit des *sections* qui suivent la structure du document :
  articles et considérants pour le RGPD, pages pour les PDF de la CNPD.
- Le découpage est *hybride* : une section courte donne un morceau ; une
  section longue est redécoupée par paragraphe, avec un plafond de taille.
- Chaque morceau garde de quoi être cité (document, article ou page) et
  retrouvé (empreinte SHA-256 du fichier brut) : c'est la suite du lineage.
- Le résultat, ``data/processed/chunks.jsonl``, n'est pas versionné dans Git
  car il contient le texte des documents CNPD (pas de rediffusion).
"""
