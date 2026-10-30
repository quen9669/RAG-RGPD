# 0006 : Découpage des textes en morceaux

- **Date** : 2026-10-30
- **Étape** : 3 (extraction, nettoyage, découpage)
- **Statut** : acceptée

## Contexte

La recherche (étape 4) comparera la question à des morceaux de texte. Un
morceau trop long mélange plusieurs idées ; trop court, il perd son
contexte. Chaque morceau doit aussi pouvoir être cité précisément.

## Option choisie : découpage hybride

1. On part de la structure : un considérant, un article, une page de PDF.
2. Les pages de PDF trop courtes (moins de 300 caractères) sont fusionnées
   avec la suivante (« Pages 3 à 5 »). Les articles ne sont jamais fusionnés.
3. Une section de plus de 1 500 caractères est redécoupée par paragraphe
   (« Article 6, paragraphes 1 à 2 »), puis par ligne de liste, puis par
   phrase si un paragraphe dépasse encore le plafond (« (suite) »).

La taille est comptée en caractères (environ 4 caractères par token), pour
ne pas dépendre du modèle d'embedding, qui n'est pas encore choisi. Les deux
seuils sont réglables (`--max-chars`, `--min-chars`).

## Options écartées

- **Par structure seule** : citations précises, mais des articles de
  plusieurs milliers de caractères donneraient des morceaux trop flous.
- **Taille fixe avec chevauchement** : simple, mais coupe au milieu des
  articles et rend les citations imprécises.

## Conséquences

- Pas de chevauchement entre morceaux : les coupures tombent sur des
  limites de paragraphe, et chaque morceau garde son titre complet
  (`heading` : document > chapitre > article) pour le contexte.
- Chaque morceau garde l'empreinte SHA-256 et le chemin du fichier brut
  (lineage) ; son identifiant change si la source change.
- Résultat dans `data/processed/chunks.jsonl`, **non versionné** (il contient
  le texte des documents CNPD), reconstructible avec `uv run rag-chunk`.
