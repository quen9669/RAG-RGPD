# 0015 — Étape 8 : domaines, filtre de recherche et seuils par domaine

Date : 2026-11-04. Validé par FRANKS (« go avec mes recommandations »).

## Contexte

L'étape 8 ajoute l'open data, le ferroviaire et le transfrontalier au corpus
du RGPD. Il faut pouvoir restreindre la recherche à un sujet, régler le
refus par sujet, et vérifier que les réponses RGPD ne se dégradent pas.

## Options choisies

1. **Étiquette de domaine** : champ `domain` dans `config/sources.toml`,
   copié sur chaque morceau (`Chunk.domain`, défaut `rgpd`) et dans l'index
   LanceDB (un seul index). Domaines : `rgpd`, `open-data`, `ferroviaire`,
   `transfrontalier`. Filtre `--domain` (répétable) sur `rag-ask` et
   `rag-eval` ; sans filtre, on cherche partout. Le filtre s'applique à la
   recherche hybride, aux articles cités et aux rangs d'explication.
   Écartées : un index par domaine (questions croisées plus lourdes), aucune
   étiquette (plus de contrôle ni de mesure par domaine).
2. **Seuil de refus par domaine** : `DOMAIN_MIN_SIMILARITY` (vide pour
   l'instant, donc 0,45 partout) et `min_similarity_for(domaines)`, qui
   prend le seuil le plus strict en cas de plusieurs domaines. `--min-similarity`
   garde priorité. À calibrer avec des questions de chaque domaine, hors-sujet
   compris. Écarté : un seuil unique non revérifié.
3. **Données structurées** (catalogues, horaires) : une fiche texte par jeu
   de données, découpée comme le reste. Prévu pour 8c, pas encore codé.
   Écartés : tables avec DuckDB, ou structuré mis de côté.
4. **Ordre** : 8a EUR-Lex, 8b Legilux, 8c catalogues et guides. Après chaque
   sous-étape, `rag-eval` sur les 52 questions RGPD **sans filtre** (le vrai
   test : les nouveaux textes ne doivent pas faire reculer le RGPD) et
   comparaison avec `evals/reference.json`.

## Conséquences

- Il faut relancer `rag-ingest documents`, `rag-chunk` puis `rag-index` sur
  le PC (réseau cloud bloqué) : l'index change de schéma (colonne `domain`).
- Les anciens `chunks.jsonl` restent lisibles (domaine `rgpd` par défaut).
- Le panneau du chat n'a pas encore de sélecteur de domaine.
