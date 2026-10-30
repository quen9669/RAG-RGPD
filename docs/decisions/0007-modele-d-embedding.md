# 0007 : Modèle d'embedding

- **Date** : 2026-10-30
- **Étape** : 4 (recherche et réponses avec citations)
- **Statut** : acceptée

## Contexte

Pour chercher par le sens, chaque morceau est transformé en embedding : une
liste de nombres qui résume son sens. Une question proche par le sens donne
un vecteur proche, même avec d'autres mots. Les textes sont en français
juridique.

## Option choisie : BGE-M3, en local

Modèle multilingue de BAAI (Beijing Academy of Artificial Intelligence),
exécuté sur le PC avec la bibliothèque sentence-transformers. Vecteurs de
dimension 1 024, normalisés (comparaison par similarité cosinus).

- Gratuit, et aucun texte ne quitte la machine à cette étape.
- Très bon en français, accepte des textes longs.

## Options écartées

- **multilingual-e5-small** : plus léger et plus rapide, mais moins précis.
  Pourra servir de point de comparaison à l'étape 5.
- **API Mistral** : rien à installer, mais clé et paiement à l'usage, et
  dépendance à un fournisseur dès la recherche.

## Conséquences

- Installe PyTorch, et environ 2 Go de modèle téléchargés au premier
  lancement (cache Hugging Face).
- Encodage de tout le corpus en quelques minutes sans carte graphique.
- Le nom du modèle est enregistré avec l'index ; une recherche avec un
  autre modèle est refusée (les vecteurs ne seraient pas comparables).
