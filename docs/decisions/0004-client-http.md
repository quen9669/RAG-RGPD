# 0004 : Bibliothèque de téléchargement

- **Date** : 2026-10-29
- **Étape** : 2 (récupération des données)
- **Statut** : acceptée

## Option choisie : httpx

- Même façon d'écrire que requests, plus moderne (délais d'attente
  configurables, mode asynchrone possible plus tard).
- Fournit `MockTransport`, qui permet de tester l'ingestion sans réseau.

## Option écartée

- **requests** : la plus connue, mais sans faux serveur intégré pour les
  tests ni mode asynchrone.

## Conséquences

- Tout passe par `PoliteClient` (`src/rag_rgpd/ingestion/http.py`) :
  respect de `robots.txt`, pause de 2 s par défaut entre deux requêtes vers
  le même site, programme identifié par son en-tête `User-Agent`.
