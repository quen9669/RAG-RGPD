# 0001 : Gestion de l'environnement Python

- **Date** : 2026-10-29
- **Étape** : 1 (environnement, dépôt, tests)
- **Statut** : acceptée

## Contexte

Le projet a besoin d'un environnement Python isolé et reproductible : même
version de Python et mêmes versions de bibliothèques sur chaque machine, pour
que les résultats (ingestion, évaluation) puissent être refaits à l'identique.

## Option choisie : uv

- Gère en un seul outil la version de Python (`.python-version`),
  l'environnement virtuel (`.venv/`) et le verrouillage des versions (`uv.lock`).
- Utilise le fichier de configuration standard `pyproject.toml`.
- Très rapide, bien reconnu par VS Code.

## Options écartées

- **venv + pip** : fourni avec Python, mais pas de vrai fichier de
  verrouillage et pas de gestion de la version de Python.
- **Poetry** : mature, mais plus lent et ne gère pas la version de Python
  (il faudrait un outil supplémentaire).

## Conséquences

- Recréer l'environnement : `uv sync`.
- Ajouter une bibliothèque : `uv add <nom>` (met à jour `pyproject.toml` et `uv.lock`).
- `uv.lock` est versionné dans Git ; `.venv/` ne l'est pas.
