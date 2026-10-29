# 0002 : Outil de tests automatisés

- **Date** : 2026-10-29
- **Étape** : 1 (environnement, dépôt, tests)
- **Statut** : acceptée

## Contexte

Il faut vérifier automatiquement, à chaque modification, que le code fait
toujours ce qu'il doit. Ces tests serviront aussi plus tard de base à la
« barrière de qualité ».

## Option choisie : pytest

- Tests écrits avec de simples `assert`, messages d'erreur clairs.
- Standard de fait dans l'écosystème Python, nombreuses extensions.
- Bien intégré à VS Code (onglet Testing).

## Option écartée

- **unittest** : fourni avec Python, mais plus verbeux (classes,
  `self.assertEqual`) et moins utilisé dans les projets récents.

## Conséquences

- pytest est une dépendance de développement : `uv add --dev pytest`.
- Lancer les tests : `uv run pytest`.
- Les tests vivent dans `tests/`, un fichier `test_<sujet>.py` par sujet.
