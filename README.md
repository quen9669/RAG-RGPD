# RAG RGPD

Assistant qui répond à des questions sur la protection des données
personnelles au Luxembourg, **en citant ses sources**, et qui répond
« je ne sais pas » quand il ne trouve rien de fiable.

Projet d'apprentissage en AI engineering, construit uniquement à partir de
données publiques. L'outil aide à l'analyse et ne remplace pas un avis juridique.

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync          # installe Python 3.13 si besoin, crée .venv/ et installe les dépendances
```

Dans VS Code, choisir l'interpréteur `.venv` (commande « Python: Select Interpreter »).

## Tests

```bash
uv run pytest
```

## Documentation

- [Journal des décisions](docs/decisions/README.md)
