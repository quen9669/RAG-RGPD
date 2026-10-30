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

## Récupérer les données

```bash
uv run rag-ingest documents             # documents listés dans config/sources.toml
uv run rag-ingest catalogue             # catalogue complet de data.public.lu
uv run rag-ingest catalogue --max-pages 1   # essai rapide
```

Les fichiers bruts vont dans `data/raw/` (non publié, voir les licences) et
chaque téléchargement est tracé dans `data/manifest.jsonl`.

## Extraire et découper les textes

```bash
uv run rag-chunk                    # dernière version de chaque document du manifeste
uv run rag-chunk --max-chars 1000   # morceaux plus petits
```

Le résultat va dans `data/processed/chunks.jsonl` (non publié) : un morceau
par ligne, avec de quoi le citer (article ou page) et remonter au fichier
brut (empreinte SHA-256).

## Documentation

- [Feuille de route](docs/roadmap.md)
- [Journal des décisions](docs/decisions/README.md)
- [Fiches sources](docs/sources/README.md)
