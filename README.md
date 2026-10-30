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

## Poser une question

```bash
uv run rag-index                      # encode les morceaux (BGE-M3) -> data/index/
uv run rag-ask --search-only "article 17"   # passages retrouvés, sans LLM ni clé
uv run rag-ask "Puis-je demander l'effacement de mes données ?"
uv run rag-ask --model claude-sonnet-5-5 "..."   # modèle plus fort si besoin
```

Le premier `rag-index` télécharge le modèle BGE-M3 (environ 2 Go).

Pour `rag-ask`, il faut une clé d'API Anthropic. Elle reste propre à ce
projet : copier `.env.example` en `.env` à la racine du dépôt et remplacer
`sk-ant-...` par la vraie clé. Le fichier `.env` est exclu de Git ; ne
jamais écrire la clé ailleurs. Une variable `ANTHROPIC_API_KEY` déjà
définie dans le terminal reste prioritaire.

La réponse cite ses sources (`[1]`, `[2]`...) ou répond « Je ne sais pas. »
avec la raison. Chaque question est tracée dans `data/logs/answers.jsonl`.

## Documentation

- [Feuille de route](docs/roadmap.md)
- [Journal des décisions](docs/decisions/README.md)
- [Fiches sources](docs/sources/README.md)
