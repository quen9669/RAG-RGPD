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

Trois niveaux, voir « Tester et comprendre dans VS Code » plus bas.

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
avec la raison. Chaque question est tracée dans `data/logs/answers.jsonl`
(journal des réponses) et `data/traces/traces.jsonl` (fiche de décision complète).

## Tester et comprendre dans VS Code

### La fiche de décision

Chaque question laisse une « fiche de décision » dans
`data/traces/traces.jsonl` (non publiée : elle contient des extraits). Pour
l'afficher :

```bash
uv run rag-ask --explain "Puis-je demander l'effacement de mes données ?"
uv run rag-ask --explain --show-prompt "..."    # + texte exact envoyé au modèle
uv run rag-ask --search-only --explain "article 17"   # gratuit, sans LLM
uv run rag-trace                 # relire la dernière fiche
uv run rag-trace --list          # les 20 dernières questions
uv run rag-trace 20261030-1015   # une fiche précise (début de l'identifiant)
```

La fiche répond, dans l'ordre : quels paramètres ; quels passages (rang,
similarité, rang dans la recherche par le sens et par mots-clés, passages
écartés juste après la coupure `k`) ; la barrière de similarité est-elle
passée ; qu'a reçu le modèle ; qu'a-t-il répondu ; chaque citation reliée à
son passage et vérifiée ; la décision et, en cas de refus, quelle barrière a
décidé ; le temps, les tokens et le coût estimé.

### Les tests, à trois niveaux

| Commande | Ce qui tourne | Coût |
|---|---|---|
| `uv run pytest` | tests rapides : faux modèle d'embedding, faux LLM | gratuit, quelques secondes |
| `uv run pytest --index -m index` | les cas de `tests/cases/questions.yaml` sur le vrai index, sans LLM | gratuit, ~1 min (chargement de BGE-M3) |
| `uv run pytest --llm -m llm` | les mêmes cas avec Claude | ~0,001 $ par cas avec Haiku |

Ajouter `-k effacement` pour un seul cas. Dans le panneau **Testing** de
VS Code (icône de fiole), chaque cas apparaît sous son nom ; les niveaux
non demandés apparaissent « sautés » avec la commande pour les lancer. En
cas d'échec, pytest affiche la fiche de décision complète.

Pour ajouter un cas, compléter `tests/cases/questions.yaml` (format expliqué
en tête du fichier). Un défaut connu se note avec `defaut_connu` : le test
devient un « échec attendu » au lieu de bloquer la suite.

### Les essais en lot

```bash
uv run rag-batch --search-only --k 4 8 12              # gratuit
uv run rag-batch --k 8 --min-similarity 0.40 0.45      # avec Claude
uv run rag-batch --model claude-haiku-5-5 claude-sonnet-5-5 --only effacement delegue
```

Chaque cas de test est lancé avec chaque combinaison de réglages. Le tableau
complet va dans `data/batches/` (CSV, s'ouvre dans Excel), un résumé
s'affiche, et chaque essai a sa fiche (`rag-trace <id>`). Les réponses de
Claude sont gardées dans un cache local (`data/cache/llm/`) : relancer le
même essai ne coûte rien (`--no-cache` pour forcer un nouvel appel).

### Débogage pas à pas et tâches

- **F5** (panneau « Exécuter et déboguer ») : poser une question sous le
  débogueur. Mettre d'abord un point d'arrêt (clic à gauche du numéro de
  ligne), par exemple dans `Searcher.search` (recherche), `answer_question`
  (barrière de similarité) ou `parse_response` (citations). Les
  configurations sont dans `.vscode/launch.json`.
- **Ctrl+Maj+P → « Tasks: Run Task »** : lancements fréquents (tests,
  question avec fiche, lots, reconstruire l'index), dans `.vscode/tasks.json`.

### Les notebooks

```bash
uv sync --group notebooks
```

Puis ouvrir un cahier de `notebooks/` et choisir le noyau `.venv` :

- `01_explorer_une_question.ipynb` : passages, scores et graphique pour une
  question ; comparaison de plusieurs `k` et du seuil ;
- `02_relire_traces_et_lots.ipynb` : toutes les fiches en tableau, refus par
  barrière, coût total, dernier essai en lot.

## Mesurer la qualité (étape 5)

```bash
uv run rag-eval --search-only          # recherche seule : gratuit, ~2 min
uv run rag-eval                        # chaîne complète + correcteur : ~1 $
uv run rag-eval --no-rerank --no-pin   # comme à l'étape 4, pour comparer
uv run rag-eval --save-reference       # la version testée devient la référence
```

`rag-eval` pose les 52 questions de `tests/cases/questions.yaml` et affiche
une dizaine d'indicateurs entre 0 et 1 (plus haut = mieux) :

| Indicateur | Ce qu'il mesure | Comment |
|---|---|---|
| source dans les k, top 3, MRR | la bonne source est retrouvée, et à quel rang | code |
| refus corrects | « je ne sais pas » exactement quand il le faut | code |
| citations vérifiées, source attendue citée | les citations tiennent et visent la bonne source | code |
| sans recopie | aucune phrase de 12 mots ou plus reprise sans guillemets | code |
| fidélité | aucune affirmation sans appui (hallucination) | correcteur LLM |
| affirmations citées | chaque affirmation porte un numéro de source | correcteur LLM |
| complétude | part des `points_cles` couverts | correcteur LLM |
| pièges évités | l'erreur décrite dans `piege` n'est pas commise | correcteur LLM |

Le correcteur est Claude Sonnet 5.5 (`--judge-model`), plus fort que Haiku
qui rédige ; il répond dans un format JSON imposé. Le rapport complet (avec
ses commentaires) va dans `data/evals/` ; les réponses à relire s'affichent
avec leur fiche (`rag-trace <id>`).

**Barrière de qualité** : `evals/reference.json` garde les scores de la
version de référence (sans aucun texte des documents, il est versionné).
Sans `--save-reference`, `rag-eval` compare à cette référence et s'arrête en
erreur si un indicateur recule (`--tolerance 0.02` accepte 2 points de bruit).

**Recherche** (décision [0011](docs/decisions/0011-evaluation-et-reclassement.md)) :
un article cité par son numéro (« article 17 ») passe en tête, puis un modèle
de reclassement local (BGE-reranker-v2-m3, environ 2 Go au premier usage)
relit les 30 premiers passages et fixe l'ordre. `--no-rerank` et `--no-pin`
désactivent ces deux étapes (aussi sur `rag-ask` et `rag-batch`).

## Documentation

- [Feuille de route](docs/roadmap.md)
- [Journal des décisions](docs/decisions/README.md)
- [Fiches sources](docs/sources/README.md)
