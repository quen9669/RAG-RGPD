# 0010 : Outils d'explicabilité et de test (traces, notebooks, cas de test)

- **Date** : 2026-10-30
- **Étape** : 4 bis (tests et explicabilité dans VS Code)
- **Statut** : acceptée

## Contexte

Pour chaque question, on veut pouvoir comprendre chaque décision du système
(passages retrouvés et pourquoi, texte exact envoyé au modèle, citations,
raison d'un refus, coût), et lancer beaucoup d'essais depuis VS Code. Trois
choix d'outils : où ranger les traces, quel type de notebook, et sous quelle
forme écrire les cas de test.

## Option choisie : tout « maison » (JSONL + Jupyter + YAML)

Choix de FRANKS.

- **Traces** : une « fiche de décision » par question, en JSON Lines dans
  `data/traces/traces.jsonl` (hors de Git : extraits CNPD). Affichage avec
  `rag-ask --explain` et `rag-trace`. Aucun service à lancer, lisible dans
  VS Code et dans pandas.
- **Notebooks** : Jupyter dans VS Code (`notebooks/`, dépendances dans le
  groupe `notebooks` : ipykernel, pandas, matplotlib).
- **Cas de test** : `tests/cases/questions.yaml`, lu par
  `pytest.mark.parametrize` (un test par cas) et par `rag-batch`. Le YAML
  accepte des commentaires, utile pour noter pourquoi un cas existe.

## Options écartées

- **Arize Phoenix (local)** : belle interface web pour les traces
  (OpenTelemetry), mais un serveur à lancer et des dépendances lourdes. On
  pourra y brancher nos traces plus tard sans rien jeter.
- **MLflow Tracing** : standard du marché pour comparer des essais, mais
  lourd et plus complexe que le besoin actuel.
- **Langfuse (auto-hébergé)** : demande une base de données et Docker.
- **marimo** : notebooks enregistrés en fichiers `.py` (propres dans Git),
  mais intégration VS Code moins aboutie que Jupyter.
- **Cas en JSONL** : plus facile à écrire par programme, moins lisible et sans
  commentaires.

## Conséquences

- `rag-ask` passe désormais par la même chaîne que la fiche de décision
  (`explain_question`) : il demande `k + 4` candidats à la recherche et
  envoie les `k` premiers au modèle. Les 4 suivants sont affichés comme
  « écartés ». Le classement hybride peut donc légèrement différer de
  l'étape 4 (la fusion RRF voit plus de candidats).
- Nouvelle barrière dans le contrôle final : une citation dont le texte ne se
  retrouve pas dans le passage désigné entraîne un refus
  (`citation_non_verifiee`).
- Tests à trois niveaux : rapides par défaut (faux modèles), `--index` (vrai
  index, sans LLM), `--llm` (Claude, payant).
- Un cache local des réponses (`data/cache/llm/`, hors de Git) évite de
  payer deux fois la même requête lors des essais en lot.
- Coût estimé à partir d'une table de prix (`explain/cost.py`) à tenir à jour
  si les prix changent.
