# 0009 : LLM et réponses citées

- **Date** : 2026-10-30
- **Étape** : 4 (recherche et réponses avec citations)
- **Statut** : acceptée

## Contexte

Le LLM (Large Language Model, modèle de langage) rédige la réponse à partir
des passages retrouvés. Il doit citer précisément ses sources et répondre
« je ne sais pas » quand les passages ne suffisent pas.

## Option choisie : Claude par API, avec les citations de l'API

- Modèle par défaut `claude-haiku-5-5` (Haiku 5.5), le moins cher
  (environ 0,001 $ par question, estimation). Choix de FRANKS : commencer
  par Haiku et monter vers Sonnet 5.5 (~0,025 $) ou Opus 5.5 (~0,05 $)
  avec `--model` si les réponses ne sont pas concluantes.
- Les passages sont envoyés comme blocs `search_result` avec citations
  activées : c'est l'API qui relie chaque phrase au passage exact qui
  l'appuie (texte cité mot pour mot), on ne dépend pas du modèle pour
  recopier des numéros.
- Avec Sonnet ou Opus, repli automatique côté serveur
  (`fallbacks: "default"`) si le modèle décline la demande à tort (Haiku
  n'offre pas ce repli).

Trois barrières contre l'hallucination :

1. **Avant le LLM** : si le meilleur passage a une similarité cosinus
   inférieure à 0,45, réponse « Je ne sais pas. » sans appel (gratuit).
   Seuil provisoire, à calibrer à l'étape 5.
2. **Consigne** : n'utiliser que les passages, sinon « Je ne sais pas. ».
3. **Contrôle** : une réponse sans aucune citation est remplacée par un refus.

## Options écartées

- **Mistral par API** : hébergé dans l'UE (Union européenne), moins cher,
  mais moins rigoureux sur les citations.
- **Local avec Ollama** : gratuit et privé, mais lent sans carte graphique
  et moins fiable en français juridique.

## Conséquences

- Clé `ANTHROPIC_API_KEY` nécessaire, rangée dans le fichier `.env` du
  projet (exclu de Git, lu par `rag-ask`), coût à l'usage (faible pour un usage
  personnel). `rag-ask --search-only` fonctionne sans clé.
- Les passages (dont des extraits CNPD) sont envoyés à l'API pour
  rédiger la réponse : usage privé, sans rediffusion.
- Dépendance à un fournisseur : le LLM est derrière une interface
  (`AnswerGenerator`), on peut le remplacer sans toucher à la recherche.
- Chaque question est tracée dans `data/logs/answers.jsonl` (journal des
  réponses, non versionné).
