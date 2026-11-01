# 0012 : Les trois agents en Python maison

- **Date** : 2026-11-01
- **Étape** : 6 (multi-agent)
- **Statut** : code livré, mesure réelle à faire (voir « Résultats »)

## Contexte

L'étape 6 sépare le travail en trois rôles : un chercheur qui trouve les
passages, un rédacteur qui écrit la réponse citée, un vérificateur qui bloque
toute affirmation sans appui. Il restait des défauts de l'étape 5 (décision
0011) : articles 9, 6 et 5 hors des 8 passages, affirmations sans appui.

## Option choisie

Choix de FRANKS : **A, code Python maison** (`src/rag_rgpd/agents/`).

- **Chercheur** (Haiku 5.5) : écrit au plus 2 reformulations de la question
  (vocabulaire juridique, numéro d'article si sûr), cherche pour chacune,
  fusionne et reclasse avec la question d'origine. Il ne reformule pas si la
  question d'origine est déjà sous la barrière de similarité (une question hors
  sujet reste refusée). S'il échoue, on retombe sur la recherche simple.
- **Rédacteur** : le `ClaudeGenerator` de l'étape 4 (Haiku 5.5), inchangé.
- **Vérificateur** (Sonnet 5.5) : liste les affirmations et dit si chacune est
  dite par un passage ; la décision est prise par le code. Si une affirmation
  est sans appui, ses remarques repartent au rédacteur (2 rédactions au
  maximum, `--rounds`) ; si la dernière version est encore refusée, ou si le
  vérificateur est en panne, réponse « Je ne sais pas. » (code
  `verification_echouee`).
- Activation : `--agents` sur `rag-ask` et `rag-eval`. La fiche de décision
  garde chaque appel dans `params.agents` (modèle, tokens, coût) ; le coût
  total inclut les trois agents.

## Options écartées

- **Claude Agent SDK** : chercheur autonome qui appelle la recherche comme un
  outil ; plus souple mais plus cher et peu testable sans LLM.
- **LangGraph** : graphe d'états standard, mais grosse dépendance pour un
  enchaînement fixe de trois étapes.

## Conséquences

- Coût par question : environ 3 appels (chercheur Haiku, rédacteur Haiku,
  vérificateur Sonnet), jusqu'à 5 avec une reprise.
- Le vérificateur peut refuser des réponses utiles (faux positifs) : la mesure
  suivra les refus à tort et la complétude.
- Reporté à l'étape 7 : consigne stricte, Sonnet comme rédacteur.

## Résultats

Pas encore mesurés : `uv run rag-eval --agents` (environ 1,5 à 2 $) à lancer
sur le PC après accord sur le coût. Comparaison avec `evals/reference.json`.
