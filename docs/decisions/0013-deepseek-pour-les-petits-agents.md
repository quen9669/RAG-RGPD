# 0013 : DeepSeek pour les petits agents (essai)

- **Date** : 2026-11-01
- **Étape** : 6 (multi-agent)
- **Statut** : code livré, qualité à mesurer sur le jeu de test

## Contexte

Le coût monte vite : les appels à Sonnet 5.5 (vérificateur, correcteur de
`rag-eval`) pèsent plus que la rédaction par Haiku. FRANKS veut essayer
DeepSeek (modèle annoncé comme « V4 Lite »). La documentation officielle
(api-docs.deepseek.com, 2026-11-01) liste `deepseek-flash` (V4.1 Flash) et
`deepseek-v4-pro`, pas de « Lite ».

## Option choisie

- Un client `DeepSeekClient` (`src/rag_rgpd/agents/deepseek.py`) qui parle à
  l'API DeepSeek (format OpenAI) et renvoie un `BetaMessage` : cache local,
  coût et lecture du JSON restent ceux des autres agents.
- Utilisable pour le **chercheur, le vérificateur et le correcteur** :
  `--researcher-model`, `--verifier-model`, `--judge-model` avec un nom
  `deepseek-*`. Clé dans `.env` : `DEEPSEEK_API_KEY`.
- Pas pour le **rédacteur** : DeepSeek n'a pas les citations natives de
  l'API Claude (blocs `search_result`) sur lesquelles repose la barrière 3.
- DeepSeek ne sait pas imposer un schéma JSON (seulement du JSON valide) : le
  schéma est recopié dans la consigne ; une réponse vide est retentée une
  fois, puis le vérificateur bloque la réponse (jamais de réponse non relue).

## Options écartées

- **DeepSeek rédacteur** : demanderait un nouveau mode de citation (marqueurs
  [n] vérifiés par code) ; hors périmètre de cet essai.
- **Point d'accès « format Anthropic » de DeepSeek** : non vérifié pour les
  citations, abandonné.

## Conséquences et risques

- Prix (par million de tokens, hors pointe) : `deepseek-flash` 0,15 $ / 0,60 $
  et `deepseek-v4-pro` 0,66 $ / 1,98 $, contre 2 $ / 10 $ pour Sonnet 5.5.
  Haiku 5.5 (0,10 $ / 0,50 $ dans `cost.py`) reste aussi bon marché que
  `deepseek-flash` : le gain est sur les rôles aujourd'hui tenus par Sonnet.
  Les tarifs doublent aux heures de pointe.
- **Données** : les serveurs sont hors de l'Union européenne. Sont envoyés les
  passages du corpus public (extraits CNPD, usage privé) et la question posée :
  ne pas y mettre de données personnelles. À inscrire au registre des risques.
- Qualité non garantie : le vérificateur Haiku bloquait à tort 3 réponses sur
  10 ; il faut mesurer DeepSeek sur le même jeu avant de l'adopter.

## Résultats

À compléter après le test sur les 10 questions.
