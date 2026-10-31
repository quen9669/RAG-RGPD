# 0011 : Évaluation maison et reclassement des passages

- **Date** : 2026-10-30
- **Étape** : 5 (mesurer la qualité)
- **Statut** : acceptée

## Contexte

L'étape 5 demande un jeu d'environ 50 questions avec les bonnes réponses, et
une mesure des erreurs : passages mal retrouvés, hallucinations (affirmations
sans appui dans les sources), citations manquantes, refus à tort. Les essais
de l'étape 4 bis ont montré des défauts précis : articles 9, 6 et 15 absents
des 8 passages envoyés, requête « article 17 » mal classée, phrases recopiées
sans guillemets, exemples de la brochure CNPD sans numéro de source, règles du
droit d'accès appliquées à l'effacement.

Deux choix : l'outil de mesure, et la façon de corriger la recherche.

## Option choisie

Choix de FRANKS : **A2**.

### Mesure : fait maison (A)

- Jeu de 52 questions dans `tests/cases/questions.yaml` (le même fichier que
  pour pytest et `rag-batch`), avec deux champs en plus : `points_cles`
  (idées attendues) et `piege` (erreur à ne pas commettre).
- Commande `rag-eval` : pose chaque question, calcule les indicateurs de
  recherche (rappel@k, top 3, MRR, refus corrects), vérifie les citations et
  les recopies sans guillemets par du code, puis fait noter chaque réponse par
  un correcteur LLM (Claude Sonnet 5.5, plus fort que Haiku qui rédige) avec
  une sortie JSON imposée : affirmations appuyées et citées, points clés
  couverts, piège évité.
- Barrière de qualité : `evals/reference.json` garde les scores de la version
  de référence (sans aucun texte des documents). `rag-eval` s'arrête en erreur
  si un indicateur recule. `--save-reference` installe une nouvelle référence.

### Recherche : articles cités + reclassement (2)

- Si la question cite un article ou un considérant par son numéro
  (« article 17 », « art. 6 »), ses morceaux passent en tête.
- Un modèle de reclassement local, **BGE-reranker-v2-m3**, relit les 30
  premiers passages de la recherche hybride en lisant question et passage
  ensemble, et fixe l'ordre final.

### Consignes du rédacteur

Trois règles ajoutées : guillemets pour toute phrase reprise mot pour mot ;
un exemple seulement s'il vient d'un passage, avec sa citation ; pas de
transposition d'un droit à un autre sans passage qui le dit.

## Options écartées

- **RAGAS** : indicateurs standard, mais calculs peu lisibles, nombreux appels
  au LLM et réglage pour Claude moins direct. Utile plus tard pour une
  comparaison ponctuelle.
- **promptfoo** : bonne interface de comparaison, mais en Node.js (le projet
  est en Python) et en doublon avec `rag-batch`.
- **Articles cités seuls** : règle le cas « article 17 », pas les questions sans
  numéro (articles 6, 9, 15).
- **Bonus de score aux articles face aux considérants** : léger, mais réglé à la
  main et fragile.

## Conséquences

- Un second modèle local d'environ 2 Go à télécharger au premier usage, et
  environ 1 à 2 secondes de plus par question sur le PC. `--no-rerank` et
  `--no-pin` (sur `rag-ask`, `rag-batch`, `rag-eval`) retrouvent l'ordre de
  l'étape 4, pour comparer.
- La similarité cosinus reste la base de la règle « je ne sais pas » : le
  reclassement change l'ordre, pas la barrière 1.
- Une évaluation complète coûte environ 1 $ (rédaction Haiku + correction
  Sonnet) ; le cache local évite de repayer une réponse inchangée.
  `--search-only` mesure la recherche gratuitement.
- Le correcteur LLM se trompe parfois : ses notes servent à comparer deux
  versions et à repérer les réponses à relire, pas à certifier une réponse.

## Résultats (mesure du 2026-10-31, 52 questions, Haiku 5.5 rédige, Sonnet 5.5 corrige)

| Indicateur | Avant (étape 4) | Après |
|---|---|---|
| source dans les 8 passages | 0,87 | 0,94 |
| source dans le top 3 | 0,70 | 0,81 |
| MRR | 0,55 | 0,65 |
| refus corrects | 0,94 | 1,00 |
| citations vérifiées | 1,00 | 1,00 |
| source attendue citée | 0,86 | 0,91 |
| fidélité (affirmations appuyées) | 0,97 | 0,97 |
| affirmations citées | 0,82 | 0,73 |
| complétude des points clés | 0,89 | 0,94 |
| pièges évités | 1,00 | 1,00 |
| sans recopie sans guillemets | 0,20 | 0,13 |

Autres essais : `pool` à 60 est moins bon que 30 (source dans les 8 passages
0,915 contre 0,936) ; avec `k` = 12, 0,936 mais top 3 et MRR restent sous
`pool` 30. Coût d'une mesure complète : environ 0,9 $.

**Limites restantes**, laissées pour la suite (boucle d'amélioration de
l'étape 7) :

- les articles 9, 6 et 5 restent hors des 8 passages (le reclassement semble
  préférer les considérants, qui expliquent en langage courant) ;
- les consignes sur les guillemets et les sources n'ont pas été suivies par
  Haiku : 41 réponses sur 52 recopient une phrase sans guillemets, et la part
  d'affirmations citées baisse avec le reclassement. Une consigne plus
  stricte est prête (`consigne-stricte.patch`, non mesurée) ; piste suivante :
  Claude Sonnet 5.5 comme rédacteur ;
- une affirmation revient sans appui : « la CNPD est l'autorité de contrôle
  luxembourgeoise ».

FRANKS a choisi de fusionner avec ces deux reculs (affirmations citées,
recopies) : la référence enregistrée (`evals/reference.json`) est cette
mesure, la barrière de qualité s'applique à partir d'elle.
