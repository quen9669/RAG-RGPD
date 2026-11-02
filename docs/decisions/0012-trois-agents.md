# 0012 : Les trois agents en Python maison

- **Date** : 2026-11-01
- **Étape** : 6 (multi-agent)
- **Statut** : acceptée, en option (`--agents`) ; vérificateur à régler à l'étape 7

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

Essais sur 10 questions (vérificateur) : Haiku bloque à tort 3 réponses sur 10
(0,06 $), DeepSeek v4-pro aussi (0,14 $, décision 0013), Sonnet 5.5 aucune
(0,17 $).

Mesure complète du 2026-11-02, 52 questions, chercheur et rédacteur Haiku 5.5,
vérificateur et correcteur Sonnet 5.5, coût 1,40 $ (environ 3 min 20 par
question sur le PC, à cause des 3 recherches avec reclassement) :

| Indicateur | Référence | Agents |
|---|---|---|
| source dans les 8 passages | 0,94 | 0,94 |
| source dans le top 3 | 0,81 | 0,79 |
| refus corrects | 1,00 | 0,94 |
| citations vérifiées | 1,00 | 1,00 |
| source attendue citée | 0,91 | 0,89 |
| fidélité | 0,97 | 0,99 |
| affirmations citées | 0,73 | 0,76 |
| complétude | 0,944 | 0,947 |
| sans recopie sans guillemets | 0,13 | 0,18 |

La barrière de qualité n'est pas franchie par la chaîne avec agents : trois
réponses de la référence sont bloquées par le vérificateur
(`delai-reponse-droits`, `effacement-limites`, `amendes`), et un cas sort du
top 3. Le chercheur ne rattrape pas les articles 9, 6 et 5.

**Décision de FRANKS (2026-11-02)** : fusionner en option. Sans `--agents`, le
comportement et la référence (`evals/reference.json`) ne changent pas. À
l'étape 7 : assouplir la consigne du vérificateur (ne pas bloquer les
reformulations évidentes), mesurer le chercheur seul, puis envisager Sonnet
comme rédacteur.
