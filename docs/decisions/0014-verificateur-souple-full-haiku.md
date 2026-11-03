# 0014 : vérificateur assoupli et agents entièrement en Haiku

- **Date** : 2026-11-03
- **Étape** : 7
- **Statut** : appliqué (PR #9 pour la consigne, cette décision pour le modèle par défaut)

## Contexte

La mesure de l'étape 6 (décision 0012) bloquait à tort 3 bonnes réponses
(delai-reponse-droits, effacement-limites, amendes) et ne laissait que
Sonnet 5.5 tenir le rôle de vérificateur, pour 1,40 $ les 52 questions.
FRANKS veut réduire le coût.

## Option choisie

- Consigne du vérificateur assouplie (PR #9) : les repères pour le lecteur
  (sens d'un considérant, sigles, niveaux d'amende) et les reformulations ne
  comptent plus comme affirmations ; une affirmation est refusée seulement si
  elle ajoute un fait, un chiffre, un délai ou une condition absents des
  passages, ou si elle transpose un droit à un autre.
- Le vérificateur par défaut devient `claude-haiku-5-5` (comme le chercheur et
  le rédacteur). Sonnet reste disponible avec `--verifier-model`.
- Le correcteur de `rag-eval` garde Sonnet par défaut, pour rester comparable
  à `evals/reference.json`.

## Mesure (2026-11-03, 52 questions, `--agents`, tout en Haiku)

- Aucun refus `verification_echouee` sur les 47 questions légitimes (avant : 3) ;
  les 5 refus sont des questions hors sujet.
- Coût : 0,15 $ (contre 1,40 $).
- Fidélité 0,968 (réf. sans agents 0,969), refus corrects 1,00, complétude 0,930
  (0,944), top 3 0,787 (0,809), source citée 0,894 (0,915), affirmations citées
  0,629 (0,734).

## Limites

- Le correcteur de cette mesure était Haiku, celui de la référence Sonnet :
  l'écart sur les affirmations citées n'est pas strictement comparable. Une
  correction avec `--judge-model claude-sonnet-5-5` reste à faire.
- Le test de 10 questions montrait une petite déduction du rédacteur (art. 18)
  que le vérificateur peut encore bloquer à raison.
- `evals/reference.json` n'est pas modifié (référence sans agents).
