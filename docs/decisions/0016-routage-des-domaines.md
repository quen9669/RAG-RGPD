# 0016 — Étape 8 : routage automatique des domaines

Date : 2026-11-05. Choisi par FRANKS sur une carte de décision.

## Contexte

Avec 1907 morceaux (RGPD, open data, ferroviaire, transfrontalier), la
recherche sans filtre laisse des textes d'un autre sujet brouiller le
classement. Mesure en Haiku (52 questions RGPD, sans filtre) : source dans
les k 0,915 contre 0,936, top 3 0,745 contre 0,787, source citée 0,872 contre
0,894. « article 17 » existe dans une dizaine de textes (corrigé en partie par
l'épinglage restreint aux documents pertinents, PR #13).

## Options

1. **Routage automatique** (choisi) : un petit modèle (Haiku, effort bas,
   environ 0,0005 $ par question) choisit le ou les domaines avant la
   recherche. Liste vide, panne ou réponse illisible : on cherche partout.
2. RGPD par défaut, élargi sur mots-clés : simple mais fragile sur les
   questions croisées.
3. Garder tel quel et prendre cette mesure comme nouvelle référence : le
   recul risque de croître à chaque ajout.

## Mise en œuvre

- `Searcher.search(..., domains=)` et `ranks(..., domains=)` : filtre propre à
  la question, sinon celui du moteur (`--domain`), sinon tout l'index.
- `agents/router.py` : agent « routeur » (schéma JSON, domaines énumérés) et
  `RoutedSearcher`, mis devant le chercheur. Une décision par question
  (mémorisée) ; son coût et son choix figurent dans la fiche de décision.
- `--route/--no-route` (actif par défaut) et `--router-model` sur `rag-ask` et
  `rag-eval` ; actif aussi dans le chat. Inactif en recherche seule (pas de
  modèle), ou si `--domain` est donné.

## Limites

- L'invite du chercheur parle encore du RGPD : ses reformulations pour une
  question ferroviaire ne sont pas adaptées. À généraliser si besoin (changer
  l'invite invalide le cache des réponses).
- Le jeu de 52 questions ne contient que du RGPD : il mesure le non-recul,
  pas la qualité sur les autres domaines (questions à écrire).
- Le routeur peut se tromper : une question RGPD classée ailleurs perd ses
  passages. D'où « dans le doute, garder rgpd ».
