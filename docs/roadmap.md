# Feuille de route

Parcours en 7 étapes, regroupées en 4 phases. Chaque phase donne un résultat
qui fonctionne seul. Avant chaque étape : explication des concepts, puis 2 ou
3 options d'outils, et c'est moi qui choisis.

## Phase 1 : Fondations (faite)

1. Environnement Python, dépôt GitHub, tests : gestion d'environnement, bonnes pratiques.
2. Récupération des données (API et téléchargements), stockage versionné : ingestion, lineage.

## Phase 2 : Premier assistant qui cite ses sources

3. Extraction du texte des PDF et pages web, nettoyage, découpage en morceaux (faite).
4. Recherche dans les textes et réponses avec citations : LLM, RAG (faite).

   **4 bis. Tests et explicabilité dans VS Code** (en cours, voir plus bas).

## Phase 3 : Mesurer la qualité

5. Jeu d'environ 50 questions avec bonnes réponses, mesure des erreurs :
   évaluation, hallucinations, barrière de qualité.

## Phase 4 : Aller plus loin

6. Les 3 agents (chercheur, rédacteur, vérificateur) : multi-agent.
7. Boucle d'amélioration, puis une expérience de fine-tuning mesurée.

Étape optionnelle : demander à l'assistant quelles règles s'appliquent à un
jeu de données du catalogue data.public.lu.

---

## Étape 4 bis : tests et explicabilité dans VS Code

### Objectif

Pour chaque question posée, pouvoir **comprendre chaque décision du système**,
et **lancer beaucoup d'essais** directement depuis VS Code, sans copier-coller
dans un terminal.

Elle se place juste après l'étape 4 (il faut d'abord un assistant qui répond)
et avant l'étape 5 : les essais faits ici deviennent la matière du jeu de
questions de l'étape 5.

### Ce que je veux voir pour chaque question

Une « fiche de décision » (une trace) qui répond à ces questions :

| Décision | Ce que la fiche montre |
|---|---|
| Quels morceaux ont été trouvés ? | La liste des morceaux récupérés, avec leur rang, leur score de similarité et leur source (article, page, document). |
| Pourquoi ceux-là ? | Les paramètres utilisés : nombre de morceaux demandés (top-k), seuil de score, filtres éventuels. Les morceaux écartés juste sous le seuil. |
| Qu'a reçu le modèle ? | Le texte exact envoyé au LLM (consignes + morceaux + question). |
| Pourquoi cette réponse ? | La réponse brute du modèle, et pour chaque citation, le morceau qu'elle désigne. Une citation qui ne correspond à aucun morceau fourni est signalée. |
| Pourquoi un refus ? | La raison précise du « je ne sais pas » : aucun morceau au-dessus du seuil, ou le modèle a jugé les morceaux insuffisants, ou une citation n'a pas pu être vérifiée. |
| Combien ça a coûté ? | Temps de chaque étape, nombre de tokens (unités de texte facturées par le fournisseur du LLM), coût estimé. |

### Ce qu'on construira

1. **La trace** : un objet Python rempli à chaque étape du traitement
   (recherche, sélection, appel au LLM, vérification des citations), enregistré
   en JSONL (une trace par ligne) dans `data/traces/`. Elle part du résultat
   `Answer` et du journal `data/logs/answers.jsonl` livrés à l'étape 4, en y
   ajoutant la requête complète, les morceaux écartés et le coût. Ces traces contiennent
   des extraits de documents CNPD, elles restent donc hors de Git comme
   `data/raw/`. Elles servent aussi de « journal des réponses » prévu par la
   gouvernance.
2. **Une commande `--explain`** : poser une question et afficher la fiche de
   décision lisiblement dans le terminal de VS Code.
3. **Des notebooks** (cahiers interactifs, dans `notebooks/`) pour explorer :
   comparer plusieurs seuils ou valeurs de top-k sur la même question,
   visualiser les scores, relire une ancienne trace.
4. **Des tests pytest à plusieurs niveaux**, visibles dans le panneau
   « Testing » de VS Code :
   - tests rapides sans LLM (un faux modèle remplace Claude) : la recherche,
     le seuil, la règle de refus, la vérification des citations ;
   - tests « de comportement » : une liste de questions dans un fichier
     (question, document ou article attendu, refus attendu ou non), chaque
     ligne devenant un test ;
   - tests qui appellent vraiment le LLM, marqués à part (`llm`) pour ne les
     lancer qu'à la demande, car ils coûtent de l'argent.
5. **La configuration VS Code** : `launch.json` pour déboguer une question pas
   à pas (points d'arrêt dans la recherche ou la génération), `tasks.json`
   pour les lancements fréquents, extensions recommandées (Python, Jupyter).
6. **Les essais en lot** : lancer N questions avec plusieurs variantes de
   paramètres et obtenir un tableau comparatif (quels morceaux, refus ou
   non, citations valides ou non). Un cache des réponses du LLM évite de
   payer deux fois le même essai.

### Outils retenus

Traces en JSONL fait maison, notebooks Jupyter, cas de test en YAML
(décision [0010](decisions/0010-outils-d-explicabilite.md)). Mode d'emploi :
section « Tester et comprendre dans VS Code » du README.

### Ce que l'étape 4 doit déjà prévoir

Pour ne pas tout réécrire ensuite, le code de l'étape 4 renvoie un résultat
structuré (et pas seulement un texte) : morceaux récupérés avec scores,
paramètres utilisés, texte envoyé au modèle, réponse, citations et raison
d'un éventuel refus. La trace complète et les outils autour viennent en 4 bis.
