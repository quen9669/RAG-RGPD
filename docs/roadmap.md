# Feuille de route

> **Le référentiel Franks est prioritaire** (`quen9669/Referentiel-Portfolio`, lecture seule) :
> `REGLES.md` et `apps/rag-rgpd.md` l'emportent sur tout ce document. Cible : API Python +
> front React + React Three Fiber (Streamlit n'est qu'une transition), IA uniquement via
> OmniRoute (clé `rag-rgpd`, 0 €), secrets dans le coffre OpenBao, visuels de `@quen9669/lignes`,
> « la source ou le silence ». Chaque commit cite l'ID de l'écart corrigé.
> Les étapes ci-dessous (1 à 8) sont conservées ; rien n'est retiré.

## Écarts du référentiel, dans l'ordre de traitement

| Ordre | ID | Prio | Sujet | Note |
|---|---|---|---|---|
| 1 | RAG-01 | P1 | LLM via OmniRoute (client compatible OpenAI, clé `rag-rgpd`), retrait d'Anthropic et DeepSeek en direct | Remplace la décision 0014 (Haiku direct). Première tâche. |
| 2 | RAG-06 | P2 | Relancer les 52 questions avec les modèles gratuits, comparer avec Haiku | Réf. Haiku : 0,15 $, fidélité 0,968. Après RAG-01. |
| 3 | RAG-09 | P2 | Secrets : coffre OpenBao seulement, plus de `.env` réel | Déclarer dans le compose, Franks range la valeur. |
| 4 | RAG-08 | P3 | `expedition.yaml`, `/health` et `/vitrine.json` (`uv run rag-contrat`) | Fait. |
| 5 | RAG-10 | P2 | Gouvernance visible : articles cités ✓, date de version, score des 52 questions | Composants de la bibliothèque. |
| 6 | RAG-05 | P2 | Charte : dominante `--line-ai`, citations roses, citation vérifiée verte ✓ | |
| 7 | RAG-04 | P2 | Front React + scène 3D « la constellation » | Après la bibliothèque (LIB-03). |
| 8 | RAG-02 | P1 | Migration sur le VPS | **Bloqué par DEC-008** (embeddings dans 12 Go). Ne pas commencer. |
| 9 | RAG-03 | P2 | Sidecar Tailscale | Avec RAG-02. |
| 10 | RAG-07 | P3 | Licence CNPD | Surveillé : l'app reste privée tant que non réglée. |

**Scène « la constellation »** (RAG-04) : textes RGPD, CNPD, ferroviaires et open data en
orbite ; les passages cités s'allument puis passent au vert ✓. Arrive après LIB-03.

**À décider, ne pas commencer (IDEE-01)** : OCR des documents scannés à l'ingestion, avec un
contrôle de confiance affiché en ✓.

La migration (RAG-02) pousse l'étape 7 (fine-tuning, en pause) et l'étape 8 derrière elle ;
les mesures de l'étape 8 se font via OmniRoute (RAG-01) et sont comparées à Haiku (RAG-06).

Parcours en 7 étapes, regroupées en 4 phases. Chaque phase donne un résultat
qui fonctionne seul. Avant chaque étape : explication des concepts, puis 2 ou
3 options d'outils, et c'est moi qui choisis.

## Phase 1 : Fondations (faite)

1. Environnement Python, dépôt GitHub, tests : gestion d'environnement, bonnes pratiques.
2. Récupération des données (API et téléchargements), stockage versionné : ingestion, lineage.

## Phase 2 : Premier assistant qui cite ses sources

3. Extraction du texte des PDF et pages web, nettoyage, découpage en morceaux (faite).
4. Recherche dans les textes et réponses avec citations : LLM, RAG (faite).

   **4 bis. Tests et explicabilité dans VS Code** (faite, voir plus bas).

## Phase 3 : Mesurer la qualité

5. Jeu d'environ 50 questions avec bonnes réponses, mesure des erreurs :
   évaluation, hallucinations, barrière de qualité (faite : jeu de 52
   questions, commande `rag-eval`, reclassement des passages, référence
   enregistrée ; recopies et citations à améliorer à l'étape 7, voir la
   décision [0011](decisions/0011-evaluation-et-reclassement.md)).

## Phase 4 : Aller plus loin

6. Les 3 agents (chercheur, rédacteur, vérificateur) : multi-agent.
7. Boucle d'amélioration, puis une expérience de fine-tuning mesurée.

Étape optionnelle : demander à l'assistant quelles règles s'appliquent à un
jeu de données du catalogue data.public.lu.

## Phase 5 : Élargir le corpus

8. Ajouter l'open data (européen et luxembourgeois) et la réglementation
   ferroviaire (européenne, luxembourgeoise, transfrontalière), avec une étiquette
   de domaine (rgpd, open-data, ferroviaire, transfrontalier), un seuil de refus et
   un jeu de questions par domaine. Validée le 2026-11-01 ; détail dans
   `roadmap/proposition-open-data-ferroviaire.md` (dossier du projet). Ingestion sur le PC.

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
