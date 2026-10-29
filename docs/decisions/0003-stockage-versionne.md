# 0003 : Stockage versionné des données brutes

- **Date** : 2026-10-29
- **Étape** : 2 (récupération des données, lineage)
- **Statut** : acceptée

## Contexte

Il faut garder chaque version des documents téléchargés et savoir, pour
chacune, d'où elle vient, quand elle a été récupérée et sous quelle licence.
C'est ce qui permettra de dire sur quelle version exacte s'appuie une réponse.

## Option choisie : dossiers datés + manifeste maison

- Fichiers bruts dans `data/raw/<source>/<AAAA-MM-JJ>/`, jamais modifiés.
- Manifeste `data/manifest.jsonl` (JSON Lines : un objet JSON par ligne) :
  identifiant, source, URL, date UTC, empreinte SHA-256, taille, type, licence.
  On ne fait qu'ajouter des lignes : l'historique complet est conservé.
- Si l'empreinte n'a pas changé depuis la dernière version, rien n'est écrit.

## Options écartées

- **DVC** : reconnu, mais un outil et des concepts en plus, disproportionné
  pour quelques dizaines de documents. Reste une évolution possible.
- **Git LFS** : quota gratuit limité, et ne trace ni la source ni la licence.

## Conséquences

- `data/raw/` n'est **pas** versionné dans Git. Le dépôt est public et les
  conditions de la CNPD interdisent la rediffusion de ses documents. Seul le
  manifeste (métadonnées) est publié.
- Les fichiers bruts restent sur la machine locale ; le manifeste et les
  empreintes permettent de vérifier qu'on retrouve bien les mêmes fichiers.
