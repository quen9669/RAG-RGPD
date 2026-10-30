# 0008 : Base de recherche : LanceDB, recherche hybride

- **Date** : 2026-10-30
- **Étape** : 4 (recherche et réponses avec citations)
- **Statut** : acceptée

## Contexte

Il faut ranger les vecteurs et retrouver vite les morceaux les plus proches
d'une question. Les questions juridiques mélangent le sens (« puis-je faire
supprimer mes données ? ») et des références exactes (« article 17 »,
« DPO »), que la recherche par le sens capte mal.

## Option choisie : LanceDB, recherche hybride

Base locale (un dossier `data/index/`, au format Lance, construit sur
Apache Arrow), sans serveur. Chaque question lance deux recherches :

- par le sens (vecteurs BGE-M3, voir 0007) ;
- par mots-clés (BM25, index plein texte en français : sans accents,
  racinisation, mots vides ignorés).

Les deux classements sont fusionnés par RRF (Reciprocal Rank Fusion). Le
texte indexé commence par le titre complet du morceau (document > chapitre
> article), pour que « article 6 » soit trouvable.

## Options écartées

- **Chroma** : la plus courante dans les tutoriels, mais recherche par
  mots-clés limitée.
- **Fait main (numpy + BM25)** : très pédagogique et suffisant pour 465
  morceaux, mais tout serait à refaire pour passer à l'échelle.

## Conséquences

- L'index est reconstruit en entier par `uv run rag-index` ; il n'est pas
  versionné (il contient le texte des documents CNPD).
- Une fiche `index_meta.json` trace le modèle d'embedding, l'empreinte de
  `chunks.jsonl` et la date (lineage).
- Le score RRF ne mesure pas la qualité absolue : la règle de refus utilise
  la similarité cosinus (voir 0009).
