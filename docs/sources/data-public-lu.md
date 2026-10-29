# Fiche source : data.public.lu

- **Organisme** : portail luxembourgeois des données ouvertes (logiciel udata).
- **Accès** : API (Application Programming Interface) JSON,
  https://data.public.lu/api/1/datasets/ , documentation : https://data.public.lu/en/docapi/
- **Ce qu'on récupère** : le catalogue, c'est-à-dire la description de chaque
  jeu de données (titre, organisme, licence, ressources), pas les données elles-mêmes.
- **Vérifié le** : 2026-10-29.

## Conditions d'usage

Le site indique : « sauf mention contraire, tout le contenu de ce site est
disponible sous licence Creative Commons CC0 » (aucune restriction). Chaque
jeu de données a sa propre licence, indiquée dans le champ `license`.

## Limites

- Pas de limite de débit documentée : on garde 2 s entre les pages.
- Certains champs (statistiques de consultation) changent souvent, donc
  l'empreinte du catalogue change presque à chaque récupération.
