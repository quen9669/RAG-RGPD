# Fiche source : EUR-Lex (texte du RGPD)

- **Organisme** : Office des publications de l'Union européenne.
- **Document** : règlement (UE) 2016/679, dit RGPD (Règlement général sur la
  protection des données), version française. Identifiant CELEX
  (numéro unique des actes européens) : `32016R0679`.
- **Page de référence (pour citer)** : https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:32016R0679
- **Adresse de téléchargement** : https://publications.europa.eu/resource/celex/32016R0679
  (Cellar, le dépôt de l'Office des publications, accès prévu pour les
  programmes), avec les en-têtes `Accept: application/xhtml+xml` et
  `Accept-Language: fra` pour obtenir le texte français en XHTML.
- **Format** : page XHTML.
- **Vérifié le** : 2026-10-29.

## Conditions d'usage

Réutilisation autorisée, y compris modification, en citant la source et en
signalant les changements : politique de réutilisation de la Commission
(décision 2011/833/UE), licence CC BY 4.0 pour le contenu éditorial.
Références : [avis juridique EUR-Lex](https://eur-lex.europa.eu/content/legal-notice/legal-notice.html),
[avis de droit d'auteur data.europa.eu](https://data.europa.eu/en/copyright-notice).
Seul le Journal officiel imprimé ou électronique fait foi.

## Limites

- Le texte initial ne contient pas les rectificatifs publiés ensuite : à
  vérifier si une version consolidée est préférable.
- Le site EUR-Lex bloque les téléchargements automatisés : le 2026-10-30,
  il a renvoyé une réponse vide. D'où le passage par Cellar.

## Ajouts de l'étape 8a (2026-11-04, à vérifier à l'ingestion)

Même source, même accès (Cellar) et même licence que le RGPD, pour :
directive (UE) 2019/1024 (données ouvertes, domaine `open-data`) ;
directives 2012/34/UE, (UE) 2016/797, 2016/798, 2016/2370 et règlements (UE)
2016/796 et 2021/782 (domaine `ferroviaire`) ; règlement (CE) 883/2004
(sécurité sociale, domaine `transfrontalier`). Identifiants CELEX dans
`config/sources.toml`.

Limites propres à ces textes :

- Ce sont les versions publiées à l'origine : plusieurs ont été modifiées
  depuis (2012/34/UE, 883/2004). Une version consolidée est à étudier.
- L'extracteur HTML a été écrit pour le RGPD ; les annexes (nombreuses dans
  les textes ferroviaires) peuvent mal sortir. À contrôler avec `rag-chunk`.
- Les STI (règlements d'exécution) ne sont pas encore listées.
