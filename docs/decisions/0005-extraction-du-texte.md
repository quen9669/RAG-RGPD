# 0005 : Extraction du texte

- **Date** : 2026-10-30
- **Étape** : 3 (extraction, nettoyage, découpage)
- **Statut** : acceptée

## Contexte

Les sources brutes sont de deux formes : le RGPD en XHTML (Office des
publications) et des PDF de la CNPD, dont certains sont mis en page comme
des diapositives. Il faut en tirer un texte propre, dans l'ordre de lecture,
en gardant la structure utile aux citations (article, page).

## Option choisie

- **PDF : PyMuPDF** (version 1.x). Rapide, bonne qualité, découpe chaque page
  en blocs positionnés et sait les remettre dans l'ordre de lecture
  (`sort=True`). La version 1.x est retenue car la 2.0 ajoute des
  dépendances lourdes (onnxruntime, numpy) dont on n'a pas besoin.
- **HTML : BeautifulSoup** (analyseur `html.parser` de Python). Le XHTML
  européen repère chapitres, articles et paragraphes par des classes CSS
  (`ti-art`, `sti-art`...) : on lit ces classes.

## Options écartées

- **pypdf** : 100 % Python et licence très permissive (BSD), mais l'ordre du
  texte est moins fiable sur les PDF de type diapositives.
- **Docling** : comprend titres et tableaux, mais installation lourde
  (modèles de plusieurs Go) et lent : disproportionné pour quelques PDF.

## Conséquences

- PyMuPDF est sous licence AGPL : sans effet pour un projet personnel publié
  en source ouverte, à revoir si le code devenait un service fermé.
- Nettoyage commun (`clean_text`) : Unicode NFC, espaces insécables et
  caractères invisibles retirés, puces de polices symboles remplacées par
  « • ». Dans les PDF : mots coupés en fin de ligne recollés, en-têtes et
  pieds de page répétés (présents sur au moins la moitié des pages) et
  numéros de page retirés.
- Pour le RGPD, le préambule, la signature et les notes de bas de page sont
  ignorés ; les considérants sont gardés (ils aident à interpréter).
- Un PDF scanné (image sans texte) ne donnerait rien : il faudrait de l'OCR.
