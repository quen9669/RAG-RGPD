"""Routage de la question vers un ou plusieurs domaines (étape 8, décision 0016).

Avec un corpus de plusieurs sujets (RGPD, open data, ferroviaire,
transfrontalier), une recherche dans tout l'index laisse des textes d'un
autre sujet brouiller le classement : « article 17 » existe dans une dizaine
de textes. Un petit modèle lit donc la question et choisit les domaines où
chercher, avant la recherche.

Garde-fous :

- en cas de doute, de panne ou de réponse illisible, on cherche partout
  (jamais de réponse inventée à la place de la recherche) ;
- une question qui touche deux sujets (vidéosurveillance en gare) reçoit
  les deux domaines ;
- le choix et son coût sont visibles dans la fiche de décision (agent
  « routeur »).
"""

from __future__ import annotations

from typing import Any

from rag_rgpd.agents.llm import JsonAgent
from rag_rgpd.retrieval.index import DEFAULT_K, DOMAINS, Hit

DEFAULT_ROUTER_MODEL = "claude-haiku-5-5"

ROUTER_PROMPT = """\
Tu aiguilles les questions d'un assistant juridique vers les bons textes.

Domaines :
- rgpd : protection des données personnelles (RGPD, droits des personnes, \
responsable du traitement, sous-traitant, délégué à la protection des \
données, violations de données, CNPD au Luxembourg, sanctions).
- open-data : données ouvertes et réutilisation des informations du secteur \
public (directive 2019/1024, licences, portails de données).
- ferroviaire : chemins de fer de l'Union européenne (gestionnaires \
d'infrastructure, entreprises ferroviaires, interopérabilité, sécurité, \
droits des voyageurs ferroviaires, vélos dans le train, retards).
- transfrontalier : travailleurs frontaliers, coordination de la sécurité \
sociale entre pays (règlement 883/2004).

Choisis le ou les domaines dont les textes peuvent contenir la réponse. Une \
question qui relève de deux sujets reçoit les deux. Si tu hésites ou si la \
question ne relève d'aucun domaine, renvoie une liste vide (on cherchera \
partout). Dans le doute entre rgpd et un autre domaine, garde les deux.
"""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "domaines": {"type": "array", "items": {"type": "string", "enum": list(DOMAINS)}}
    },
    "required": ["domaines"],
    "additionalProperties": False,
}


def make_router(client: Any = None, *, model: str = DEFAULT_ROUTER_MODEL,
                cache: Any = None) -> JsonAgent:
    """Construit le petit agent qui choisit les domaines."""
    return JsonAgent("routeur", ROUTER_PROMPT, SCHEMA, model=model, client=client,
                     cache=cache, effort="low")


class RoutedSearcher:
    """Moteur de recherche qui choisit d'abord les domaines (même interface que ``Searcher``)."""

    def __init__(self, inner: Any, router: JsonAgent) -> None:
        """Enveloppe un ``Searcher``.

        Args:
            inner: moteur de recherche (accepte ``domains`` à chaque appel).
            router: agent de routage.
        """
        self._inner = inner
        self._router = router
        self._steps: list[dict[str, Any]] = []
        self._routes: dict[str, tuple[str, ...] | None] = {}
        self.meta = inner.meta

    @property
    def settings(self) -> dict[str, object]:
        """Réglages de la recherche, plus le modèle du routeur."""
        return {**getattr(self._inner, "settings", {}), "router_model": self._router.model}

    @property
    def reranker(self) -> Any:
        """Modèle de reclassement du moteur sous-jacent."""
        return getattr(self._inner, "reranker", None)

    def take_steps(self) -> list[dict[str, Any]]:
        """Renvoie (et vide) le suivi du routeur depuis le dernier appel."""
        steps, self._steps = self._steps, []
        return steps

    def route(self, question: str) -> tuple[str, ...] | None:
        """Domaines où chercher, ou ``None`` pour tout l'index (une fois par question)."""
        if question in self._routes:
            return self._routes[question]
        domains: tuple[str, ...] | None = None
        try:
            call = self._router.call(f"Question : {question}")
        except ValueError as error:
            self._steps.append({"agent": "routeur", "error": str(error), "cost_usd": 0.0})
        else:
            chosen = tuple(d for d in dict.fromkeys(call.data.get("domaines", [])) if d in DOMAINS)
            domains = chosen or None
            self._steps.append({**call.step, "domaines": list(chosen)})
        self._routes[question] = domains
        return domains

    def search(self, question: str, *, k: int = DEFAULT_K) -> list[Hit]:
        """Recherche dans les domaines choisis pour la question."""
        return self._inner.search(question, k=k, domains=self.route(question))

    def ranks(self, question: str, *, depth: int = 50) -> dict[str, tuple[int | None, int | None]]:
        """Rangs dans les mêmes domaines que la recherche."""
        return self._inner.ranks(question, depth=depth, domains=self.route(question))
