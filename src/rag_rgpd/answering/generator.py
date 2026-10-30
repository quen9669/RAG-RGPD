"""Rédaction d'une réponse citée, ou « je ne sais pas ».

Trois barrières contre l'hallucination (réponse inventée présentée comme
vraie), de la moins chère à la plus fine :

1. **Avant le LLM** : si le meilleur passage retrouvé est trop éloigné de la
   question (similarité sous le seuil), on répond « je ne sais pas » sans
   appeler le modèle.
2. **Dans la consigne** : le modèle ne doit utiliser que les passages fournis
   et répondre exactement « Je ne sais pas. » s'ils ne suffisent pas.
3. **Après le LLM** : une réponse sans aucune citation est traitée comme un
   refus. On ne publie jamais une affirmation qu'aucun passage n'appuie.

Les passages sont envoyés à Claude sous forme de blocs ``search_result`` avec
les citations activées : c'est l'API qui relie chaque phrase de la réponse au
passage exact qui l'appuie (texte cité mot pour mot), au lieu de compter sur
le modèle pour recopier correctement des numéros de source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_rgpd.retrieval.index import DEFAULT_MIN_SIMILARITY, Hit

DEFAULT_MODEL = "claude-opus-5-5"

REFUSAL_TEXT = "Je ne sais pas."

SYSTEM_PROMPT = """\
Tu es un assistant qui répond à des questions sur la protection des données \
personnelles au Luxembourg (RGPD et documents de la CNPD).

Règles :
- Réponds en français, en langage simple, en expliquant chaque sigle à sa \
première utilisation.
- Appuie-toi uniquement sur les résultats de recherche fournis dans le \
message. N'utilise pas tes connaissances générales, même si tu es sûr de toi.
- Chaque affirmation doit pouvoir être reliée à un passage précis.
- Si les passages ne permettent pas de répondre, réponds exactement \
« Je ne sais pas. » et rien d'autre. Une réponse partielle est acceptable \
seulement si tu dis clairement ce que les passages ne couvrent pas.
- Tu n'es pas un avocat : ne présente pas ta réponse comme un avis juridique.
"""


@dataclass(frozen=True)
class Citation:
    """Lien entre un extrait de la réponse et le passage qui l'appuie.

    Attributes:
        source_number: numéro du passage dans la liste des sources (à
            partir de 1, comme les marqueurs ``[1]`` de la réponse).
        cited_text: phrase du passage reprise mot pour mot.
    """

    source_number: int
    cited_text: str


@dataclass(frozen=True)
class Answer:
    """Réponse à une question, avec ses sources.

    Attributes:
        question: question posée.
        text: réponse, avec des marqueurs ``[n]`` renvoyant aux sources.
        refused: ``True`` si l'assistant répond « je ne sais pas ».
        reason: pourquoi il a refusé (vide sinon).
        hits: passages retrouvés et montrés au modèle, dans l'ordre des
            numéros de source.
        citations: citations relevées dans la réponse.
        model: modèle qui a réellement répondu (vide si non appelé).
    """

    question: str
    text: str
    refused: bool
    reason: str = ""
    hits: list[Hit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    model: str = ""

    @property
    def cited_numbers(self) -> list[int]:
        """Numéros des sources effectivement citées, sans doublon."""
        return sorted({c.source_number for c in self.citations})


class AnswerGenerator(Protocol):
    """Ce que l'assistant attend d'un LLM.

    Un protocole pour pouvoir changer de fournisseur (Mistral, modèle local)
    sans toucher au reste : c'est la mesure prévue dans le registre des
    risques contre la dépendance à un fournisseur.
    """

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        """Rédige une réponse à partir des passages retrouvés."""
        ...


def refuse(question: str, reason: str, hits: list[Hit] | None = None, model: str = "") -> Answer:
    """Construit une réponse « je ne sais pas » avec sa raison."""
    return Answer(
        question=question, text=REFUSAL_TEXT, refused=True, reason=reason,
        hits=hits or [], model=model,
    )


def search_result_blocks(hits: list[Hit]) -> list[dict[str, Any]]:
    """Présente les passages au modèle sous forme de résultats de recherche.

    ``source`` porte l'identifiant du morceau : c'est lui que l'API renvoie
    dans chaque citation, ce qui permet de remonter sans ambiguïté au
    morceau, puis au fichier brut (lineage).
    """
    return [
        {
            "type": "search_result",
            "source": hit.chunk.chunk_id,
            "title": hit.chunk.heading,
            "content": [{"type": "text", "text": hit.chunk.text}],
            "citations": {"enabled": True},
        }
        for hit in hits
    ]


def parse_response(question: str, hits: list[Hit], response: Any) -> Answer:
    """Transforme la réponse de l'API en ``Answer``.

    La réponse arrive découpée en blocs de texte ; ceux qui s'appuient sur
    un passage portent une liste ``citations``. On recolle les blocs en
    ajoutant ``[n]`` après chaque bloc cité.

    Args:
        question: question posée.
        hits: passages envoyés, dans le même ordre que dans la requête.
        response: message renvoyé par ``client.beta.messages.create``.

    Returns:
        La réponse, ou un refus si le modèle a refusé ou n'a rien cité.
    """
    model = getattr(response, "model", "")
    if response.stop_reason == "refusal":
        return refuse(question, "le modèle a décliné la demande", hits, model)

    parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        # On ignore les blocs de réflexion et les blocs « fallback » (passage
        # de relais à un autre modèle) : seul le texte compose la réponse.
        if block.type != "text":
            continue
        parts.append(block.text)
        numbers = []
        for cite in block.citations or []:
            if getattr(cite, "type", "") != "search_result_location":
                continue
            number = cite.search_result_index + 1
            citations.append(Citation(source_number=number, cited_text=cite.cited_text))
            numbers.append(number)
        if numbers:
            parts.append("".join(f"[{n}]" for n in sorted(set(numbers))))

    text = "".join(parts).strip()
    if text.startswith("Je ne sais pas"):
        return refuse(question, "les passages trouvés ne permettent pas de répondre", hits, model)
    if not citations:
        # Barrière 3 : une réponse qui ne cite rien n'est pas vérifiable.
        return refuse(question, "le modèle a répondu sans citer de source", hits, model)
    return Answer(
        question=question, text=text, refused=False, hits=hits,
        citations=citations, model=model,
    )


class ClaudeGenerator:
    """Rédaction de la réponse par Claude, avec citations vérifiées par l'API."""

    def __init__(
        self,
        client: Any = None,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
    ) -> None:
        """Prépare le client.

        Args:
            client: client ``anthropic.Anthropic`` (un faux dans les tests).
                Par défaut, un client qui lit la clé dans la variable
                d'environnement ``ANTHROPIC_API_KEY``.
            model: modèle Claude à utiliser.
            effort: profondeur de réflexion (``low`` à ``max``). ``medium``
                suffit pour reformuler des passages fournis ; à mesurer à
                l'étape 5.
        """
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model
        self.effort = effort

    def generate(self, question: str, hits: list[Hit]) -> Answer:
        """Rédige une réponse citée (voir ``AnswerGenerator.generate``)."""
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": self.effort},
            # Si le modèle décline la demande (filtre de sécurité trop
            # prudent), l'API la relance elle-même sur un modèle de secours.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        *search_result_blocks(hits),
                        {"type": "text", "text": f"Question : {question}"},
                    ],
                }
            ],
        )
        return parse_response(question, hits, response)


def answer_question(
    question: str,
    hits: list[Hit],
    generator: AnswerGenerator,
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> Answer:
    """Répond à une question, ou refuse sans appeler le LLM si rien n'est proche.

    Args:
        question: question posée.
        hits: passages retrouvés par la recherche.
        generator: LLM chargé de rédiger.
        min_similarity: seuil de la barrière 1.

    Returns:
        La réponse citée ou un refus motivé.
    """
    best = max((h.similarity for h in hits), default=0.0)
    if best < min_similarity:
        return refuse(
            question,
            f"aucun passage assez proche de la question "
            f"(meilleure similarité {best:.2f} < seuil {min_similarity:.2f})",
            hits,
        )
    return generator.generate(question, hits)
