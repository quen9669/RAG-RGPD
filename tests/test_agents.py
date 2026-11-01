"""Tests rapides de l'étape 6 : chercheur, rédacteur, vérificateur.

Faux client Anthropic scripté et faux modèles locaux : aucun réseau, aucun coût.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_rgpd.agents.llm import JsonAgent
from rag_rgpd.agents.pipeline import VERIFICATION_FAILED, AgentGenerator
from rag_rgpd.agents.researcher import AgentSearcher, make_researcher, merge_hits
from rag_rgpd.agents.verifier import make_verifier, verify
from rag_rgpd.answering.generator import ClaudeGenerator
from rag_rgpd.explain.trace import ANSWERED, REFUSED, explain_question
from rag_rgpd.retrieval.index import Hit, Searcher, build_index
from fakes import CHUNKS, FakeEmbedder, FakeReranker, response, text_block


class ScriptedClient:
    """Faux client Anthropic : renvoie ses réponses l'une après l'autre."""

    def __init__(self, *replies: SimpleNamespace) -> None:
        self.requests: list[dict[str, Any]] = []
        self._replies = list(replies)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return self._replies.pop(0)


def json_reply(data: dict[str, Any], model: str = "claude-haiku-5-5") -> SimpleNamespace:
    return response(text_block(json.dumps(data)), model=model,
                    input_tokens=1000, output_tokens=100)


def verdict_reply(*claims: tuple[str, bool]) -> SimpleNamespace:
    return json_reply({"affirmations": [{"texte": t, "appuyee": ok} for t, ok in claims]},
                      model="claude-sonnet-5-5")


def writer_reply(text: str = "Le consentement est requis") -> SimpleNamespace:
    cited = CHUNKS[0].text[:30]
    return response(text_block(text, (0, cited)), model="claude-haiku-5-5",
                    input_tokens=2000, output_tokens=100)


HITS = [Hit(chunk=c, similarity=0.8, score=0.03) for c in CHUNKS]


@pytest.fixture
def searcher(tmp_path: Path) -> Searcher:
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index")
    return Searcher(tmp_path / "index", FakeEmbedder())


# --- Vérificateur ------------------------------------------------------------


def make_answer() -> Any:
    from rag_rgpd.answering.generator import parse_response

    return parse_response("Faut-il consentir ?", HITS, writer_reply())


def test_verificateur_accepte_si_tout_est_appuye() -> None:
    agent = make_verifier(ScriptedClient(verdict_reply(("a", True), ("b", True))))

    verdict = verify(agent, make_answer())

    assert verdict.ok and verdict.claims == 2
    assert verdict.step["cost_usd"] > 0


def test_verificateur_bloque_une_affirmation_sans_appui() -> None:
    agent = make_verifier(ScriptedClient(verdict_reply(("a", True), ("inventée", False))))

    verdict = verify(agent, make_answer())

    assert not verdict.ok
    assert verdict.unsupported == ["inventée"]


def test_verificateur_bloque_si_sa_reponse_est_illisible() -> None:
    client = ScriptedClient(response(text_block("pas du json")))

    verdict = verify(make_verifier(client), make_answer())

    assert not verdict.ok
    assert "vérification impossible" in verdict.unsupported[0]


def test_message_du_verificateur_numerote_les_passages() -> None:
    client = ScriptedClient(verdict_reply(("a", True)))
    verify(make_verifier(client), make_answer())

    message = client.requests[0]["messages"][0]["content"]
    assert "[1] RGPD" in message and "Réponse de l'assistant" in message


# --- Orchestrateur -----------------------------------------------------------


def make_generator(writer_replies: list[SimpleNamespace], verdicts: list[SimpleNamespace],
                   rounds: int = 2) -> tuple[AgentGenerator, ScriptedClient]:
    writer_client = ScriptedClient(*writer_replies)
    writer = ClaudeGenerator(writer_client)
    return AgentGenerator(writer, make_verifier(ScriptedClient(*verdicts)),
                          max_rounds=rounds), writer_client


def test_reponse_verifiee_du_premier_coup() -> None:
    generator, _ = make_generator([writer_reply()], [verdict_reply(("a", True))])

    answer = generator.generate("Faut-il consentir ?", HITS)

    assert not answer.refused
    assert [s["agent"] for s in answer.steps] == ["rédacteur", "vérificateur"]


def test_reprise_avec_les_remarques_du_verificateur() -> None:
    generator, writer_client = make_generator(
        [writer_reply("première version"), writer_reply("seconde version")],
        [verdict_reply(("inventée", False)), verdict_reply(("a", True))],
    )

    answer = generator.generate("Faut-il consentir ?", HITS)

    assert not answer.refused
    assert "seconde version" in answer.text
    second = writer_client.requests[1]["messages"][0]["content"][-1]["text"]
    assert "inventée" in second
    assert answer.question == "Faut-il consentir ?"  # sans les remarques
    assert answer.usage["input_tokens"] == 4000  # deux rédactions comptées


def test_blocage_apres_le_dernier_tour() -> None:
    generator, _ = make_generator(
        [writer_reply(), writer_reply()],
        [verdict_reply(("x", False)), verdict_reply(("x", False))],
    )

    answer = generator.generate("Faut-il consentir ?", HITS)

    assert answer.refused and answer.refusal_code == VERIFICATION_FAILED
    assert answer.text == "Je ne sais pas."
    assert "x" in answer.reason


def test_un_tour_suffit_si_demande() -> None:
    generator, writer_client = make_generator([writer_reply()], [verdict_reply(("x", False))],
                                              rounds=1)

    answer = generator.generate("Faut-il consentir ?", HITS)

    assert answer.refused and len(writer_client.requests) == 1


def test_un_refus_du_redacteur_n_appelle_pas_le_verificateur() -> None:
    refusal = response(text_block("Je ne sais pas."))
    generator, _ = make_generator([refusal], [])

    answer = generator.generate("Hors sujet ?", HITS)

    assert answer.refused
    assert [s["agent"] for s in answer.steps] == ["rédacteur"]


# --- Chercheur ---------------------------------------------------------------


def test_fusion_sans_doublon_et_sans_reclasseur() -> None:
    a, b, c, d = HITS
    merged = merge_hits("q", [[a, b], [b, c], [c, d]])

    # b et c apparaissent dans deux listes : ils passent devant a et d.
    assert {h.chunk.chunk_id for h in merged[:2]} == {b.chunk.chunk_id, c.chunk.chunk_id}
    assert len(merged) == 4


def test_fusion_garde_les_articles_epingles_en_tete() -> None:
    from dataclasses import replace

    pinned = replace(HITS[1], pinned=True)
    merged = merge_hits("effacement", [[HITS[0], pinned], [HITS[2]]], FakeReranker("Brochure"))

    assert merged[0].chunk.chunk_id == pinned.chunk.chunk_id


def test_le_chercheur_ajoute_les_passages_des_reformulations(searcher: Searcher) -> None:
    client = ScriptedClient(json_reply({"requetes": ["délégué à la protection des données"]}))
    agent_searcher = AgentSearcher(searcher, make_researcher(client), min_similarity=0.0)

    plain = searcher.search("qui surveille mes données ?", k=1)
    hits = agent_searcher.search("qui surveille mes données ?", k=4)

    assert len(hits) >= len(plain)
    assert "Article 37" in {h.chunk.label for h in hits}
    steps = agent_searcher.take_steps()
    assert steps[0]["requetes"] == ["délégué à la protection des données"]
    assert agent_searcher.take_steps() == []


def test_pas_de_reformulation_sous_la_barriere(searcher: Searcher) -> None:
    client = ScriptedClient()
    agent_searcher = AgentSearcher(searcher, make_researcher(client), min_similarity=2.0)

    agent_searcher.search("recette de crêpes", k=3)

    assert client.requests == []


def test_chercheur_en_panne_retombe_sur_la_recherche_simple(searcher: Searcher) -> None:
    client = ScriptedClient(response(text_block("illisible")))
    agent_searcher = AgentSearcher(searcher, make_researcher(client), min_similarity=0.0)

    hits = agent_searcher.search("effacement des données", k=2)

    assert hits == searcher.search("effacement des données", k=2)
    assert "error" in agent_searcher.take_steps()[0]


# --- Chaîne complète ---------------------------------------------------------


def test_fiche_de_decision_avec_les_trois_agents(searcher: Searcher) -> None:
    researcher = make_researcher(ScriptedClient(json_reply({"requetes": ["licéité du traitement"]})))
    generator, _ = make_generator([writer_reply()], [verdict_reply(("a", True))])
    agent_searcher = AgentSearcher(searcher, researcher, min_similarity=0.0)

    trace, answer = explain_question("consentement au traitement", agent_searcher, generator,
                                     k=4, min_similarity=0.0)

    assert trace.decision == ANSWERED
    agents = [s["agent"] for s in trace.params["agents"]]
    assert agents == ["chercheur", "rédacteur", "vérificateur"]
    # coût = rédacteur (dans l'usage) + chercheur + vérificateur
    assert trace.cost_usd > trace.params["agents"][0]["cost_usd"]


def test_fiche_blocage_du_verificateur(searcher: Searcher) -> None:
    generator, _ = make_generator([writer_reply(), writer_reply()],
                                  [verdict_reply(("x", False))] * 2)

    trace, _ = explain_question("consentement au traitement", searcher, generator,
                                k=4, min_similarity=0.0)

    assert trace.decision == REFUSED and trace.refusal_code == VERIFICATION_FAILED


def test_json_agent_utilise_le_cache(tmp_path: Path) -> None:
    from rag_rgpd.answering.cache import ResponseCache

    from anthropic.types.beta import BetaMessage

    real = BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": '{"requetes": []}'}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 30},
    })
    client = ScriptedClient(real)
    agent = JsonAgent("test", "consigne", {"type": "object"}, model="claude-haiku-5-5",
                      client=client, cache=ResponseCache(tmp_path))

    first = agent.call("q")
    second = agent.call("q")

    assert len(client.requests) == 1
    assert first.step["cost_usd"] > 0 and second.step["cost_usd"] == 0.0
