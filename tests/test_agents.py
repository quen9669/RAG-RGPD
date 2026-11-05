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
from rag_rgpd.agents.router import RoutedSearcher, make_router
from rag_rgpd.agents.verifier import make_verifier, verify
from rag_rgpd.answering.generator import ClaudeGenerator
from rag_rgpd.explain.trace import ANSWERED, REFUSED, explain_question
from rag_rgpd.retrieval.index import Hit, Searcher, build_index
from fakes import CHUNKS, FakeEmbedder, FakeReranker, make_chunk, response, text_block


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


# --- DeepSeek ----------------------------------------------------------------


def _deepseek(reply: dict[str, Any] | list[dict[str, Any]], key: str | None = "sk-test"):
    import httpx

    from rag_rgpd.agents.deepseek import DeepSeekClient

    replies = [reply] if isinstance(reply, dict) else list(reply)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=replies.pop(0))

    client = DeepSeekClient(api_key=key, http=httpx.Client(transport=httpx.MockTransport(handler)))
    return client, seen


def _ds_reply(content: str) -> dict[str, Any]:
    return {"id": "x", "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50,
                      "prompt_cache_hit_tokens": 400}}


def test_deepseek_repond_comme_un_client_anthropic() -> None:
    client, seen = _deepseek(_ds_reply('{"affirmations": [{"texte": "a", "appuyee": true}]}'))
    agent = make_verifier(client, model="deepseek-flash")

    verdict = verify(agent, make_answer())

    assert verdict.ok
    body = json.loads(seen[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert "affirmations" in body["messages"][0]["content"]  # le schéma est dans la consigne
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    usage = verdict.step["usage"]
    assert usage["input_tokens"] == 600 and usage["cache_read_input_tokens"] == 400
    assert 0 < verdict.step["cost_usd"] < 0.001


def test_deepseek_reponse_vide_puis_reussie() -> None:
    client, _ = _deepseek([_ds_reply(""), _ds_reply('{"requetes": []}')])

    call = make_researcher(client, model="deepseek-flash").call("q")

    assert call.data == {"requetes": []}


def test_deepseek_toujours_vide_bloque_la_reponse() -> None:
    client, _ = _deepseek([_ds_reply(""), _ds_reply("")])

    verdict = verify(make_verifier(client, model="deepseek-flash"), make_answer())

    assert not verdict.ok and "vérification impossible" in verdict.unsupported[0]


def test_deepseek_sans_cle() -> None:
    client, _ = _deepseek(_ds_reply("{}"), key=None)

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        client.beta.messages.create(model="deepseek-flash", system="s", messages=[])


def test_choix_du_client_selon_le_modele(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_rgpd.agents.deepseek import DeepSeekClient, is_deepseek, make_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert is_deepseek("deepseek-v4-pro") and not is_deepseek("claude-haiku-5-5")
    assert isinstance(make_client("deepseek-flash"), DeepSeekClient)
    assert not isinstance(make_client("claude-haiku-5-5"), DeepSeekClient)


# --- Routeur de domaines (étape 8) -------------------------------------------

@pytest.fixture
def mixed_searcher(tmp_path: Path) -> Searcher:
    rail = make_chunk(
        9, "Article 17", "Entretien des voies",
        "Le gestionnaire de l'infrastructure ferroviaire assure l'entretien des voies.",
        document_id="infra-ue", domain="ferroviaire", chunk_id="infra-ue-abcdef12-0009",
    )
    build_index([*CHUNKS, rail], FakeEmbedder(), tmp_path / "index")
    return Searcher(tmp_path / "index", FakeEmbedder())


def test_le_routeur_limite_la_recherche_aux_domaines_choisis(mixed_searcher: Searcher) -> None:
    client = ScriptedClient(json_reply({"domaines": ["rgpd"]}))
    routed = RoutedSearcher(mixed_searcher, make_router(client))

    hits = routed.search("entretien des voies du gestionnaire", k=5)

    assert {h.chunk.domain for h in hits} == {"rgpd"}
    steps = routed.take_steps()
    assert steps[0]["agent"] == "routeur"
    assert steps[0]["domaines"] == ["rgpd"]


def test_le_routeur_est_appele_une_fois_par_question(mixed_searcher: Searcher) -> None:
    client = ScriptedClient(json_reply({"domaines": ["ferroviaire"]}))
    routed = RoutedSearcher(mixed_searcher, make_router(client))

    routed.search("entretien des voies", k=3)
    routed.ranks("entretien des voies")

    assert len(client.requests) == 1


def test_liste_vide_ou_panne_du_routeur_cherche_partout(mixed_searcher: Searcher) -> None:
    for reply in (json_reply({"domaines": []}), response(text_block("illisible"))):
        routed = RoutedSearcher(mixed_searcher, make_router(ScriptedClient(reply)))

        hits = routed.search("vélo entretien des voies délégué", k=5)

        assert {h.chunk.domain for h in hits} == {"rgpd", "ferroviaire"}


def test_le_suivi_du_routeur_remonte_par_le_chercheur(mixed_searcher: Searcher) -> None:
    routed = RoutedSearcher(mixed_searcher, make_router(ScriptedClient(
        json_reply({"domaines": ["rgpd"]}))))
    agent_searcher = AgentSearcher(routed, None)

    agent_searcher.search("effacement des données", k=2)

    assert [s["agent"] for s in agent_searcher.take_steps()] == ["routeur"]
