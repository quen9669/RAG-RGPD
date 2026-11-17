"""Tests de la réponse citée, des refus et des commandes rag-index / rag-ask."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_rgpd.answering import cli
from rag_rgpd.answering.generator import (
    MODEL_INSUFFICIENT,
    NO_CITATION,
    NO_RELEVANT_PASSAGE,
    REFUSAL_TEXT,
    SAFETY_REFUSAL,
    ClaudeGenerator,
    answer_question,
    parse_response,
    passages_text,
)
from rag_rgpd.processing.pipeline import write_chunks
from rag_rgpd.retrieval.index import Hit
from fakes import CHUNKS, FakeClient, FakeEmbedder, FakeReranker, cited_block, response, text_block

HITS = [Hit(chunk=c, similarity=0.8 - i / 10, score=0.03) for i, c in enumerate(CHUNKS)]


class ExplodingGenerator:
    def generate(self, question: str, hits: list[Hit]) -> Any:
        raise AssertionError("le LLM n'aurait pas dû être appelé")


def test_passages_envoyes_numerotes() -> None:
    assert passages_text(HITS[:2]).startswith(f"[1] {CHUNKS[0].heading}\n{CHUNKS[0].text}\n\n[2] ")


def test_reponse_citee_avec_marqueurs() -> None:
    reply = response(
        SimpleNamespace(type="thinking", thinking=""),
        text_block(json.dumps({"segments": [
            {"texte": "Vous pouvez demander l'effacement de vos données", "source": 2,
             "extrait": "droit d'obtenir"},
            {"texte": ", sans délai excessif", "source": 2, "extrait": "meilleurs délais"},
            {"texte": ".", "source": 1, "extrait": "a consenti"},
        ]})),
    )

    answer = parse_response("effacement ?", HITS, reply)

    assert not answer.refused
    assert answer.text == (
        "Vous pouvez demander l'effacement de vos données[2], sans délai excessif[2].[1]"
    )
    assert answer.cited_numbers == [1, 2]
    assert answer.citations[0].chunk_id == CHUNKS[1].chunk_id
    assert json.loads(answer.raw_text)["segments"][0]["source"] == 2
    assert answer.model == "claude-test"


def test_reponse_sans_citation_refusee() -> None:
    answer = parse_response("q", HITS, response(cited_block("Selon moi, oui.")))

    assert answer.refused
    assert answer.text == REFUSAL_TEXT
    assert answer.refusal_code == NO_CITATION


def test_reponse_illisible_refusee() -> None:
    answer = parse_response("q", HITS, response(text_block("pas du json")))

    assert answer.refusal_code == NO_CITATION and "illisible" in answer.reason


def test_je_ne_sais_pas_du_modele() -> None:
    answer = parse_response("q", HITS, response(text_block('{"segments": []}')))

    assert answer.refusal_code == MODEL_INSUFFICIENT


def test_refus_de_securite() -> None:
    answer = parse_response("q", HITS, response(stop_reason="refusal"))

    assert answer.refusal_code == SAFETY_REFUSAL


def test_barriere_de_similarite_evite_l_appel() -> None:
    far = [Hit(chunk=CHUNKS[0], similarity=0.2, score=0.03)]

    answer = answer_question("Quelle est la capitale du Pérou ?", far, ExplodingGenerator(),
                             min_similarity=0.45)

    assert answer.refusal_code == NO_RELEVANT_PASSAGE
    assert "0.20 < seuil 0.45" in answer.reason
    assert answer.request == {}
    assert answer.params["min_similarity"] == 0.45


def test_requete_claude() -> None:
    client = FakeClient(response(cited_block("Oui", (0, "a consenti"))))
    generator = ClaudeGenerator(client, model="claude-opus-5-5")

    answer = answer_question("Le consentement suffit-il ?", HITS[:2], generator,
                             params={"k": 2})

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert "fallbacks" not in request
    content = request["messages"][0]["content"]
    assert content.startswith("Passages :\n\n[1] ") and "[2] " in content
    assert content.endswith("Question : Le consentement suffit-il ?")
    assert answer.text == "Oui[1]"
    assert answer.request["messages"] == request["messages"]
    assert answer.params == {"k": 2, "min_similarity": 0.45,
                             "llm_model": "claude-opus-5-5", "effort": "medium"}


def test_haiku_par_defaut_sans_repli() -> None:
    client = FakeClient(response(cited_block("Oui", (0, "a consenti"))))

    ClaudeGenerator(client).generate("q", HITS[:1])

    assert client.requests[0]["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in client.requests[0]


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply: SimpleNamespace) -> Path:
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    monkeypatch.setattr(cli, "make_embedder", lambda name: FakeEmbedder(name))
    monkeypatch.setattr(cli, "make_reranker", lambda name: FakeReranker())
    monkeypatch.setattr(cli, "make_generator", lambda model, cache=None: ClaudeGenerator(FakeClient(reply)))
    cli.index_main(["--chunks", str(tmp_path / "chunks.jsonl"),
                    "--index-dir", str(tmp_path / "index")])
    return tmp_path / "index"


def test_rag_ask_affiche_les_sources_et_journalise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup(tmp_path, monkeypatch, response(
        cited_block("Il faut désigner un délégué", (0, "désigne un délégué"))))
    monkeypatch.setenv("OMNIROUTE_API_KEY", "test")
    log = tmp_path / "answers.jsonl"

    cli.ask_main(["délégué à la protection des données", "--index-dir", str(index_dir),
                  "--k", "2", "--min-similarity", "0.1", "--log", str(log), "--no-route",
                  "--env-file", str(tmp_path / "absent.env")])

    out = capsys.readouterr().out
    assert "Il faut désigner un délégué[1]" in out
    assert "[1] RGPD > Chapitre II : Principes > Article 37" in out
    assert "ne remplace pas un avis juridique" in out
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["cited"] == [CHUNKS[2].chunk_id]
    assert record["retrieved"][0]["rank"] == 1
    assert record["params"]["k"] == 2
    assert record["embedding_model"] == "BAAI/bge-m3"
    assert not record["refused"]


def test_rag_ask_search_only_sans_cle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup(tmp_path, monkeypatch, response())
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)

    cli.ask_main(["effacement", "--index-dir", str(index_dir), "--search-only", "--k", "1",
                  "--env-file", str(tmp_path / "absent.env")])

    assert "Article 17" in capsys.readouterr().out


def test_rag_ask_sans_cle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index_dir = _setup(tmp_path, monkeypatch, response())
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="OMNIROUTE_API_KEY"):
        cli.ask_main(["effacement", "--index-dir", str(index_dir),
                      "--env-file", str(tmp_path / "absent.env")])


def test_rag_ask_lit_la_cle_dans_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup(tmp_path, monkeypatch, response(
        cited_block("Oui", (0, "droit d'obtenir"))))
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("OMNIROUTE_API_KEY=sk-ant-test\n", encoding="utf-8")

    cli.ask_main(["effacement", "--index-dir", str(index_dir), "--k", "1",
                  "--min-similarity", "0", "--no-route", "--env-file", str(env_file),
                  "--log", str(tmp_path / "log.jsonl")])

    assert "Oui[1]" in capsys.readouterr().out
    monkeypatch.delenv("OMNIROUTE_API_KEY")  # load_dotenv l'a posée pour tout le processus


def test_erreur_api_lisible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_rgpd.agents.omniroute import OmniRouteError

    class BrokeGenerator:
        model = "claude-haiku-5-5"

        def generate(self, question: str, hits: list[Hit]) -> Any:
            raise OmniRouteError("appel à OmniRoute impossible : 401")

    index_dir = _setup(tmp_path, monkeypatch, response())
    monkeypatch.setattr(cli, "make_generator", lambda model, cache=None: BrokeGenerator())
    monkeypatch.setenv("OMNIROUTE_API_KEY", "test")

    with pytest.raises(SystemExit, match="401"):
        cli.ask_main(["effacement", "--index-dir", str(index_dir), "--min-similarity", "0",
                      "--no-route", "--env-file", str(tmp_path / "absent.env")])
