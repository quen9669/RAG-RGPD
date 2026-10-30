"""Tests rapides de l'étape 4 bis : fiche de décision, cache, coût, cas, lots.

Faux modèle d'embedding et faux LLM : aucun réseau, aucun coût.
"""

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag_rgpd.answering import cli as ask_cli
from rag_rgpd.answering.cache import ResponseCache
from rag_rgpd.answering.generator import (
    NO_RELEVANT_PASSAGE,
    UNVERIFIED_CITATION,
    ClaudeGenerator,
    parse_response,
)
from rag_rgpd.explain import cli as explain_cli
from rag_rgpd.explain.batch import make_variants, run_batch, summarize, write_csv
from rag_rgpd.explain.cases import Case, first_match_rank, load_cases, matches
from rag_rgpd.explain.cost import estimate_cost
from rag_rgpd.explain.render import format_prompt, format_trace, format_trace_list
from rag_rgpd.explain.trace import (
    ANSWERED,
    REFUSED,
    SEARCH_ONLY,
    explain_question,
    find_trace,
    read_traces,
    write_trace,
)
from rag_rgpd.processing.pipeline import write_chunks
from rag_rgpd.retrieval.index import Hit, Searcher, build_index
from fakes import CHUNKS, FakeEmbedder, FakeGenerator, response, text_block

HITS = [Hit(chunk=c, similarity=0.8 - i / 10, score=0.03) for i, c in enumerate(CHUNKS)]
ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def searcher(tmp_path: Path) -> Searcher:
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index")
    return Searcher(tmp_path / "index", FakeEmbedder())


# --- Vérification des citations et tokens -----------------------------------


def test_citation_verifiee() -> None:
    answer = parse_response("q", HITS, response(text_block("Oui", (1, "droit d'obtenir"))))

    assert answer.citations[0].verified
    assert not answer.refused


def test_citation_absente_du_passage_entraine_un_refus() -> None:
    answer = parse_response("q", HITS, response(text_block("Oui", (1, "texte inventé"))))

    assert answer.refused
    assert answer.refusal_code == UNVERIFIED_CITATION
    assert "[2]" in answer.reason
    assert not answer.citations[0].verified


def test_citation_vers_un_passage_inexistant() -> None:
    answer = parse_response("q", HITS[:1], response(text_block("Oui", (5, "x"))))

    assert answer.refusal_code == UNVERIFIED_CITATION
    assert answer.citations[0].chunk_id == ""


def test_citation_tolere_les_espaces() -> None:
    answer = parse_response("q", HITS, response(text_block("Oui", (1, "droit  d'obtenir\nl'effacement"))))

    assert answer.citations[0].verified


def test_tokens_releves() -> None:
    answer = parse_response("q", HITS, response(text_block("Oui", (0, "a consenti")),
                                                input_tokens=1200, output_tokens=80))

    assert answer.usage["input_tokens"] == 1200
    assert answer.usage["output_tokens"] == 80
    assert answer.usage["cache_read_input_tokens"] == 0


def test_cout_estime() -> None:
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    assert estimate_cost("claude-haiku-5-5", usage) == pytest.approx(0.60)
    assert estimate_cost("claude-opus-5-5", {"cache_read_input_tokens": 1_000_000}) == pytest.approx(0.40)
    assert estimate_cost("modele-inconnu", usage) is None
    assert estimate_cost("modele-inconnu", {}) == 0.0


# --- Cache des réponses -----------------------------------------------------


def _api_message(cited: str) -> Any:
    from anthropic.types.beta import BetaMessage

    return BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "Oui", "citations": [{
            "type": "search_result_location", "cited_text": cited, "search_result_index": 0,
            "source": CHUNKS[0].chunk_id, "title": "t", "start_block_index": 0,
            "end_block_index": 1}]}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 30},
    })


def test_cache_evite_le_second_appel(tmp_path: Path) -> None:
    calls: list[dict] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return _api_message("a consenti")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
    generator = ClaudeGenerator(client, cache=ResponseCache(tmp_path / "cache"))

    first = generator.generate("q", HITS[:1])
    second = generator.generate("q", HITS[:1])
    other = generator.generate("autre question", HITS[:1])

    assert len(calls) == 2  # « q » une seule fois, puis « autre question »
    assert not first.from_cache
    assert second.from_cache
    assert not other.from_cache
    assert second.text == first.text == "Oui[1]"
    assert second.usage == first.usage


def test_cle_de_cache_independante_de_l_ordre() -> None:
    assert ResponseCache.key({"a": 1, "b": [1, 2]}) == ResponseCache.key({"b": [1, 2], "a": 1})
    assert ResponseCache.key({"a": 1}) != ResponseCache.key({"a": 2})


# --- Fiche de décision ------------------------------------------------------


def test_fiche_complete(searcher: Searcher) -> None:
    trace, answer = explain_question("droit à l'effacement", searcher, FakeGenerator(),
                                     k=2, extra=1, min_similarity=0.1,
                                     params={"embedding_model": "faux-modele"})

    assert trace.decision == ANSWERED
    assert [p.sent for p in trace.passages] == [True, True, False]
    assert trace.passages[0].label == "Article 17"
    assert trace.passages[0].meaning_rank is not None
    assert trace.passages[0].cited_as == [1]
    assert trace.citations[0]["label"] == "Article 17"
    assert trace.citations[0]["verified"]
    assert trace.params["k"] == 2
    assert trace.params["llm_model"] == "claude-haiku-5-5"
    assert trace.index["embedding_model"] == "faux-modele"
    assert trace.cost_usd == pytest.approx(2000 * 0.10 / 1e6 + 100 * 0.50 / 1e6)
    assert set(trace.timings) == {"search", "ranks", "llm", "total"}
    assert trace.answer == answer.text


def test_fiche_refus_avant_le_llm(searcher: Searcher) -> None:
    trace, _ = explain_question("effacement", searcher, FakeGenerator(), min_similarity=0.99)

    assert trace.decision == REFUSED
    assert trace.refusal_code == NO_RELEVANT_PASSAGE
    assert trace.request == {}
    assert trace.cost_usd == 0.0
    text = format_trace(trace)
    assert "non, refus sans appeler le modèle" in text
    assert NO_RELEVANT_PASSAGE in text


def test_fiche_recherche_seule(searcher: Searcher) -> None:
    trace, answer = explain_question("délégué", searcher, None, k=2)

    assert answer is None
    assert trace.decision == SEARCH_ONLY
    assert "Recherche seule" in format_trace(trace)


def test_affichage_de_la_fiche(searcher: Searcher) -> None:
    trace, _ = explain_question("droit à l'effacement", searcher, FakeGenerator(),
                                k=2, extra=1, min_similarity=0.1)

    text = format_trace(trace)
    for title in ("1. Paramètres", "2. Passages retrouvés", "3. Barrière 1",
                  "4. Texte envoyé", "5. Réponse brute", "6. Citations", "7. Décision", "8. Coût"):
        assert title in text
    assert "écartés juste après la coupure" in text
    assert "✓ vérifiée" in text
    assert "--show-prompt" in text
    assert "consigne" in format_trace(trace, show_prompt=True)


def test_texte_exact_envoye(searcher: Searcher) -> None:
    client_reply = response(text_block("Oui", (0, "La personne")))
    from fakes import FakeClient

    generator = ClaudeGenerator(FakeClient(client_reply))
    trace, _ = explain_question("effacement", searcher, generator, k=2, min_similarity=0.0)

    prompt = format_prompt(trace)
    assert "[CONSIGNE SYSTÈME]" in prompt
    assert "[PASSAGE 1] source=" in prompt
    assert "[MESSAGE] Question : effacement" in prompt


def test_traces_ecrites_et_relues(searcher: Searcher, tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    first, _ = explain_question("effacement", searcher, None)
    second, _ = explain_question("délégué", searcher, FakeGenerator(), min_similarity=0.0)
    write_trace(first, path)
    write_trace(second, path)

    traces = read_traces(path)
    assert [t.question for t in traces] == ["effacement", "délégué"]
    assert traces[1].passages[0].chunk_id == second.passages[0].chunk_id
    assert find_trace(traces, "last").question == "délégué"
    assert find_trace(traces, first.trace_id[:18]).trace_id in {first.trace_id, second.trace_id}
    with pytest.raises(LookupError):
        find_trace(traces, "1999")
    with pytest.raises(LookupError):
        find_trace([], "last")
    assert "délégué" in format_trace_list(traces)


# --- Cas de test et essais en lot --------------------------------------------


def test_correspondance_des_sources() -> None:
    assert matches("Article 17", "Article 17", "rgpd-fr")
    assert matches("Article 17", "Article 17, paragraphe 1", "rgpd-fr")
    assert matches("Article 17", "Article 17 (suite)", "rgpd-fr")
    assert not matches("Article 17", "Article 170", "rgpd-fr")
    assert matches("brochure", "Page 3", "brochure")
    case = Case("c", "q", sources=("Article 37",))
    assert first_match_rank(case, [(1, "Article 6", "rgpd-fr"), (2, "Article 37", "rgpd-fr")]) == 2


def test_fichier_des_cas_du_depot_valide() -> None:
    cases = load_cases(ROOT / "tests" / "cases" / "questions.yaml")

    assert len(cases) >= 10
    assert any(c.refused for c in cases)
    assert all(c.sources or c.refused for c in cases)


def test_cas_en_double_refuses(tmp_path: Path) -> None:
    path = tmp_path / "cas.yaml"
    path.write_text("- {id: a, question: x}\n- {id: a, question: y}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="double"):
        load_cases(path)


def test_essais_en_lot(searcher: Searcher, tmp_path: Path) -> None:
    cases = [
        Case("effacement", "droit à l'effacement", sources=("Article 17",)),
        Case("perou", "capitale du Pérou", refused=True),
    ]
    variants = make_variants([1, 2], [0.1], ["claude-haiku-5-5", ""])
    made: list[str] = []

    def factory(model: str) -> FakeGenerator:
        made.append(model)
        return FakeGenerator()

    rows = run_batch(cases, variants, searcher, factory, trace_path=tmp_path / "t.jsonl")

    assert len(rows) == 8
    assert made == ["claude-haiku-5-5"]  # un seul LLM par modèle
    assert len(read_traces(tmp_path / "t.jsonl")) == 8
    effacement = [r for r in rows if r["cas"] == "effacement"]
    assert all(r["source_ok"] for r in effacement)
    assert {r["decision"] for r in rows} == {ANSWERED, SEARCH_ONLY, REFUSED}
    summary = summarize(rows)
    assert "k=1 seuil=0.10 claude-haiku-5-5" in summary
    write_csv(rows, tmp_path / "lot.csv")
    with (tmp_path / "lot.csv").open(encoding="utf-8-sig") as f:
        assert next(csv.reader(f, delimiter=";"))[0] == "cas"


# --- Commandes ----------------------------------------------------------------


def _setup_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    monkeypatch.setattr(ask_cli, "make_embedder", lambda name: FakeEmbedder(name))
    monkeypatch.setattr(ask_cli, "make_generator", lambda model, cache=None: FakeGenerator())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    ask_cli.index_main(["--chunks", str(tmp_path / "chunks.jsonl"),
                        "--index-dir", str(tmp_path / "index")])
    return tmp_path / "index"


def test_rag_ask_explain_puis_rag_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup_cli(tmp_path, monkeypatch)
    traces = tmp_path / "traces.jsonl"

    ask_cli.ask_main(["droit à l'effacement", "--explain", "--index-dir", str(index_dir),
                      "--min-similarity", "0.1", "--trace", str(traces),
                      "--log", str(tmp_path / "log.jsonl"),
                      "--env-file", str(tmp_path / "absent.env")])
    out = capsys.readouterr().out
    assert "FICHE DE DÉCISION" in out
    assert "Sources :" in out
    trace_id = json.loads(traces.read_text(encoding="utf-8"))["trace_id"]
    assert f"rag-trace {trace_id}" in out

    explain_cli.trace_main(["--trace", str(traces)])
    assert trace_id in capsys.readouterr().out
    explain_cli.trace_main(["--list", "--trace", str(traces)])
    assert "droit à l'effacement" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="aucune trace"):
        explain_cli.trace_main(["inexistant", "--trace", str(traces)])


def test_rag_ask_search_only_ecrit_une_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup_cli(tmp_path, monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    traces = tmp_path / "traces.jsonl"

    ask_cli.ask_main(["effacement", "--search-only", "--explain", "--index-dir", str(index_dir),
                      "--trace", str(traces), "--env-file", str(tmp_path / "absent.env")])

    assert "Recherche seule" in capsys.readouterr().out
    assert read_traces(traces)[0].decision == SEARCH_ONLY


def test_rag_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    index_dir = _setup_cli(tmp_path, monkeypatch)
    cases = tmp_path / "cas.yaml"
    cases.write_text(
        "- {id: effacement, question: \"droit à l'effacement\", sources: [Article 17]}\n"
        "- {id: dpo, question: délégué, sources: [Article 6]}\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "lot.csv"

    explain_cli.batch_main(["--cases", str(cases), "--k", "1", "2", "--min-similarity", "0.1",
                            "--index-dir", str(index_dir), "--out", str(out_csv),
                            "--trace", str(tmp_path / "t.jsonl"),
                            "--env-file", str(tmp_path / "absent.env")])

    out = capsys.readouterr().out
    assert "2 questions × 2 réglages = 4 essais" in out
    assert "À regarder de près" in out  # « dpo » attend l'article 6 : échec voulu
    assert out_csv.exists()
