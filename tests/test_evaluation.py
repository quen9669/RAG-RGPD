"""Tests rapides de l'étape 5 : indicateurs, correcteur, barrière de qualité.

Faux modèles partout : aucun réseau, aucun coût.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_rgpd.answering import cli as ask_cli
from rag_rgpd.evaluation import cli as eval_cli
from rag_rgpd.evaluation.judge import ClaudeJudge, Judgement, judge_message, verbatim_copies
from rag_rgpd.evaluation.metrics import compare, format_summary, score_case, summarize
from rag_rgpd.explain.cases import Case, load_cases
from rag_rgpd.explain.trace import explain_question
from rag_rgpd.processing.pipeline import write_chunks
from rag_rgpd.retrieval.index import Searcher, build_index
from fakes import CHUNKS, FakeClient, FakeEmbedder, FakeGenerator, FakeReranker, response

ROOT = Path(__file__).resolve().parent.parent
PASSAGE = ("La personne concernée a le droit d'obtenir du responsable du traitement "
           "l'effacement de données à caractère personnel la concernant dans les meilleurs délais.")


@pytest.fixture
def searcher(tmp_path: Path) -> Searcher:
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index")
    return Searcher(tmp_path / "index", FakeEmbedder())


def _judgement(**values) -> Judgement:
    base = {"claims": [{"texte": "a", "appuyee": True, "citee": True}],
            "key_points": [{"point": "p", "couvert": True}, {"point": "q", "couvert": False}]}
    return Judgement(**{**base, **values})


# --- Recopies sans guillemets ------------------------------------------------


def test_recopie_sans_guillemets_reperee() -> None:
    answer = f"Selon le RGPD, {PASSAGE} [1]."

    copies = verbatim_copies(answer, [PASSAGE])

    assert len(copies) == 1
    assert copies[0].startswith("la personne concernée a le droit")


def test_recopie_entre_guillemets_acceptee() -> None:
    answer = f"Le RGPD dit : « {PASSAGE} » [1]."

    assert verbatim_copies(answer, [PASSAGE]) == []


def test_formule_courte_acceptee() -> None:
    answer = "Le responsable du traitement doit effacer les données dans les meilleurs délais [1]."

    assert verbatim_copies(answer, [PASSAGE]) == []


def test_part_des_affirmations() -> None:
    judgement = _judgement(claims=[{"texte": "a", "appuyee": True, "citee": False},
                                   {"texte": "b", "appuyee": False, "citee": False}])

    assert judgement.support_rate == 0.5 and judgement.citation_rate == 0.0
    assert _judgement(claims=[]).support_rate is None


def test_reformulation_acceptee() -> None:
    answer = "Vous pouvez demander à l'organisme d'effacer vos données rapidement [1]."

    assert verbatim_copies(answer, [PASSAGE]) == []


# --- Correcteur ---------------------------------------------------------------


def _judge_reply(data: dict) -> SimpleNamespace:
    reply = response(SimpleNamespace(type="text", text=json.dumps(data), citations=None),
                     model="claude-sonnet-5-5", input_tokens=3000, output_tokens=400)
    reply.model_dump = lambda mode="json": {}  # pas de cache dans ce test
    return reply


def test_correcteur_lit_le_json(searcher: Searcher) -> None:
    case = Case(id="e", question="effacement de mes données", sources=("Article 17",),
                key_points=("effacement",), trap="mélanger avec l'accès")
    trace, _ = explain_question(case.question, searcher, FakeGenerator())
    client = FakeClient(_judge_reply({
        "affirmations": [{"texte": "x", "appuyee": False, "citee": True}],
        "points_cles": [{"point": "effacement", "couvert": True}],
        "piege_evite": False, "commentaire": "affirmation non appuyée",
    }))

    judgement = ClaudeJudge(client).judge(case, trace)

    assert not judgement.faithful
    assert judgement.support_rate == 0.0
    assert judgement.citation_rate == 1.0
    assert judgement.unsupported == ["x"]
    assert judgement.completeness == 1.0
    assert not judgement.trap_avoided
    request = client.requests[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert "Piège : mélanger avec l'accès" in request["messages"][0]["content"]
    assert request["fallbacks"] == "default"


def test_message_du_correcteur_numerote_les_passages(searcher: Searcher) -> None:
    case = Case(id="e", question="droit à l'effacement")
    trace, _ = explain_question(case.question, searcher, FakeGenerator())

    message = judge_message(case, trace)

    assert "[1] RGPD > " in message
    assert "Points clés attendus :\n(aucun)" in message


# --- Indicateurs ----------------------------------------------------------------


def test_indicateurs_d_une_reponse(searcher: Searcher) -> None:
    case = Case(id="e", question="droit à l'effacement de mes données",
                sources=("Article 17",), trap="piège")
    trace, _ = explain_question(case.question, searcher, FakeGenerator())

    row = score_case(case, trace, _judgement(trap_avoided=False), min_similarity=0.1)

    assert row["rang"] == 1 and row["mrr"] == 1.0 and row["source_top3"]
    assert row["refus_ok"] and row["citations_ok"] and row["source_citee"]
    assert row["fidele"] == 1.0 and row["toutes_citees"] == 1.0
    assert row["completude"] == 0.5 and row["non_appuyees"] == []
    assert row["piege_ok"] is False


def test_indicateurs_en_recherche_seule(searcher: Searcher) -> None:
    case = Case(id="h", question="capitale du Pérou", refused=True)
    trace, _ = explain_question(case.question, searcher, None)

    row = score_case(case, trace, None, min_similarity=0.45)

    assert row["refus_ok"]
    assert row["source_k"] is None and row["fidele"] is None


def test_resume_ignore_les_cas_sans_objet() -> None:
    rows = [{"source_k": True, "mrr": 1.0, "refus_ok": True, "fidele": None, "cout_usd": 0.01},
            {"source_k": False, "mrr": 0.0, "refus_ok": True, "fidele": None, "cout_usd": 0.02}]

    summary = summarize(rows)

    assert summary["source_k"] == 0.5
    assert summary["refus_ok"] == 1.0
    assert summary["fidele"] is None
    assert summary["questions"] == 2 and summary["cout_usd"] == 0.03


def test_barriere_de_qualite() -> None:
    reference = {"source_k": 0.8, "fidele": 0.9, "completude": None}
    better = {"source_k": 0.9, "fidele": 0.9, "completude": 0.7}
    worse = {"source_k": 0.9, "fidele": 0.85, "completude": 0.7}

    assert compare(better, reference) == []
    assert compare(worse, reference) == ["fidélité (affirmations appuyées) : 0.90 -> 0.85"]
    assert compare(worse, reference, tolerance=0.05) == []
    assert "↑" in format_summary(better, reference)


# --- Commande rag-eval ----------------------------------------------------------


def test_rag_eval_reference_puis_barriere(tmp_path: Path, monkeypatch, capsys) -> None:
    write_chunks(CHUNKS, tmp_path / "chunks.jsonl")
    build_index(CHUNKS, FakeEmbedder(), tmp_path / "index")
    monkeypatch.setattr(ask_cli, "make_embedder", lambda name: FakeEmbedder(name))
    monkeypatch.setattr(ask_cli, "make_reranker", lambda name: FakeReranker())
    cases = tmp_path / "cases.yaml"
    cases.write_text(
        "- id: effacement\n  question: droit à l'effacement de mes données\n"
        "  sources: [Article 17]\n"
        "- id: delegue\n  question: désigner un délégué\n  sources: [Article 37]\n",
        encoding="utf-8",
    )
    common = ["--cases", str(cases), "--index-dir", str(tmp_path / "index"),
              "--embedding-model", "faux-modele", "--search-only",
              "--reference", str(tmp_path / "reference.json"),
              "--trace", str(tmp_path / "traces.jsonl"), "--min-similarity", "0.1"]

    eval_cli.eval_main([*common, "--out", str(tmp_path / "r1.json"), "--save-reference"])
    reference = json.loads((tmp_path / "reference.json").read_text(encoding="utf-8"))
    assert reference["summary"]["source_k"] == 1.0
    assert set(reference["cases"][0]) == set(eval_cli.REFERENCE_COLUMNS)
    assert reference["settings"]["reranker"] == "faux-reclasseur"

    eval_cli.eval_main([*common, "--out", str(tmp_path / "r2.json")])
    assert "Barrière de qualité franchie" in capsys.readouterr().out

    # Une version qui perd l'article 37 est bloquée par la barrière.
    cases.write_text(cases.read_text(encoding="utf-8").replace("Article 37", "Article 99"),
                     encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        eval_cli.eval_main([*common, "--out", str(tmp_path / "r3.json")])
    assert stopped.value.code == 1
    assert "BARRIÈRE DE QUALITÉ" in capsys.readouterr().out


def test_jeu_d_evaluation_complet() -> None:
    cases = load_cases(ROOT / "tests" / "cases" / "questions.yaml")

    assert len(cases) >= 50
    answerable = [c for c in cases if not c.refused]
    assert all(c.sources for c in answerable)
    assert all(c.key_points for c in answerable)
    assert any(c.trap for c in cases)
    assert sum(c.refused for c in cases) >= 5
