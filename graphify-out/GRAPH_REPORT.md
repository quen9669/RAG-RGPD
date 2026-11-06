# Graph Report - rag-rgpd  (2026-11-06)

## Corpus Check
- 96 files · ~48,620 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 9 file(s) not represented in the graph (top: (none) 2, .ipynb 2, .example 1)

## Summary
- 1080 nodes · 2650 edges · 81 communities (57 shown, 24 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 329 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `7fdd505f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_agents.py
- test_explain.py
- documents.py
- Searcher
- index.py
- PoliteClient
- RawStore
- generator.py
- trace.py
- ManifestEntry
- Trace
- eval_main
- test_evaluation.py
- DeepSeekClient
- explain_question
- Case
- app.py
- processing/pipeline.py
- batch.py
- .call
- RoutedSearcher
- score_case
- evaluation/cli.py
- pdf.py
- sources/README.md
- RAG RGPD
- Hit
- chunking.py
- ChatEngine
- chunk_document
- .search
- httpx
- re
- answering/cli.py
- engine.py
- chunk_sections
- Section
- test_extraction.py
- searcher
- Feuille de route
- test_behaviour.py
- test_catalogue.py
- agents/pipeline.py
- test_chat_app.py
- metrics.py
- 0011 : Évaluation maison et reclassement des passages
- AnswerGenerator
- coulisses.py
- rerank.py
- conftest.py
- 0012 : Les trois agents en Python maison
- ResponseCache
- 0001 : Gestion de l'environnement Python
- 0002 : Outil de tests automatisés
- 0003 : Stockage versionné des données brutes
- 0005 : Extraction du texte
- 0006 : Découpage des textes en morceaux
- 0007 : Modèle d'embedding
- 0008 : Base de recherche : LanceDB, recherche hybride
- 0009 : LLM et réponses citées
- decisions/README.md
- 0014 : vérificateur assoupli et agents entièrement en Haiku
- 0016 — Étape 8 : routage automatique des domaines
- 0010 : Outils d'explicabilité et de test (traces, notebooks, cas de test)
- 0015 — Étape 8 : domaines, filtre de recherche et seuils par domaine
- pathlib
- make_citation
- Any
- compare
- Reranker
- _domain_filter
- fakes.py
- CLAUDE.md
- engine
- test_seuil_par_domaine
- rag-rgpd

## God Nodes (most connected - your core abstractions)
1. `Searcher` - 57 edges
2. `explain_question()` - 40 edges
3. `FakeEmbedder` - 39 edges
4. `Hit` - 37 edges
5. `response()` - 32 edges
6. `ResponseCache` - 30 edges
7. `Trace` - 29 edges
8. `text_block()` - 26 edges
9. `Answer` - 25 edges
10. `RawStore` - 25 edges

## Surprising Connections (you probably didn't know these)
- `Mise en œuvre` --references--> `RoutedSearcher`  [INFERRED]
  docs/decisions/0016-routage-des-domaines.md → src/rag_rgpd/agents/router.py
- `Ce qu'on construira` --references--> `Answer`  [INFERRED]
  docs/roadmap.md → src/rag_rgpd/answering/generator.py
- `Conséquences` --references--> `AnswerGenerator`  [INFERRED]
  docs/decisions/0009-llm-et-citations.md → src/rag_rgpd/answering/generator.py
- `Option choisie` --references--> `ClaudeGenerator`  [INFERRED]
  docs/decisions/0012-trois-agents.md → src/rag_rgpd/answering/generator.py
- `Conséquences` --references--> `explain_question()`  [INFERRED]
  docs/decisions/0010-outils-d-explicabilite.md → src/rag_rgpd/explain/trace.py

## Import Cycles
- None detected.

## Communities (81 total, 24 thin omitted)

### Community 0 - "test_agents.py"
Cohesion: 0.07
Nodes (41): AgentSearcher, make_researcher(), merge_hits(), make_router(), make_verifier(), Verdict, verifier_message(), verify() (+33 more)

### Community 1 - "test_explain.py"
Cohesion: 0.08
Nodes (34): ClaudeGenerator, parse_response(), FakeClient, FakeReranker, response(), text_block(), test_un_refus_du_redacteur_n_appelle_pas_le_verificateur(), _setup() (+26 more)

### Community 2 - "documents.py"
Cohesion: 0.09
Nodes (14): DocumentSource, extension_for(), ingest_documents(), load_sources(), main(), Chunk, media_type(), read_chunks() (+6 more)

### Community 3 - "Searcher"
Cohesion: 0.17
Nodes (20): Searcher, FakeEmbedder, index_dir(), mixed_index(), test_article_cite_dans_plusieurs_textes_garde_le_texte_pertinent(), test_article_cite_peut_etre_ignore(), test_article_cite_place_en_tete(), test_domaine_invalide_refuse() (+12 more)

### Community 4 - "index.py"
Cohesion: 0.11
Nodes (8): Embedder, build_index(), _file_sha256(), IndexMeta, IndexMismatchError, read_meta(), _schema(), test_fiche_de_l_index_tracee()

### Community 5 - "PoliteClient"
Cohesion: 0.09
Nodes (8): 0004 : Bibliothèque de téléchargement, Conséquences, Option choisie : httpx, Option écartée, PoliteClient, make_client(), factory(), test_choix_du_client_selon_le_modele()

### Community 6 - "RawStore"
Cohesion: 0.12
Nodes (6): ingest_catalogue(), main(), EmptyResponseError, RobotsDisallowedError, RawStore, test_instantane_trie_et_trace()

### Community 7 - "generator.py"
Cohesion: 0.16
Nodes (4): is_deepseek(), make_client(), best_similarity(), usage_dict()

### Community 8 - "trace.py"
Cohesion: 0.11
Nodes (6): estimate_cost(), find_trace(), read_traces(), _total_cost(), _trace_id(), write_trace()

### Community 9 - "ManifestEntry"
Cohesion: 0.16
Nodes (8): ManifestEntry, sha256_hex(), latest_documents(), _save(), test_contenu_identique_pas_de_doublon(), test_contenu_modifie_nouvelle_version_datee(), test_deux_versions_le_meme_jour_ne_s_ecrasent_pas(), test_enregistre_fichier_et_ligne_de_manifeste()

### Community 10 - "Trace"
Cohesion: 0.16
Nodes (10): format_prompt(), format_trace(), format_trace_list(), _passage_line(), _rank(), _section(), _short(), Passage (+2 more)

### Community 11 - "eval_main"
Cohesion: 0.14
Nodes (9): add_agent_options(), add_router_options(), with_agents(), with_router(), add_search_options(), ask_main(), open_searcher(), eval_main() (+1 more)

### Community 12 - "test_evaluation.py"
Cohesion: 0.13
Nodes (12): verbatim_copies(), _judge_reply(), _judgement(), searcher(), test_formule_courte_acceptee(), test_indicateurs_d_une_reponse(), test_indicateurs_en_recherche_seule(), test_part_des_affirmations() (+4 more)

### Community 13 - "DeepSeekClient"
Cohesion: 0.11
Nodes (9): 0013 : DeepSeek pour les petits agents (essai), Conséquences et risques, Contexte, Option choisie, Options écartées, Résultats, DeepSeekClient, DeepSeekError (+1 more)

### Community 14 - "explain_question"
Cohesion: 0.15
Nodes (11): explain_question(), _pages(), SearchEngine, FakeGenerator, test_correcteur_lit_le_json(), searcher(), test_affichage_de_la_fiche(), test_fiche_complete() (+3 more)

### Community 15 - "Case"
Cohesion: 0.15
Nodes (10): judge_message(), Case, first_match_rank(), load_cases(), matches(), test_jeu_d_evaluation_complet(), test_message_du_correcteur_numerote_les_passages(), test_cas_en_double_refuses() (+2 more)

### Community 16 - "app.py"
Cohesion: 0.20
Nodes (8): _api_error_message(), load_env(), get_engine(), _label(), main(), render_coulisses(), render_reply(), Reply

### Community 17 - "processing/pipeline.py"
Cohesion: 0.13
Nodes (5): _article_label(), _blocks(), _classes(), _flat(), _heading()

### Community 18 - "batch.py"
Cohesion: 0.18
Nodes (7): make_variants(), run_batch(), summarize(), Variant, write_csv(), test_essais_en_lot(), factory()

### Community 21 - "score_case"
Cohesion: 0.12
Nodes (5): run_eval(), _write_json(), score_case(), summarize(), test_resume_ignore_les_cas_sans_objet()

### Community 22 - "evaluation/cli.py"
Cohesion: 0.20
Nodes (3): require_api_key(), batch_main(), trace_main()

### Community 23 - "pdf.py"
Cohesion: 0.17
Nodes (7): _block_text(), extract_pdf(), _repeated_blocks(), _signature(), _pdf(), test_pdf_page_vide_ignoree(), test_pdf_une_section_par_page_et_entetes_retires()

### Community 24 - "sources/README.md"
Cohesion: 0.12
Nodes (12): Conditions d'usage, Conséquences pour le projet, Fiche source : CNPD, Limites, Conditions d'usage, Fiche source : data.public.lu, Limites, Ajouts de l'étape 8a (2026-11-04, à vérifier à l'ingestion) (+4 more)

### Community 25 - "RAG RGPD"
Cohesion: 0.12
Nodes (16): Documentation, Débogage pas à pas et tâches, Extraire et découper les textes, Installation, La fiche de décision, Les essais en lot, Les notebooks, Les tests, à trois niveaux (+8 more)

### Community 26 - "Hit"
Cohesion: 0.22
Nodes (6): Answer, answer_question(), refuse(), Hit, ExplodingGenerator, test_barriere_de_similarite_evite_l_appel()

### Community 27 - "chunking.py"
Cohesion: 0.18
Nodes (7): _group_label(), _hard_split(), _pack_strings(), Piece, _split_section(), _Unit, _units()

### Community 28 - "ChatEngine"
Cohesion: 0.21
Nodes (7): stages(), timing_text(), ChatEngine, test_coulisses_refus_barriere_1(), test_coulisses_reponse(), test_refus_sans_source(), test_reponse_avec_sources()

### Community 29 - "chunk_document"
Cohesion: 0.26
Nodes (8): chunk_document(), IntegrityError, _store_with_rgpd(), test_commande_rag_chunk(), test_ecriture_relecture_utf8(), test_fichier_modifie_detecte(), test_morceaux_tracables_et_citables(), test_version_vide_ne_masque_pas_la_bonne()

### Community 30 - ".search"
Cohesion: 0.13
Nodes (7): cited_labels(), embedding_text(), _label_filter(), _label_position(), _row_to_chunk(), test_articles_cites_reconnus(), test_le_texte_encode_porte_le_titre_complet()

### Community 31 - "httpx"
Cohesion: 0.19
Nodes (6): test_attend_entre_deux_requetes_vers_le_meme_hote(), test_refuse_une_reponse_vide_ou_non_200(), test_refuse_une_url_interdite_par_robots(), handler(), test_sans_robots_txt_tout_est_permis(), test_transmet_les_en_tetes_demandes()

### Community 32 - "re"
Cohesion: 0.18
Nodes (4): clean_text(), test_clean_text_normalise_espaces_et_caracteres_invisibles(), test_clean_text_puces_apostrophes_et_guillemets(), test_version_suit_le_format_semver()

### Community 33 - "answering/cli.py"
Cohesion: 0.19
Nodes (5): format_answer(), format_source(), index_main(), _pages(), _utf8_output()

### Community 34 - "engine.py"
Cohesion: 0.18
Nodes (4): log_answer(), _pages(), Source, to_reply()

### Community 35 - "chunk_sections"
Cohesion: 0.32
Nodes (10): chunk_sections(), _article(), _page(), test_page_longue_pas_fusionnee(), test_pages_courtes_fusionnees_mais_pas_les_articles(), test_paragraphe_trop_long_coupe_aux_lignes_puis_suite(), test_plafond_trop_petit_refuse(), test_section_courte_un_seul_morceau() (+2 more)

### Community 36 - "Section"
Cohesion: 0.18
Nodes (4): _merge_short_pages(), _Builder, Paragraph, Section

### Community 37 - "test_extraction.py"
Cohesion: 0.24
Nodes (9): extract_eu_act(), test_acte_ue_article_titre_contexte_et_paragraphes(), test_acte_ue_considerant_sans_espace_parasite(), test_acte_ue_considerants_puis_articles(), test_acte_ue_format_recent_avec_prefixe_oj(), test_acte_ue_retours_a_la_ligne_du_fichier_ignores(), test_acte_ue_section_dans_chapitre_et_fin_du_texte(), test_annexes_apres_la_signature_sont_extraites() (+1 more)

### Community 39 - "Feuille de route"
Cohesion: 0.18
Nodes (11): Ce qu'on construira, Ce que je veux voir pour chaque question, Ce que l'étape 4 doit déjà prévoir, Feuille de route, Objectif, Outils retenus, Phase 1 : Fondations (faite), Phase 2 : Premier assistant qui cite ses sources (+3 more)

### Community 40 - "test_behaviour.py"
Cohesion: 0.22
Nodes (3): _params(), test_recherche(), test_reponse()

### Community 41 - "test_catalogue.py"
Cohesion: 0.24
Nodes (4): fetch_catalogue(), _faux_api(), test_max_pages_limite_la_recuperation(), test_suit_toutes_les_pages()

### Community 43 - "test_chat_app.py"
Cohesion: 0.20
Nodes (3): test_ancienne_reponse_sans_fiche_ne_plante_pas(), test_panneau_coulisses_rend_une_reponse(), script()

### Community 45 - "0011 : Évaluation maison et reclassement des passages"
Cohesion: 0.22
Nodes (9): 0011 : Évaluation maison et reclassement des passages, Consignes du rédacteur, Conséquences, Contexte, Mesure : fait maison (A), Option choisie, Options écartées, Recherche : articles cités + reclassement (2) (+1 more)

### Community 47 - "coulisses.py"
Cohesion: 0.25
Nodes (3): agent_rows(), passage_rows(), Stage

### Community 50 - "0012 : Les trois agents en Python maison"
Cohesion: 0.29
Nodes (6): 0012 : Les trois agents en Python maison, Conséquences, Contexte, Option choisie, Options écartées, Résultats

### Community 52 - "0001 : Gestion de l'environnement Python"
Cohesion: 0.33
Nodes (5): 0001 : Gestion de l'environnement Python, Conséquences, Contexte, Option choisie : uv, Options écartées

### Community 53 - "0002 : Outil de tests automatisés"
Cohesion: 0.33
Nodes (5): 0002 : Outil de tests automatisés, Conséquences, Contexte, Option choisie : pytest, Option écartée

### Community 54 - "0003 : Stockage versionné des données brutes"
Cohesion: 0.33
Nodes (5): 0003 : Stockage versionné des données brutes, Conséquences, Contexte, Option choisie : dossiers datés + manifeste maison, Options écartées

### Community 55 - "0005 : Extraction du texte"
Cohesion: 0.33
Nodes (5): 0005 : Extraction du texte, Conséquences, Contexte, Option choisie, Options écartées

### Community 56 - "0006 : Découpage des textes en morceaux"
Cohesion: 0.33
Nodes (5): 0006 : Découpage des textes en morceaux, Conséquences, Contexte, Option choisie : découpage hybride, Options écartées

### Community 57 - "0007 : Modèle d'embedding"
Cohesion: 0.33
Nodes (5): 0007 : Modèle d'embedding, Conséquences, Contexte, Option choisie : BGE-M3, en local, Options écartées

### Community 58 - "0008 : Base de recherche : LanceDB, recherche hybride"
Cohesion: 0.33
Nodes (5): 0008 : Base de recherche : LanceDB, recherche hybride, Conséquences, Contexte, Option choisie : LanceDB, recherche hybride, Options écartées

### Community 59 - "0009 : LLM et réponses citées"
Cohesion: 0.33
Nodes (5): 0009 : LLM et réponses citées, Conséquences, Contexte, Option choisie : Claude par API, avec les citations de l'API, Options écartées

### Community 61 - "0014 : vérificateur assoupli et agents entièrement en Haiku"
Cohesion: 0.33
Nodes (5): 0014 : vérificateur assoupli et agents entièrement en Haiku, Contexte, Limites, Mesure (2026-11-03, 52 questions, `--agents`, tout en Haiku), Option choisie

### Community 62 - "0016 — Étape 8 : routage automatique des domaines"
Cohesion: 0.33
Nodes (5): 0016 — Étape 8 : routage automatique des domaines, Contexte, Limites, Mise en œuvre, Options

### Community 63 - "0010 : Outils d'explicabilité et de test (traces, notebooks, cas de test)"
Cohesion: 0.40
Nodes (5): 0010 : Outils d'explicabilité et de test (traces, notebooks, cas de test), Conséquences, Contexte, Option choisie : tout « maison » (JSONL + Jupyter + YAML), Options écartées

### Community 64 - "0015 — Étape 8 : domaines, filtre de recherche et seuils par domaine"
Cohesion: 0.40
Nodes (4): 0015 — Étape 8 : domaines, filtre de recherche et seuils par domaine, Conséquences, Contexte, Options choisies

### Community 66 - "make_citation"
Cohesion: 0.40
Nodes (3): Citation, make_citation(), _normalize()

### Community 68 - "compare"
Cohesion: 0.40
Nodes (3): compare(), format_summary(), test_barriere_de_qualite()

## Knowledge Gaps
- **1 isolated node(s):** `rag-rgpd`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 459 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **24 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Hit` connect `Hit` to `test_agents.py`, `answering/cli.py`, `test_explain.py`, `make_citation`, `Any`, `index.py`, `documents.py`, `generator.py`, `trace.py`, `agents/pipeline.py`, `AnswerGenerator`, `explain_question`, `.call`, `RoutedSearcher`, `.search`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 46 inferred relationships involving `Searcher` (e.g. with `get_engine()` and `Embedder`) actually correct?**
  _`Searcher` has 46 INFERRED edges - model-reasoned connections that need verification._
- **What connects `rag-rgpd` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_agents.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0700152207001522 - nodes in this community are weakly interconnected._
- **Why does `Searcher` connect `Searcher` to `test_agents.py`, `answering/cli.py`, `test_explain.py`, `index.py`, `Reranker`, `_domain_filter`, `fakes.py`, `searcher`, `eval_main`, `test_chat_app.py`, `test_evaluation.py`, `engine`, `explain_question`, `app.py`, `Case`, `batch.py`, `.search`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `explain_question()` (e.g. with `Conséquences` and `Answer`) actually correct?**
  _`explain_question()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `test_explain.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07782898105478751 - nodes in this community are weakly interconnected._