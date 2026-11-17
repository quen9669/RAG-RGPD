## Référentiel Franks (prioritaire, lecture seule)

Ce projet suit le référentiel `C:\Users\franc\Dev\Projets\portfolio-franks` (dépôt `quen9669/Referentiel-Portfolio`). On tend absolument vers lui.
Avant toute modification d'infra, d'IA, de secrets, de données ou d'interface :
1. Relire `REGLES.md` (R-01 à R-26, non négociables) et `apps/rag-rgpd.md`, en commençant par la **section 4 « Écarts à corriger »**.
2. Le moins de code possible (méthode Ponytail) et sobriété en tokens.
3. Visuels : uniquement la bibliothèque `@quen9669/lignes`. IA : uniquement OmniRoute. Secrets : uniquement le coffre OpenBao.
4. Citer l'ID de l'écart dans le commit (`fix(XXX-01): …`) et mettre à jour `expedition.yaml`.
Ne jamais modifier le référentiel depuis cette session. Si une règle gêne, le signaler à Franks.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
