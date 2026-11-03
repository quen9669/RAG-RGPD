"""Commande ``rag-chat`` : ouvre le chat dans le navigateur."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Lance Streamlit sur la page du chat (à lancer à la racine du dépôt)."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit("Streamlit manque : lancer `uv sync --extra ui` puis `uv run --extra ui rag-chat`.")
    app = Path(__file__).with_name("app.py")
    sys.argv = ["streamlit", "run", str(app), "--server.headless", "false",
                "--browser.gatherUsageStats", "false", *(argv or sys.argv[1:])]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
