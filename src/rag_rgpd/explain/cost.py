"""Estimation du coût d'un appel au LLM.

Le fournisseur facture au token (unité de texte d'environ 3 à 4 caractères
en français), avec un prix différent pour le texte envoyé (entrée) et le
texte produit (sortie). Les tokens relus depuis le cache de l'API coûtent
10 % du prix d'entrée, ceux écrits dans ce cache 125 %.

Les prix sont ceux publiés par Anthropic en octobre 2026, en dollars par
million de tokens. Ils changent : c'est une estimation, la facture fait foi.
"""

from __future__ import annotations

PRICES_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    # modèle: (entrée, sortie)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-fable-5-1": (10.00, 50.00),
}

CACHE_READ_FACTOR = 0.1
CACHE_WRITE_FACTOR = 1.25


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    """Coût estimé d'un appel, en dollars.

    Args:
        model: modèle qui a répondu.
        usage: tokens comptés par l'API (voir ``Answer.usage``).

    Returns:
        Le coût, ``0.0`` si rien n'a été consommé, ``None`` si le prix du
        modèle n'est pas connu.
    """
    if not any(usage.values()):
        return 0.0
    if model not in PRICES_USD_PER_MILLION:
        return None
    price_in, price_out = PRICES_USD_PER_MILLION[model]
    total = (
        usage.get("input_tokens", 0) * price_in
        + usage.get("cache_read_input_tokens", 0) * price_in * CACHE_READ_FACTOR
        + usage.get("cache_creation_input_tokens", 0) * price_in * CACHE_WRITE_FACTOR
        + usage.get("output_tokens", 0) * price_out
    )
    return total / 1_000_000
