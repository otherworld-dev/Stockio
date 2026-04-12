"""Correlation filter — prevents doubling up on the same macro exposure.

Hardcoded correlation groups based on stable long-term forex relationships.
No live calculation needed — these correlations hold over decades.
"""

from __future__ import annotations

import structlog

from stockio.broker.models import Signal

log = structlog.get_logger()

# Instruments that move together (correlated via same underlying factor)
_CORRELATION_GROUPS: dict[str, list[str]] = {
    # All inversely correlated with USD — going long multiple = concentrated USD short
    # USD_CHF moves inversely to these (long USD_CHF ≈ short EUR_USD)
    "usd_short": ["EUR_USD", "GBP_USD", "AUD_USD", "NZD_USD", "USD_CHF"],
    # All move with risk appetite vs JPY — going long multiple = concentrated JPY short
    "jpy_short": ["USD_JPY", "EUR_JPY", "GBP_JPY", "AUD_JPY"],
    # Commodity currencies — correlated via commodity prices
    "commodity_fx": ["AUD_USD", "NZD_USD", "USD_CAD"],
}

# High pairwise correlations (|r| > 0.7)
_HIGH_CORRELATION_PAIRS: dict[tuple[str, str], float] = {
    ("EUR_USD", "GBP_USD"): 0.85,
    ("AUD_USD", "NZD_USD"): 0.90,
    ("EUR_JPY", "USD_JPY"): 0.70,
    ("GBP_JPY", "USD_JPY"): 0.75,
    ("AUD_JPY", "USD_JPY"): 0.70,
    ("GBP_JPY", "EUR_JPY"): 0.80,
    ("AUD_JPY", "EUR_JPY"): 0.75,
    ("EUR_USD", "USD_CHF"): 0.90,  # Inverse correlation (same USD exposure)
    ("GBP_USD", "USD_CHF"): 0.80,
}


def filter_correlated_signals(
    ranked: list[Signal],
    open_instruments: set[str],
) -> list[Signal]:
    """Filter out signals that would duplicate exposure to already-taken positions.

    Takes the ranked signal list (highest confidence first) and removes signals
    that are highly correlated with a higher-ranked signal or an open position.
    """
    accepted: list[Signal] = []
    taken: set[str] = set(open_instruments)

    for signal in ranked:
        inst = signal.instrument

        # Check if this instrument is highly correlated with anything already taken
        is_correlated = False
        for existing in taken:
            # Check direct pair correlation
            pair = (min(inst, existing), max(inst, existing))
            if pair in _HIGH_CORRELATION_PAIRS:
                is_correlated = True
                log.debug(
                    "correlation_filtered",
                    instrument=inst,
                    correlated_with=existing,
                    correlation=_HIGH_CORRELATION_PAIRS[pair],
                )
                break

            # Check if same direction in same group
            for group_name, members in _CORRELATION_GROUPS.items():
                if inst in members and existing in members:
                    # Both in same group — check if same direction would stack
                    is_correlated = True
                    log.debug(
                        "group_filtered",
                        instrument=inst,
                        correlated_with=existing,
                        group=group_name,
                    )
                    break
            if is_correlated:
                break

        if not is_correlated:
            accepted.append(signal)
            taken.add(inst)

    if len(accepted) < len(ranked):
        log.info(
            "correlation_filter_applied",
            before=len(ranked),
            after=len(accepted),
            filtered=[s.instrument for s in ranked if s not in accepted],
        )

    return accepted
