import logging
from typing import List

from .state import AgentState, PriceBar, PatternSignal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _detect_simple_sideways(prices: List[PriceBar]) -> List[PatternSignal]:
    """
    Super simple pattern detector:
    - If net change < 3% over period → 'sideways_range'.
    """
    if len(prices) < 3:
        logger.warning("Not enough price data (%d points) to detect patterns.", len(prices))
        return []

    first = prices[0]
    last = prices[-1]
    if first["close"] == 0:
        return []

    pct_change = (last["close"] - first["close"]) / first["close"]

    if abs(pct_change) < 0.03:  # <3% move
        signal: PatternSignal = {
            "name": "sideways_range",
            "label": "Sideways / Range-Bound",
            "direction": "neutral",
            "start_date": first["date"],
            "end_date": last["date"],
            "confidence": 0.6,
            "notes": "Net price change over the period was modest, indicating a mostly sideways or range-bound market.",
        }
        return [signal]

    return []


def detect_patterns_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `prices` from state.
    - Detects simple price patterns.
    - Writes `patterns` back into state.
    """
    prices: List[PriceBar] = state.get("prices", []) or []

    if len(prices) < 3:
        logger.warning("Not enough price data to detect patterns.")
        new_state = dict(state)
        new_state["patterns"] = []
        return new_state

    patterns = _detect_simple_sideways(prices)
    logger.info("Detected %d patterns from price data.", len(patterns))

    new_state = dict(state)
    new_state["patterns"] = patterns
    return new_state
