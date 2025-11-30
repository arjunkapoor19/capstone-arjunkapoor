import logging
from typing import List, Optional

from .state import AgentState, PriceBar, PatternSignal

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _compute_return(prices: List[PriceBar]) -> float:
    """Simple total return over the period."""
    if len(prices) < 2:
        return 0.0
    start = prices[0]["close"]
    end = prices[-1]["close"]
    if start == 0:
        return 0.0
    return (end - start) / start


def _find_local_extrema(prices: List[PriceBar]) -> tuple[Optional[int], Optional[int]]:
    """
    Find indices of min and max close price.
    Returns (min_idx, max_idx).
    """
    if not prices:
        return None, None

    closes = [p["close"] for p in prices]
    min_idx = closes.index(min(closes))
    max_idx = closes.index(max(closes))
    return min_idx, max_idx


def detect_patterns(prices: List[PriceBar]) -> List[PatternSignal]:
    """
    Very simple 'technical pattern detection':
    - Bullish trend / breakout (strong positive return)
    - Bearish trend / breakdown (strong negative return)
    - Volatile / choppy (large swings but small net change)
    """
    patterns: List[PatternSignal] = []

    if len(prices) < 2:
        logger.warning("Not enough price data to detect patterns.")
        return patterns

    total_ret = _compute_return(prices)
    min_idx, max_idx = _find_local_extrema(prices)
    start_date = prices[0]["date"]
    end_date = prices[-1]["date"]

    # Thresholds (tunable, but good enough for capstone)
    bullish_threshold = 0.05   # +5%
    bearish_threshold = -0.05  # -5%

    # 1) Trend / breakout pattern
    if total_ret >= bullish_threshold:
        patterns.append(
            PatternSignal(
                name="bullish_breakout",
                label="Bullish Breakout / Uptrend",
                start_date=start_date,
                end_date=end_date,
                confidence=min(1.0, abs(total_ret) * 5),  # crude scaling
                direction="bullish",
                notes=(
                    f"Price increased by approximately {total_ret*100:.1f}% "
                    f"between {start_date} and {end_date}."
                ),
            )
        )
    elif total_ret <= bearish_threshold:
        patterns.append(
            PatternSignal(
                name="bearish_breakdown",
                label="Bearish Breakdown / Downtrend",
                start_date=start_date,
                end_date=end_date,
                confidence=min(1.0, abs(total_ret) * 5),
                direction="bearish",
                notes=(
                    f"Price decreased by approximately {total_ret*100:.1f}% "
                    f"between {start_date} and {end_date}."
                ),
            )
        )
    else:
        patterns.append(
            PatternSignal(
                name="sideways_range",
                label="Sideways / Range-Bound",
                start_date=start_date,
                end_date=end_date,
                confidence=0.6,
                direction="neutral",
                notes=(
                    "Net price change over the period was modest, indicating "
                    "a mostly sideways or range-bound market."
                ),
            )
        )

    # 2) Swing pattern based on local min/max
    if min_idx is not None and max_idx is not None and min_idx != max_idx:
        low_bar = prices[min_idx]
        high_bar = prices[max_idx]
        swing_ret = (high_bar["close"] - low_bar["close"]) / low_bar["close"]

        if swing_ret > 0.08:  # >8% swing
            swing_name = "strong_up_swing" if min_idx < max_idx else "strong_down_swing"
            direction = "bullish" if swing_name == "strong_up_swing" else "bearish"
            patterns.append(
                PatternSignal(
                    name=swing_name,
                    label="Strong Price Swing",
                    start_date=low_bar["date"],
                    end_date=high_bar["date"],
                    confidence=min(1.0, abs(swing_ret) * 4),
                    direction=direction,
                    notes=(
                        f"Detected a strong {'up' if direction=='bullish' else 'down'} swing "
                        f"of about {swing_ret*100:.1f}% between {low_bar['date']} "
                        f"and {high_bar['date']}."
                    ),
                )
            )

    return patterns


def detect_patterns_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `prices` from state
    - Writes detected `patterns` back into state
    """
    prices: List[PriceBar] = state.get("prices", [])

    if not prices:
        logger.warning("No price data in state; skipping pattern detection.")
        new_state: AgentState = dict(state)
        new_state["patterns"] = []
        return new_state

    patterns = detect_patterns(prices)
    new_state: AgentState = dict(state)
    new_state["patterns"] = patterns
    logger.info("Detected %d patterns from price data.", len(patterns))
    return new_state
