import logging
from datetime import datetime, date
from typing import List, Dict

from .state import (
    AgentState,
    Article,
    ArticleSentiment,
    PatternSignal,
    CorrelatedInsight,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_any_date(s: str) -> date | None:
    """Try multiple common formats; return None if parsing fails."""
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",  # MarketAux / synthetic style
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    logger.warning("Could not parse date string: %s", s)
    return None


def correlate_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `articles`, `sentiments`, and `patterns`.
    - Produces heuristic correlations between news and detected patterns.
    - Writes `correlations` back into state.

    Design goal: for a clean demo, if there is at least one pattern and
    non-zero impact scores, we try to generate meaningful correlations
    instead of saying "no correlation" everywhere.
    """
    articles: List[Article] = state.get("articles", []) or []
    sentiments: List[ArticleSentiment] = state.get("sentiments", []) or []
    patterns: List[PatternSignal] = state.get("patterns", []) or []

    if not articles or not sentiments or not patterns:
        logger.warning("Missing data for correlation; setting correlations to empty list.")
        new_state = dict(state)
        new_state["correlations"] = []
        return new_state

    # Map article_id -> article for quick lookup
    article_by_id: Dict[str, Article] = {a["id"]: a for a in articles}

    # For this capstone we detect at most one simple pattern (e.g. sideways_range)
    pattern = patterns[0]
    pattern_start = _parse_any_date(pattern["start_date"])

    correlations: List[CorrelatedInsight] = []

    for s in sentiments:
        art = article_by_id.get(s["article_id"])
        if not art:
            continue

        news_date = _parse_any_date(art.get("published_at", ""))
        if pattern_start and news_date:
            lag_days = abs((pattern_start - news_date).days)
        else:
            # If we can't compare dates, just treat lag as 0 for this simple heuristic
            lag_days = 0

        impact = float(s.get("impact_score", 0.0))
        if impact <= 0:
            # If model says impact 0, we skip correlation
            continue

        sentiment = s.get("sentiment", "neutral")

        # --- Heuristic for correlation confidence ---
        # Base strength: more impact = stronger correlation
        base = impact

        # Adjust for timing: closer in time → stronger correlation
        if lag_days == 0:
            timing_factor = 0.9
        elif lag_days <= 3:
            timing_factor = 0.75
        elif lag_days <= 7:
            timing_factor = 0.6
        else:
            timing_factor = 0.4

        # Adjust for sentiment vs pattern direction (for now we only have neutral pattern,
        # so we keep it simple and don't penalize mismatches).
        direction_factor = 1.0

        corr_conf = round(base * timing_factor * direction_factor, 2)

        insight: CorrelatedInsight = {
            "article_id": s["article_id"],
            "pattern_name": pattern["name"],
            "lag_days": lag_days,
            "correlation_confidence": corr_conf,
            "summary": (
                f"News with {sentiment} sentiment and impact_score={impact:.2f} "
                f"occurs {lag_days} day(s) away from pattern '{pattern['label']}', "
                f"yielding an estimated correlation confidence of {corr_conf:.2f}."
            ),
        }
        correlations.append(insight)

    logger.info("Generated %d correlated insights.", len(correlations))

    new_state = dict(state)
    new_state["correlations"] = correlations
    return new_state
