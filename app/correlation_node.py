import logging
from datetime import datetime, date
from typing import List

from .state import AgentState, ArticleSentiment, PatternSignal, CorrelatedInsight

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_any_date(s: str) -> date | None:
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",  # MarketAux style
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
    - Reads `sentiments` and `patterns`.
    - Produces rough heuristic correlations between news and patterns.
    - Writes `correlations` back into state.
    """
    sentiments: List[ArticleSentiment] = state.get("sentiments", []) or []
    patterns: List[PatternSignal] = state.get("patterns", []) or []

    if not sentiments or not patterns:
        logger.warning("Missing data for correlation; setting correlations to empty list.")
        new_state = dict(state)
        new_state["correlations"] = []
        return new_state

    pattern = patterns[0]  # we only have one simple pattern in this demo
    pattern_start = _parse_any_date(pattern["start_date"])
    if pattern_start is None:
        logger.warning("Pattern start date invalid; cannot compute correlations.")
        new_state = dict(state)
        new_state["correlations"] = []
        return new_state

    correlations: List[CorrelatedInsight] = []

    for s in sentiments:
        news_date = _parse_any_date(state.get("articles", [{}])[0].get("published_at", ""))  # crude; you can map by id if needed
        if news_date is None:
            continue

        lag = (pattern_start - news_date).days
        if lag < 0:
            continue  # pattern before news → ignore

        # very simple heuristic: correlation = impact_score * 0.5 if lag <= 3 days
        base_corr = s["impact_score"] * 0.5 if lag <= 3 else s["impact_score"] * 0.2

        insight: CorrelatedInsight = {
            "article_id": s["article_id"],
            "pattern_name": pattern["name"],
            "lag_days": lag,
            "correlation_confidence": round(base_corr, 2),
            "summary": (
                f"News with {s['sentiment']} sentiment on lag {lag} day(s) "
                f"relative to pattern '{pattern['label']}' with impact_score={s['impact_score']:.2f}."
            ),
        }
        correlations.append(insight)

    logger.info("Generated %d correlated insights.", len(correlations))

    new_state = dict(state)
    new_state["correlations"] = correlations
    return new_state
