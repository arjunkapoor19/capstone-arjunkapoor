import logging
from typing import List, Dict, Optional
from datetime import datetime, date

from .state import (
    AgentState,
    Article,
    ArticleSentiment,
    PatternSignal,
    CorrelatedInsight,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_any_date(s: str) -> Optional[date]:
    """
    Try to parse strings like:
    - '2024-11-29'
    - '2024-11-29T09:30:00Z'
    - '2024-11-29T09:30:00'
    """
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue

    logger.warning("Could not parse date string: %s", s)
    return None


def _sentiment_direction(sentiment: str) -> str:
    """
    Map sentiment label to directional bias.
    """
    if sentiment == "positive":
        return "bullish"
    if sentiment == "negative":
        return "bearish"
    return "neutral"


def correlate_news_and_patterns(
    articles: List[Article],
    sentiments: List[ArticleSentiment],
    patterns: List[PatternSignal],
    max_lag_days: int = 7,
) -> List[CorrelatedInsight]:
    """
    Very simple heuristic:
    - For each article sentiment, find patterns whose start_date is within
      [0, max_lag_days] days AFTER the article publish date.
    - Compute a correlation_confidence based on:
        * impact_score of the news
        * confidence of the pattern
        * whether sentiment direction matches pattern direction
    """
    if not sentiments or not patterns:
        logger.info("No sentiments or patterns to correlate.")
        return []

    article_by_id: Dict[str, Article] = {a["id"]: a for a in articles}

    insights: List[CorrelatedInsight] = []

    for s in sentiments:
        article_id = s["article_id"]
        article = article_by_id.get(article_id)
        if not article:
            logger.warning("No article found for sentiment with id %s", article_id)
            continue

        art_date = _parse_any_date(article.get("published_at", ""))
        if art_date is None:
            continue

        news_dir = _sentiment_direction(s["sentiment"])

        for p in patterns:
            pat_start = _parse_any_date(p.get("start_date", ""))
            if pat_start is None:
                continue

            lag_days = (pat_start - art_date).days

            if lag_days < 0 or lag_days > max_lag_days:
                continue

            base_score = s["impact_score"] * p["confidence"]

            pat_dir = p.get("direction", "neutral")
            if news_dir == "neutral" or pat_dir == "neutral":
                direction_factor = 0.7
            elif news_dir == pat_dir:
                direction_factor = 1.0
            else:
                direction_factor = 0.4  # sentiment conflicts with pattern

            lag_penalty = max(0.4, 1.0 - (lag_days * 0.1))

            corr = base_score * direction_factor * lag_penalty
            corr = min(1.0, max(0.0, corr))

            if corr <= 0.05:
                continue

            summary = (
                f"News with {s['sentiment']} sentiment on {art_date.isoformat()} "
                f"(impact_score={s['impact_score']:.2f}) is followed by pattern "
                f"'{p['label']}' starting {lag_days} day(s) later, suggesting a "
                f"possible {'bullish' if pat_dir=='bullish' else 'bearish' if pat_dir=='bearish' else 'neutral'} "
                f"reaction to this event."
            )

            insight: CorrelatedInsight = {
                "article_id": article_id,
                "pattern_name": p["name"],
                "correlation_confidence": corr,
                "lag_days": lag_days,
                "summary": summary,
            }
            insights.append(insight)

    insights.sort(key=lambda x: x["correlation_confidence"], reverse=True)

    logger.info("Generated %d correlated insights.", len(insights))
    return insights


def correlate_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `articles`, `sentiments`, and `patterns`
    - Writes `correlations` back into state
    """
    articles: List[Article] = state.get("articles", [])
    sentiments: List[ArticleSentiment] = state.get("sentiments", [])
    patterns: List[PatternSignal] = state.get("patterns", [])

    if not articles or not sentiments or not patterns:
        logger.warning("Missing data for correlation; setting correlations to empty list.")
        new_state: AgentState = dict(state)
        new_state["correlations"] = []
        return new_state

    correlations = correlate_news_and_patterns(articles, sentiments, patterns)
    new_state: AgentState = dict(state)
    new_state["correlations"] = correlations
    return new_state
