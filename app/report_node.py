import logging
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


def _build_article_index(articles: List[Article]) -> Dict[str, Article]:
    """Map article_id -> Article for easy lookup."""
    return {a["id"]: a for a in articles}


def _format_article_block(
    article: Article,
    sentiments_for_article: List[ArticleSentiment],
    correlations_for_article: List[CorrelatedInsight],
) -> str:
    """Format a markdown block summarizing one article and its impact."""
    sentiment_summary_lines = []
    for s in sentiments_for_article:
        sentiment_summary_lines.append(
            f"- **Sentiment:** {s['sentiment'].title()} "
            f"(confidence: {s['confidence']:.2f}, impact: {s['impact_score']:.2f})\n"
            f"  - Tags: {', '.join(s['event_tags']) if s['event_tags'] else 'None'}\n"
            f"  - Reasoning: {s['reasoning']}"
        )

    if not sentiment_summary_lines:
        sentiment_block = "_No sentiment analysis available for this article._"
    else:
        sentiment_block = "\n".join(sentiment_summary_lines)

    corr_lines = []
    for c in correlations_for_article:
        corr_lines.append(
            f"- Related pattern: `{c['pattern_name']}` "
            f"(lag: {c['lag_days']} day(s), correlation: {c['correlation_confidence']:.2f})\n"
            f"  - {c['summary']}"
        )

    if not corr_lines:
        corr_block = "_No strong pattern correlation identified for this article._"
    else:
        corr_block = "\n".join(corr_lines)

    block = f"""
### 📰 {article.get('title', 'Untitled article')}

- **Source:** {article.get('source', 'Unknown')}  
- **Published at:** {article.get('published_at', 'Unknown')}  
- **URL:** {article.get('url', '')}

**Sentiment & Event Analysis**

{sentiment_block}

**Linked Market Reactions**

{corr_block}
"""
    return block.strip()


def _format_patterns_section(patterns: List[PatternSignal]) -> str:
    """Summarize detected technical patterns."""
    if not patterns:
        return "_No clear technical patterns detected in this date range._"

    lines = []
    for p in patterns:
        lines.append(
            f"- **{p['label']}** (`{p['name']}`) from {p['start_date']} to {p['end_date']}  \n"
            f"  Direction: **{p['direction'].title()}**, confidence: {p['confidence']:.2f}  \n"
            f"  Notes: {p['notes']}"
        )
    return "\n".join(lines)


def _format_high_level_summary(
    sentiments: List[ArticleSentiment],
    patterns: List[PatternSignal],
    correlations: List[CorrelatedInsight],
) -> str:
    """Create a short high-level narrative summary."""
    if not sentiments and not patterns:
        return (
            "There was not enough news or price data to form a meaningful summary "
            "for this period."
        )

    num_pos = sum(1 for s in sentiments if s["sentiment"] == "positive")
    num_neg = sum(1 for s in sentiments if s["sentiment"] == "negative")
    num_neu = sum(1 for s in sentiments if s["sentiment"] == "neutral")

    dominant_sentiment = "mixed/neutral"
    if num_pos > num_neg and num_pos > num_neu:
        dominant_sentiment = "overall positive"
    elif num_neg > num_pos and num_neg > num_neu:
        dominant_sentiment = "overall negative"

    num_bull = sum(1 for p in patterns if p["direction"] == "bullish")
    num_bear = sum(1 for p in patterns if p["direction"] == "bearish")

    if num_bull > num_bear:
        market_tone = "price action tilted bullish"
    elif num_bear > num_bull:
        market_tone = "price action tilted bearish"
    else:
        market_tone = "price action remained relatively balanced or sideways"

    top_corr = correlations[0] if correlations else None

    lines = [
        f"- **News tone:** {dominant_sentiment} "
        f"({num_pos} positive / {num_neg} negative / {num_neu} neutral articles).",
        f"- **Market tone:** {market_tone}.",
    ]

    if top_corr:
        lines.append(
            f"- **Strongest link:** Article `{top_corr['article_id']}` → pattern "
            f"`{top_corr['pattern_name']}` with correlation score "
            f"{top_corr['correlation_confidence']:.2f} after {top_corr['lag_days']} day(s)."
        )

    return "\n".join(lines)


def generate_report_markdown(state: AgentState) -> str:
    """
    Build a human-readable markdown report from the agent state.
    """
    ticker = state.get("ticker", "UNKNOWN")
    start_date = state.get("start_date", "UNKNOWN")
    end_date = state.get("end_date", "UNKNOWN")

    articles: List[Article] = state.get("articles", []) or []
    sentiments: List[ArticleSentiment] = state.get("sentiments", []) or []
    patterns: List[PatternSignal] = state.get("patterns", []) or []
    correlations: List[CorrelatedInsight] = state.get("correlations", []) or []

    article_index = _build_article_index(articles)

    sentiments_by_article: Dict[str, List[ArticleSentiment]] = {}
    for s in sentiments:
        sentiments_by_article.setdefault(s["article_id"], []).append(s)

    correlations_by_article: Dict[str, List[CorrelatedInsight]] = {}
    for c in correlations:
        correlations_by_article.setdefault(c["article_id"], []).append(c)

    header = f"# 📈 Stock Market News-Pattern Intelligence Report: {ticker}\n\n"
    header += f"**Date range:** {start_date} → {end_date}\n\n"

    summary_section = "## 🔍 High-Level Summary\n\n"
    summary_section += _format_high_level_summary(sentiments, patterns, correlations)
    summary_section += "\n\n"

    pattern_section = "## 📊 Detected Technical Patterns\n\n"
    pattern_section += _format_patterns_section(patterns)
    pattern_section += "\n\n"

    articles_section = "## 📰 News Events & Their Market Impact\n\n"

    if not articles:
        articles_section += "_No news articles were retrieved for this period._\n"
    else:
        article_blocks = []
        for article_id, article in article_index.items():
            article_blocks.append(
                _format_article_block(
                    article=article,
                    sentiments_for_article=sentiments_by_article.get(article_id, []),
                    correlations_for_article=correlations_by_article.get(article_id, []),
                )
            )
        articles_section += "\n\n".join(article_blocks)

    takeaway_section = "## 🎯 Trader Takeaways (Qualitative)\n\n"
    takeaway_section += (
        "This report does not provide financial advice, but highlights how news "
        "and technical patterns interacted over the selected period. Traders may "
        "consider:\n"
        "- Whether strong negative or positive events consistently precede large moves.\n"
        "- Which types of news (earnings, regulation, macro) seem most impactful.\n"
        "- How quickly the stock tends to react to different categories of news.\n"
    )

    full_report = header + summary_section + pattern_section + articles_section + "\n\n" + takeaway_section
    return full_report.strip()


def generate_report_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads ticker, date range, articles, sentiments, patterns, correlations
    - Writes a markdown report into `report_markdown`
    """
    logger.info("Generating final markdown report from state.")
    report_md = generate_report_markdown(state)

    new_state: AgentState = dict(state)
    new_state["report_markdown"] = report_md
    return new_state
