import json
import logging
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from .state import AgentState, Article, ArticleSentiment
from .analysis_schemas import SentimentOutput

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _build_sentiment_prompt(article: Article) -> str:
    return json.dumps(
        {
            "title": article["title"],
            "summary": article["summary"],
            "full_text": article["full_text"],
            "ticker": article["ticker"],
        }
    )


def _analyze_single_article(article: Article) -> ArticleSentiment:
    """
    Call the LLM once for a single article and return structured sentiment.
    Robust to model/JSON failures: falls back to neutral.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial news sentiment analyst. "
                        "Given a news article about a specific stock, you output a "
                        "JSON object with fields: sentiment, confidence, event_tags, "
                        "impact_score, reasoning."
                    ),
                },
                {
                    "role": "user",
                    "content": _build_sentiment_prompt(article),
                },
            ],
            temperature=0.2,
        )

        raw_content = (resp.choices[0].message.content or "").strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        struct = SentimentOutput.model_validate_json(raw_content)

    except Exception as e:
        logger.error(
            "Sentiment analysis failed for article '%s': %s",
            article.get("title", "UNKNOWN"),
            e,
        )
        struct = SentimentOutput(
            sentiment="neutral",
            confidence=0.0,
            event_tags=[],
            impact_score=0.0,
            reasoning="Fallback neutral sentiment due to model or parsing error.",
        )

    result: ArticleSentiment = {
        "article_id": article["id"],
        "sentiment": struct.sentiment,
        "confidence": float(struct.confidence),
        "event_tags": list(struct.event_tags),
        "impact_score": float(struct.impact_score),
        "reasoning": struct.reasoning,
    }
    return result


def analyze_sentiment_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `articles` from state.
    - Runs LLM-based sentiment analysis per article.
    - Writes `sentiments` back into state.
    """
    articles: List[Article] = state.get("articles", []) or []

    if not articles:
        logger.warning("No articles found in state; skipping sentiment analysis.")
        new_state = dict(state)
        new_state["sentiments"] = []
        return new_state

    sentiments: List[ArticleSentiment] = []
    for a in articles:
        sentiments.append(_analyze_single_article(a))

    logger.info("Completed sentiment analysis for %d articles.", len(sentiments))

    new_state = dict(state)
    new_state["sentiments"] = sentiments
    return new_state
