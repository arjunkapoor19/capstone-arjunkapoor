import json
import logging
import os
from typing import List

from openai import OpenAI

from .state import AgentState, Article, ArticleSentiment
from .analysis_schemas import SentimentOutput
from .prompts import SENTIMENT_SYSTEM_PROMPT, build_sentiment_user_prompt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _analyze_single_article(article: Article) -> ArticleSentiment:
    """
    Call the LLM to analyze a single article and return an ArticleSentiment dict.
    """
    user_prompt = build_sentiment_user_prompt(article)

    logger.info("Analyzing sentiment for article: %s", article.get("title", ""))

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SENTIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    raw_content = (response.choices[0].message.content or "").strip()

    try:
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`")
            raw_content = raw_content.replace("json", "", 1).strip()

        sentiment_struct = SentimentOutput.model_validate_json(raw_content)

    except Exception as e:
        logger.error("Failed to parse structured sentiment output: %s", e)
        logger.debug("Raw model output was: %s", raw_content)

        sentiment_struct = SentimentOutput(
            article_id=article["id"],
            sentiment="neutral",
            confidence=0.0,
            event_tags=[],
            impact_score=0.0,
            reasoning="Fallback neutral sentiment due to parsing error.",
        )

    result: ArticleSentiment = {
        "article_id": sentiment_struct.article_id or article["id"],
        "sentiment": sentiment_struct.sentiment,
        "confidence": float(sentiment_struct.confidence),
        "event_tags": list(sentiment_struct.event_tags),
        "impact_score": float(sentiment_struct.impact_score),
        "reasoning": sentiment_struct.reasoning,
    }

    return result


def analyze_sentiment_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads `articles` from state
    - Runs LLM sentiment + event extraction
    - Writes `sentiments` list back into state
    """
    articles: List[Article] = state.get("articles", [])

    if not articles:
        logger.warning("No articles found in state; skipping sentiment analysis.")
        new_state: AgentState = dict(state)
        new_state["sentiments"] = []
        return new_state

    sentiments: List[ArticleSentiment] = []

    for article in articles:
        try:
            sentiment = _analyze_single_article(article)
            sentiments.append(sentiment)
        except Exception as e:
            logger.exception(
                "Unexpected error while analyzing article %s: %s",
                article.get("id"),
                e,
            )

    new_state: AgentState = dict(state)
    new_state["sentiments"] = sentiments
    logger.info("Completed sentiment analysis for %d articles.", len(sentiments))
    return new_state
