import os
import logging
from typing import List

import requests
from openai import OpenAI
from dotenv import load_dotenv

from .state import Article, AgentState

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _generate_synthetic_articles(ticker: str, start_date: str, end_date: str) -> List[Article]:
    logger.warning("Falling back to synthetic news generation for %s", ticker)

    prompt = f"""
    Generate 2 realistic financial news events for stock {ticker}
    between {start_date} and {end_date}.
    Each must include:
    - title
    - summary
    - full_text
    - source
    - ISO timestamp (published_at)
    Return a JSON array, no extra text.
    """

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You generate plausible but synthetic stock news."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    import json
    content = (resp.choices[0].message.content or "").strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Synthetic news is not a list")
    except Exception as e:
        logger.error("Synthetic JSON parse error: %s", e)
        data = []

    articles: List[Article] = []
    for idx, a in enumerate(data):
        articles.append(
            {
                "id": f"{ticker}-synthetic-{idx}",
                "ticker": ticker,
                "title": a.get("title", ""),
                "url": "",
                "published_at": a.get("published_at", f"{start_date}T00:00:00Z"),
                "source": a.get("source", "SyntheticWire"),
                "summary": a.get("summary", ""),
                "full_text": a.get("full_text", a.get("summary", "")),
            }
        )
    logger.info("Generated %d synthetic articles for %s.", len(articles), ticker)
    return articles


def fetch_news_for_ticker(ticker: str, start_date: str, end_date: str, max_results: int = 20) -> List[Article]:
    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        logger.error("MARKETAUX_API_KEY missing — using synthetic fallback.")
        return _generate_synthetic_articles(ticker, start_date, end_date)

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "entities": ticker,
        "filter_entities": "true",
        "language": "en",
        "limit": max_results,
        "api_token": api_key,
    }

    try:
        logger.info("Fetching MarketAux news: %s", ticker)
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()

        res = r.json()
        news_data = res.get("data", [])

        if not isinstance(news_data, list):
            logger.warning("MarketAux returned unexpected data format")
            return _generate_synthetic_articles(ticker, start_date, end_date)

        if len(news_data) == 0:
            logger.warning("MarketAux returned zero articles for %s", ticker)
            return _generate_synthetic_articles(ticker, start_date, end_date)

        # Filter out garbage / non-ticker-related articles
        filtered = []
        ticker_l = ticker.lower()
        extra_keywords = [ticker_l]
        if ticker.upper() == "AAPL":
            extra_keywords.append("apple")

        for idx, n in enumerate(news_data):
            text = " ".join(
                [
                    str(n.get("title", "")),
                    str(n.get("description", "")),
                    str(n.get("snippet", "")),
                ]
            ).lower()

            if any(kw in text for kw in extra_keywords):
                filtered.append(n)

        if not filtered:
            logger.warning(
                "MarketAux articles not clearly about %s; falling back to synthetic ticker-specific news.",
                ticker,
            )
            return _generate_synthetic_articles(ticker, start_date, end_date)

        articles: List[Article] = []
        for idx, n in enumerate(filtered):
            articles.append(
                {
                    "id": f"{ticker}-{idx}",
                    "ticker": ticker,
                    "title": n.get("title", ""),
                    "url": n.get("url", ""),
                    "published_at": n.get("published_at", f"{start_date}T00:00:00Z"),
                    "source": n.get("source", "Unknown"),
                    "summary": n.get("description", n.get("snippet", "")),
                    "full_text": n.get("description", n.get("snippet", "")),
                }
            )

        logger.info("Fetched %d relevant real news articles for %s", len(articles), ticker)
        return articles

    except Exception as e:
        logger.error("MarketAux error: %s — using synthetic fallback.", e)
        return _generate_synthetic_articles(ticker, start_date, end_date)


def fetch_news_node(state: AgentState) -> AgentState:
    ticker = state["ticker"]
    start_date = state["start_date"]
    end_date = state["end_date"]

    articles = fetch_news_for_ticker(ticker, start_date, end_date)

    new_state = dict(state)
    new_state["articles"] = articles
    return new_state
