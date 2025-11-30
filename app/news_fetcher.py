import os
import logging
from typing import List
from datetime import datetime

import requests

from .state import Article, AgentState


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NewsFetchError(Exception):
    """Custom exception for news fetching issues."""
    pass


def _mock_articles(ticker: str, start_date: str, end_date: str) -> List[Article]:
    """
    Fallback data so the project still works without a real API key.
    Useful for demo / offline runs.
    """
    logger.warning("Using MOCK news articles because no NEWS_API_KEY is set.")

    return [
        {
            "id": f"{ticker}-mock-1",
            "ticker": ticker,
            "title": f"{ticker} beats earnings expectations",
            "url": "https://example.com/mock-earnings",
            "published_at": f"{start_date}T09:30:00Z",
            "source": "MockNewsWire",
            "summary": f"Analysts react as {ticker} posts strong quarterly results.",
            "full_text": (
                f"{ticker} reported quarterly earnings above market expectations, "
                "driven by strong demand and improved margins."
            ),
        },
        {
            "id": f"{ticker}-mock-2",
            "ticker": ticker,
            "title": f"{ticker} faces regulatory scrutiny over new product",
            "url": "https://example.com/mock-regulation",
            "published_at": f"{end_date}T14:15:00Z",
            "source": "MockFinanceDaily",
            "summary": f"Regulators review {ticker}'s latest product launch for compliance.",
            "full_text": (
                f"Regulators announced a review into {ticker}'s latest product, "
                "raising concerns about potential delays and legal challenges."
            ),
        },
    ]


def fetch_news_for_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    max_results: int = 20,
) -> List[Article]:
    """
    Fetch news for a given ticker and date range.

    If NEWS_API_KEY is not set, falls back to mock data so the rest of
    the LangGraph pipeline can still be demonstrated.
    """
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        return _mock_articles(ticker, start_date, end_date)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "apiKey": api_key,
    }

    try:
        logger.info("Fetching news for %s from %s to %s", ticker, start_date, end_date)
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            raise NewsFetchError(f"News API returned non-ok status: {data!r}")

        articles_raw = data.get("articles", [])
        articles: List[Article] = []

        for idx, a in enumerate(articles_raw):
            published_at = a.get("publishedAt") or f"{start_date}T00:00:00Z"
            source_name = (a.get("source") or {}).get("name", "Unknown")

            article: Article = {
                "id": f"{ticker}-{idx}",
                "ticker": ticker,
                "title": a.get("title") or "No title",
                "url": a.get("url") or "",
                "published_at": published_at,
                "source": source_name,
                "summary": a.get("description") or "",
                "full_text": a.get("content") or a.get("description") or "",
            }
            articles.append(article)

        logger.info("Fetched %d articles for %s", len(articles), ticker)
        return articles

    except requests.RequestException as e:
        logger.error("Network error while fetching news: %s", e)
        return _mock_articles(ticker, start_date, end_date)
    except Exception as e:
        logger.exception("Unexpected error while fetching news: %s", e)
        return _mock_articles(ticker, start_date, end_date)


def fetch_news_node(state: AgentState) -> AgentState:
    """
    LangGraph node: reads ticker & date range from state,
    fetches news, and writes `articles` back into the state.
    """
    ticker = state.get("ticker")
    start_date = state.get("start_date")
    end_date = state.get("end_date")

    if not ticker or not start_date or not end_date:
        logger.error("Missing required input fields in state: %s", state)
        raise ValueError("ticker, start_date, and end_date must be set in state before fetch_news_node.")

    articles = fetch_news_for_ticker(ticker=ticker, start_date=start_date, end_date=end_date)

    new_state: AgentState = dict(state)
    new_state["articles"] = articles
    return new_state
