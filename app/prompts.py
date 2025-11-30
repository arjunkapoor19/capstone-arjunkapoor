
from textwrap import dedent
from .state import Article


SENTIMENT_SYSTEM_PROMPT = dedent("""
    You are an expert financial news analyst.
    Your job is to read news articles about a specific stock
    and extract structured information that explains how this news
    is likely to influence the stock's price.

    Be especially careful to:
    - Focus ONLY on information relevant to the stock's performance.
    - Distinguish between truly impactful events and minor noise.
    - Consider both short-term and medium-term price impact.
    
    You MUST respond with valid JSON matching the provided schema.
""").strip()


SENTIMENT_USER_PROMPT_TEMPLATE = dedent("""
    Analyze the following news article about stock "{ticker}".

    Return a JSON object with:
      - sentiment: "positive", "neutral", or "negative"
      - confidence: a number between 0 and 1
      - event_tags: list of short tags like ["earnings", "acquisition", "lawsuit"]
      - impact_score: number between 0 and 1 representing how strongly this news is likely to affect the stock's price
      - reasoning: a short explanation in plain English

    Article metadata:
      - Title: {title}
      - Source: {source}
      - Published at: {published_at}
      - URL: {url}

    Article text:
    ---
    {full_text}
    ---
""").strip()


def build_sentiment_user_prompt(article: Article) -> str:
    """
    Helper to fill the user prompt template for a single article.
    """
    return SENTIMENT_USER_PROMPT_TEMPLATE.format(
        ticker=article["ticker"],
        title=article.get("title", "N/A"),
        source=article.get("source", "Unknown"),
        published_at=article.get("published_at", "Unknown"),
        url=article.get("url", ""),
        full_text=article.get("full_text") or article.get("summary") or "",
    )
