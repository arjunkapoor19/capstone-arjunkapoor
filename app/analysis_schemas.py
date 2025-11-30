
from typing import List, Literal
from pydantic import BaseModel, Field

from .state import SentimentLabel


class SentimentOutput(BaseModel):
    """
    Structured output for sentiment + event extraction from a single article.
    This matches the ArticleSentiment TypedDict shape from state.py.
    """
    article_id: str = Field(
        ...,
        description="ID of the article this analysis refers to.",
    )
    sentiment: SentimentLabel = Field(
        ...,
        description="Overall sentiment of the article toward the stock: positive, neutral, or negative.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence in the sentiment label between 0 and 1.",
    )
    event_tags: List[str] = Field(
        default_factory=list,
        description=(
            "Short tags describing key events, e.g. "
            "['earnings', 'guidance_cut', 'acquisition', 'lawsuit', 'downgrade']."
        ),
    )
    impact_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How impactful this news is likely to be on the stock price (0 to 1).",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why you chose this sentiment and impact.",
    )


class BatchSentimentOutput(BaseModel):
    """
    Optional wrapper if we want the LLM to output a list in one shot.
    You can use this if you batch multiple articles into a single call.
    """
    results: List[SentimentOutput]
