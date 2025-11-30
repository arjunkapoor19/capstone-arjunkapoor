from typing import List, Literal, Optional, TypedDict
from datetime import datetime


SentimentLabel = Literal["positive", "neutral", "negative"]

class Article(TypedDict):
    """Raw news article data fetched from tools/APIs."""
    id: str                
    ticker: str
    title: str
    url: str
    published_at: str      
    source: str
    summary: str           
    full_text: str         


class ArticleSentiment(TypedDict):
    """LLM-extracted sentiment + events for one article."""
    article_id: str
    sentiment: SentimentLabel
    confidence: float                  
    event_tags: List[str]             
    impact_score: float               
    reasoning: str                     


class PriceBar(TypedDict):
    """Basic OHLCV data for one trading day."""
    date: str                          
    open: float
    high: float
    low: float
    close: float
    volume: int


class PatternSignal(TypedDict):
    """
    Detected technical pattern within the date range.
    Example names: "bullish_breakout", "bearish_reversal", "gap_up", etc.
    """
    name: str                          
    label: str                         
    start_date: str                    
    end_date: str                      
    confidence: float                  
    direction: Literal["bullish", "bearish", "neutral"]
    notes: str                         


class CorrelatedInsight(TypedDict):
    """
    Link between a news event and a technical pattern.
    """
    article_id: str
    pattern_name: str
    correlation_confidence: float      
    lag_days: int                      
    summary: str                       


# ─────────────────────────────
# LangGraph State
# ─────────────────────────────

class AgentState(TypedDict, total=False):
    """
    Global state passed between LangGraph nodes.
    total=False allows us to gradually fill this over the workflow.
    """

    # User input
    ticker: str
    start_date: str        
    end_date: str          

    # Data fetched from tools/APIs
    articles: List[Article]
    prices: List[PriceBar]

    # LLM analysis
    sentiments: List[ArticleSentiment]
    patterns: List[PatternSignal]
    correlations: List[CorrelatedInsight]

    # Final output
    report_markdown: str   
