from langgraph.graph import StateGraph, END

from .state import AgentState
from .news_fetcher import fetch_news_node
from .sentiment_node import analyze_sentiment_node
from .price_fetcher import fetch_prices_node
from .pattern_detection import detect_patterns_node
from .correlation_node import correlate_node
from .report_node import generate_report_node


def build_graph():
    """
    Build the LangGraph workflow for the Stock Market News-Pattern Intelligence Agent.

    Flow:
      1. fetch_news_node
      2. analyze_sentiment_node
      3. fetch_prices_node
      4. detect_patterns_node
      5. correlate_node
      6. generate_report_node
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("fetch_news", fetch_news_node)
    graph.add_node("analyze_sentiment", analyze_sentiment_node)
    graph.add_node("fetch_prices", fetch_prices_node)
    graph.add_node("detect_patterns", detect_patterns_node)
    graph.add_node("correlate", correlate_node)
    graph.add_node("generate_report", generate_report_node)

    graph.set_entry_point("fetch_news")

    graph.add_edge("fetch_news", "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "fetch_prices")
    graph.add_edge("fetch_prices", "detect_patterns")
    graph.add_edge("detect_patterns", "correlate")
    graph.add_edge("correlate", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()
