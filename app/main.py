import os
import logging
from pprint import pprint

from .state import AgentState
from .graph import build_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_example():
    """
    Run an example end-to-end pipeline for a given ticker and date range.
    """
    ticker = "AAPL"
    start_date ="2024-11-20"
    end_date = "2024-11-29"

    initial_state: AgentState = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
    }

    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        logger.critical("Pipeline crashed unexpectedly: %s", e, exc_info=True)
        return

    report = final_state.get("report_markdown", "")
    print("\n" + "=" * 80)
    print(f"FINAL REPORT FOR {ticker}")
    print("=" * 80 + "\n")
    print(report)
    print("\n" + "=" * 80)
    print("Raw final state keys:")
    pprint(list(final_state.keys()))
    print("=" * 80)


if __name__ == "__main__":
    run_example()
