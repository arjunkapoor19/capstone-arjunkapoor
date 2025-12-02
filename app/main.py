import os
from pprint import pprint

from .state import AgentState
from .graph import build_graph


def run_example():
    """
    Run an example end-to-end pipeline for a given ticker and date range.
    Adjust the ticker and dates as you like for your demo.
    """
    ticker = os.getenv("EXAMPLE_TICKER", "AAPL")
    start_date = os.getenv("EXAMPLE_START_DATE", "2024-11-20")
    end_date = os.getenv("EXAMPLE_END_DATE", "2024-11-29")

    initial_state: AgentState = {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
    }

    graph = build_graph()
    final_state = graph.invoke(initial_state)

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
