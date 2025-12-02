import logging
from typing import List

from .state import AgentState, PriceBar

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _fetch_price_history(ticker: str, start_date: str, end_date: str) -> List[PriceBar]:
    """
    Fetch OHLCV price history using yfinance.
    Returns an empty list on any failure instead of raising.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance is not installed; cannot fetch real price data.")
        return []

    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,  # prevent future default warnings
        )
    except Exception as e:
        logger.error("Failed to fetch price history for %s: %s", ticker, e)
        return []

    if data is None or data.empty:
        logger.warning("No price data returned for %s in given date range.", ticker)
        return []

    prices: List[PriceBar] = []
    for idx_date, row in data.iterrows():
        try:
            prices.append(
                {
                    "date": idx_date.strftime("%Y-%m-%d"),
                    "open": float(row["Open"].item()),
                    "high": float(row["High"].item()),
                    "low": float(row["Low"].item()),
                    "close": float(row["Close"].item()),
                    "volume": int(row.get("Volume", 0).item() if hasattr(row.get("Volume", None), "item") else int(row.get("Volume", 0))),
                }
            )
        except Exception as e:
            logger.warning("Skipping malformed OHLCV row at %s for %s: %s", idx_date, ticker, e)
            continue

    logger.info("Fetched %d OHLCV bars for %s", len(prices), ticker)
    return prices


def fetch_prices_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads ticker & date range from state.
    - Fetches price history.
    - Writes `prices` back into state.
    """
    ticker = state.get("ticker")
    start_date = state.get("start_date")
    end_date = state.get("end_date")

    if not ticker or not start_date or not end_date:
        logger.error("Missing inputs for price fetch: %s", state)
        new_state = dict(state)
        new_state["prices"] = []
        return new_state

    prices = _fetch_price_history(ticker, start_date, end_date)

    new_state = dict(state)
    new_state["prices"] = prices
    return new_state
