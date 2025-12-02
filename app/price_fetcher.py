import logging
from typing import List

from .state import PriceBar, AgentState

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    logger.warning("yfinance not installed; falling back to mock price data.")
    YF_AVAILABLE = False


def _mock_prices(ticker: str, start_date: str, end_date: str) -> List[PriceBar]:
    """
    Fallback OHLCV data so the pipeline still runs in demos
    without internet or yfinance.
    """
    logger.warning("Using MOCK price data for %s from %s to %s.", ticker, start_date, end_date)

    # Super simple fake trend: day 1 = 100, then up and down a bit
    base_close = 100.0
    days = [
        ("2024-11-25", base_close * 0.98),
        ("2024-11-26", base_close * 1.02),
        ("2024-11-27", base_close * 1.05),
        ("2024-11-28", base_close * 1.08),
        ("2024-11-29", base_close * 1.03),
    ]

    prices: List[PriceBar] = []
    for date, close in days:
        bar: PriceBar = {
            "date": date,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000,
        }
        prices.append(bar)

    return prices


def fetch_price_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> List[PriceBar]:
    """
    Fetch daily OHLCV price history for the given ticker and date range.
    Uses yfinance if available; otherwise falls back to mock data.
    """
    if not YF_AVAILABLE:
        return _mock_prices(ticker, start_date, end_date)

    try:
        logger.info("Fetching price history for %s from %s to %s", ticker, start_date, end_date)
        data = yf.download(
            tickers=ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
        )

        if data.empty:
            logger.warning("No price data returned for %s, using mock data.", ticker)
            return _mock_prices(ticker, start_date, end_date)

        prices: List[PriceBar] = []

        for idx, (date, row) in enumerate(data.iterrows()):
            bar: PriceBar = {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"].item()),
                "high": float(row["High"].item()),
                "low": float(row["Low"].item()),
                "close": float(row["Close"].item()),
                "volume": int(row.get("Volume", 0).item() if "Volume" in row else 0),
            }
            prices.append(bar)

        logger.info("Fetched %d OHLCV bars for %s", len(prices), ticker)
        return prices

    except Exception as e:
        logger.exception("Error fetching prices for %s: %s. Using mock data.", ticker, e)
        return _mock_prices(ticker, start_date, end_date)


def fetch_prices_node(state: AgentState) -> AgentState:
    """
    LangGraph node:
    - Reads ticker, start_date, end_date
    - Writes `prices` into state
    """
    ticker = state.get("ticker")
    start_date = state.get("start_date")
    end_date = state.get("end_date")

    if not ticker or not start_date or not end_date:
        raise ValueError("ticker, start_date, and end_date must be set in state before fetch_prices_node.")

    prices = fetch_price_history(ticker=ticker, start_date=start_date, end_date=end_date)

    new_state: AgentState = dict(state)
    new_state["prices"] = prices
    return new_state
