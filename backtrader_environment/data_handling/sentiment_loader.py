"""Sentiment data handling for Alpha Vantage News Sentiment API.

This module provides functions to load and cache sentiment data for stocks.
The sentiment scores are used as features for the SVR model as described
in Konzeption Section 3.2.2.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT_DIR = os.path.join(ROOT_DIR, "data", "sentiment")
API_KEY_FILE = os.path.join(ROOT_DIR, "alpha_vantage_key.txt")

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"


def get_api_key() -> str:
    """Load Alpha Vantage API key from file."""
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE, "r") as f:
            return f.read().strip()

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    if api_key:
        return api_key

    raise ValueError(
        f"Alpha Vantage API key not found. "
        f"Either create {API_KEY_FILE} or set ALPHA_VANTAGE_API_KEY environment variable."
    )


def download_sentiment_data(ticker: str, time_from: str = None, time_to: str = None) -> list[dict]:
    """Download sentiment data from Alpha Vantage News Sentiment API.

    Args:
        ticker: Stock ticker symbol
        time_from: Start date in format YYYYMMDDTHHMM (optional)
        time_to: End date in format YYYYMMDDTHHMM (optional)

    Returns:
        List of sentiment records with date and score
    """
    if not _REQUESTS_AVAILABLE:
        raise ImportError("requests library required for API calls")

    api_key = get_api_key()

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key,
        "limit": 1000,
    }
    if time_from:
        params["time_from"] = time_from
    if time_to:
        params["time_to"] = time_to

    response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params)
    data = response.json()

    if "feed" not in data:
        if "Note" in data:
            raise RuntimeError(f"API limit reached: {data['Note']}")
        if "Error Message" in data:
            raise RuntimeError(f"API error: {data['Error Message']}")
        return []

    sentiment_records = []
    for item in data["feed"]:
        time_published = item.get("time_published", "")
        if not time_published:
            continue

        ticker_sentiments = item.get("ticker_sentiment", [])
        for ts in ticker_sentiments:
            if ts.get("ticker") == ticker:
                score = float(ts.get("ticker_sentiment_score", 0))
                date_str = time_published[:8]
                try:
                    date = datetime.strptime(date_str, "%Y%m%d")
                    sentiment_records.append({
                        "date": date,
                        "sentiment": score,
                        "relevance": float(ts.get("relevance_score", 0)),
                    })
                except ValueError:
                    continue

    return sentiment_records


def load_sentiment_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Load sentiment data from cache or download if not available.

    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD format)
        end: End date (YYYY-MM-DD format)

    Returns:
        DataFrame with DatetimeIndex and 'sentiment' column
    """
    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    cache_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
    else:
        df = pd.DataFrame(columns=["sentiment"])
        df.index.name = "date"

    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    return df


def save_sentiment_data(ticker: str, records: list[dict]):
    """Save sentiment records to cache file.

    Args:
        ticker: Stock ticker symbol
        records: List of sentiment records with date and sentiment keys
    """
    os.makedirs(SENTIMENT_DIR, exist_ok=True)
    cache_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment.csv")

    if os.path.exists(cache_path):
        existing_df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
    else:
        existing_df = pd.DataFrame(columns=["sentiment"])
        existing_df.index.name = "date"

    if records:
        new_df = pd.DataFrame(records)
        new_df = new_df.set_index("date")
        new_df = new_df.groupby(level=0).mean()

        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        combined_df = combined_df.sort_index()
    else:
        combined_df = existing_df

    combined_df.to_csv(cache_path)


def download_and_cache_sentiment(ticker: str, start: str, end: str, rate_limit_delay: float = 12.0):
    """Download sentiment data for a date range and cache it.

    Alpha Vantage free tier has rate limits (5 calls/minute, 500/day).
    This function handles pagination and rate limiting.

    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        rate_limit_delay: Seconds to wait between API calls
    """
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    all_records = []
    current_date = start_dt

    while current_date < end_dt:
        chunk_end = min(current_date + timedelta(days=30), end_dt)

        time_from = current_date.strftime("%Y%m%dT0000")
        time_to = chunk_end.strftime("%Y%m%dT2359")

        print(f"  Downloading {ticker} sentiment: {current_date.date()} to {chunk_end.date()}")

        try:
            records = download_sentiment_data(ticker, time_from, time_to)
            all_records.extend(records)
        except RuntimeError as e:
            print(f"    Warning: {e}")
            break

        current_date = chunk_end + timedelta(days=1)
        time.sleep(rate_limit_delay)

    if all_records:
        save_sentiment_data(ticker, all_records)
        print(f"  Saved {len(all_records)} sentiment records for {ticker}")
    else:
        print(f"  No sentiment data found for {ticker}")


def get_sentiment_for_model(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Get sentiment data formatted for SVR model.

    Returns DataFrame with DatetimeIndex and 'sentiment' column.
    If no sentiment data is available, returns empty DataFrame.

    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date

    Returns:
        DataFrame suitable for SVRPredictionModel.set_sentiment_data()
    """
    try:
        df = load_sentiment_data(ticker, start, end)
        if len(df) > 0 and "sentiment" in df.columns:
            return df[["sentiment"]]
    except Exception:
        pass

    return pd.DataFrame(columns=["sentiment"])


def generate_placeholder_sentiment(dates: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Generate placeholder sentiment data for testing when API is not available.

    This creates synthetic sentiment data that follows realistic patterns:
    - Mean around 0 (neutral)
    - Some autocorrelation (sentiment trends)
    - Occasional spikes (news events)

    Args:
        dates: DatetimeIndex of trading days
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sentiment scores
    """
    np.random.seed(seed)

    n = len(dates)
    base = np.random.randn(n) * 0.1

    for i in range(1, n):
        base[i] = 0.7 * base[i - 1] + 0.3 * base[i]

    n_spikes = n // 50
    spike_indices = np.random.choice(n, n_spikes, replace=False)
    base[spike_indices] += np.random.choice([-0.5, 0.5], n_spikes)

    base = np.clip(base, -1, 1)

    return pd.DataFrame({"sentiment": base}, index=dates)
