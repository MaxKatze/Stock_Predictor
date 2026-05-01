"""Unified data loading with train/validation/test split."""

import os
import pandas as pd
import yaml

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
ASSETS_DIR = os.path.join(ROOT_DIR, "data", "assets")
MACRO_DIR = os.path.join(ROOT_DIR, "data", "macro")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_stock_data(ticker: str, start: str = None, end: str = None) -> pd.DataFrame:
    """Load stock OHLCV data from CSV cache."""
    filepath = os.path.join(ASSETS_DIR, f"{ticker}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No data for {ticker}. Run download_assets.py first.")

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def load_macro_data(factor_key: str, start: str = None, end: str = None) -> pd.Series:
    """Load macroeconomic factor data."""
    filepath = os.path.join(MACRO_DIR, f"{factor_key}.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No macro data for {factor_key}. Run download_assets.py first.")

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    series = df["Close"]
    if start:
        series = series[series.index >= start]
    if end:
        series = series[series.index <= end]
    return series


def get_data_split(ticker: str, config: dict = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split stock data into train/validation/test sets (60/10/30).

    Returns:
        (train_df, val_df, test_df)
    """
    if config is None:
        config = load_config()

    train_start = config.get("training_start", "2020-01-02")
    train_end = config.get("training_end", "2023-12-29")
    val_start = config.get("validation_start", "2024-01-02")
    val_end = config.get("validation_end", "2024-06-28")
    test_start = config.get("test_start", "2024-07-01")
    test_end = config.get("test_end", "2025-12-31")

    df = load_stock_data(ticker, start=train_start, end=test_end)

    train_df = df[(df.index >= train_start) & (df.index <= train_end)]
    val_df = df[(df.index >= val_start) & (df.index <= val_end)]
    test_df = df[(df.index >= test_start) & (df.index <= test_end)]

    return train_df, val_df, test_df


def get_selected_stocks(config: dict = None) -> list[str]:
    """Load selected stocks from selection results or config."""
    if config is None:
        config = load_config()

    selection_path = os.path.join(ROOT_DIR, "data", "selection", "selected_stocks.json")
    if os.path.exists(selection_path):
        import json
        with open(selection_path) as f:
            return json.load(f)

    return config.get("assets", [])
