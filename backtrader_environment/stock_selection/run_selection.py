"""Run SDE-based stock selection and save results."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handling.data_loader import load_config, load_stock_data, load_macro_data
from data_handling.sp500_universe import get_sp500_2020_tickers
from stock_selection.sde_stock_selector import SDEStockSelector


def main():
    config = load_config()
    sde_config = config.get("sde", {})

    train_start = config.get("training_start", "2020-01-02")
    train_end = config.get("training_end", "2023-12-29")

    tickers = get_sp500_2020_tickers()
    print(f"Loading data for {len(tickers)} S&P 500 stocks...")

    stock_data = {}
    for ticker in tickers:
        try:
            df = load_stock_data(ticker, start=train_start, end=train_end)
            if len(df) >= 500:
                stock_data[ticker] = df["Close"]
        except FileNotFoundError:
            continue

    print(f"Loaded {len(stock_data)} stocks with sufficient data.")

    macro_factors_config = config.get("macro_factors", {})
    factor_data = {}
    for key in macro_factors_config:
        try:
            series = load_macro_data(key, start=train_start, end=train_end)
            if len(series) >= 500:
                factor_data[key] = series
        except FileNotFoundError:
            print(f"  WARNING: No macro data for '{key}'. Skipping.")

    print(f"Loaded {len(factor_data)} macro factors.")

    if len(stock_data) < 20 or len(factor_data) < 2:
        print("ERROR: Not enough data. Download data first with download_assets.py")
        sys.exit(1)

    selector = SDEStockSelector(
        num_simulations=sde_config.get("num_simulations", 100),
        group_size=sde_config.get("group_size", 4),
        num_factors=sde_config.get("num_factors", 2),
        num_selected=sde_config.get("num_selected", 10),
    )

    print(f"Running SDE selection ({sde_config.get('num_simulations', 100)} simulations)...")
    selected = selector.select(stock_data, factor_data)

    print(f"\nSelected {len(selected)} stocks:")
    for i, ticker in enumerate(selected, 1):
        print(f"  {i}. {ticker}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "selection")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "selected_stocks.json")

    with open(output_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
