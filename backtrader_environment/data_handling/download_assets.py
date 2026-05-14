import os
import yaml
import yfinance as yf


header_row_real = "Date,Adj Close,Close,High,Low,Open,Volume\n"
header_row_adjusted = "Date,Close,High,Low,Open,Volume\n"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(ROOT_DIR, "data/assets")
os.makedirs(CACHE_DIR, exist_ok=True)

MACRO_DIR = os.path.join(ROOT_DIR, "data/macro")
os.makedirs(MACRO_DIR, exist_ok=True)
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

ASSETS = cfg.get("assets", [])
START_DATE = cfg.get("start_date", cfg.get("training_start", "2020-01-02"))
END_DATE = cfg.get("end_date", cfg.get("test_end", "2025-12-31"))
FORCE_UPDATE = cfg.get("force_update", False)
AUTO_ADJUST = cfg.get("auto_adjust", True)
MACRO_FACTORS = cfg.get("macro_factors", {})


def download_asset(symbol, start=START_DATE, end=END_DATE, force_update=FORCE_UPDATE, auto_adjust=AUTO_ADJUST):
    """Download and normalize one symbol into the local CSV cache."""
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")

    if os.path.exists(filepath) and not force_update:
        print(f"{symbol} already cached.")
        return filepath

    print(f"Downloading {symbol}...")
    try:
        data = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust, progress=False)
    except Exception as e:
        print(f"  ERROR downloading {symbol}: {e}")
        return None

    if data.empty:
        print(f"  WARNING: No data for {symbol}")
        return None

    data.to_csv(filepath)

    header = header_row_adjusted if auto_adjust else header_row_real
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines = lines[3:]
    lines.insert(0, header)
    with open(filepath, "w") as f:
        f.writelines(lines)

    return filepath


def download_universe(tickers: list[str], start=START_DATE, end=END_DATE,
                      force_update=FORCE_UPDATE, auto_adjust=AUTO_ADJUST, batch_size=50):
    """Download a large set of tickers in batches."""
    results = {}
    total = len(tickers)
    for i in range(0, total, batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}: "
              f"{batch[0]}..{batch[-1]}")
        for symbol in batch:
            path = download_asset(symbol, start=start, end=end,
                                  force_update=force_update, auto_adjust=auto_adjust)
            if path:
                results[symbol] = path
    print(f"Downloaded {len(results)}/{total} tickers successfully.")
    return results


def download_macro_factors(factors: dict = None, start=START_DATE, end=END_DATE, force_update=FORCE_UPDATE):
    """Download macroeconomic factor data.

    Args:
        factors: dict mapping factor_key -> ticker symbol
                 e.g. {"oil": "CL=F", "gold": "GC=F", ...}
    """
    if factors is None:
        factors = MACRO_FACTORS

    for key, symbol in factors.items():
        filepath = os.path.join(MACRO_DIR, f"{key}.csv")

        if os.path.exists(filepath) and not force_update:
            print(f"Macro factor '{key}' already cached.")
            continue

        print(f"Downloading macro factor '{key}' ({symbol})...")

        if symbol == "T10YIE":
            _download_fred_series(symbol, filepath, start, end)
        else:
            try:
                data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            except Exception as e:
                print(f"  ERROR downloading {key}: {e}")
                continue

            if data.empty:
                print(f"  WARNING: No data for {key} ({symbol})")
                continue

            data.to_csv(filepath)
            with open(filepath, "r") as f:
                lines = f.readlines()
            lines = lines[3:]
            lines.insert(0, header_row_adjusted)
            with open(filepath, "w") as f:
                f.writelines(lines)

    print("Macro factor download complete.")


def _download_fred_series(series_id: str, filepath: str, start: str, end: str):
    """Download a FRED series (e.g. T10YIE breakeven inflation)."""
    try:
        import pandas_datareader.data as web
        data = web.DataReader(series_id, "fred", start, end)
        data.columns = ["Close"]
        data.index.name = "Date"
        data.to_csv(filepath)
    except ImportError:
        print(f"  pandas_datareader not installed. Skipping {series_id}.")
        print("  Install with: pip install pandas-datareader")
    except Exception as e:
        print(f"  ERROR downloading FRED {series_id}: {e}")


if __name__ == '__main__':
    from sp500_universe import get_sp500_2020_tickers

    print("=" * 70)
    print("DATA DOWNLOAD FOR STOCK SELECTION AND EVALUATION")
    print("=" * 70)
    print(f"Time period: {START_DATE} to {END_DATE}")
    print()

    # 1. Download S&P 500 universe
    sp500_tickers = get_sp500_2020_tickers()
    print(f"[1/2] Downloading S&P 500 universe ({len(sp500_tickers)} stocks)...")
    print("-" * 70)
    download_universe(sp500_tickers)
    print()

    # 2. Download macroeconomic factors
    print(f"[2/2] Downloading macroeconomic factors...")
    print("-" * 70)
    if MACRO_FACTORS:
        for key, symbol in MACRO_FACTORS.items():
            print(f"  {key}: {symbol}")
        print()
        download_macro_factors()
    else:
        print("  WARNING: No macro_factors defined in config.yaml")
    print()

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print("Data locations:")
    print("  - Stocks:  data/assets/")
    print("  - Factors: data/macro/")
