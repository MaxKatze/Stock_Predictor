import os
import yaml
import yfinance as yf
from .file_format_yahoo import header_row_real, header_row_adjusted

# cache folder
CACHE_DIR = "data/assets"
os.makedirs(CACHE_DIR, exist_ok=True)

# read config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

ASSETS = cfg.get("assets", [])
START_DATE = cfg.get("start_date", "2023-01-01")
END_DATE = cfg.get("end_date", "2027-12-31")
FORCE_UPDATE = cfg.get("force_update", False)
AUTO_ADJUST = cfg.get("auto_adjust", True)


def download_asset(symbol, start=START_DATE, end=END_DATE, force_update=FORCE_UPDATE, auto_adjust=AUTO_ADJUST):
    """download assets"""
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    ticker = yf.Ticker(symbol)
    info = ticker.info
    print(symbol, "Currency:", info.get("currency"))

    if os.path.exists(filepath) and not force_update:
        print(f"{symbol} already cached.")
        return filepath
    
    print(f"Downloading {symbol}...")
    data = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust)

    data.to_csv(filepath)

    # correct yahoo finance data
    header = header_row_adjusted if auto_adjust else header_row_real
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines = lines[3:]
    lines.insert(0, header)
    with open(filepath, "w") as f:
        f.writelines(lines)

    return filepath


if __name__ == '__main__':
    cached_files = [download_asset(sym) for sym in ASSETS]
    print("Fertig! Dateien sind im Cache:", cached_files)
