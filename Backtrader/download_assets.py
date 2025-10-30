import os
import yfinance as yf

#cache folder
CACHE_DIR = "data/assets"
os.makedirs(CACHE_DIR, exist_ok=True)

# list of assets
ASSETS = ["ORCL", "AAPL", "MSFT"]

# Zeitraum für die historischen Daten
START_DATE = "1995-01-01"
END_DATE = "2025-12-31"

def download_asset(symbol, start=START_DATE, end=END_DATE, force_update=False):
    """download assets"""
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    if os.path.exists(filepath) and not force_update:
        print(f"{symbol} already cached.")
        return filepath
    
    print(f"Downloading {symbol}...")
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    data.to_csv(filepath)
    return filepath


#check assets and download
if __name__ == '__main__':
    cached_files = [download_asset(sym) for sym in ASSETS]
    print("Fertig! Dateien sind im Cache:", cached_files)
