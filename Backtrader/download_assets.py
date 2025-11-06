import os
import yfinance as yf
from file_format_yahoo import header_row_real, header_row_adjusted

#cache folder
CACHE_DIR = "data/assets"
os.makedirs(CACHE_DIR, exist_ok=True)

# list of assets
ASSETS = ["ORCL", "AAPL", "MSFT", "SAP"]

# Zeitraum für die historischen Daten
START_DATE = "1995-01-01"
END_DATE = "2027-12-31"

def download_asset(symbol, start=START_DATE, end=END_DATE, force_update=False, auto_adjust=True):
    """download assets"""
    filepath = os.path.join(CACHE_DIR, f"{symbol}.csv")
    
    ticker = yf.Ticker(symbol)
    info = ticker.info
    print(symbol, "Currency:", info.get("currency"))

    if os.path.exists(filepath) and not force_update:
        print(f"{symbol} already cached.")
        return filepath
    
    print(f"Downloading {symbol}...")
    data = yf.download(symbol, start=start, end=end, auto_adjust=auto_adjust) # auto adjusted prices (no dividend/splits)

    data.to_csv(filepath)

    #correct yahoo finance data, somehow header is wrong, TODO: correct it with pandas, this somehow weird:
    if auto_adjust:
        header = header_row_adjusted
    else:
        header = header_row_real
    
    with open(filepath, "r") as f:
        lines = f.readlines()
    lines = lines[3:]
    lines.insert(0, header)
    with open(filepath, "w") as f:
        f.writelines(lines)

    return filepath


#check assets and download
if __name__ == '__main__':
    cached_files = [download_asset(sym, force_update=True, auto_adjust=True) for sym in ASSETS]
    print("Fertig! Dateien sind im Cache:", cached_files)
