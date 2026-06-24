import pandas as pd
import requests
from logger import get_logger

logger = get_logger("fetch_universe")

def get_nifty500_tickers() -> list:
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        tickers = [symbol + ".NS" for symbol in df["Symbol"].tolist()]
        logger.info(f"Fetched {len(tickers)} tickers from Nifty 500.")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching Nifty 500 tickers: {e}")
        return []
    
    
