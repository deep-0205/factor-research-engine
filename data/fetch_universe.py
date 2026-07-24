import pandas as pd
import requests
from logger import get_logger

logger = get_logger("fetch_universe")

def get_nifty500_tickers() -> list:
    """
    Fetch Nifty 500 tickers from NSE official source.
    
    CRITICAL:
    - Uses official NSE archives URL
    - Adds .NS suffix for NSE-listed stocks
    - Validates response and handles failures gracefully
    - Returns empty list on error (caller handles)
    
    Returns:
        List of ticker symbols (e.g., ["INFY.NS", "RELIANCE.NS", ...])
    """
    
    url = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        
        df = pd.read_csv(pd.io.common.StringIO(response.text))
        
        if df.empty:
            logger.warning("Nifty 500 data is empty from remote source")
            return []
        
        # Add NSE suffix
        tickers = [symbol + ".NS" for symbol in df["Symbol"].tolist()]
        
        # Validate
        if not tickers:
            logger.warning("No tickers extracted from Nifty 500 list")
            return []
        
        logger.info(
            f"Successfully fetched {len(tickers)} tickers from Nifty 500 | "
            f"Sample: {tickers[:5]}"
        )
        return tickers
        
    except requests.exceptions.Timeout:
        logger.error("Timeout fetching Nifty 500 tickers (10s)")
        return []
    except requests.exceptions.ConnectionError:
        logger.error("Connection error fetching Nifty 500 tickers")
        return []
    except Exception as e:
        logger.error(f"Error fetching Nifty 500 tickers: {e}")
        return []

