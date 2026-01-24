"""
External Market Data Fetcher

Fetches BTC and S&P 500 prices for training data.
Uses free APIs: CoinGecko (BTC) and Alpha Vantage (S&P 500).
"""

import requests
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Free API endpoints
COINGECKO_BTC_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
ALPHA_VANTAGE_SP500_URL = "https://www.alphavantage.co/query"

# Database path
EXTERNAL_MARKETS_DB = "external_markets.db"


def init_database(db_path: str = EXTERNAL_MARKETS_DB):
    """Initialize database for external market prices."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # BTC prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS btc_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp)
        )
    """)
    
    # S&P 500 prices table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sp500_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(timestamp)
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_btc_timestamp 
        ON btc_prices(timestamp DESC)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_sp500_timestamp 
        ON sp500_prices(timestamp DESC)
    """)
    
    conn.commit()
    conn.close()
    logger.info(f"External markets database initialized: {db_path}")


def fetch_btc_price() -> Optional[float]:
    """Fetch current BTC price from CoinGecko (free, no API key needed)."""
    try:
        response = requests.get(COINGECKO_BTC_URL, timeout=5)
        response.raise_for_status()
        data = response.json()
        btc_price = data.get("bitcoin", {}).get("usd")
        if btc_price:
            return float(btc_price)
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch BTC price: {e}")
        return None


def fetch_sp500_price() -> Optional[float]:
    """Fetch current S&P 500 price from Alpha Vantage (free tier, 5 calls/min)."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        logger.warning("ALPHA_VANTAGE_API_KEY not set - skipping S&P 500 fetch")
        return None
    
    try:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": "SPY",  # S&P 500 ETF (more reliable than ^GSPC)
            "apikey": api_key,
        }
        response = requests.get(ALPHA_VANTAGE_SP500_URL, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        # Alpha Vantage format
        quote = data.get("Global Quote", {})
        price_str = quote.get("05. price") or quote.get("price")
        if price_str:
            return float(price_str)
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 price: {e}")
        return None


def store_price(db_path: str, table: str, timestamp: datetime, price: float):
    """Store price in database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            f"INSERT OR IGNORE INTO {table} (timestamp, price) VALUES (?, ?)",
            (timestamp.isoformat(), price)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Error storing {table} price: {e}")
        conn.rollback()
    finally:
        conn.close()


def get_recent_prices(
    db_path: str,
    table: str,
    hours: int = 2.5,
    interval_minutes: int = 5,
) -> List[Tuple[datetime, float]]:
    """
    Get recent prices for training.
    
    Args:
        db_path: Database path
        table: Table name ('btc_prices' or 'sp500_prices')
        hours: Hours of history to fetch
        interval_minutes: Target interval between prices
        
    Returns:
        List of (timestamp, price) tuples, sorted by timestamp
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    
    try:
        cursor.execute(
            f"""
            SELECT timestamp, price 
            FROM {table}
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (cutoff,)
        )
        rows = cursor.fetchall()
        
        # Convert to datetime objects
        prices = []
        for timestamp_str, price in rows:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                prices.append((timestamp, float(price)))
            except Exception as e:
                logger.warning(f"Error parsing timestamp {timestamp_str}: {e}")
                continue
        
        return prices
    except Exception as e:
        logger.error(f"Error fetching {table} prices: {e}")
        return []
    finally:
        conn.close()


def fetch_and_store_markets():
    """Fetch and store current BTC and S&P 500 prices."""
    db_path = EXTERNAL_MARKETS_DB
    
    # Initialize database if needed
    if not Path(db_path).exists():
        init_database(db_path)
    
    timestamp = datetime.utcnow()
    
    # Fetch BTC
    btc_price = fetch_btc_price()
    if btc_price:
        store_price(db_path, "btc_prices", timestamp, btc_price)
        logger.info(f"Stored BTC price: ${btc_price:,.2f}")
    else:
        logger.warning("Failed to fetch BTC price")
    
    # Fetch S&P 500 (with rate limiting)
    sp500_price = fetch_sp500_price()
    if sp500_price:
        store_price(db_path, "sp500_prices", timestamp, sp500_price)
        logger.info(f"Stored S&P 500 price: ${sp500_price:,.2f}")
    else:
        logger.warning("Failed to fetch S&P 500 price (may need API key)")
    
    # Rate limit for Alpha Vantage (5 calls/min)
    if sp500_price:
        time.sleep(12)  # Wait 12 seconds to avoid rate limit


if __name__ == "__main__":
    # Initialize and fetch
    init_database()
    fetch_and_store_markets()
