import os
# Set tokenizers parallelism before any imports that might use tokenizers
# This prevents warnings when subprocesses are spawned
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import requests
import time
import sqlite3
import socket

# Force IPv4 for DNS resolution if needed (helps with some network configurations)
# This can help when nslookup works but Python's DNS resolver fails
try:
    # Monkey-patch socket.getaddrinfo to prefer IPv4
    original_getaddrinfo = socket.getaddrinfo
    def getaddrinfo_ipv4(*args, **kwargs):
        results = original_getaddrinfo(*args, **kwargs)
        # Filter to prefer IPv4
        ipv4_results = [r for r in results if r[0] == socket.AF_INET]
        if ipv4_results:
            return ipv4_results
        return results
    # Only apply if DNS resolution is failing - comment out if not needed
    # socket.getaddrinfo = getaddrinfo_ipv4
except Exception:
    pass  # If patching fails, continue without it
from threading import Thread
from flask import Flask, jsonify, render_template, request
import threading
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sol_price_fetcher import SOLPriceFetcher
from statistics import mean, stdev
import logging
import traceback
import pytz
from river.tree import HoeffdingAdaptiveTreeRegressor
import pickle
from signal_performance_tracker import SignalPerformanceTracker
from consensus_tracker import ConsensusTracker

# RL Agent (optional)
try:
    from rl_agent.prediction_manager import PredictionManager
    from rl_agent.attention_logger import AttentionLogger
    from rl_agent.risk_manager import RiskManager
    from rl_agent.explainability import RuleExtractor, SHAPExplainer
    from rl_agent.integration import RLAgentIntegration
    from rl_agent.model_manager import ModelManager
    from rl_agent.retraining_scheduler import RetrainingScheduler
    from rl_agent.model import TradingActorCritic
    from rl_agent.prediction_generator import generate_15m_price_prediction
    RL_AGENT_AVAILABLE = True
except ImportError:
    RL_AGENT_AVAILABLE = False
    logger.warning("RL agent module not available. Install dependencies: pip install torch gymnasium")

# Global RL agent integration instance
rl_agent_integration = None
rl_model_manager = None
rl_retraining_scheduler = None

# News sentiment analyzer (optional)
try:
    from news_sentiment import NewsSentimentAnalyzer
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False
    logger.warning("news_sentiment module not available. Install dependencies: pip install feedparser sentence-transformers scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Ensure the required environment variables are set
if not os.path.exists('.env'):
    raise FileNotFoundError("Missing .env file. Please create a .env file with the required environment variables.")
# Ensure the required environment variables are set
required_env_vars = ["HELIUS_API_KEY", "SOLANA_WALLET_ADDRESS", "LIVECOINWATCH_API_KEY", "DATABASE_PATH"]
for var in required_env_vars:
    if var not in os.environ:
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Global variables to control the price fetching
price_fetcher = None
price_fetch_thread = None
price_fetch_active = False

# Global variables for news sentiment
news_analyzer = None
news_fetch_thread = None
news_fetch_active = False

# Global variables for RL agent decision loop
rl_decision_thread = None
rl_decision_active = False

# Global variables for prediction actuals update loop
prediction_update_thread = None
prediction_update_active = False

# For timezones - which suck.
utc = pytz.utc
#eastern = pytz.timezone("US/Eastern")  # or use tzlocal()
# We should use tzlocal or something better for local timezone, but for now...
eastern = pytz.timezone("America/New_York")  # or use tzlocal()


# Load environment variables from .env file
load_dotenv()

# Flask app to collect and display Solana rewards from Helius API
app = Flask(__name__)

# Constants from environment variables
API_KEY = os.getenv("HELIUS_API_KEY")
WALLET = os.getenv("SOLANA_WALLET_ADDRESS")
LIVECOINWATCH_API = os.getenv("LIVECOINWATCH_API_KEY")
DB_PATH = os.getenv("DATABASE_PATH", "rewards.db")  # Default fallback

# Validate required environment variables
required_env_vars = {
    "HELIUS_API_KEY": API_KEY,
    "SOLANA_WALLET_ADDRESS": WALLET,
    "LIVECOINWATCH_API_KEY": LIVECOINWATCH_API
}

for var_name, var_value in required_env_vars.items():
    if not var_value:
        raise ValueError(f"Missing required environment variable: {var_name}")

ORCA_POOLS = {
    "EUuUbDcafPrmVTD5M6qoJAoyyNbihBhugADAxRMn5he9",
    "2WLWEuKDgkDUccTpbwYp1GToYktiSB1cXvreHUwiSUVP"
}

ORCA_WHIRLPOOL_PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"

# Initialize SQLite database
def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS collect_fees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        signature TEXT,
        timestamp INTEGER,
        fee_payer TEXT,
        token_mint TEXT,
        token_amount REAL,
        from_token_account TEXT,
        to_token_account TEXT,
        from_user_account TEXT,
        to_user_account TEXT,
        UNIQUE(signature, token_mint, to_user_account) ON CONFLICT IGNORE
    )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS tokens (
            mint TEXT PRIMARY KEY,
            symbol TEXT,
            name TEXT,
            decimals INTEGER
        )
        ''')
    # Init prediction DB
    prediction_conn = sqlite3.connect("sol_prices.db")
    prediction_cursor = prediction_conn.cursor()
    prediction_cursor.execute('''
        CREATE TABLE IF NOT EXISTS sol_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_rate REAL NOT NULL,
            actual_rate REAL NOT NULL,
            error REAL NOT NULL,
            mae REAL NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Create indexes for sol_predictions table
    prediction_cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_predictions_timestamp ON sol_predictions(timestamp)')
    prediction_cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_predictions_created_at ON sol_predictions(created_at)')
    prediction_conn.commit()
    prediction_conn.close()

    # Create mSOL tracking tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS msol_balance_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            balance REAL NOT NULL,
            balance_usd REAL,
            transaction_signature TEXT,
            snapshot_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS msol_conversions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signature TEXT UNIQUE,
            timestamp INTEGER NOT NULL,
            amount_sol REAL,
            amount_msol REAL NOT NULL,
            conversion_rate REAL,
            transaction_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create indexes for rewards.db tables
    c.execute('CREATE INDEX IF NOT EXISTS idx_collect_fees_timestamp ON collect_fees(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_collect_fees_token_mint ON collect_fees(token_mint)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_collect_fees_to_user ON collect_fees(to_user_account)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_collect_fees_composite ON collect_fees(token_mint, timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_collect_fees_signature ON collect_fees(signature)')
    
    # Create indexes for mSOL tables
    c.execute('CREATE INDEX IF NOT EXISTS idx_msol_snapshots_timestamp ON msol_balance_snapshots(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_msol_conversions_timestamp ON msol_conversions(timestamp)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_msol_conversions_signature ON msol_conversions(signature)')

    conn.commit()
    conn.close()

def seed_tokens():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    known_tokens = [
        ("So11111111111111111111111111111111111111112", "SOL", "Solana", 9),
        ("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "USDC", "USD Coin", 6)
    ]
    for mint, symbol, name, decimals in known_tokens:
        c.execute('''
            INSERT OR IGNORE INTO tokens (mint, symbol, name, decimals)
            VALUES (?, ?, ?, ?)
        ''', (mint, symbol, name, decimals))
    conn.commit()
    conn.close()


def fetch_helius_transactions(wallet, limit=50):
    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    params = {
        "api-key": API_KEY,
        "limit": limit
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()

# mSOL tracking constants
MSOL_MINT = "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"
SOL_MINT = "So11111111111111111111111111111111111111112"

def fetch_wallet_token_balance(wallet, token_mint, retries=3):
    """
    Fetch current token balance for a specific mint from Helius API.
    Returns the balance as a float, or 0.0 if not found.
    """
    url = f"https://api.helius.xyz/v0/addresses/{wallet}/balances"
    params = {
        "api-key": API_KEY
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Find the token in the tokens array
            if 'tokens' in data:
                for token in data['tokens']:
                    if token.get('mint') == token_mint:
                        # Return balance accounting for decimals
                        amount = token.get('amount', 0)
                        decimals = token.get('decimals', 9)
                        return amount / (10 ** decimals)
            
            return 0.0
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching token balance (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Error fetching token balance for {token_mint}: Timeout after {retries} attempts")
                return 0.0
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error fetching token balance (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Error fetching token balance for {token_mint}: Connection failed after {retries} attempts")
                return 0.0
        except Exception as e:
            logger.error(f"Error fetching token balance for {token_mint}: {e}")
            return 0.0
    
    return 0.0

def fetch_msol_transactions(wallet, since_timestamp=None, limit=1000, retries=3):
    """
    Fetch historical transactions involving mSOL from Helius API.
    Returns list of transactions that have mSOL token transfers.
    Helius API pagination uses transaction signatures, not timestamps.
    """
    all_transactions = []
    url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
    # Use a smaller batch size per request to avoid 400 errors
    batch_size = min(100, limit)  # Start with 100, which is known to work
    params = {
        "api-key": API_KEY,
        "limit": batch_size
    }
    
    max_pages = 50  # Prevent infinite loops
    page_count = 0
    
    while len(all_transactions) < limit and page_count < max_pages:
        page_count += 1
        attempt = 0
        success = False
        
        while attempt < retries and not success:
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                transactions = response.json()
                success = True
                
                if not transactions:
                    return all_transactions[:limit]
                
                # Filter transactions that involve mSOL and check timestamp
                for txn in transactions:
                    txn_timestamp = txn.get('timestamp', 0)
                    
                    # If we have a since_timestamp, stop if we've gone past it
                    if since_timestamp and txn_timestamp < since_timestamp:
                        # We've reached transactions older than our start date
                        return all_transactions[:limit]
                    
                    # Check if transaction involves mSOL
                    if 'tokenTransfers' in txn:
                        for transfer in txn['tokenTransfers']:
                            if transfer.get('mint') == MSOL_MINT:
                                all_transactions.append(txn)
                                break
                
                # Check if we need to paginate
                if len(transactions) < batch_size:
                    return all_transactions[:limit]
                
                # Use the oldest transaction's signature for pagination (Helius uses 'before' with signature)
                if transactions:
                    last_signature = transactions[-1].get('signature')
                    if last_signature:
                        params["before"] = last_signature
                        # Keep batch_size consistent for subsequent requests
                        params["limit"] = batch_size
                    else:
                        return all_transactions[:limit]
                else:
                    return all_transactions[:limit]
                    
            except requests.exceptions.Timeout:
                attempt += 1
                logger.warning(f"Timeout fetching mSOL transactions page {page_count} (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Timeout fetching mSOL transactions after {retries} attempts")
                    return all_transactions[:limit]
            except requests.exceptions.ConnectionError as e:
                attempt += 1
                logger.warning(f"Connection error fetching mSOL transactions page {page_count} (attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Connection failed fetching mSOL transactions after {retries} attempts")
                    return all_transactions[:limit]
            except requests.exceptions.HTTPError as e:
                # Log the full error response for 400 errors
                if e.response.status_code == 400:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"400 Bad Request from Helius API: {error_detail}")
                        logger.error(f"Request URL: {url}")
                        logger.error(f"Request params: {params}")
                    except:
                        logger.error(f"400 Bad Request from Helius API: {e.response.text}")
                else:
                    logger.error(f"HTTP error fetching mSOL transactions: {e}")
                return all_transactions[:limit]
            except Exception as e:
                logger.error(f"Error fetching mSOL transactions: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return all_transactions[:limit]
    
    return all_transactions[:limit]

def parse_msol_conversion(transaction):
    """
    Parse a transaction to extract mSOL transfer details.
    Returns dict with transfer info or None if no significant mSOL movement.
    Only tracks transfers > 0.01 mSOL to filter out staking reward dust.
    """
    if 'tokenTransfers' not in transaction:
        return None

    # Minimum mSOL amount to consider a "real" transfer (filters out staking rewards dust)
    # Lower threshold to capture more transfers while still filtering out micro-dust
    MIN_MSOL_AMOUNT = 0.0001  # 0.0001 mSOL minimum (captures down to ~$0.02 transfers)
    
    msol_transfers = []
    sol_transfers = []

    for transfer in transaction['tokenTransfers']:
        mint = transfer.get('mint')
        # Helius returns tokenAmount already in decimal form (e.g., 0.004 mSOL, not 4000000)
        # So we don't need to divide by decimals
        actual_amount = transfer.get('tokenAmount', 0)

        from_account = transfer.get('fromUserAccount')
        to_account = transfer.get('toUserAccount')
        wallet = WALLET

        if mint == MSOL_MINT:
            # Only track significant mSOL transfers (> MIN_MSOL_AMOUNT)
            if actual_amount < MIN_MSOL_AMOUNT:
                continue  # Skip dust amounts
                
            # Check if mSOL is coming TO our wallet
            if to_account == wallet:
                msol_transfers.append({
                    'amount': actual_amount,
                    'direction': 'in',
                    'from': from_account,
                    'to': to_account
                })
            # Check if mSOL is going FROM our wallet
            elif from_account == wallet:
                msol_transfers.append({
                    'amount': actual_amount,
                    'direction': 'out',
                    'from': from_account,
                    'to': to_account
                })
        elif mint == SOL_MINT:
            # Check if SOL is going FROM our wallet (conversion to mSOL)
            if from_account == wallet:
                sol_transfers.append({
                    'amount': actual_amount,
                    'from': from_account,
                    'to': to_account
                })

    # Only track if there are significant mSOL transfers
    if msol_transfers:
        # Calculate net mSOL change
        msol_in = sum(t['amount'] for t in msol_transfers if t['direction'] == 'in')
        msol_out = sum(t['amount'] for t in msol_transfers if t['direction'] == 'out')
        net_msol = msol_in - msol_out
        
        # Determine transaction type
        sol_amount = sum(t['amount'] for t in sol_transfers) if sol_transfers else 0
        
        if sol_amount > 0 and msol_in > 0:
            tx_type = 'swap'  # SOL -> mSOL conversion
        elif msol_in > 0:
            tx_type = 'transfer_in'  # Received mSOL
        elif msol_out > 0:
            tx_type = 'transfer_out'  # Sent mSOL
        else:
            tx_type = 'unknown'
        
        logger.info(f"mSOL transfer: {tx_type} {net_msol:.4f} mSOL (in: {msol_in:.4f}, out: {msol_out:.4f})")
        
        return {
            'signature': transaction.get('signature'),
            'timestamp': transaction.get('timestamp'),
            'amount_sol': sol_amount,
            'amount_msol': net_msol,  # Net change (positive = gain, negative = loss)
            'msol_in': msol_in,
            'msol_out': msol_out,
            'conversion_rate': msol_in / sol_amount if sol_amount > 0 else 0,
            'transaction_type': tx_type
        }
    
    return None

def parse_collect_fees_event(event):
    """
    Print COLLECT_FEES event details.
    """
    logger.info(f"Transaction Signature: {event['signature']}")
    logger.info(f"Timestamp: {event['timestamp']}")
    logger.info(f"Fee Payer: {event['feePayer']}")
    logger.info(f"Type: {event['type']}")

    if 'tokenTransfers' in event:
        logger.info("Token Transfers (Rewards Claimed):")
        for t in event['tokenTransfers']:
            amount = t.get('tokenAmount')
            mint = t.get('mint')
            from_account = t.get('fromTokenAccount')
            to_account = t.get('toTokenAccount')
            logger.info(f"  - {amount} tokens of mint {mint}")
            logger.info(f"    From: {from_account} → To: {to_account}")
    else:
        logger.info("No token transfers found.")
    logger.info("-" * 40)


def insert_collect_fee(event):
    """
    Insert collect_fee event with USD value tracking.
    Now captures USD value at time of redemption for accurate portfolio tracking.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if 'tokenTransfers' not in event:
        conn.close()
        return
    
    # Get current SOL price for USD conversion
    try:
        sol_price_data = get_sol_price_data()
        current_sol_price = sol_price_data.get('rate', 0) if sol_price_data else 0
    except Exception as e:
        logger.warning(f"Could not fetch SOL price for redemption value: {e}")
        current_sol_price = 0
    
    for t in event['tokenTransfers']:
        amount = t.get('tokenAmount')
        mint = t.get('mint')
        from_token_account = t.get('fromTokenAccount')
        to_token_account = t.get('toTokenAccount')
        from_user_account = t.get('fromUserAccount')
        to_user_account = t.get('toUserAccount')
        
        # Calculate USD value at redemption
        usd_value = 0.0
        sol_price_at_redemption = 0.0
        usdc_price_at_redemption = 1.0
        
        SOL_MINT = 'So11111111111111111111111111111111111111112'
        USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        
        if mint == SOL_MINT:
            # SOL redemption
            sol_price_at_redemption = current_sol_price
            usd_value = amount * current_sol_price if current_sol_price > 0 else 0
        elif mint == USDC_MINT:
            # USDC redemption (1:1 with USD)
            usd_value = amount
            sol_price_at_redemption = current_sol_price  # Store for reference
            usdc_price_at_redemption = 1.0
        
        redemption_date = datetime.utcfromtimestamp(event['timestamp']).isoformat() if event.get('timestamp') else None
        
        try:
            # Check if columns exist (for backward compatibility)
            cursor.execute("PRAGMA table_info(collect_fees)")
            columns = [col[1] for col in cursor.fetchall()]
            has_usd_columns = 'usd_value_at_redemption' in columns
            
            if has_usd_columns:
                cursor.execute('''
                    INSERT OR IGNORE INTO collect_fees (
                        signature, timestamp, fee_payer, token_mint,
                        token_amount, from_token_account, to_token_account,
                        from_user_account, to_user_account,
                        usd_value_at_redemption, sol_price_at_redemption,
                        usdc_price_at_redemption, redemption_date
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event['signature'],
                    event['timestamp'],
                    event['feePayer'],
                    mint,
                    amount,
                    from_token_account,
                    to_token_account,
                    from_user_account,
                    to_user_account,
                    usd_value,
                    sol_price_at_redemption,
                    usdc_price_at_redemption,
                    redemption_date
                ))
            else:
                # Fallback for old schema
                cursor.execute('''
                    INSERT OR IGNORE INTO collect_fees (
                        signature, timestamp, fee_payer, token_mint,
                        token_amount, from_token_account, to_token_account,
                        from_user_account, to_user_account
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event['signature'],
                    event['timestamp'],
                    event['feePayer'],
                    mint,
                    amount,
                    from_token_account,
                    to_token_account,
                    from_user_account,
                    to_user_account
                ))
        except Exception as e:
            logger.error(f"DB insert error: {e}")
    conn.commit()
    conn.close()

def get_sol_price_data():
    url = 'https://api.livecoinwatch.com/coins/single'
    headers = {
        'content-type': 'application/json',
        'x-api-key': LIVECOINWATCH_API
    }
    payload = {
        "currency": "USD",
        "code": "SOL",
        "meta": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        # Do NOT crash the request/page when upstream price API is slow/down.
        logger.warning(f"LiveCoinWatch SOL fetch failed: {e}. Falling back to last stored price.")
        try:
            conn = sqlite3.connect("sol_prices.db")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rate, timestamp, delta_hour, delta_day, delta_week FROM sol_prices "
                "ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            conn.close()
            if not row:
                return {"rate": 0, "delta": {"hour": 1.0, "day": 1.0, "week": 1.0}, "source": "fallback_none"}
            rate, ts, dh, dd, dw = row
            return {
                "rate": float(rate) if rate is not None else 0,
                "delta": {
                    "hour": float(dh) if dh is not None else 1.0,
                    "day": float(dd) if dd is not None else 1.0,
                    "week": float(dw) if dw is not None else 1.0,
                },
                "timestamp": ts,
                "source": "fallback_db",
            }
        except Exception as db_e:
            logger.error(f"Fallback SOL price lookup failed: {db_e}")
            return {"rate": 0, "delta": {"hour": 1.0, "day": 1.0, "week": 1.0}, "source": "fallback_error"}

def get_msol_price_data():
    """
    Fetch current mSOL price from LiveCoinWatch API.
    Returns price data similar to get_sol_price_data().
    """
    url = 'https://api.livecoinwatch.com/coins/single'
    headers = {
        'content-type': 'application/json',
        'x-api-key': LIVECOINWATCH_API
    }
    payload = {
        "currency": "USD",
        "code": "MSOL",  # mSOL ticker on LiveCoinWatch
        "meta": True
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch mSOL price from LiveCoinWatch: {e}")
        # Fallback: Use SOL price (mSOL typically trades very close to SOL)
        try:
            sol_data = get_sol_price_data()
            return sol_data  # Return SOL price as fallback
        except:
            return {'rate': 0}

def update_msol_balance_snapshot(wallet, snapshot_type='periodic', transaction_signature=None):
    """
    Fetch current mSOL balance and create a snapshot in the database.
    """
    try:
        balance = fetch_wallet_token_balance(wallet, MSOL_MINT)
        if balance is None:
            balance = 0.0
        
        # Get current mSOL price for USD conversion
        try:
            msol_price_data = get_msol_price_data()
            msol_price = msol_price_data.get('rate', 0) if msol_price_data else 0
        except Exception as e:
            logger.warning(f"Could not fetch mSOL price for snapshot: {e}")
            msol_price = 0
        
        balance_usd = balance * msol_price
        
        # Get current timestamp
        current_timestamp = int(datetime.now().timestamp())
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO msol_balance_snapshots (
                    timestamp, balance, balance_usd, transaction_signature, snapshot_type
                ) VALUES (?, ?, ?, ?, ?)
            ''', (current_timestamp, balance, balance_usd, transaction_signature, snapshot_type))
            conn.commit()
            logger.info(f"mSOL balance snapshot created: {balance} mSOL (${balance_usd:.2f})")
        except sqlite3.IntegrityError:
            # Handle duplicate timestamps gracefully - update instead
            cursor.execute('''
                UPDATE msol_balance_snapshots
                SET balance = ?, balance_usd = ?, transaction_signature = ?, snapshot_type = ?
                WHERE timestamp = ?
            ''', (balance, balance_usd, transaction_signature, snapshot_type, current_timestamp))
            conn.commit()
            logger.debug(f"mSOL balance snapshot updated: {balance} mSOL (${balance_usd:.2f})")
        except Exception as e:
            logger.error(f"Error inserting mSOL balance snapshot: {e}")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Error updating mSOL balance snapshot: {e}")

def process_msol_transaction(transaction):
    """
    Process a transaction to extract mSOL conversion and update balance.
    """
    conversion = parse_msol_conversion(transaction)
    
    if not conversion:
        return False
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Insert conversion record
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO msol_conversions (
                    signature, timestamp, amount_sol, amount_msol, conversion_rate, transaction_type
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conversion['signature'],
                conversion['timestamp'],
                conversion['amount_sol'],
                conversion['amount_msol'],
                conversion['conversion_rate'],
                conversion['transaction_type']
            ))
            conn.commit()
            logger.info(f"mSOL conversion recorded: {conversion['amount_msol']} mSOL from {conversion['amount_sol']} SOL")
        except sqlite3.IntegrityError:
            # Already exists, skip
            logger.debug(f"Conversion {conversion['signature']} already recorded")
        except Exception as e:
            logger.error(f"Error inserting mSOL conversion: {e}")
        
        # Update balance snapshot for this transaction
        # Calculate new balance by fetching current balance
        # (In catch-up, we'll reconstruct balances chronologically)
        if conversion['timestamp']:
            update_msol_balance_snapshot(
                WALLET,
                snapshot_type='conversion',
                transaction_signature=conversion['signature']
            )
        
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error processing mSOL transaction: {e}")
        return False

def catchup_msol_history(wallet, start_date=None):
    """
    Fetch and process historical mSOL transactions to reconstruct balance history.
    """
    try:
        # Get start date from environment or parameter
        if start_date is None:
            start_date_str = os.getenv("MSOL_TRACKING_START_DATE")
            if start_date_str:
                try:
                    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                except ValueError:
                    logger.error(f"Invalid MSOL_TRACKING_START_DATE format: {start_date_str}. Expected YYYY-MM-DD")
                    return {"status": "error", "message": "Invalid date format"}
            else:
                # Default to 90 days ago if not specified
                start_date = datetime.now() - timedelta(days=90)
                logger.info("No MSOL_TRACKING_START_DATE set, defaulting to 90 days ago")
        
        # Convert to Unix timestamp
        start_timestamp = int(start_date.timestamp())
        logger.info(f"Starting mSOL history catch-up from {start_date.strftime('%Y-%m-%d')} (timestamp: {start_timestamp})")
        
        # Fetch transactions in batches
        all_transactions = []
        batch_size = 1000
        max_transactions = 10000  # Limit to prevent excessive API calls
        
        try:
            transactions = fetch_msol_transactions(wallet, since_timestamp=start_timestamp, limit=max_transactions)
            all_transactions.extend(transactions)
            logger.info(f"Fetched {len(transactions)} mSOL-related transactions")
        except Exception as e:
            logger.error(f"Error fetching mSOL transactions: {e}")
            return {"status": "error", "message": str(e)}
        
        # Sort transactions by timestamp (oldest first) for chronological processing
        all_transactions.sort(key=lambda x: x.get('timestamp', 0))
        
        # Process transactions and reconstruct balance
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        current_balance = 0.0
        processed_count = 0
        conversion_count = 0
        
        # Get existing balance at start date (if any)
        cursor.execute('''
            SELECT balance FROM msol_balance_snapshots
            WHERE timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (start_timestamp,))
        existing = cursor.fetchone()
        if existing:
            current_balance = existing[0]
            logger.info(f"Found existing balance at start date: {current_balance} mSOL")
        
        # Get mSOL price ONCE at start (not per transaction!)
        try:
            msol_price_data = get_msol_price_data()
            msol_price = msol_price_data.get('rate', 0) if msol_price_data else 0
            logger.info(f"Using mSOL price: ${msol_price:.2f}")
        except:
            msol_price = 0
            logger.warning("Could not fetch mSOL price, USD values will be 0")
        
        significant_transfers = []
        
        for transaction in all_transactions:
            conversion = parse_msol_conversion(transaction)
            if conversion:
                significant_transfers.append(conversion)
                
                # Update balance based on net mSOL change (can be positive or negative)
                old_balance = current_balance
                current_balance += conversion['amount_msol']
                
                logger.info(f"Transfer: {conversion['transaction_type']} {conversion['amount_msol']:.4f} mSOL | Balance: {old_balance:.4f} -> {current_balance:.4f}")
                
                # Ensure balance doesn't go negative
                if current_balance < 0:
                    logger.warning(f"Balance went negative ({current_balance}), resetting to 0")
                    current_balance = 0
                
                # Insert conversion
                try:
                    cursor.execute('''
                        INSERT OR IGNORE INTO msol_conversions (
                            signature, timestamp, amount_sol, amount_msol, conversion_rate, transaction_type
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        conversion['signature'],
                        conversion['timestamp'],
                        conversion['amount_sol'],
                        conversion['amount_msol'],
                        conversion['conversion_rate'],
                        conversion['transaction_type']
                    ))
                    conversion_count += 1
                except Exception as e:
                    logger.debug(f"Conversion already exists or error: {e}")
                
                # Create snapshot (USD calculated with single price fetch)
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO msol_balance_snapshots (
                            timestamp, balance, balance_usd, transaction_signature, snapshot_type
                        ) VALUES (?, ?, ?, ?, ?)
                    ''', (
                        conversion['timestamp'],
                        current_balance,
                        current_balance * msol_price,
                        conversion['signature'],
                        'catchup'
                    ))
                except Exception as e:
                    logger.debug(f"Error creating snapshot: {e}")
                
                processed_count += 1
        
        conn.commit()
        conn.close()
        
        # Get actual current balance from Helius for comparison
        actual_balance = fetch_wallet_token_balance(wallet, MSOL_MINT)
        logger.info(f"=== mSOL Catchup Summary ===")
        logger.info(f"Calculated balance from transfers: {current_balance:.4f} mSOL")
        logger.info(f"Actual balance from Helius: {actual_balance:.4f} mSOL")
        logger.info(f"Significant transfers found: {len(significant_transfers)}")
        
        # If there's a big discrepancy, use the actual balance
        if abs(actual_balance - current_balance) > 0.1:
            logger.warning(f"Discrepancy detected! Using actual balance from Helius.")
            current_balance = actual_balance
        
        # Create final snapshot with current balance
        update_msol_balance_snapshot(wallet, snapshot_type='catchup')
        
        logger.info(f"mSOL catch-up complete: {processed_count} transactions processed, {conversion_count} conversions recorded")
        return {
            "status": "success",
            "transactions_processed": processed_count,
            "conversions_recorded": conversion_count,
            "calculated_balance": current_balance,
            "actual_balance": actual_balance,
            "significant_transfers": len(significant_transfers)
        }
    except Exception as e:
        logger.error(f"Error in mSOL catch-up: {e}")
        return {"status": "error", "message": str(e)}

def fetch_newer_than(wallet, since_signature, max_pages=10, batch_size=100):
    after = since_signature
    all_events = []

    for _ in range(max_pages):
        url = f"https://api.helius.xyz/v0/addresses/{wallet}/transactions"
        params = {
            "api-key": API_KEY,
            "limit": batch_size,
            "until": after
        }

        logger.info(f"Fetching transactions newer than {after}...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        txns = response.json()

        if not txns:
            break

        txns.reverse()
        all_events.extend(txns)
        after = txns[0]["signature"]

    return all_events

@app.route('/backfill_newer')
def backfill_newer():
    # This signature should also be configurable if needed
    last_known_sig = os.getenv("LAST_KNOWN_SIGNATURE")
    events = fetch_newer_than(WALLET, last_known_sig, max_pages=50)

    count = 0
    for event in events:
        if event.get("type") == "COLLECT_FEES":
            insert_collect_fee(event)
            count += 1

    return jsonify({"status": "Backfill complete", "new_events": count})

def get_redemption_frequency():
    """Get days between each collection with redemption frequency data"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    SELECT
        signature,
        timestamp,
        LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
        (timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) / 86400.0 as days_since_last_collection
    FROM collect_fees
    ORDER BY timestamp
    ''')
    results = c.fetchall()
    conn.close()
    return results

def get_average_redemption_frequency():
    """Get average days between collections"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    SELECT 
        signature,
        timestamp
    FROM collect_fees
    ORDER BY signature, timestamp
    ''')
    results = c.fetchall()
    conn.close()

    if len(results) < 2:
        return 0

    total_days = 0
    count = 0

    for i in range(1, len(results)):
        if results[i][0] == results[i-1][0]:  # Same signature
            days_diff = (results[i][1] - results[i-1][1]) / 86400.0
            total_days += days_diff
            count += 1

    return total_days / count if count > 0 else 0

def get_daily_earning_rates():
    """Get daily earning breakdown by date"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT 
            DATE(datetime(timestamp, 'unixepoch')) as collection_date,
            COUNT(*) as collections_per_day,
            SUM(CASE WHEN token_mint = 'So11111111111111111111111111111111111111112' THEN token_amount ELSE 0 END) as sol_per_day,
            SUM(CASE WHEN token_mint = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' THEN token_amount ELSE 0 END) as usdc_per_day
        FROM collect_fees 
        GROUP BY DATE(datetime(timestamp, 'unixepoch'))
        ORDER BY collection_date
    ''')
    results = c.fetchall()
    conn.close()
    return results

def get_collection_patterns():
    """Get collection patterns by hour of day (in local timezone)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Convert UTC timestamps to local timezone (Eastern)
    # SQLite doesn't have great timezone support, so we'll do it in Python
    c.execute('''
        SELECT timestamp
        FROM collect_fees
    ''')
    all_timestamps = [row[0] for row in c.fetchall()]
    
    # Count by hour in local timezone
    hour_counts = {}
    for ts in all_timestamps:
        # Convert UTC unix timestamp to local time
        utc_dt = datetime.utcfromtimestamp(ts)
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(eastern)
        hour = local_dt.hour
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    # Convert to list of tuples matching original format
    results = [(hour, count) for hour, count in sorted(hour_counts.items())]
    conn.close()
    return results

def get_collection_statistics():
    """Get comprehensive collection statistics"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get current SOL price for USD conversions
    sol_data = get_sol_price_data()
    current_sol_price = sol_data.get('rate', 0) if sol_data else 0
    
    stats = {}
    
    # Best and worst days
    c.execute('''
        SELECT 
            DATE(datetime(timestamp, 'unixepoch')) as collection_date,
            COUNT(*) as collections,
            SUM(CASE WHEN token_mint = 'So11111111111111111111111111111111111111112' THEN token_amount ELSE 0 END) as sol_amount,
            SUM(CASE WHEN token_mint = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' THEN token_amount ELSE 0 END) as usdc_amount
        FROM collect_fees
        GROUP BY collection_date
        ORDER BY collections DESC
        LIMIT 1
    ''')
    best_day = c.fetchone()
    if best_day:
        stats['best_day'] = {
            'date': best_day[0],
            'collections': best_day[1],
            'sol': best_day[2],
            'usdc': best_day[3]
        }
    
    # Largest single collection
    c.execute('''
        SELECT 
            datetime(timestamp, 'unixepoch') as collection_time,
            token_mint,
            token_amount
        FROM collect_fees
        ORDER BY token_amount DESC
        LIMIT 1
    ''')
    largest = c.fetchone()
    if largest:
        stats['largest_collection'] = {
            'time': largest[0],
            'token': 'SOL' if largest[1].startswith('So1') else 'USDC',
            'amount': largest[2]
        }
    
    # Average collection size
    c.execute('''
        SELECT 
            AVG(CASE WHEN token_mint = 'So11111111111111111111111111111111111111112' THEN token_amount ELSE 0 END) as avg_sol,
            AVG(CASE WHEN token_mint = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v' THEN token_amount ELSE 0 END) as avg_usdc,
            COUNT(*) as total_collections
        FROM collect_fees
    ''')
    avg_row = c.fetchone()
    if avg_row:
        stats['avg_collection'] = {
            'sol': avg_row[0] or 0,
            'usdc': avg_row[1] or 0,
            'total': avg_row[2] or 0
        }
    
    # Current streak (consecutive days with collections)
    c.execute('''
        SELECT DISTINCT DATE(datetime(timestamp, 'unixepoch')) as collection_date
        FROM collect_fees
        ORDER BY collection_date DESC
    ''')
    dates = [row[0] for row in c.fetchall()]
    
    if dates:
        streak = 0
        current_date = datetime.now().date()
        for i, date_str in enumerate(dates):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            expected_date = current_date - timedelta(days=i)
            if date_obj == expected_date:
                streak += 1
            else:
                break
        stats['current_streak'] = streak
    
    # Token ratio - check if usd_value_at_redemption column exists
    c.execute("PRAGMA table_info(collect_fees)")
    columns = [col[1] for col in c.fetchall()]
    has_usd_value = 'usd_value_at_redemption' in columns
    
    if has_usd_value:
        # Use stored USD values (most accurate - uses price at time of redemption)
        c.execute('''
            SELECT 
                token_mint,
                SUM(usd_value_at_redemption) as total_usd_value
            FROM collect_fees
            WHERE token_mint IN ('So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
            AND usd_value_at_redemption IS NOT NULL
            GROUP BY token_mint
        ''')
        token_usd_totals = {}
        for row in c.fetchall():
            if row[0].startswith('So1'):
                token_usd_totals['SOL'] = row[1] or 0
            else:
                token_usd_totals['USDC'] = row[1] or 0
    else:
        # Fallback: calculate USD values using current SOL price
        c.execute('''
            SELECT 
                token_mint,
                SUM(token_amount) as total_amount
            FROM collect_fees
            WHERE token_mint IN ('So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
            GROUP BY token_mint
        ''')
        token_usd_totals = {}
        for row in c.fetchall():
            if row[0].startswith('So1'):
                # Convert SOL to USD using current price
                token_usd_totals['SOL'] = (row[1] or 0) * current_sol_price
            else:
                # USDC is 1:1 with USD
                token_usd_totals['USDC'] = row[1] or 0
    
    total_usd_value = sum(token_usd_totals.values())
    if total_usd_value > 0:
        stats['token_ratio'] = {
            'sol_pct': (token_usd_totals.get('SOL', 0) / total_usd_value) * 100,
            'usdc_pct': (token_usd_totals.get('USDC', 0) / total_usd_value) * 100
        }
    
    # Weekly patterns (day of week) - in local timezone
    c.execute('''
        SELECT timestamp
        FROM collect_fees
    ''')
    all_timestamps = [row[0] for row in c.fetchall()]
    
    # Count by day of week in local timezone
    day_counts = {}  # {day_of_week: (collections_count, days_count)}
    day_dates = {}   # {day_of_week: set of dates}
    
    for ts in all_timestamps:
        # Convert UTC unix timestamp to local time
        utc_dt = datetime.utcfromtimestamp(ts)
        local_dt = utc_dt.replace(tzinfo=pytz.utc).astimezone(eastern)
        day_of_week = local_dt.weekday()  # 0=Monday, 6=Sunday
        date_str = local_dt.strftime('%Y-%m-%d')
        
        if day_of_week not in day_counts:
            day_counts[day_of_week] = 0
            day_dates[day_of_week] = set()
        
        day_counts[day_of_week] += 1
        day_dates[day_of_week].add(date_str)
    
    # Convert to list matching original format: (day_of_week, collections_count, days_count)
    weekly_patterns = [
        (day, day_counts[day], len(day_dates[day]))
        for day in sorted(day_counts.keys())
    ]
    stats['weekly_patterns'] = weekly_patterns
    
    # Find best day and hour for collections
    if weekly_patterns:
        best_day_data = max(weekly_patterns, key=lambda x: x[1])
        best_day = best_day_data[0]  # 0=Monday, 6=Sunday
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        stats['best_day'] = day_names[best_day]
    
    # Get collection patterns for best hour
    collection_patterns = get_collection_patterns()
    if collection_patterns:
        best_hour_data = max(collection_patterns, key=lambda x: x[1])
        best_hour = best_hour_data[0]
        stats['best_hour'] = best_hour
    
    # Last 7 days collection stats (USD value and days)
    seven_days_ago = datetime.now() - timedelta(days=7)
    seven_days_ago_ts = int(seven_days_ago.timestamp())
    
    # Check if usd_value_at_redemption column exists
    c.execute("PRAGMA table_info(collect_fees)")
    columns = [col[1] for col in c.fetchall()]
    has_usd_value = 'usd_value_at_redemption' in columns
    
    if has_usd_value:
        # Use stored USD values (most accurate)
        c.execute('''
            SELECT 
                COUNT(*) as collections,
                COUNT(DISTINCT DATE(datetime(timestamp, 'unixepoch'))) as days_active,
                SUM(usd_value_at_redemption) as total_usd
            FROM collect_fees
            WHERE timestamp >= ?
            AND usd_value_at_redemption IS NOT NULL
        ''', (seven_days_ago_ts,))
        recent = c.fetchone()
        total_usd = recent[2] if recent and recent[2] else 0
    else:
        # Fallback: calculate USD values
        c.execute('''
            SELECT 
                COUNT(*) as collections,
                COUNT(DISTINCT DATE(datetime(timestamp, 'unixepoch'))) as days_active
            FROM collect_fees
            WHERE timestamp >= ?
        ''', (seven_days_ago_ts,))
        recent = c.fetchone()
        
        # Calculate USD value
        c.execute('''
            SELECT 
                token_mint,
                SUM(token_amount) as total_amount
            FROM collect_fees
            WHERE timestamp >= ?
            GROUP BY token_mint
        ''', (seven_days_ago_ts,))
        token_totals = {}
        for row in c.fetchall():
            if row[0].startswith('So1'):
                token_totals['SOL'] = (row[1] or 0) * current_sol_price
            else:
                token_totals['USDC'] = row[1] or 0
        total_usd = sum(token_totals.values())
    
    if recent:
        stats['last_7_days'] = {
            'collections': recent[0],
            'days_active': recent[1],
            'total_usd': total_usd
        }
    
    conn.close()
    return stats

@app.route('/')
def home():
    """New modern home page with summary and quick links"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Get totals
    c.execute('''
        SELECT token_mint, SUM(token_amount)
        FROM collect_fees
        WHERE token_mint IN ('So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
        GROUP BY token_mint
    ''')
    results = c.fetchall()

    c.execute('SELECT MIN(timestamp) FROM collect_fees')
    since_timestamp = c.fetchone()[0]

    # Monthly summary
    c.execute('''
        SELECT strftime('%Y-%m', timestamp, 'unixepoch') AS month,
               token_mint,
               SUM(token_amount) AS total_amount
        FROM collect_fees
        WHERE token_mint IN (
            'So11111111111111111111111111111111111111112',
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        )
        GROUP BY month, token_mint
        ORDER BY month
    ''')
    monthly_rows = c.fetchall()

    conn.close()

    totals = {}
    for mint, amount in results:
        if mint == 'So11111111111111111111111111111111111111112':
            totals['SOL'] = amount
        elif mint == 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':
            totals['USDC'] = amount

    since_date = datetime.utcfromtimestamp(since_timestamp).strftime('%Y-%m-%d') if since_timestamp else 'N/A'
    sol_data = get_sol_price_data()
    sol_price = sol_data.get('rate', 0)
    
    # Calculate deltas
    sol_deltas = {
        'hour': sol_data.get('delta', {}).get('hour', 1.0),
        'day': sol_data.get('delta', {}).get('day', 1.0),
        'week': sol_data.get('delta', {}).get('week', 1.0)
    }

    # Analytics - reuse logic from index route
    analytics = {
        'avg_days_between_collections': get_average_redemption_frequency(),
        'daily_earnings': get_daily_earning_rates(),
        'collection_patterns': get_collection_patterns(),
        'redemption_frequency': get_redemption_frequency(),
        'collection_stats': get_collection_statistics()
    }

    # Monthly summary
    monthly_summary = []
    monthly_data = {}
    for month, mint, amount in monthly_rows:
        if month not in monthly_data:
            monthly_data[month] = {'SOL': 0, 'USDC': 0}
        if mint == 'So11111111111111111111111111111111111111112':
            monthly_data[month]['SOL'] = amount
        elif mint == 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':
            monthly_data[month]['USDC'] = amount

    for month, values in monthly_data.items():
        usd_value = (values['SOL'] * sol_price) + values['USDC']
        monthly_summary.append({
            'month': month,
            'usd_value': usd_value
        })

    return render_template(
        "home.html",
        sol=totals.get('SOL', 0),
        usdc=totals.get('USDC', 0),
        sol_price=sol_price,
        total_usd=(totals.get('SOL', 0) * sol_price) + totals.get('USDC', 0),
        since_date=since_date,
        sol_deltas=sol_deltas,
        analytics=analytics,
        monthly_summary=monthly_summary
    )

@app.route('/orca')
def index():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Existing code for totals...
    c.execute('''
        SELECT token_mint, SUM(token_amount)
        FROM collect_fees
        WHERE token_mint IN ('So11111111111111111111111111111111111111112', 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v')
        GROUP BY token_mint
    ''')
    results = c.fetchall()

    c.execute('SELECT MIN(timestamp) FROM collect_fees')
    since_timestamp = c.fetchone()[0]


    # Get SOL price data first (needed for calculations)
    sol_data = get_sol_price_data()
    sol_price = sol_data.get('rate', 0)
    
    # --- Summarize USD per day (combine SOL + USDC) ---
    # Use USD value if available, otherwise calculate using current SOL price
    # Note: Can't join sol_prices table (different database), so use current price for fallback
    c.execute('''
        SELECT date(timestamp, 'unixepoch') AS day,
               token_mint,
               SUM(token_amount) AS total_amount
        FROM collect_fees
        WHERE token_mint IN (
            'So11111111111111111111111111111111111111112',
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        )
        GROUP BY day, token_mint
        ORDER BY day
    ''')
    rows = c.fetchall()

    # --- Monthly breakdown (per token) ---
    c.execute('''
        SELECT strftime('%Y-%m', timestamp, 'unixepoch') AS month,
               token_mint,
               SUM(token_amount) AS total_amount
        FROM collect_fees
        WHERE token_mint IN (
            'So11111111111111111111111111111111111111112',
            'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        )
        GROUP BY month, token_mint
        ORDER BY month
    ''')
    monthly_rows = c.fetchall()

    conn.close()

    totals = {}
    for mint, amount in results:
        if mint == 'So11111111111111111111111111111111111111112':
            totals['SOL'] = amount
        elif mint == 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':
            totals['USDC'] = amount

    since_date = datetime.utcfromtimestamp(since_timestamp).strftime('%Y-%m-%d') if since_timestamp else 'N/A'
    # sol_data already fetched above for daily summary calculation

    # Add analytics data
    analytics = {
        'avg_days_between_collections': get_average_redemption_frequency(),
        'daily_earnings': get_daily_earning_rates(),
        'collection_patterns': get_collection_patterns(),
        'redemption_frequency': get_redemption_frequency(),
        'collection_stats': get_collection_statistics()
    }
    # # Debug prints
    # print("Analytics debug:")
    # for key, value in analytics.items():
    #     print(f"{key}: {type(value)} - {value}")

    # --- Combine USD values by day ---
    daily_summary = {}
    for day, mint, amount in rows:
        usd_value = amount * sol_data['rate'] if mint.startswith('So1') else amount
        if day not in daily_summary:
            daily_summary[day] = 0
        daily_summary[day] += usd_value

    # --- Combine monthly data into USD ---
    monthly_summary = {}
    for month, mint, amount in monthly_rows:
        usd_value = amount * sol_data['rate'] if mint.startswith('So1') else amount
        monthly_summary[month] = monthly_summary.get(month, 0) + usd_value

    monthly_summary_list = [
        {'month': month, 'usd_value': value}
        for month, value in sorted(monthly_summary.items())
    ]


    # Convert to sorted list for template
    daily_summary_list = [
        {'day': day, 'usd_value': value} 
        for day, value in sorted(daily_summary.items())
    ]

    # --- Query mSOL growth data for chart ---
    msol_conn = sqlite3.connect(DB_PATH)
    msol_cursor = msol_conn.cursor()
    
    # Get all mSOL balance snapshots, ordered by timestamp
    msol_cursor.execute('''
        SELECT timestamp, balance, balance_usd
        FROM msol_balance_snapshots
        ORDER BY timestamp ASC
    ''')
    msol_rows = msol_cursor.fetchall()
    logger.info(f"Found {len(msol_rows)} mSOL balance snapshots in database")
    msol_conn.close()
    
    # Format data for TradingView charts: { time: timestamp_in_seconds, value: balance }
    # Just pass the raw balance - no transformations
    msol_growth_data = []
    
    # Get current mSOL price for USD values
    try:
        msol_price_data = get_msol_price_data()
        current_msol_price = msol_price_data.get('rate', 0) if msol_price_data else 0
    except:
        current_msol_price = 0
    
    logger.info(f"Processing {len(msol_rows)} mSOL balance snapshots for chart")
    
    for timestamp, balance, balance_usd in msol_rows:
        # Just use the raw balance value - no transformations
        balance_float = float(balance) if balance is not None else 0.0
        balance_usd_float = float(balance_usd) if balance_usd is not None else (balance_float * current_msol_price if current_msol_price > 0 else 0.0)
        
        msol_growth_data.append({
            'time': int(timestamp),
            'value': balance_float,  # Raw balance value
            'balance': balance_float,
            'balance_usd': balance_usd_float
        })
    
    logger.info(f"Prepared {len(msol_growth_data)} data points for mSOL chart")
    
    # Sort by timestamp to ensure chronological order
    msol_growth_data.sort(key=lambda x: x['time'])
    
    # If we have data, log a sample for debugging
    if msol_growth_data:
        logger.info(f"Sample mSOL data point: time={msol_growth_data[0]['time']}, balance={msol_growth_data[0]['balance']}, value={msol_growth_data[0]['value']}")
        logger.info(f"Last mSOL data point: time={msol_growth_data[-1]['time']}, balance={msol_growth_data[-1]['balance']}, value={msol_growth_data[-1]['value']}")
    
    # Get current mSOL balance for display
    current_msol_balance = fetch_wallet_token_balance(WALLET, MSOL_MINT)
    # Get mSOL price (not SOL price) for accurate USD conversion
    try:
        msol_price_data = get_msol_price_data()
        current_msol_price = msol_price_data.get('rate', 0) if msol_price_data else 0
    except Exception as e:
        logger.warning(f"Could not fetch mSOL price for display: {e}")
        current_msol_price = sol_price  # Fallback to SOL price
    current_msol_usd = current_msol_balance * current_msol_price if current_msol_balance else 0
    
    # Calculate monthly growth rate (initialize to None)
    msol_monthly_growth_rate = None
    if msol_growth_data and len(msol_growth_data) > 0:
        # Find first meaningful balance (> 0.01 mSOL) to avoid dust amounts skewing the calculation
        first_balance = None
        first_timestamp = None
        MIN_MEANINGFUL_BALANCE = 0.01  # 0.01 mSOL minimum
        
        for data_point in msol_growth_data:
            if data_point['balance'] >= MIN_MEANINGFUL_BALANCE:
                first_balance = data_point['balance']
                first_timestamp = data_point['time']
                break
        
        # Fallback to first data point if no meaningful balance found
        if first_balance is None:
            first_balance = msol_growth_data[0]['balance']
            first_timestamp = msol_growth_data[0]['time']
            logger.warning(f"No meaningful starting balance found, using first data point: {first_balance:.6f}")
        
        # Use current balance (from API, more accurate than last data point)
        final_balance = current_msol_balance if current_msol_balance else msol_growth_data[-1]['balance']
        current_timestamp = int(datetime.now().timestamp())
        
        logger.info(f"Calculating growth rate: first_balance={first_balance:.4f}, final_balance={final_balance:.4f}, first_timestamp={first_timestamp}, current_timestamp={current_timestamp}")
        
        # Calculate time difference in months
        time_diff_seconds = current_timestamp - first_timestamp
        months_elapsed = time_diff_seconds / (30.44 * 24 * 3600)  # Average days per month
        
        logger.info(f"Time difference: {time_diff_seconds} seconds = {months_elapsed:.2f} months")
        
        if months_elapsed > 0:
            if first_balance > 0 and final_balance > 0:
                # For very short periods (< 0.1 months = ~3 days), use simple linear projection
                # For longer periods, use compound growth formula
                if months_elapsed < 0.1:
                    # Simple linear: (change / initial) / months * 100
                    change = final_balance - first_balance
                    monthly_growth_rate = (change / first_balance) / months_elapsed * 100
                    logger.info(f"Using linear growth rate (short period): {monthly_growth_rate:.2f}%")
                else:
                    # Compound growth rate: ((final/initial)^(1/months) - 1) * 100
                    growth_factor = final_balance / first_balance
                    # Cap growth factor to prevent astronomical rates from tiny starting balances
                    if growth_factor > 1000:
                        logger.warning(f"Very large growth factor ({growth_factor:.2f}), capping calculation")
                        # Use a more conservative calculation for extreme cases
                        monthly_growth_rate = min((growth_factor ** (1 / months_elapsed) - 1) * 100, 1000.0)
                    else:
                        monthly_growth_rate = (growth_factor ** (1 / months_elapsed) - 1) * 100
                    logger.info(f"Using compound growth rate: {monthly_growth_rate:.2f}%")
                
                # Sanity check: cap at 200% per month (anything higher is likely a calculation error)
                if monthly_growth_rate > 200:
                    logger.warning(f"Growth rate {monthly_growth_rate:.2f}% seems unrealistic, capping at 200%")
                    monthly_growth_rate = 200.0
                
                msol_monthly_growth_rate = monthly_growth_rate
                logger.info(f"mSOL monthly growth rate: {monthly_growth_rate:.2f}% (from {first_balance:.4f} to {final_balance:.4f} over {months_elapsed:.2f} months)")
            elif first_balance == 0 and final_balance > 0:
                # Started at zero, calculate as if starting from a tiny amount to avoid division issues
                # Use a very small starting balance (0.0001 mSOL) for calculation
                tiny_start = 0.0001
                growth_factor = final_balance / tiny_start
                monthly_growth_rate = (growth_factor ** (1 / months_elapsed) - 1) * 100
                msol_monthly_growth_rate = monthly_growth_rate
                logger.info(f"mSOL monthly growth rate (started from ~0): {monthly_growth_rate:.2f}%")
            elif final_balance <= 0:
                # Balance went to zero or negative
                msol_monthly_growth_rate = -100.0
                logger.warning(f"mSOL balance went to zero or negative: {final_balance}")
            else:
                logger.warning(f"Could not calculate growth rate: first_balance={first_balance}, final_balance={final_balance}")
        else:
            logger.warning(f"Invalid time difference for growth rate calculation: {months_elapsed} months")
    else:
        logger.warning("No mSOL growth data available for growth rate calculation")
    
    logger.info(f"Final msol_monthly_growth_rate value: {msol_monthly_growth_rate}")

    # # --- Prepare chart data for Chart.js ---
    # chart_data = {}
    # for token in ['SOL', 'USDC']:
    #     token_data = [row for row in daily_summary if row['token'] == token]
    #     chart_data[token] = {
    #         'labels': [row['day'] for row in token_data],
    #         'values': [round(row['usd_value'], 2) for row in token_data]
    #     }

    return render_template('index.html', 
        sol=totals.get('SOL', 0),
        usdc=totals.get('USDC', 0),
        sol_price=sol_data['rate'],
        sol_img=sol_data['png64'],
        sol_deltas=sol_data['delta'],
        total_usd=(totals.get('SOL', 0) * sol_data['rate']) + totals.get('USDC', 0),
        since_date=since_date,
        analytics=analytics,  # Pass analytics data to template
        daily_summary=daily_summary_list,  # Pass daily summary to template
        monthly_summary=monthly_summary_list,
        msol_growth_data=msol_growth_data,  # Pass mSOL growth data for chart (will be converted to JSON in template)
        current_msol_balance=current_msol_balance or 0,
        current_msol_usd=current_msol_usd,
        msol_price=current_msol_price,  # mSOL price for crosshair USD calculation
        msol_monthly_growth_rate=msol_monthly_growth_rate,  # Monthly growth rate percentage
        msol_data_count=len(msol_growth_data)  # Debug: pass count to template
        # chart_data=json.dumps(chart_data)  # Pass chart data to template
    )

def get_predictions(cursor, time_threshold=None): # Renamed limit to time_threshold
    if time_threshold is None: # Default behavior if no time_threshold is passed
        # Calculate time_threshold for the last 24 hours if not provided
        time_threshold = (datetime.utcnow() - timedelta(days=1)).isoformat()

    cursor.execute(f"""
        SELECT timestamp, predicted_rate, actual_rate, error, mae
        FROM sol_predictions
        WHERE created_at >= ?
        ORDER BY created_at DESC
    """, (time_threshold,))
    rows = cursor.fetchall()
    predictions = []
    for row in rows:
        pred_ts = datetime.fromisoformat(row[0]).strftime("%b %d, %I:%M %p")
        predictions.append({
            "timestamp": pred_ts,
            "predicted": round(row[1], 4),
            "actual": round(row[2], 4),
            "error": round(row[3], 4),
            "mae": round(row[4], 4)
        })
    return predictions


def get_bandits(cursor, limit="24h"): # limit is now a string like "1h", "24h", "1w", etc.
    window=limit
    now = datetime.utcnow()
    # Updated delta_map to include "1h"
    delta_map = {
        "1h": timedelta(hours=1),
        "24h": timedelta(hours=24),
        "1w": timedelta(weeks=1),
        "1m": timedelta(days=30), # Approx 1 month
        "1y": timedelta(days=365) # Approx 1 year
    }
    # Default to 24 hours if the limit string is not recognized
    delta = delta_map.get(window, timedelta(hours=24))
    time_threshold_dt = now - delta
    time_threshold_iso = time_threshold_dt.isoformat()

    # Now includes rate pulled from JSON
    # The 'timestamp' in bandit_logs is expected to be ISO format string
    cursor.execute("""
        SELECT 
            timestamp, action, reward, 
            prediction_buy, prediction_sell, prediction_hold,
            json_extract(data_json, '$.rate') AS rate
        FROM bandit_logs
        WHERE timestamp >= ?
        ORDER BY created_at DESC
    """, (time_threshold_iso,)) # Use the ISO formatted time_threshold_iso

    rows = cursor.fetchall()
    bandit_logs = []

    for row in rows:
        log_ts = datetime.fromisoformat(row[0]).strftime("%b %d, %I:%M %p")
        bandit_logs.append({
            "timestamp": log_ts,
            "action": row[1],
            "reward": round(row[2], 4),
            "prediction_buy": round(row[3], 4) if row[3] is not None else None,
            "prediction_sell": round(row[4], 4) if row[4] is not None else None,
            "prediction_hold": round(row[5], 4) if row[5] is not None else None,
            "rate": round(row[6], 4) if row[6] is not None else None
        })



    return bandit_logs

def load_bandit_state():
    try:
        with open("bandit_state.json", "r") as f:
            state = json.load(f)
        return state
    except FileNotFoundError:
        print("bandit_state.json not found. Using default values.")
        return {
            "last_action": "unknown",
            "entry_price": 0.0,
            "position_open": False,
            "fee": 0.001,
            "portfolio": {
                "sol_balance": 0.0,
                "usd_balance": 0.0,
                "total_cost_basis": 0.0,
                "realized_pnl": 0.0
            }
        }
    except json.JSONDecodeError:
        print("bandit_state.json is malformed. Check the file.")
        return {
            "last_action": "error",
            "entry_price": 0.0,
            "position_open": False,
            "fee": 0.001,
            "portfolio": {
                "sol_balance": 0.0,
                "usd_balance": 0.0,
                "total_cost_basis": 0.0,
                "realized_pnl": 0.0
            }
        }

@app.route('/orca/msol/catchup')
def msol_catchup():
    """
    Manual trigger for mSOL history catch-up process.
    """
    try:
        result = catchup_msol_history(WALLET)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in mSOL catch-up endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/orca/msol/reset')
def msol_reset():
    """
    Clear mSOL tracking data and re-run catchup.
    Use this if the data seems incorrect.
    
    Query params:
        start_date: Optional start date in YYYY-MM-DD format (e.g., ?start_date=2025-06-27)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Clear existing mSOL data
        cursor.execute('DELETE FROM msol_balance_snapshots')
        cursor.execute('DELETE FROM msol_conversions')
        conn.commit()
        
        deleted_snapshots = cursor.rowcount
        logger.info(f"Cleared mSOL tracking data")
        conn.close()
        
        # Get start date from query param or env var
        start_date = None
        start_date_str = request.args.get('start_date')
        if start_date_str:
            try:
                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
                logger.info(f"Using start date from query param: {start_date_str}")
            except ValueError:
                return jsonify({"status": "error", "message": f"Invalid date format: {start_date_str}. Use YYYY-MM-DD"}), 400
        
        # Re-run catchup
        result = catchup_msol_history(WALLET, start_date=start_date)
        result['cleared_records'] = True
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in mSOL reset endpoint: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/orca/msol/debug')
def msol_debug():
    """
    Debug endpoint to see what's in the mSOL tables and raw transaction data.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Count records
        cursor.execute('SELECT COUNT(*) FROM msol_balance_snapshots')
        snapshot_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM msol_conversions')
        conversion_count = cursor.fetchone()[0]
        
        # Get sample snapshots
        cursor.execute('''
            SELECT timestamp, balance, balance_usd, snapshot_type 
            FROM msol_balance_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_snapshots = [{'timestamp': r[0], 'balance': r[1], 'balance_usd': r[2], 'type': r[3]} for r in cursor.fetchall()]
        
        # Get sample conversions
        cursor.execute('''
            SELECT timestamp, amount_msol, transaction_type 
            FROM msol_conversions 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_conversions = [{'timestamp': r[0], 'amount_msol': r[1], 'type': r[2]} for r in cursor.fetchall()]
        
        # Get balance range in snapshots
        cursor.execute('SELECT MIN(balance), MAX(balance) FROM msol_balance_snapshots')
        balance_range = cursor.fetchone()
        
        conn.close()
        
        # Also fetch a few raw transactions from Helius to see what we're getting
        raw_transactions = []
        try:
            transactions = fetch_msol_transactions(WALLET, limit=5)
            for txn in transactions[:5]:
                raw_transactions.append({
                    'signature': txn.get('signature', '')[:20] + '...',
                    'timestamp': txn.get('timestamp'),
                    'type': txn.get('type'),
                    'tokenTransfers': txn.get('tokenTransfers', []),
                    'nativeTransfers': txn.get('nativeTransfers', []),
                })
        except Exception as e:
            raw_transactions = [{'error': str(e)}]
        
        return jsonify({
            "snapshot_count": snapshot_count,
            "conversion_count": conversion_count,
            "balance_range": {"min": balance_range[0], "max": balance_range[1]},
            "recent_snapshots": recent_snapshots,
            "recent_conversions": recent_conversions,
            "raw_transactions_sample": raw_transactions,
            "wallet": WALLET,
            "msol_mint": MSOL_MINT
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/orca/msol/balance')
def msol_balance():
    """
    Get current mSOL balance and USD value.
    """
    try:
        balance = fetch_wallet_token_balance(WALLET, MSOL_MINT)
        
        # Get mSOL price for USD conversion (not SOL price)
        try:
            msol_price_data = get_msol_price_data()
            msol_price = msol_price_data.get('rate', 0) if msol_price_data else 0
        except Exception as e:
            logger.warning(f"Could not fetch mSOL price: {e}")
            msol_price = 0
        
        balance_usd = balance * msol_price
        
        return jsonify({
            "balance": balance,
            "balance_usd": balance_usd,
            "msol_price": msol_price,
            "timestamp": int(datetime.now().timestamp())
        })
    except Exception as e:
        logger.error(f"Error fetching mSOL balance: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/sol-tracker")
def sol_tracker():
    selected_range = request.args.get("range", "day")
    # Updated range_map to use timedelta for time-based filtering
    # Use timezone-aware datetime for consistency
    from datetime import timezone
    now = datetime.now(timezone.utc)
    range_map = {
        "hour": now - timedelta(hours=1),
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
        "month": now - timedelta(days=30),  # Approx month
        "year": now - timedelta(days=365)   # Approx year
    }
    time_threshold = range_map.get(selected_range, now - timedelta(days=1))

    fetcher = SOLPriceFetcher()
    # CRITICAL FIX: Don't use LIMIT for time-based queries - it cuts off recent data!
    # The LIMIT with ORDER BY ASC returns the OLDEST N records, excluding recent data.
    # For time-based ranges, we want ALL data in that time range, not a limited subset.
    # Only use limit for "year" range where we might need sampling for performance.
    
    # Convert threshold to string format matching DB storage (naive ISO, no timezone)
    # Timestamps in DB are stored as datetime.now().isoformat() which is naive (no timezone)
    time_threshold_str = time_threshold.isoformat()
    # Remove timezone info to match DB format
    if '+' in time_threshold_str:
        time_threshold_str = time_threshold_str.split('+')[0]
    elif time_threshold_str.endswith('Z'):
        time_threshold_str = time_threshold_str[:-1]
    
    logger.info(f"Fetching price history for range '{selected_range}' with threshold: {time_threshold_str}")
    
    # Only apply limit for year range (for performance with very large datasets)
    # For all other ranges, get ALL data in the time window
    use_limit = (selected_range == "year")
    max_data_points = 20000 if use_limit else None
    
    all_data = fetcher.get_price_history(
        time_threshold=time_threshold_str,
        limit=max_data_points
    )
    logger.info(f"get_price_history returned {len(all_data) if all_data else 0} records")
    
    # Log the actual time span of returned data for debugging
    if all_data and len(all_data) > 0:
        try:
            first_ts = all_data[0][0]
            last_ts = all_data[-1][0]
            if isinstance(first_ts, str):
                first_dt = datetime.fromisoformat(first_ts.replace('Z', '+00:00'))
            else:
                first_dt = first_ts
            if isinstance(last_ts, str):
                last_dt = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            else:
                last_dt = last_ts
            if first_dt.tzinfo is None:
                first_dt = first_dt.replace(tzinfo=timezone.utc)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            time_span = last_dt - first_dt
            logger.info(f"Data time span: {time_span} (from {first_dt.isoformat()} to {last_dt.isoformat()})")
        except Exception as e:
            logger.warning(f"Could not calculate data time span: {e}")
    fetcher.close()

    timestamps = []
    prices = []
    unix_timestamps = []  # Initialize Unix timestamps list
    
    if all_data and len(all_data) > 0:  # Ensure all_data is not None and not empty
        for row in all_data:  # oldest to newest
            # Optimized query now returns (timestamp, rate) tuple
            ts = row[0]  # timestamp column (first column)
            price = row[1]  # rate column (second column)
            # Ensure timestamp is valid before formatting
            if ts:
                try:
                    # Parse timestamp - handle both timezone-aware and naive timestamps
                    if isinstance(ts, str):
                        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    else:
                        dt = ts
                    # Ensure timezone-aware for consistent timestamp calculation
                    if dt.tzinfo is None:
                        from datetime import timezone
                        dt = dt.replace(tzinfo=timezone.utc)
                    short_ts = dt.strftime("%b %d, %I:%M %p")
                    timestamps.append(short_ts)
                    prices.append(float(price))  # Ensure price is float
                    # Store Unix timestamp for TradingView charts
                    unix_ts = int(dt.timestamp())
                    unix_timestamps.append(unix_ts)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping row with invalid data: {ts}, {price} - {e}")
            else:
                logger.warning("Skipping row with null timestamp.")
    
    # Debug logging
    logger.info(f"Loaded {len(timestamps)} timestamps, {len(prices)} prices, {len(unix_timestamps)} unix timestamps")
    if len(unix_timestamps) > 0:
        logger.info(f"Sample unix timestamp: {unix_timestamps[0]}")
        logger.info(f"First timestamp: {timestamps[0] if timestamps else 'N/A'}, First price: {prices[0] if prices else 'N/A'}")
        logger.info(f"Last timestamp: {timestamps[-1] if timestamps else 'N/A'}, Last price: {prices[-1] if prices else 'N/A'}")
        
        # Check if we got the expected amount of data for the selected range
        if unix_timestamps and len(unix_timestamps) > 1:
            try:
                first_unix = unix_timestamps[0]
                last_unix = unix_timestamps[-1]
                actual_span_seconds = last_unix - first_unix
                actual_span = timedelta(seconds=actual_span_seconds)
                expected_span = {
                    "hour": timedelta(hours=1),
                    "day": timedelta(days=1),
                    "week": timedelta(weeks=1),
                    "month": timedelta(days=30),
                    "year": timedelta(days=365)
                }.get(selected_range, timedelta(days=1))
                
                if actual_span < expected_span * 0.5:  # If we got less than 50% of expected
                    logger.warning(f"⚠️ Data span mismatch: Expected ~{expected_span}, got {actual_span} "
                                 f"({actual_span_seconds/3600:.1f} hours). "
                                 f"This might indicate insufficient data in database or limit being hit.")
            except Exception as e:
                logger.debug(f"Could not compare time spans: {e}")
    else:
        logger.warning(f"No data processed! all_data length: {len(all_data) if all_data else 'None'}")

    if not prices: # Handle case with no price data for the selected range
        logger.warning(f"No price data found for range: {selected_range} with threshold {time_threshold_iso}.")
        logger.warning(f"Attempting fallback: fetching last {max_data_points} records regardless of time threshold.")
        # Fallback: try fetching without time threshold
        try:
            fetcher2 = SOLPriceFetcher()
            all_data_fallback = fetcher2.get_price_history(time_threshold=None, limit=max_data_points)
            fetcher2.close()
            if all_data_fallback and len(all_data_fallback) > 0:
                logger.info(f"Fallback query returned {len(all_data_fallback)} records")
                for row in all_data_fallback[-max_data_points:]:  # Take most recent
                    ts = row[0]
                    price = row[1]
                    if ts:
                        try:
                            if isinstance(ts, str):
                                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                            else:
                                dt = ts
                            if dt.tzinfo is None:
                                from datetime import timezone
                                dt = dt.replace(tzinfo=timezone.utc)
                            short_ts = dt.strftime("%b %d, %I:%M %p")
                            timestamps.append(short_ts)
                            prices.append(float(price))
                            unix_ts = int(dt.timestamp())
                            unix_timestamps.append(unix_ts)
                        except Exception as e:
                            logger.warning(f"Error processing fallback row: {e}")
        except Exception as e:
            logger.error(f"Fallback query also failed: {e}")

    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()

    # Get predictions for backward compatibility (deprecated but still needed for template)
    predictions = get_predictions(cursor, time_threshold=time_threshold.isoformat())
    
    # Get latest bandit action (just the most recent one, not all logs)
    latest_bandit = None
    cursor.execute("""
        SELECT 
            timestamp, action, reward, 
            prediction_buy, prediction_sell, prediction_hold
        FROM bandit_logs
        ORDER BY created_at DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    if row:
        log_ts = datetime.fromisoformat(row[0]).strftime("%b %d, %I:%M %p")
        latest_bandit = {
            "timestamp": log_ts,
            "action": row[1],
            "reward": round(row[2], 4) if row[2] is not None else None,
            "prediction_buy": round(row[3], 4) if row[3] is not None else None,
            "prediction_sell": round(row[4], 4) if row[4] is not None else None,
            "prediction_hold": round(row[5], 4) if row[5] is not None else None
        }
    
    # Empty list for backward compatibility (we don't show the full log list anymore)
    bandit_logs = []

    conn.close()

    # Get the ACTUAL current/latest price (not from filtered time range)
    # This should always be the most recent price in the database, regardless of selected time range
    actual_current_price = None
    latest_timestamp_obj = None
    price_24h_ago = None  # Price from exactly 24 hours ago (for 24h change calculation)
    try:
        conn = sqlite3.connect("sol_prices.db")
        cursor = conn.cursor()
        
        # Get the most recent price
        cursor.execute("""
            SELECT rate, timestamp 
            FROM sol_prices 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest_row = cursor.fetchone()
        if latest_row:
            actual_current_price = round(latest_row[0], 4)
            latest_timestamp = latest_row[1]
            
            # Parse latest timestamp
            if isinstance(latest_timestamp, str):
                latest_timestamp_obj = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
            else:
                latest_timestamp_obj = latest_timestamp
            if latest_timestamp_obj.tzinfo is None:
                from datetime import timezone
                latest_timestamp_obj = latest_timestamp_obj.replace(tzinfo=timezone.utc)
            
            # Calculate 24 hours ago from the latest timestamp (use already parsed timestamp_obj)
            if latest_timestamp_obj is None:
                from datetime import timezone
                if isinstance(latest_timestamp, str):
                    latest_dt = datetime.fromisoformat(latest_timestamp.replace('Z', '+00:00'))
                else:
                    latest_dt = latest_timestamp
                if latest_dt.tzinfo is None:
                    latest_dt = latest_dt.replace(tzinfo=timezone.utc)
                latest_timestamp_obj = latest_dt
            
            # Get price from 24 hours ago
            price_24h_ago_time = (latest_timestamp_obj - timedelta(hours=24)).isoformat()
            cursor.execute("""
                SELECT rate 
                FROM sol_prices 
                WHERE timestamp <= ?
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (price_24h_ago_time,))
            price_24h_row = cursor.fetchone()
            if price_24h_row:
                price_24h_ago = round(price_24h_row[0], 4)
        
        conn.close()
    except Exception as e:
        logger.warning(f"Could not fetch latest price from database: {e}")
        # Fallback: use get_sol_price_data() for live price
        try:
            sol_data = get_sol_price_data()
            actual_current_price = round(sol_data.get('rate', 0), 4) if sol_data else None
            # Try to get 24h change from API data
            if sol_data and 'delta' in sol_data and 'day' in sol_data['delta']:
                # Calculate 24h ago price from delta
                delta_day = sol_data['delta'].get('day', 1.0)
                if actual_current_price and delta_day != 1.0:
                    price_24h_ago = round(actual_current_price / delta_day, 4)
        except Exception as e2:
            logger.error(f"Could not fetch live price from API: {e2}")
            # Last resort: use the last price from the filtered range if available
            actual_current_price = round(prices[-1], 4) if prices else None

    # Stats calculations
    def simple_moving_average(prices, window):
        if len(prices) < window:
            return None
        return round(mean(prices[-window:]), 4)
    
    def exponential_moving_average(prices, period):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        multiplier = 2.0 / (period + 1)
        ema = mean(prices[:period])
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return round(ema, 4)
    
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow + signal:
            return None, None, None
        ema_fast = exponential_moving_average(prices, fast)
        ema_slow = exponential_moving_average(prices, slow)
        if ema_fast is None or ema_slow is None:
            return None, None, None
        macd_line = ema_fast - ema_slow
        
        # For signal line, we need MACD history - simplified version
        # Use last 'signal' periods of prices to approximate signal line
        if len(prices) >= slow + signal:
            recent_prices = prices[-(slow + signal):]
            signal_ema = exponential_moving_average(recent_prices, signal)
            if signal_ema:
                # Approximate signal line using recent price momentum
                signal_line = signal_ema * 0.01  # Simplified approximation
            else:
                signal_line = None
        else:
            signal_line = None
        
        histogram = macd_line - signal_line if signal_line else None
        return round(macd_line, 4), round(signal_line, 4) if signal_line else None, round(histogram, 4) if histogram else None
    
    def calculate_bollinger_bands(prices, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        sma = mean(prices[-period:])
        std = stdev(prices[-period:])
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        return round(upper_band, 4), round(sma, 4), round(lower_band, 4)
    
    def calculate_momentum(prices, period=10):
        """Calculate price momentum (rate of change)"""
        if len(prices) < period + 1:
            return None
        return round(((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100, 2)

    sma_1h = simple_moving_average(prices, 12) if len(prices) >= 12 else None
    sma_4h = simple_moving_average(prices, 48) if len(prices) >= 48 else None
    sma_24h = simple_moving_average(prices, 288) if len(prices) >= 288 else None
    ema_12 = exponential_moving_average(prices, 12) if len(prices) >= 12 else None
    ema_26 = exponential_moving_average(prices, 26) if len(prices) >= 26 else None
    
    # MACD
    macd_line, macd_signal, macd_histogram = calculate_macd(prices) if len(prices) >= 35 else (None, None, None)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices) if len(prices) >= 20 else (None, None, None)
    
    # Momentum
    momentum_10 = calculate_momentum(prices, 10) if len(prices) >= 11 else None

    # Calculate RSI using the function from sol_price_fetcher
    from sol_price_fetcher import calculate_rsi
    import numpy as np
    
    rsi_value = None
    rsi_values = []  # For chart display - RSI series
    # Note: unix_timestamps is already initialized and populated above, don't reinitialize it
    
    if len(prices) >= 15:  # Need at least 15 prices for 14-period RSI
        # Calculate current RSI (last value)
        rsi_value = calculate_rsi(prices, period=14)
        
        # Calculate RSI for the full series (for chart)
        # RSI can only be calculated starting from index 14 (period)
        rsi_period = 14
        rsi_values = [None] * rsi_period  # First 14 values are None
        
        # Calculate RSI for each subsequent point
        for i in range(rsi_period, len(prices)):
            # Get prices up to current point
            prices_slice = prices[:i+1]
            rsi_at_point = calculate_rsi(prices_slice, period=rsi_period)
            rsi_values.append(rsi_at_point)
    

    # Calculate price position relative to SMAs and Bollinger Bands
    # Use the actual current price for indicators (not filtered by time range)
    current_price_for_indicators = actual_current_price if actual_current_price is not None else (prices[-1] if prices else None)
    price_vs_sma_1h = (current_price_for_indicators / sma_1h - 1) * 100 if (current_price_for_indicators and sma_1h and sma_1h != 0) else None
    price_vs_sma_4h = (current_price_for_indicators / sma_4h - 1) * 100 if (current_price_for_indicators and sma_4h and sma_4h != 0) else None
    price_vs_sma_24h = (current_price_for_indicators / sma_24h - 1) * 100 if (current_price_for_indicators and sma_24h and sma_24h != 0) else None
    
    # Price position in Bollinger Bands (%)
    bb_position = None
    if current_price_for_indicators and bb_upper and bb_lower and bb_upper != bb_lower:
        bb_position = ((current_price_for_indicators - bb_lower) / (bb_upper - bb_lower)) * 100
    
    # Initialize stats with default/empty values
    stats = {
        "current_price": None,
        "price_start": None,
        "percent_change": 0,
        "percent_change_24h": None,
        "percent_change_display": None,
        "high": None,
        "low": None,
        "range": None,
        "sma_1h": sma_1h,
        "sma_4h": sma_4h,
        "sma_24h": sma_24h,
        "ema_12": ema_12,
        "ema_26": ema_26,
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_histogram,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "bb_position": bb_position,
        "momentum_10": momentum_10,
        "price_vs_sma_1h": price_vs_sma_1h,
        "price_vs_sma_4h": price_vs_sma_4h,
        "price_vs_sma_24h": price_vs_sma_24h,
        "std_dev": 0,
        "avg_delta": 0,
        "rsi": rsi_value
    }

    if prices: # Only calculate these if there is price data
        # Use actual_current_price for the header display (always latest)
        # Use prices[-1] for period-specific calculations (last price in selected range)
        period_end_price = prices[-1]  # Last price in the selected time range
        price_start = prices[0]  # First price in the selected time range
        
        # Set current_price to the actual latest price (not filtered by time range)
        stats["current_price"] = actual_current_price if actual_current_price is not None else round(period_end_price, 4)
        stats["price_start"] = round(price_start, 4)
        
        # Calculate 24-hour change (always based on current price vs 24 hours ago, regardless of selected range)
        if actual_current_price is not None and price_24h_ago is not None and price_24h_ago != 0:
            percent_change_24h = round(((actual_current_price - price_24h_ago) / price_24h_ago) * 100, 2)
            stats["percent_change_24h"] = percent_change_24h
        else:
            stats["percent_change_24h"] = None
        
        # For period-specific percent change, use the period's start and end prices
        # This shows the change within the selected time range
        if price_start != 0: # Avoid division by zero
            # Calculate percent change for the SELECTED TIME RANGE (period_end_price vs price_start)
            percent_change = round(((period_end_price - price_start) / price_start) * 100, 2)
            stats["percent_change"] = percent_change
        else:
            stats["percent_change"] = 0 # Or some other indicator like 'N/A' if preferred
        
        # Use 24h change for display if available, otherwise fall back to period change
        if stats.get("percent_change_24h") is not None:
            stats["percent_change_display"] = stats["percent_change_24h"]
        else:
            stats["percent_change_display"] = stats["percent_change"]

        high_price = round(max(prices), 4)
        low_price = round(min(prices), 4)
        stats["high"] = high_price
        stats["low"] = low_price
        stats["range"] = round(high_price - low_price, 4)

        if len(prices) > 1:
            stats["std_dev"] = round(stdev(prices), 4)
            stats["avg_delta"] = round(mean([abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]), 4)
        else: # Handle cases with single price point
            stats["std_dev"] = 0
            stats["avg_delta"] = 0 # Or None, depending on desired display

    # Update SMAs in stats dict again, ensuring they are correctly assigned
    stats["sma_1h"] = sma_1h # These are already calculated based on available `prices`
    stats["sma_4h"] = sma_4h
    stats["sma_24h"] = sma_24h

    # Initialize signal performance tracker and update metrics
    signal_tracker = SignalPerformanceTracker()
    
    # Log RSI signals if we have RSI data (only log if signal changed or is new)
    if rsi_value is not None and actual_current_price is not None:
        # Determine current RSI signal
        current_rsi_signal = None
        if rsi_value < 30:
            current_rsi_signal = 'rsi_buy'
        elif rsi_value > 70:
            current_rsi_signal = 'rsi_sell'
        else:
            current_rsi_signal = 'rsi_hold'
        
        # Check if this signal is different from the last logged signal
        # (to avoid logging duplicates on every page load)
        try:
            recent_signals = signal_tracker.get_recent_signals(limit=1, signal_type=current_rsi_signal)
            should_log = True
            if recent_signals:
                # Check if last signal was within last 5 minutes (likely same signal)
                last_signal_time = datetime.fromisoformat(recent_signals[0]['timestamp'])
                time_diff = (datetime.utcnow() - last_signal_time).total_seconds()
                if time_diff < 300:  # 5 minutes
                    should_log = False
            
            if should_log:
                signal_tracker.log_signal(
                    current_rsi_signal,
                    actual_current_price,
                    metadata={'rsi': rsi_value, 'signal_strength': 'strong' if (rsi_value < 30 or rsi_value > 70) else 'neutral'}
                )
        except Exception as e:
            logger.warning(f"Error checking recent RSI signals: {e}")
            # Log anyway if check fails
            signal_tracker.log_signal(
                current_rsi_signal,
                actual_current_price,
                metadata={'rsi': rsi_value, 'signal_strength': 'strong' if (rsi_value < 30 or rsi_value > 70) else 'neutral'}
            )
    
    # Log bandit signal if available (only if different from last)
    if latest_bandit and latest_bandit.get('action') and actual_current_price is not None:
        bandit_signal_type = f"bandit_{latest_bandit['action']}"
        try:
            recent_bandit_signals = signal_tracker.get_recent_signals(limit=1, signal_type=bandit_signal_type)
            should_log_bandit = True
            if recent_bandit_signals:
                last_signal_time = datetime.fromisoformat(recent_bandit_signals[0]['timestamp'])
                time_diff = (datetime.utcnow() - last_signal_time).total_seconds()
                if time_diff < 300:  # 5 minutes
                    should_log_bandit = False
            
            if should_log_bandit:
                signal_tracker.log_signal(
                    bandit_signal_type,
                    actual_current_price,
                    metadata={
                        'reward': latest_bandit.get('reward'),
                        'prediction_buy': latest_bandit.get('prediction_buy'),
                        'prediction_sell': latest_bandit.get('prediction_sell'),
                        'prediction_hold': latest_bandit.get('prediction_hold')
                    }
                )
        except Exception as e:
            logger.warning(f"Error checking recent bandit signals: {e}")
            # Log anyway if check fails
            signal_tracker.log_signal(
                bandit_signal_type,
                actual_current_price,
                metadata={
                    'reward': latest_bandit.get('reward'),
                    'prediction_buy': latest_bandit.get('prediction_buy'),
                    'prediction_sell': latest_bandit.get('prediction_sell'),
                    'prediction_hold': latest_bandit.get('prediction_hold')
                }
            )
    
    # Update performance metrics (for signals that are old enough)
    signal_tracker.update_performance_metrics()
    
    # Check for consensus signals (all 7 indicators agree)
    consensus_tracker = ConsensusTracker()
    consensus_signal, indicator_signals = consensus_tracker.check_consensus(
        rsi=rsi_value,
        price_vs_sma_1h=price_vs_sma_1h,
        price_vs_sma_4h=price_vs_sma_4h,
        price_vs_sma_24h=price_vs_sma_24h,
        macd_line=macd_line,
        macd_signal=macd_signal,
        bb_position=bb_position,
        momentum_10=momentum_10,
        current_price=actual_current_price if actual_current_price else (prices[-1] if prices else 0),
        indicator_values={
            'rsi': rsi_value,
            'price_vs_sma_1h': price_vs_sma_1h,
            'price_vs_sma_4h': price_vs_sma_4h,
            'price_vs_sma_24h': price_vs_sma_24h,
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'bb_position': bb_position,
            'momentum_10': momentum_10,
        }
    )
    
    # Log consensus signal if detected (only if different from last one)
    if consensus_signal and actual_current_price is not None:
        try:
            recent_consensus = consensus_tracker.get_recent_consensus_signals(limit=1)
            should_log = True
            if recent_consensus:
                last_signal_time = datetime.fromisoformat(recent_consensus[0]['timestamp'])
                time_diff = (datetime.utcnow() - last_signal_time).total_seconds()
                if time_diff < 300:  # 5 minutes - avoid duplicate logs
                    should_log = False
            
            if should_log:
                consensus_tracker.log_consensus_signal(
                    signal_type=consensus_signal,
                    price=actual_current_price,
                    indicator_values={
                        'rsi': rsi_value,
                        'price_vs_sma_1h': price_vs_sma_1h,
                        'price_vs_sma_4h': price_vs_sma_4h,
                        'price_vs_sma_24h': price_vs_sma_24h,
                        'macd_line': macd_line,
                        'macd_signal': macd_signal,
                        'bb_position': bb_position,
                        'momentum_10': momentum_10,
                    },
                    indicator_signals=indicator_signals
                )
                logger.info(f"🎯 CONSENSUS {consensus_signal} SIGNAL DETECTED! All 7 indicators agree.")
        except Exception as e:
            logger.warning(f"Error logging consensus signal: {e}")
    
    # Update consensus returns (for signals old enough)
    try:
        consensus_tracker.update_returns()
    except Exception as e:
        logger.warning(f"Error updating consensus returns: {e}")
    
    # Get consensus statistics
    consensus_stats = consensus_tracker.get_consensus_stats()
    consensus_buy_stats = consensus_tracker.get_consensus_stats('BUY')
    consensus_sell_stats = consensus_tracker.get_consensus_stats('SELL')
    
    # Get performance statistics for display
    performance_stats = signal_tracker.get_performance_stats()
    
    # Get stats for each signal type we care about
    rsi_buy_stats = signal_tracker.get_performance_stats('rsi_buy', hours_later=24)
    rsi_sell_stats = signal_tracker.get_performance_stats('rsi_sell', hours_later=24)
    rsi_hold_stats = signal_tracker.get_performance_stats('rsi_hold', hours_later=24)
    bandit_buy_stats = signal_tracker.get_performance_stats('bandit_buy', hours_later=24)
    bandit_sell_stats = signal_tracker.get_performance_stats('bandit_sell', hours_later=24)
    bandit_hold_stats = signal_tracker.get_performance_stats('bandit_hold', hours_later=24)
    
    # Get news sentiment features
    news_features = None
    if NEWS_ANALYZER_AVAILABLE:
        try:
            global news_analyzer
            if news_analyzer is None:
                news_analyzer = NewsSentimentAnalyzer()
            
            # Force refresh if news is stale (older than 1 hour)
            # This ensures RL agent has fresh data for training
            try:
                # Check if news is stale (older than 1 hour)
                is_stale = news_analyzer.is_news_stale(stale_hours=1)
                force_fetch = is_stale
                
                if force_fetch:
                    logger.info("News is stale (older than 1 hour), forcing fetch...")
                
                # Try to fetch fresh news (force if stale, otherwise respect cooldown)
                articles = news_analyzer.fetch_news(force=force_fetch)
                if articles:
                    # Process and store new articles
                    current_price = actual_current_price if actual_current_price else (prices[-1] if prices else 100.0)
                    processed = news_analyzer.process_and_store_news(articles, current_price)
                    if processed > 0:
                        logger.info(f"Processed {processed} new news articles on sol-tracker page")
                    else:
                        logger.warning(f"Fetched {len(articles)} articles but processed 0 - may indicate duplicate filtering")
                elif force_fetch:
                    logger.error("Forced news fetch returned no articles - this indicates a problem with RSS feeds or network")
                    logger.error("Check RSS feed URLs in news_feeds.json and network connectivity")
                else:
                    logger.debug("News fetch skipped (in cooldown period)")
            except Exception as e:
                logger.error(f"Error fetching news on sol-tracker page: {e}")
                logger.error(traceback.format_exc())
            
            # Get recent news features
            news_features = news_analyzer.get_recent_news_features(hours=24, crypto_only=True)
        except Exception as e:
            logger.warning(f"Failed to get news features: {e}")
            news_features = {
                "news_sentiment_score": 0.0,
                "news_sentiment_label": "neutral",
                "news_count": 0,
                "news_positive_count": 0,
                "news_negative_count": 0,
                "news_crypto_count": 0,
            }
    else:
        news_features = {
            "news_sentiment_score": 0.0,
            "news_sentiment_label": "neutral",
            "news_count": 0,
            "news_positive_count": 0,
            "news_negative_count": 0,
            "news_crypto_count": 0,
        }

    # Final validation before passing to template
    if len(unix_timestamps) != len(timestamps):
        logger.error(f"CRITICAL: unix_timestamps length ({len(unix_timestamps)}) doesn't match timestamps length ({len(timestamps)})")
        logger.error("This will cause the chart to fail. Check timestamp conversion logic.")
    else:
        logger.info(f"Data validated: {len(unix_timestamps)} timestamps ready for chart")

    return render_template(
        "sol_tracker.html",
        timestamps=timestamps,
        prices=prices,
        stats=stats,
        predictions=predictions,
        bandit_logs=bandit_logs,  # Empty list - we don't show full logs
        latest_bandit=latest_bandit,  # Latest bandit recommendation
        selected_range=selected_range,
        rsi_values=rsi_values,  # Add RSI values for chart
        unix_timestamps=unix_timestamps,  # Add Unix timestamps for TradingView
        performance_stats={
            'rsi_buy': rsi_buy_stats,
            'rsi_sell': rsi_sell_stats,
            'rsi_hold': rsi_hold_stats,
            'bandit_buy': bandit_buy_stats,
            'bandit_sell': bandit_sell_stats,
            'bandit_hold': bandit_hold_stats
        },
        consensus_stats={
            'all': consensus_stats,
            'buy': consensus_buy_stats,
            'sell': consensus_sell_stats
        },
        consensus_signal=consensus_signal,
        news_features=news_features
    )


@app.route('/api/latest-prediction', methods=['GET'])
def get_latest_prediction():
    """
    DEPRECATED: Price prediction has been disabled due to irrational value returns.
    This endpoint returns historical predictions if available, but no new predictions
    are being generated.
    """
    try:
        conn = sqlite3.connect("sol_prices.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, predicted_rate, actual_rate, error, mae, created_at
            FROM sol_predictions
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            prediction = {
                "timestamp": row[0],
                "predicted_rate": row[1],
                "actual_rate": row[2],
                "error": row[3],
                "mae": row[4],
                "created_at": row[5],
                "deprecated": True,
                "message": "Price prediction has been disabled. This is historical data only."
            }
            return jsonify(prediction)
        else:
            return jsonify({
                "error": "No predictions found",
                "deprecated": True,
                "message": "Price prediction has been disabled. No historical data available."
            }), 404
    except sqlite3.Error as e:
        logger.error(f"Database error fetching latest prediction: {e}")
        return jsonify({"error": "Database error"}), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching latest prediction: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/api/signal-performance', methods=['GET'])
def get_signal_performance():
    """
    Get performance statistics for trading signals
    
    Query parameters:
    - signal_type: Optional filter by signal type (e.g., 'rsi_buy', 'bandit_sell')
    - hours_later: Time horizon for performance (1, 4, 24, or 168 for 7d). Default: 24
    """
    try:
        signal_type = request.args.get('signal_type', None)
        hours_later = int(request.args.get('hours_later', 24))
        
        tracker = SignalPerformanceTracker()
        stats = tracker.get_performance_stats(signal_type, hours_later)
        
        return jsonify({
            'success': True,
            'signal_type': signal_type,
            'hours_later': hours_later,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error fetching signal performance: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/predictions', methods=['GET'])
def get_rl_agent_predictions():
    """
    Get RL agent multi-horizon return predictions.
    
    Query parameters:
    - limit: Number of recent predictions to return (default: 10)
    - hours: Number of hours to look back for accuracy stats (default: 24)
    - chart: If 'true', returns data formatted for chart display
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        limit = int(request.args.get('limit', 10))
        hours = int(request.args.get('hours', 24))
        chart_format = request.args.get('chart', 'false').lower() == 'true'
        
        # Use same database path as rest of app (rewards.db by default)
        db_path = os.getenv("DATABASE_PATH", "rewards.db")
        prediction_manager = PredictionManager(db_path=db_path)
        
        if chart_format:
            # Return data formatted for chart
            chart_data = prediction_manager.get_predictions_for_chart(hours=hours)
            return jsonify({
                'success': True,
                'chart_data': chart_data
            })
        else:
            # Return latest predictions and accuracy stats
            latest_predictions = prediction_manager.get_latest_predictions(limit=limit)
            current_prediction = prediction_manager.get_current_prediction()
            accuracy_stats = prediction_manager.get_prediction_accuracy_stats(hours=hours)
            
            return jsonify({
                'success': True,
                'current_prediction': current_prediction,
                'recent_predictions': latest_predictions,
                'accuracy_stats': accuracy_stats
            })
    except Exception as e:
        logger.error(f"Error fetching RL agent predictions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sol-price/predict-15m', methods=['GET'])
def predict_sol_price_15m():
    """
    Predict SOL price 15 minutes into the future.
    
    Returns:
        JSON with current price, predicted price, return percentage, confidence, and method used.
    """
    global rl_agent_integration
    
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        # Check if RL agent integration is initialized
        if rl_agent_integration is None:
            return jsonify({
                'success': False,
                'error': 'RL agent model not loaded. Train model first.',
                'note': 'Use /api/rl-agent/status to check model availability'
            }), 503
        
        # Get current market state
        prices, price_features = rl_agent_integration.get_price_data(hours=24)
        news_data = rl_agent_integration.get_news_data(hours=24, max_headlines=20)
        
        if not prices:
            return jsonify({
                'success': False,
                'error': 'No price data available'
            }), 503
        
        current_price = price_features["current_price"]
        
        if current_price <= 0:
            return jsonify({
                'success': False,
                'error': 'Invalid current price'
            }), 503
        
        # Calculate position state (default values for prediction)
        position_state = {
            "position_size": rl_agent_integration.current_position,
            "portfolio_value": rl_agent_integration.portfolio_value,
            "entry_price": rl_agent_integration.entry_price,
            "time_since_last_trade": 0.0,
            "unrealized_pnl": 0.0,
        }
        
        # Generate 15-minute prediction
        pred_15m, confidence_15m, predicted_price_15m, method, class_info = generate_15m_price_prediction(
            model=rl_agent_integration.model,
            state_encoder=rl_agent_integration.state_encoder,
            price_data=prices,
            price_features=price_features,
            news_data=news_data,
            position_state=position_state,
            current_price=current_price,
            timestamp=datetime.now(),
            device=rl_agent_integration.device,
        )
        
        response = {
            'success': True,
            'current_price': float(current_price),
            'predicted_price_15m': float(predicted_price_15m),
            'predicted_return_15m': float(pred_15m),
            'confidence_15m': float(confidence_15m),
            'timestamp': datetime.now().isoformat(),
            'prediction_horizon_minutes': 15,
            'method': method
        }
        
        # Include classification info if available
        if class_info:
            response['classification'] = class_info
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error generating 15-minute price prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/attention', methods=['GET'])
def get_rl_agent_attention():
    """
    Get attention weights and influential headlines for RL agent decisions.
    
    Query parameters:
    - decision_id: Specific decision ID to get attention for
    - top_k: Number of top headlines to return (default: 5)
    - limit: Number of recent decisions to return (default: 10)
    - cluster: If 'true', returns attention aggregated by cluster
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        decision_id = request.args.get('decision_id', type=int)
        top_k = int(request.args.get('top_k', 5))
        limit = int(request.args.get('limit', 10))
        cluster_view = request.args.get('cluster', 'false').lower() == 'true'
        
        attention_logger = AttentionLogger()
        
        if decision_id:
            # Get attention for specific decision
            headlines = attention_logger.get_top_headlines_for_decision(decision_id, top_k=top_k)
            return jsonify({
                'success': True,
                'decision_id': decision_id,
                'headlines': headlines
            })
        elif cluster_view:
            # Get attention aggregated by cluster
            hours = int(request.args.get('hours', 24))
            cluster_stats = attention_logger.get_attention_by_cluster(hours=hours)
            
            # If no cluster stats, try to run clustering on recent news
            if not cluster_stats or len(cluster_stats) == 0:
                try:
                    from news_sentiment import NewsSentimentAnalyzer
                    news_analyzer = NewsSentimentAnalyzer()
                    clusters = news_analyzer.cluster_news_topics(hours=168, n_clusters=10)  # Cluster last week
                    if clusters:
                        logger.info(f"Generated {len(clusters)} news clusters")
                except Exception as e:
                    logger.debug(f"Could not run clustering: {e}")
            
            return jsonify({
                'success': True,
                'cluster_stats': cluster_stats
            })
        else:
            # Get recent attention logs
            recent_attention = attention_logger.get_recent_attention(limit=limit)
            return jsonify({
                'success': True,
                'recent_decisions': recent_attention
            })
    except Exception as e:
        logger.error(f"Error fetching RL agent attention: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/risk', methods=['GET'])
def get_rl_agent_risk():
    """
    Get risk metrics for RL agent.
    
    Returns current risk status including:
    - Position size
    - Trade frequency
    - Daily P&L
    - Risk limits status
    - Uncertainty metrics
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        # Get the actual risk manager instance from RL agent integration if available
        global rl_agent_integration
        if rl_agent_integration and hasattr(rl_agent_integration, 'risk_manager'):
            risk_manager = rl_agent_integration.risk_manager
        else:
            # Fallback: create new instance and load state from database
            risk_manager = RiskManager()
            
            # Try to load risk state from database (rl_agent_decisions table)
            try:
                conn = sqlite3.connect("sol_prices.db")
                cursor = conn.cursor()
                
                # Get recent trades to populate trade history
                cursor.execute("""
                    SELECT timestamp 
                    FROM rl_agent_decisions 
                    WHERE action IN ('BUY', 'SELL')
                    ORDER BY timestamp DESC 
                    LIMIT 100
                """)
                trade_rows = cursor.fetchall()
                
                # Populate trade history
                for row in trade_rows:
                    try:
                        trade_time = datetime.fromisoformat(row[0])
                        risk_manager.trade_times.append(trade_time)
                    except:
                        pass
                
                # Get current price for portfolio value estimation
                cursor.execute("SELECT rate FROM sol_prices ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                current_price = result[0] if result else 100.0
                
                # Try to get actual portfolio value from recent decisions
                # Calculate portfolio value based on recent trades
                cursor.execute("""
                    SELECT action, current_price, timestamp
                    FROM rl_agent_decisions
                    WHERE action IN ('BUY', 'SELL')
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                last_trade = cursor.fetchone()
                
                conn.close()
                
                # Initialize with a default portfolio value if not set
                if risk_manager.daily_start_value is None:
                    # Use current price * assumed position size, or default
                    portfolio_value = current_price * 100  # Assume 100 SOL portfolio
                    risk_manager.reset_daily_tracking(portfolio_value)
                    risk_manager.current_portfolio_value = portfolio_value
                else:
                    # Update current portfolio value based on current price
                    # This makes the dashboard show changes
                    if last_trade and last_trade[1]:
                        # Estimate portfolio value from last trade price
                        base_value = last_trade[1] * 100  # Assume 100 SOL
                        # Adjust for price change since last trade
                        price_change = (current_price - last_trade[1]) / last_trade[1] if last_trade[1] > 0 else 0
                        risk_manager.current_portfolio_value = base_value * (1 + price_change)
                    else:
                        # Fallback: update based on current price
                        risk_manager.current_portfolio_value = current_price * 100
            except Exception as e:
                logger.warning(f"Could not load risk state from database: {e}")
                # Initialize with defaults
                risk_manager.reset_daily_tracking(10000.0)  # Default $10k portfolio
        
        metrics = risk_manager.get_risk_metrics()
        
        return jsonify({
            'success': True,
            'risk_metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error fetching RL agent risk metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/rules', methods=['GET'])
def get_rl_agent_rules():
    """
    Get discovered trading rules from RL agent.
    
    Query parameters:
    - action: Filter by action (BUY, SELL, HOLD)
    - min_win_rate: Minimum win rate (0-1)
    - limit: Number of rules to return (default: 20)
    - sort_by: Sort by 'win_rate' or 'sample_size' (default: 'win_rate')
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        action = request.args.get('action')
        min_win_rate = float(request.args.get('min_win_rate', 0.0))
        limit = int(request.args.get('limit', 20))
        
        # Use same database path as rest of app (rewards.db by default)
        db_path = os.getenv("DATABASE_PATH", "rewards.db")
        rule_extractor = RuleExtractor(db_path=db_path)
        rules = rule_extractor.get_discovered_rules(
            action=action,
            min_win_rate=min_win_rate,
            limit=limit,
        )
        
        # If no rules found, try to extract rules from recent decisions
        if len(rules) == 0:
            try:
                logger.info("No rules found, attempting to extract rules from recent decisions...")
                # Check how many decisions with actual returns we have
                # CRITICAL FIX: Allow rules with just 1h returns (24h takes 24 hours!)
                # Use same database path as rest of app
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Count decisions with at least 1h returns (for immediate rule extraction)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM rl_agent_decisions d
                    JOIN rl_prediction_accuracy pa ON d.id = pa.decision_id
                    WHERE pa.actual_return_1h IS NOT NULL
                """)
                count_1h = cursor.fetchone()[0]
                
                # Count decisions with both 1h and 24h returns (for complete rules)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM rl_agent_decisions d
                    JOIN rl_prediction_accuracy pa ON d.id = pa.decision_id
                    WHERE pa.actual_return_1h IS NOT NULL
                    AND pa.actual_return_24h IS NOT NULL
                """)
                count_both = cursor.fetchone()[0]
                
                # Count total decisions
                cursor.execute("SELECT COUNT(*) FROM rl_agent_decisions")
                total_decisions = cursor.fetchone()[0]
                
                # Count total predictions
                cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
                total_predictions = cursor.fetchone()[0]
                
                conn.close()
                
                logger.info(f"Rule extraction diagnostics: {total_decisions} total decisions, {total_predictions} total predictions, {count_1h} with 1h returns, {count_both} with both 1h and 24h returns")
                
                # Use 1h returns for rule extraction (much faster - only need 1 hour wait)
                # Lower threshold if we don't have enough data
                min_samples = 10 if count_1h < 30 else 20  # Lowered significantly
                
                if count_1h >= min_samples:
                    extracted_rules = rule_extractor.extract_rules_from_decisions(
                        min_samples=min_samples,
                        max_depth=5,
                        min_samples_split=5,  # Lowered from 10
                    )
                    if extracted_rules:
                        # Evaluate and store rules
                        rule_extractor.store_rules(extracted_rules)
                        # Get rules again after extraction
                        rules = rule_extractor.get_discovered_rules(
                            action=action,
                            min_win_rate=min_win_rate,
                            limit=limit,
                        )
                        logger.info(f"Extracted and stored {len(rules)} new rules")
                    else:
                        logger.warning(f"Rule extraction returned no rules despite {count_1h} samples with 1h returns")
                else:
                    logger.warning(f"Not enough decisions with 1h returns ({count_1h} < {min_samples}). Need at least {min_samples} decisions with 1h actual returns. Total: {total_decisions} decisions, {total_predictions} predictions.")
            except Exception as e:
                logger.error(f"Could not extract rules: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Add diagnostic info if no rules
        response_data = {
            'success': True,
            'rules': rules,
            'count': len(rules)
        }
        
        if len(rules) == 0:
            # Add diagnostic info
            try:
                # Use same database path as rest of app
                db_path = os.getenv("DATABASE_PATH", "rewards.db")
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Count total decisions
                cursor.execute("SELECT COUNT(*) FROM rl_agent_decisions")
                total_decisions = cursor.fetchone()[0]
                
                # Count decisions with 1h returns (for rule extraction)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM rl_agent_decisions d
                    JOIN rl_prediction_accuracy pa ON d.id = pa.decision_id
                    WHERE pa.actual_return_1h IS NOT NULL
                """)
                decisions_with_1h_returns = cursor.fetchone()[0]
                
                # Count decisions with both 1h and 24h returns
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM rl_agent_decisions d
                    JOIN rl_prediction_accuracy pa ON d.id = pa.decision_id
                    WHERE pa.actual_return_1h IS NOT NULL
                    AND pa.actual_return_24h IS NOT NULL
                """)
                decisions_with_both_returns = cursor.fetchone()[0]
                
                # Count total predictions
                cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
                total_predictions = cursor.fetchone()[0]
                
                # Count predictions needing updates
                now = datetime.now()
                one_hour_ago = (now - timedelta(hours=1)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM rl_prediction_accuracy
                    WHERE datetime(timestamp) <= datetime(?)
                    AND actual_return_1h IS NULL
                """, (one_hour_ago,))
                predictions_needing_1h_update = cursor.fetchone()[0]
                
                conn.close()
                
                response_data['diagnostics'] = {
                    'total_decisions': total_decisions,
                    'total_predictions': total_predictions,
                    'decisions_with_1h_returns': decisions_with_1h_returns,
                    'decisions_with_both_returns': decisions_with_both_returns,
                    'predictions_needing_1h_update': predictions_needing_1h_update,
                    'message': f'Found {total_decisions} decisions, {total_predictions} predictions. {decisions_with_1h_returns} have 1h returns (need {min_samples} for rules). {predictions_needing_1h_update} predictions need 1h updates.'
                }
            except Exception as e:
                logger.debug(f"Could not get rule diagnostics: {e}")
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error fetching RL agent rules: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/diagnostics', methods=['GET'])
def get_rl_agent_diagnostics():
    """
    Get detailed diagnostics about the RL agent model, including prediction health.
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        diagnostics = {
            'model_loaded': rl_agent_integration is not None,
            'predictions_available': False,
            'latest_prediction': None,
            'prediction_health': 'unknown',
            'recommendations': []
        }
        
        if rl_agent_integration is None:
            diagnostics['recommendations'].append('RL agent model not loaded. Check if model exists in models/rl_agent/')
            return jsonify({
                'success': True,
                'diagnostics': diagnostics
            })
        
        # Check for recent predictions
        # Use same database path as rest of app (rewards.db by default)
        db_path = os.getenv("DATABASE_PATH", "rewards.db")
        prediction_manager = PredictionManager(db_path=db_path)
        latest_pred = prediction_manager.get_current_prediction()
        
        if latest_pred:
            diagnostics['predictions_available'] = True
            diagnostics['latest_prediction'] = {
                'timestamp': latest_pred.get('timestamp'),
                'pred_1h': latest_pred.get('predicted_return_1h'),
                'pred_24h': latest_pred.get('predicted_return_24h'),
                'conf_1h': latest_pred.get('predicted_confidence_1h'),
                'conf_24h': latest_pred.get('predicted_confidence_24h'),
            }
            
            # Check prediction health
            pred_1h = latest_pred.get('predicted_return_1h', 0)
            pred_24h = latest_pred.get('predicted_return_24h', 0)
            
            if abs(pred_1h) < 1e-6 and abs(pred_24h) < 1e-6:
                diagnostics['prediction_health'] = 'unhealthy'
                diagnostics['recommendations'].extend([
                    'Predictions are zero - auxiliary heads may not be trained',
                    'Run: python scripts/fix_auxiliary_heads.py --checkpoint models/rl_agent/checkpoint_*.pt',
                    'Or retrain with: python scripts/train_rl_agent.py --enable-auxiliary --epochs 10'
                ])
            elif abs(pred_1h) < 0.001 or abs(pred_24h) < 0.001:
                diagnostics['prediction_health'] = 'weak'
                diagnostics['recommendations'].append('Predictions are very small - consider retraining with auxiliary losses')
            else:
                diagnostics['prediction_health'] = 'healthy'
        else:
            diagnostics['recommendations'].append('No predictions found. Make a decision first: POST /api/rl-agent/decision')
        
        # Check accuracy stats
        accuracy_stats = prediction_manager.get_prediction_accuracy_stats(hours=24)
        diagnostics['accuracy_stats'] = accuracy_stats
        
        if accuracy_stats['count'] == 0:
            diagnostics['recommendations'].append('No accuracy data available - predictions need time to mature (1h+ for 1h predictions, 24h+ for 24h predictions)')
        
        return jsonify({
            'success': True,
            'diagnostics': diagnostics
        })
    except Exception as e:
        logger.error(f"Error in RL agent diagnostics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/status', methods=['GET'])
def get_rl_agent_status():
    """Get RL agent status including model info and scheduler status."""
    global rl_model_manager, rl_retraining_scheduler
    
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        status = {
            'model_loaded': rl_agent_integration is not None,
        }
        
        if rl_model_manager:
            status['model_info'] = rl_model_manager.get_model_info()
        
        if rl_retraining_scheduler:
            status['scheduler'] = rl_retraining_scheduler.get_status()
        
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        logger.error(f"Error getting RL agent status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predictions/diagnostics', methods=['GET'])
def get_predictions_diagnostics():
    """Get detailed diagnostics about prediction storage and updates."""
    try:
        import sqlite3
        from datetime import datetime, timedelta
        
        pred_db_path = os.getenv("DATABASE_PATH", "rewards.db")
        conn = sqlite3.connect(pred_db_path)
        cursor = conn.cursor()
        
        # Count total predictions
        cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
        total_predictions = cursor.fetchone()[0]
        
        # Count predictions with actual returns
        cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_1h IS NOT NULL")
        with_1h = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_24h IS NOT NULL")
        with_24h = cursor.fetchone()[0]
        
        # Count predictions needing updates
        now = datetime.now()
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        twenty_four_hours_ago = (now - timedelta(hours=24)).isoformat()
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM rl_prediction_accuracy
            WHERE datetime(timestamp) <= datetime(?)
            AND actual_return_1h IS NULL
            AND price_at_prediction IS NOT NULL
        """, (one_hour_ago,))
        needing_1h = cursor.fetchone()[0]
        
        cursor.execute("""
            SELECT COUNT(*) 
            FROM rl_prediction_accuracy
            WHERE datetime(timestamp) <= datetime(?)
            AND actual_return_24h IS NULL
            AND price_at_prediction IS NOT NULL
        """, (twenty_four_hours_ago,))
        needing_24h = cursor.fetchone()[0]
        
        # Get oldest prediction without 1h return
        cursor.execute("""
            SELECT timestamp, price_at_prediction
            FROM rl_prediction_accuracy
            WHERE actual_return_1h IS NULL
            AND price_at_prediction IS NOT NULL
            ORDER BY timestamp ASC
            LIMIT 1
        """)
        oldest_missing_1h = cursor.fetchone()
        
        # Get most recent prediction
        cursor.execute("""
            SELECT timestamp, price_at_prediction, actual_return_1h, actual_return_24h
            FROM rl_prediction_accuracy
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        most_recent = cursor.fetchone()
        
        conn.close()
        
        # Check if update loop is running
        global prediction_update_active
        loop_running = prediction_update_active
        
        return jsonify({
            'success': True,
            'diagnostics': {
                'total_predictions': total_predictions,
                'with_1h_returns': with_1h,
                'with_24h_returns': with_24h,
                'needing_1h_update': needing_1h,
                'needing_24h_update': needing_24h,
                'update_loop_running': loop_running,
                'oldest_missing_1h': {
                    'timestamp': oldest_missing_1h[0] if oldest_missing_1h else None,
                    'price': oldest_missing_1h[1] if oldest_missing_1h else None,
                    'age_hours': ((datetime.now() - datetime.fromisoformat(oldest_missing_1h[0])) / timedelta(hours=1)) if oldest_missing_1h else None
                } if oldest_missing_1h else None,
                'most_recent_prediction': {
                    'timestamp': most_recent[0] if most_recent else None,
                    'price': most_recent[1] if most_recent else None,
                    'has_1h': most_recent[2] is not None if most_recent else None,
                    'has_24h': most_recent[3] is not None if most_recent else None,
                } if most_recent else None,
            }
        })
    except Exception as e:
        logger.error(f"Error getting prediction diagnostics: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/consensus/stats', methods=['GET'])
def get_consensus_stats():
    """Get statistics on consensus signals (when all 7 indicators agree)."""
    try:
        consensus_tracker = ConsensusTracker()
        
        signal_type = request.args.get('type', None)  # 'BUY', 'SELL', or None for all
        
        stats = consensus_tracker.get_consensus_stats(signal_type)
        recent_signals = consensus_tracker.get_recent_consensus_signals(limit=20)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'recent_signals': recent_signals,
            'signal_type': signal_type or 'all'
        })
    except Exception as e:
        logger.error(f"Error getting consensus stats: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/retrain', methods=['POST'])
def trigger_retrain():
    """Manually trigger RL agent retraining."""
    global rl_retraining_scheduler
    
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    if rl_retraining_scheduler is None:
        return jsonify({
            'success': False,
            'error': 'Retraining scheduler not initialized'
        }), 503
    
    try:
        run_async = request.json.get('async', True) if request.is_json else True
        rl_retraining_scheduler.trigger_retrain(run_async=run_async)
        
        return jsonify({
            'success': True,
            'message': 'Retraining triggered',
            'async': run_async
        })
    except Exception as e:
        logger.error(f"Error triggering retrain: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/reload', methods=['POST'])
def reload_rl_agent_model():
    """Reload the RL agent model (useful after new training or checkpoint deployment)."""
    global rl_agent_integration, rl_model_manager
    
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    if rl_model_manager is None:
        return jsonify({
            'success': False,
            'error': 'Model manager not initialized'
        }), 503
    
    try:
        from rl_agent.model import TradingActorCritic
        
        # Reload model (will auto-detect newer checkpoints)
        model_kwargs = {
            "price_window_size": 60,
            "num_indicators": 9,  # FIX #3: stationary features only
            "embedding_dim": 384,
            "max_news_headlines": 20,
            "num_actions": 3,
        }
        
        model = rl_model_manager.load_current_model(
            model_class=TradingActorCritic,
            model_kwargs=model_kwargs,
        )
        
        if model:
            # Reinitialize integration with reloaded model
            rl_agent_integration = RLAgentIntegration(
                model=model,
                device="cpu",
            )
            
            checkpoint_path = rl_model_manager.current_model_path
            checkpoint_version = rl_model_manager.current_model_version
            
            logger.info(f"✅ RL agent model reloaded: {checkpoint_path.name} (version: {checkpoint_version})")
            
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'checkpoint': str(checkpoint_path),
                'checkpoint_name': checkpoint_path.name if checkpoint_path else None,
                'version': checkpoint_version,
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load model. Check logs for details.'
            }), 500
            
    except Exception as e:
        logger.error(f"Error reloading RL agent model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/decision', methods=['POST', 'GET'])
def make_rl_agent_decision():
    """
    Make a trading decision using the RL agent.
    
    POST: Make a new decision
    GET: Get the latest decision
    """
    global rl_agent_integration
    
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        if request.method == 'POST':
            # Initialize integration if needed
            if rl_agent_integration is None:
                return jsonify({
                    'success': False,
                    'error': 'RL agent model not loaded. Train model first.',
                    'note': 'Use /api/rl-agent/status to check model availability',
                    'steps': [
                        '1. Train model: python scripts/train_rl_agent.py --epochs 10',
                        '2. Restart app to load trained model',
                        '3. Make decision: POST /api/rl-agent/decision'
                    ]
                }), 503
            
            # Make decision
            decision = rl_agent_integration.make_decision()
            
            return jsonify({
                'success': True,
                'decision': decision
            })
        else:
            # GET: Return latest decision
            # Use same database as integration (rewards.db by default)
            db_path = os.getenv("DATABASE_PATH", "rewards.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Check if confidence column exists (for backward compatibility)
            cursor.execute("PRAGMA table_info(rl_agent_decisions)")
            columns = [col[1] for col in cursor.fetchall()]
            has_confidence = 'confidence' in columns
            
            if has_confidence:
                cursor.execute("""
                    SELECT id, timestamp, action, confidence, current_price,
                           predicted_return_1h, predicted_return_24h
                    FROM rl_agent_decisions
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
            else:
                # Fallback: use action_probabilities or default confidence
                cursor.execute("""
                    SELECT id, timestamp, action, action_probabilities, current_price,
                           predicted_return_1h, predicted_return_24h
                    FROM rl_agent_decisions
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return jsonify({
                    'success': True,
                    'decision': {
                        'decision_id': row[0],
                        'timestamp': row[1],
                        'action': row[2],
                        'confidence': row[3],
                        'current_price': row[4],
                        'prediction_1h': row[5],
                        'prediction_24h': row[6],
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No decisions found',
                    'message': 'No RL agent decisions have been made yet. Train the model and make a decision via POST /api/rl-agent/decision',
                    'note': 'This is expected if the model has not been trained or no decisions have been made yet.'
                }), 404
                
    except Exception as e:
        logger.error(f"Error in RL agent decision: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/news/force-fetch', methods=['POST'])
def force_news_fetch():
    """
    Manually trigger a news fetch (bypasses cooldown).
    Useful for debugging and ensuring fresh news.
    """
    global news_analyzer
    
    if not NEWS_ANALYZER_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'News analyzer not available'
        }), 503
    
    try:
        if news_analyzer is None:
            news_analyzer = NewsSentimentAnalyzer()
        
        # Get current SOL price
        if price_fetcher:
            price_data = price_fetcher.fetch_sol_price()
            current_price = price_data.get('rate', 0.0) if price_data else 0.0
        else:
            conn = sqlite3.connect("sol_prices.db")
            cursor = conn.cursor()
            cursor.execute("SELECT rate FROM sol_prices ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            current_price = result[0] if result else 0.0
            conn.close()
        
        if current_price <= 0:
            return jsonify({
                'success': False,
                'error': 'No valid SOL price available'
            }), 400
        
        # Force fetch news
        articles = news_analyzer.fetch_news(force=True)
        if articles:
            processed = news_analyzer.process_and_store_news(articles, current_price)
            return jsonify({
                'success': True,
                'articles_fetched': len(articles),
                'articles_processed': processed,
                'message': f'Fetched {len(articles)} articles, processed {processed} new ones'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'No articles returned from RSS feeds. Check feed URLs and network connectivity.',
                'articles_fetched': 0,
                'articles_processed': 0
            }), 400
            
    except Exception as e:
        logger.error(f"Error in force news fetch: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/rl-agent/feature-importance', methods=['GET'])
def get_rl_agent_feature_importance():
    """
    Get feature importance using SHAP values.
    
    Returns ranking of features by importance.
    """
    if not RL_AGENT_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'RL agent module not available'
        }), 503
    
    try:
        # This is a placeholder - would need actual model and states
        # For now, return example structure
        return jsonify({
            'success': True,
            'feature_importance': {
                'rsi': 0.25,
                'news_sentiment': 0.20,
                'momentum': 0.15,
                'sma_ratio': 0.10,
                'volatility': 0.08,
            },
            'note': 'Feature importance computation requires trained model and state data'
        })
    except Exception as e:
        logger.error(f"Error fetching feature importance: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/latest-bandit-action', methods=['GET'])
def get_latest_bandit_action():
    try:
        conn = sqlite3.connect("sol_prices.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, action, reward, prediction_buy, prediction_sell, prediction_hold, data_json, created_at
            FROM bandit_logs
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()

        if row:
            # Attempt to parse data_json, but allow it to be None if missing or invalid
            parsed_data_json = None
            if row[6]:
                try:
                    parsed_data_json = json.loads(row[6])
                except json.JSONDecodeError as json_err:
                    logger.warning(f"Could not parse data_json for bandit log created at {row[7]}: {json_err}")
                    # Keep parsed_data_json as None

            bandit_action = {
                "timestamp": row[0],
                "action": row[1],
                "reward": row[2],
                "prediction_buy": row[3],
                "prediction_sell": row[4],
                "prediction_hold": row[5],
                "data_json": parsed_data_json,
                "created_at": row[7]
            }
            return jsonify(bandit_action)
        else:
            return jsonify({"error": "No bandit actions found"}), 404
    except sqlite3.Error as e:
        logger.error(f"Database error fetching latest bandit action: {e}")
        return jsonify({"error": "Database error"}), 500
    except Exception as e:
        logger.error(f"Unexpected error fetching latest bandit action: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

def seconds_until_midnight():
    now = datetime.utcnow()
    tomorrow = now + timedelta(days=1)
    midnight = datetime.combine(tomorrow.date(), datetime.min.time())
    return (midnight - now).total_seconds()

def setup_sol_price_fetcher():
    global price_fetcher, price_fetch_active

    try:
        price_fetcher = SOLPriceFetcher()
        fast_mode = os.getenv("FAST_MODE", "Y").upper() != "N"

        if not fast_mode:
            interval_minutes = 1 # TODO - Set this globally or pull it from .env
            logger.info("FAST_MODE disabled, using static interval of 1 minute")
        else:
            credits = price_fetcher.get_credits()
            if credits:
                remaining = credits.get("dailyCreditsRemaining", 0)
                limit = credits.get("dailyCreditsLimit", 10000)
                logger.info(f"Initial credits: {remaining}/{limit}")

                seconds_remaining_today = seconds_until_midnight()
                safe_buffer = 0.85  # Leave 15% unused
                safe_credits = int(remaining * safe_buffer)

                if safe_credits > 0:
                    interval_seconds = max(seconds_remaining_today / safe_credits, 15)  # min 15 sec
                    interval_minutes = round(interval_seconds / 60, 2)
                else:
                    interval_minutes = 10  # fallback if no credits
            else:
                interval_minutes = 1  # fallback if no credit info

            logger.info(f"FAST_MODE enabled, using dynamic interval: {interval_minutes} minutes")

        start_sol_price_collection(interval_minutes)

    except Exception as e:
        logger.error(f"Error setting up SOL price fetcher: {e}")
        logger.error(traceback.format_exc())




def start_sol_price_collection(interval_minutes):
    """Start the SOL price collection in a separate thread"""
    global price_fetch_thread, price_fetch_active

    if price_fetch_active:
        logger.info("SOL price collection already active")
        return

    price_fetch_active = True
    price_fetch_thread = threading.Thread(
        target=sol_price_fetch_loop, 
        args=(interval_minutes,), 
        daemon=True
    )
    price_fetch_thread.start()
    logger.info(f"Started SOL price collection with {interval_minutes} minute interval")

def check_price_data_gap():
    """Check if there's a significant gap in price data (server was offline)."""
    try:
        conn = sqlite3.connect("sol_prices.db")
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp FROM sol_prices ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        
        if result:
            last_timestamp_str = result[0]
            try:
                # Parse timestamp (could be ISO format or SQLite datetime)
                if 'T' in last_timestamp_str:
                    last_timestamp = datetime.fromisoformat(last_timestamp_str.replace('Z', '+00:00'))
                else:
                    last_timestamp = datetime.strptime(last_timestamp_str, '%Y-%m-%d %H:%M:%S')
                
                # Make timezone-aware for comparison
                from datetime import timezone
                now = datetime.now(timezone.utc)
                if last_timestamp.tzinfo is None:
                    last_timestamp = last_timestamp.replace(tzinfo=timezone.utc)
                
                gap = now - last_timestamp
                gap_hours = gap.total_seconds() / 3600
                
                if gap_hours > 1:  # More than 1 hour gap
                    logger.warning(f"Price data gap detected: {gap_hours:.1f} hours since last price update")
                    logger.warning(f"Last price timestamp: {last_timestamp_str}, Current time: {now.isoformat()}")
                    logger.warning("This indicates the server was offline. Price data will resume from now.")
                    return gap_hours
                else:
                    logger.debug(f"Price data is current (gap: {gap_hours:.2f} hours)")
            except Exception as e:
                logger.warning(f"Could not parse last timestamp '{last_timestamp_str}': {e}")
        else:
            logger.info("No price data found in database - this appears to be a fresh start")
    except Exception as e:
        logger.error(f"Error checking price data gap: {e}")
    
    return 0

def sol_price_fetch_loop(initial_interval_minutes):
    global price_fetcher, price_fetch_active

    fast_mode = os.getenv("FAST_MODE", "Y").upper() != "N"
    interval_seconds = initial_interval_minutes * 60
    cycle_count = 0
    check_credits_every = 1000  # only used in fast mode
    
    # Check for data gap on startup
    gap_hours = check_price_data_gap()
    if gap_hours > 0:
        logger.info(f"Server was offline for {gap_hours:.1f} hours. Resuming price collection...")

    while price_fetch_active:
        try:
            if price_fetcher:
                price_data = price_fetcher.fetch_sol_price()
                if price_data:
                    logger.info(f"SOL price: ${price_data['rate']:.2f}")
                else:
                    logger.warning("Failed to fetch SOL price")

            if fast_mode:
                cycle_count += 1
                if cycle_count % check_credits_every == 0:
                    if price_fetcher is not None:
                        credits = price_fetcher.get_credits()
                        if credits:
                            remaining = credits.get("dailyCreditsRemaining", 0)
                            limit = credits.get("dailyCreditsLimit", 10000)
                            seconds_remaining = seconds_until_midnight()
                            safe_buffer = 0.85
                            safe_credits = int(remaining * safe_buffer)
                            if safe_credits > 0:
                                new_interval = max(seconds_remaining / safe_credits, 15)  # min 15 sec
                                interval_seconds = new_interval
                                logger.info(f"Adjusted interval to {interval_seconds/60:.2f} minutes based on credits {remaining}/{limit}")
                            else:
                                interval_seconds = 600  # fallback 10 min if no credits left
                                logger.warning("No safe credits left, setting interval to 10 minutes")
                        else:
                            logger.warning("Could not retrieve credits info")
                    else:
                        logger.warning("price_fetcher is None, cannot fetch credits")
                    cycle_count = 0

            logger.info(f"Sleeping for {interval_seconds:.2f} seconds before next fetch")
            time.sleep(interval_seconds)

        except Exception as e:
            logger.error(f"Error in SOL price fetch loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying

def background_fetch_loop():
    # Background loop to fetch transactions and insert COLLECT_FEES events
    # Configurable fetch interval (in seconds)
    fetch_interval = int(os.getenv("FETCH_INTERVAL_SECONDS", "7200"))  # Default 2 hours
    
    # mSOL snapshot interval (in hours, default 1 hour)
    msol_snapshot_interval_hours = int(os.getenv("MSOL_SNAPSHOT_INTERVAL_HOURS", "1"))
    msol_snapshot_interval_seconds = msol_snapshot_interval_hours * 3600
    last_msol_snapshot = 0

    while True:
        try:
            transactions = fetch_helius_transactions(WALLET)
            for event in transactions:
                if event.get('type') == 'COLLECT_FEES':
                    insert_collect_fee(event)
                # Also check for mSOL conversions
                process_msol_transaction(event)
        except requests.RequestException as e:
            logger.info(f"Error fetching transactions: {e}")

        # Update mSOL balance snapshot periodically
        current_time = time.time()
        if current_time - last_msol_snapshot >= msol_snapshot_interval_seconds:
            try:
                update_msol_balance_snapshot(WALLET, snapshot_type='periodic')
                last_msol_snapshot = current_time
            except Exception as e:
                logger.error(f"Error updating mSOL balance snapshot: {e}")

        time.sleep(fetch_interval)

def start_background_fetch():
    # Initialize the database and seed tokens
    logger.info("Initializing database and seeding tokens...")
    init_db()
    seed_tokens()
    
    # Optionally trigger mSOL catch-up on startup if MSOL_TRACKING_START_DATE is set
    # Skip if we already have recent snapshot data (within last 24 hours)
    if os.getenv("MSOL_TRACKING_START_DATE"):
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*), MAX(timestamp) FROM msol_balance_snapshots 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            recent_count, last_snapshot = cursor.fetchone()
            conn.close()
            
            if recent_count and recent_count > 0:
                logger.info(f"mSOL catchup skipped - {recent_count} recent snapshots exist (last: {last_snapshot})")
            else:
                logger.info("MSOL_TRACKING_START_DATE detected, triggering catch-up in background...")
                def msol_catchup_thread():
                    try:
                        catchup_msol_history(WALLET)
                    except Exception as e:
                        logger.error(f"Error in startup mSOL catch-up: {e}")
                catchup_thread = threading.Thread(target=msol_catchup_thread, daemon=True)
                catchup_thread.start()
        except Exception as e:
            logger.error(f"Error checking mSOL snapshot status: {e}")
    
    fetch_thread = threading.Thread(target=background_fetch_loop, daemon=True)
    fetch_thread.start()

def news_fetch_loop():
    """Background loop to fetch and process news articles."""
    global news_analyzer, news_fetch_active
    
    if not NEWS_ANALYZER_AVAILABLE:
        logger.warning("News analyzer not available. Skipping news fetching.")
        return
    
    # Check for news every 6 minutes (cooldown is handled by analyzer to respect rate limits)
    # The analyzer has a 5-minute cooldown, so checking every 6 minutes ensures we catch
    # when the cooldown expires quickly while not being too aggressive
    check_interval = 6 * 60  # 6 minutes in seconds
    
    try:
        news_analyzer = NewsSentimentAnalyzer()
        logger.info("News sentiment analyzer initialized")
    except Exception as e:
        logger.error(f"Failed to initialize news analyzer: {e}")
        return
    
    # Perform initial fetch immediately on startup
    logger.info("Performing initial news fetch on startup...")
    try:
        if price_fetcher:
            price_data = price_fetcher.fetch_sol_price()
            current_price = price_data.get('rate', 0.0) if price_data else 0.0
        else:
            conn = sqlite3.connect("sol_prices.db")
            cursor = conn.cursor()
            cursor.execute("SELECT rate FROM sol_prices ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            current_price = result[0] if result else 0.0
            conn.close()
        
        if current_price > 0:
            articles = news_analyzer.fetch_news(force=True)  # Force initial fetch
            if articles:
                processed = news_analyzer.process_and_store_news(articles, current_price)
                logger.info(f"Initial fetch: Processed {processed} news articles")
            else:
                logger.info("Initial fetch: No new articles found")
        else:
            logger.warning("Initial fetch skipped: no valid SOL price available")
    except Exception as e:
        logger.error(f"Error in initial news fetch: {e}")
        logger.error(traceback.format_exc())
    
    logger.info("News fetch background loop started - will check every 6 minutes")
    
    while news_fetch_active:
        try:
            # Check if news is stale (older than 1 hour) - force fetch if so
            is_stale = news_analyzer.is_news_stale(stale_hours=1)
            force_fetch = is_stale
            
            if force_fetch:
                logger.info("News is stale (older than 1 hour), forcing fetch...")
            else:
                logger.debug("News is fresh, will attempt fetch (respecting cooldown if needed)")
            
            # Get current SOL price for tracking
            if price_fetcher:
                price_data = price_fetcher.fetch_sol_price()
                current_price = price_data.get('rate', 0.0) if price_data else 0.0
            else:
                # Fallback: get from database
                conn = sqlite3.connect("sol_prices.db")
                cursor = conn.cursor()
                cursor.execute("SELECT rate FROM sol_prices ORDER BY timestamp DESC LIMIT 1")
                result = cursor.fetchone()
                current_price = result[0] if result else 0.0
                conn.close()
            
            if current_price > 0:
                logger.info("Checking for news articles in background loop...")
                # Always attempt to fetch - fetch_news() handles cooldown internally
                # Force fetch if news is stale (older than 1 hour), otherwise respect cooldown
                # This ensures regular fetching every ~6 minutes (after 5-min cooldown expires)
                try:
                    articles = news_analyzer.fetch_news(force=force_fetch)
                    if articles:  # Only process if we got articles (not in cooldown)
                        processed = news_analyzer.process_and_store_news(articles, current_price)
                        if processed > 0:
                            logger.info(f"Background loop: Processed {processed} new news articles")
                        else:
                            logger.debug(f"Background loop: Fetched {len(articles)} articles but processed 0 (duplicates)")
                    elif force_fetch:
                        logger.error("Background loop: Forced news fetch returned no articles - check RSS feeds and network")
                        logger.error("This may indicate: 1) RSS feed URLs are broken, 2) Network connectivity issues, 3) Feeds are empty")
                    else:
                        logger.debug("Background loop: News fetch skipped (in cooldown period, will retry in ~6 minutes)")
                except Exception as fetch_error:
                    logger.error(f"Error fetching news in background loop: {fetch_error}")
                    logger.error(traceback.format_exc())
            else:
                logger.warning("Cannot fetch news: no valid SOL price available")
            
        except Exception as e:
            logger.error(f"Error in news fetch loop: {e}")
            logger.error(traceback.format_exc())
            # Continue running even after errors
        
        # Sleep for check_interval using a single sleep call
        # Break the sleep into smaller chunks to allow for graceful shutdown
        sleep_chunks = 60  # Check every minute if we should stop
        chunk_duration = check_interval / sleep_chunks
        
        for _ in range(sleep_chunks):
            if not news_fetch_active:
                break
            time.sleep(chunk_duration)

def start_news_fetch():
    """Start the news fetching in a separate thread."""
    global news_fetch_thread, news_fetch_active
    
    if not NEWS_ANALYZER_AVAILABLE:
        logger.info("News analyzer not available. Skipping news fetch thread.")
        return
    
    if news_fetch_active:
        logger.info("News fetch already active")
        return
    
    news_fetch_active = True
    news_fetch_thread = threading.Thread(target=news_fetch_loop, daemon=True)
    news_fetch_thread.start()
    logger.info("Started news sentiment fetching")

def rl_decision_loop():
    """Background loop to make RL agent decisions regularly and generate predictions."""
    global rl_agent_integration, rl_decision_active
    
    if not RL_AGENT_AVAILABLE:
        logger.warning("RL agent not available. Skipping decision loop.")
        return
    
    # Decision interval: make a decision every hour (configurable)
    decision_interval = int(os.getenv("RL_DECISION_INTERVAL_MINUTES", "60"))  # Default: 60 minutes
    decision_interval_seconds = decision_interval * 60
    
    logger.info(f"RL agent decision loop started - will make decisions every {decision_interval} minutes")
    
    while rl_decision_active:
        try:
            if rl_agent_integration is None:
                logger.debug("RL agent integration not initialized yet, waiting...")
                time.sleep(60)  # Check every minute
                continue
            
            # Make a decision (this also generates predictions)
            try:
                logger.info("Making scheduled RL agent decision...")
                decision = rl_agent_integration.make_decision()
                
                action = decision.get('action', 'N/A')
                confidence = decision.get('confidence', 0)
                pred_1h = decision.get('predicted_return_1h', 0)
                pred_24h = decision.get('predicted_return_24h', 0)
                
                logger.info(
                    f"✅ Scheduled decision made: {action} "
                    f"(confidence: {confidence:.2f}, "
                    f"1h: {pred_1h*100:.2f}%, 24h: {pred_24h*100:.2f}%)"
                )
            except Exception as e:
                logger.error(f"Error making RL agent decision: {e}")
                logger.error(traceback.format_exc())
                # Continue running even after errors
            
        except Exception as e:
            logger.error(f"Error in RL decision loop: {e}")
            logger.error(traceback.format_exc())
        
        # Sleep for decision interval, checking every minute if we should stop
        sleep_chunks = decision_interval  # Check every minute
        chunk_duration = 60  # 1 minute
        
        for _ in range(sleep_chunks):
            if not rl_decision_active:
                break
            time.sleep(chunk_duration)

def start_rl_decision_loop():
    """Start the RL agent decision loop in a separate thread."""
    global rl_decision_thread, rl_decision_active
    
    if not RL_AGENT_AVAILABLE:
        logger.info("RL agent not available. Skipping decision loop thread.")
        return
    
    if rl_decision_active:
        logger.info("RL decision loop already active")
        return
    
    rl_decision_active = True
    rl_decision_thread = threading.Thread(target=rl_decision_loop, daemon=True)
    rl_decision_thread.start()
    logger.info("Started RL agent decision loop")

def update_prediction_actuals_loop():
    """Background loop to update predictions with actual return values."""
    global prediction_update_active
    
    if not RL_AGENT_AVAILABLE:
        logger.warning("RL agent not available. Skipping prediction update loop.")
        return
    
    # Run every 15 minutes
    update_interval_seconds = 15 * 60
    
    logger.info("Prediction actuals update loop started - will check every 15 minutes")
    
    while prediction_update_active:
        try:
            from rl_agent.prediction_manager import PredictionManager
            # Use same database path as rest of app
            db_path = os.getenv("DATABASE_PATH", "sol_prices.db")
            prediction_manager = PredictionManager(db_path=db_path)
            
            # Get predictions that need updating
            # 1h predictions: older than 1h but no actual_return_1h
            # 24h predictions: older than 24h but no actual_return_24h
            # Use same database path as rest of app (rewards.db by default)
            pred_db_path = os.getenv("DATABASE_PATH", "rewards.db")
            conn = sqlite3.connect(pred_db_path)
            cursor = conn.cursor()
            
            now = datetime.now()
            one_hour_ago = (now - timedelta(hours=1)).isoformat()
            twenty_four_hours_ago = (now - timedelta(hours=24)).isoformat()
            
            # CRITICAL FIX: Use datetime comparison instead of string comparison
            # SQLite datetime comparison works better with ISO format strings
            # Also check for predictions that might have been missed
            
            # Find predictions needing 1h updates (older than 1 hour)
            cursor.execute("""
                SELECT id, timestamp, price_at_prediction
                FROM rl_prediction_accuracy
                WHERE datetime(timestamp) <= datetime(?)
                AND actual_return_1h IS NULL
                AND price_at_prediction IS NOT NULL
                AND price_at_prediction > 0
                ORDER BY timestamp DESC
                LIMIT 100
            """, (one_hour_ago,))
            
            predictions_1h = cursor.fetchall()
            
            # Find predictions needing 24h updates (older than 24 hours)
            cursor.execute("""
                SELECT id, timestamp, price_at_prediction
                FROM rl_prediction_accuracy
                WHERE datetime(timestamp) <= datetime(?)
                AND actual_return_24h IS NULL
                AND price_at_prediction IS NOT NULL
                AND price_at_prediction > 0
                ORDER BY timestamp DESC
                LIMIT 100
            """, (twenty_four_hours_ago,))
            
            predictions_24h = cursor.fetchall()
            
            # Log how many predictions need updating (always log, not just when there are some)
            if predictions_1h or predictions_24h:
                logger.info(f"🔄 Found {len(predictions_1h)} predictions needing 1h updates, {len(predictions_24h)} needing 24h updates")
            else:
                # Check if there are any predictions at all
                cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
                total_preds = cursor.fetchone()[0]
                if total_preds > 0:
                    logger.debug(f"Prediction update loop: No predictions need updating (total: {total_preds})")
            
            conn.close()
            
            updated_count = 0
            
            # Update 1h predictions
            for pred_id, pred_timestamp, price_at_pred in predictions_1h:
                try:
                    # Parse timestamp (handle both with and without timezone)
                    if 'Z' in pred_timestamp:
                        pred_dt = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
                    elif '+' in pred_timestamp or pred_timestamp.endswith('00:00'):
                        pred_dt = datetime.fromisoformat(pred_timestamp)
                    else:
                        # Naive datetime (no timezone)
                        pred_dt = datetime.fromisoformat(pred_timestamp)
                    
                    target_dt = pred_dt + timedelta(hours=1)
                    
                    # Get price 1 hour later (within 1 hour window for better matching)
                    price_conn = sqlite3.connect("sol_prices.db")
                    price_cursor = price_conn.cursor()
                    
                    # Use string comparison for timestamps (SQLite stores as TEXT)
                    # Wider window for better matching: 30 min before to 1 hour after
                    target_start = (target_dt - timedelta(minutes=30)).isoformat()
                    target_end = (target_dt + timedelta(hours=1)).isoformat()
                    
                    # Use julianday for better time matching (handles timezone issues)
                    price_cursor.execute("""
                        SELECT rate, timestamp
                        FROM sol_prices
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY ABS(julianday(timestamp) - julianday(?))
                        LIMIT 1
                    """, (target_start, target_end, target_dt.isoformat()))
                    
                    price_row = price_cursor.fetchone()
                    price_conn.close()
                    
                    if price_row and price_at_pred and price_at_pred > 0:
                        price_1h_later = price_row[0]
                        actual_return_1h = (price_1h_later - price_at_pred) / price_at_pred
                        # Clamp to ±100%
                        actual_return_1h = max(-1.0, min(1.0, actual_return_1h))
                        
                        prediction_manager.update_actual_returns(
                            prediction_id=pred_id,
                            actual_return_1h=actual_return_1h,
                            price_1h_later=price_1h_later,
                        )
                        updated_count += 1
                        if updated_count % 10 == 0:
                            logger.info(f"✅ Updated {updated_count} predictions so far...")
                        logger.debug(f"Updated prediction {pred_id} with 1h actual return: {actual_return_1h:.4f}")
                except Exception as e:
                    logger.warning(f"Error updating 1h actual for prediction {pred_id}: {e}")
            
            # Update 24h predictions
            for pred_id, pred_timestamp, price_at_pred in predictions_24h:
                try:
                    # Parse timestamp (handle both with and without timezone)
                    if 'Z' in pred_timestamp:
                        pred_dt = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
                    elif '+' in pred_timestamp or pred_timestamp.endswith('00:00'):
                        pred_dt = datetime.fromisoformat(pred_timestamp)
                    else:
                        # Naive datetime (no timezone)
                        pred_dt = datetime.fromisoformat(pred_timestamp)
                    
                    target_dt = pred_dt + timedelta(hours=24)
                    
                    # Get price 24 hours later (within 2 hour window for better matching)
                    price_conn = sqlite3.connect("sol_prices.db")
                    price_cursor = price_conn.cursor()
                    
                    # Use string comparison for timestamps (SQLite stores as TEXT)
                    # Wider window for better matching: 1 hour before to 2 hours after
                    target_start = (target_dt - timedelta(hours=1)).isoformat()
                    target_end = (target_dt + timedelta(hours=2)).isoformat()
                    
                    # Use julianday for better time matching (handles timezone issues)
                    price_cursor.execute("""
                        SELECT rate, timestamp
                        FROM sol_prices
                        WHERE timestamp >= ? AND timestamp <= ?
                        ORDER BY ABS(julianday(timestamp) - julianday(?))
                        LIMIT 1
                    """, (target_start, target_end, target_dt.isoformat()))
                    
                    price_row = price_cursor.fetchone()
                    price_conn.close()
                    
                    if price_row and price_at_pred and price_at_pred > 0:
                        price_24h_later = price_row[0]
                        actual_return_24h = (price_24h_later - price_at_pred) / price_at_pred
                        # Clamp to ±100%
                        actual_return_24h = max(-1.0, min(1.0, actual_return_24h))
                        
                        prediction_manager.update_actual_returns(
                            prediction_id=pred_id,
                            actual_return_24h=actual_return_24h,
                            price_24h_later=price_24h_later,
                        )
                        updated_count += 1
                        if updated_count % 10 == 0:
                            logger.info(f"✅ Updated {updated_count} predictions so far...")
                        logger.debug(f"Updated prediction {pred_id} with 24h actual return: {actual_return_24h:.4f}")
                except Exception as e:
                    logger.warning(f"Error updating 24h actual for prediction {pred_id}: {e}")
            
            if updated_count > 0:
                logger.info(f"✅ Updated {updated_count} predictions with actual returns")
            else:
                # Log diagnostic info if no updates (helps debug why it's not working)
                if predictions_1h or predictions_24h:
                    logger.debug(f"Found {len(predictions_1h)} predictions needing 1h updates, {len(predictions_24h)} needing 24h updates, but updated 0. Check price matching logic.")
                else:
                    # Check if there are any predictions at all
                    conn = sqlite3.connect(pred_db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
                    total_preds = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_1h IS NOT NULL")
                    with_1h = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_24h IS NOT NULL")
                    with_24h = cursor.fetchone()[0]
                    conn.close()
                    logger.debug(f"Prediction update loop: {total_preds} total predictions, {with_1h} with 1h returns, {with_24h} with 24h returns")
            
        except Exception as e:
            logger.error(f"Error in prediction actuals update loop: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Sleep for update interval, checking every minute if we should stop
        sleep_chunks = update_interval_seconds // 60  # Check every minute
        chunk_duration = 60  # 1 minute
        
        for _ in range(sleep_chunks):
            if not prediction_update_active:
                break
            time.sleep(chunk_duration)

def start_prediction_update_loop():
    """Start the prediction actuals update loop in a separate thread."""
    global prediction_update_thread, prediction_update_active
    
    if not RL_AGENT_AVAILABLE:
        logger.info("RL agent not available. Skipping prediction update loop thread.")
        return
    
    if prediction_update_active:
        logger.info("Prediction update loop already active")
        return
    
    prediction_update_active = True
    prediction_update_thread = threading.Thread(target=update_prediction_actuals_loop, daemon=True)
    prediction_update_thread.start()
    logger.info("Started prediction actuals update loop")

def initialize_rl_agent():
    """Initialize RL agent model manager and scheduler."""
    global rl_agent_integration, rl_model_manager, rl_retraining_scheduler
    
    if not RL_AGENT_AVAILABLE:
        logger.info("RL agent not available, skipping initialization")
        return
    
    try:
        # Initialize model manager
        logger.info("Initializing RL agent model manager...")
        rl_model_manager = ModelManager(
            model_dir="models/rl_agent",
            archive_dir="models/rl_agent/archive",
            retention_days=30,
        )
        
        # Try to load current model
        model_kwargs = {
            "price_window_size": 60,
            "num_indicators": 9,  # FIX #3: stationary features only
            "embedding_dim": 384,
            "max_news_headlines": 20,
            "num_actions": 3,
        }
        
        # Check for low-memory mode (skip auto-deploy to reduce memory usage)
        low_memory = os.environ.get('RL_LOW_MEMORY', '').lower() in ('1', 'true', 'yes')
        if low_memory:
            logger.info("🔋 Low memory mode: skipping auto-deploy check")
        
        model = rl_model_manager.load_current_model(
            model_class=TradingActorCritic,
            model_kwargs=model_kwargs,
            skip_auto_deploy=low_memory,
        )
        
        if model:
            # Initialize integration with loaded model
            rl_agent_integration = RLAgentIntegration(
                model=model,
                device="cpu",
            )
            logger.info("✅ RL agent model loaded and ready")
            
            # Make an initial decision on startup so the model is immediately useful
            try:
                logger.info("Making initial RL agent decision...")
                initial_decision = rl_agent_integration.make_decision()
                logger.info(f"✅ Initial decision made: {initial_decision.get('action', 'N/A')} "
                          f"(confidence: {initial_decision.get('confidence', 0):.2f})")
            except Exception as e:
                logger.warning(f"Failed to make initial decision: {e}")
                import traceback
                logger.error(traceback.format_exc())  # Use error level to see full traceback
            
            # Start the decision loop to make regular decisions and generate predictions
            start_rl_decision_loop()
            
            # Start the prediction actuals update loop to track accuracy
            start_prediction_update_loop()
        else:
            logger.info("⚠️ No trained RL agent model found. Will wait for scheduled training.")
        
        # Initialize retraining scheduler.
        # CRITICAL: do NOT retrain (or prep training data) just because the app started.
        # Retraining is opt-in via env var to prevent memory exhaustion on startup.
        logger.info("Initializing RL agent retraining scheduler...")
        retraining_enabled = os.environ.get("RL_RETRAINING_ENABLED", "").lower() in ("1", "true", "yes")
        if not retraining_enabled:
            logger.info("🔕 RL retraining disabled (set RL_RETRAINING_ENABLED=1 to enable scheduled retraining)")

        # Use smaller default epochs when running in low-memory mode (safe default for Macs)
        default_epochs = 3 if low_memory else 10
        training_epochs = int(os.environ.get("RL_RETRAIN_EPOCHS", str(default_epochs)))

        rl_retraining_scheduler = RetrainingScheduler(
            model_manager=rl_model_manager,
            interval_days=7,  # Weekly retraining
            enabled=retraining_enabled,
            training_epochs=training_epochs,
        )
        
        # Log scheduler status
        status = rl_retraining_scheduler.get_status()
        if status["next_retrain_time"]:
            logger.info(f"📅 Next retraining scheduled for: {status['next_retrain_time']}")
        else:
            logger.info("📅 Retraining schedule initialized (no immediate training)")
        
        # Start scheduler (checks every hour) ONLY if enabled.
        # When disabled, the manual retrain endpoint can still be used to trigger retraining.
        if retraining_enabled:
            rl_retraining_scheduler.start_scheduler(check_interval_seconds=3600)
            logger.info("✅ RL agent retraining scheduler started (weekly, non-blocking)")
        else:
            logger.info("Skipping retraining scheduler loop (disabled)")
        
    except Exception as e:
        logger.error(f"Error initializing RL agent: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Only run startup tasks if we're in the actual Flask process (not the reloader parent)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        logger.info("Starting background fetch thread...")
        start_background_fetch()

        logger.info("Setting up SOL price fetcher...")
        setup_sol_price_fetcher()
        
        logger.info("Starting news sentiment fetching...")
        start_news_fetch()
        
        logger.info("Initializing RL agent...")
        initialize_rl_agent()
    else:
        logger.info("Skipping background tasks in parent process")

    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5030"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    app.run(host=host, port=port, debug=debug)