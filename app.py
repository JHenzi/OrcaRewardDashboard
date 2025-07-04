import requests
import time
import sqlite3
from threading import Thread
from flask import Flask, jsonify, render_template, request
import threading
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from sol_price_fetcher import SOLPriceFetcher
from statistics import mean, stdev
import logging
import traceback
import pytz


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
    prediction_conn.commit()
    prediction_conn.close()

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
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

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
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    if 'tokenTransfers' not in event:
        conn.close()
        return
    for t in event['tokenTransfers']:
        amount = t.get('tokenAmount')
        mint = t.get('mint')
        from_token_account = t.get('fromTokenAccount')
        to_token_account = t.get('toTokenAccount')
        from_user_account = t.get('fromUserAccount')
        to_user_account = t.get('toUserAccount')
        try:
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
            logger.info(f"DB insert error: {e}")
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
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

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
    """Get collection patterns by hour of day"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT 
            CAST(strftime('%H', datetime(timestamp, 'unixepoch')) AS INTEGER) as hour_of_day,
            COUNT(*) as collections_count
        FROM collect_fees 
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    ''')
    results = c.fetchall()
    conn.close()
    return results

@app.route('/')
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

    conn.close()

    totals = {}
    for mint, amount in results:
        if mint == 'So11111111111111111111111111111111111111112':
            totals['SOL'] = amount
        elif mint == 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':
            totals['USDC'] = amount

    since_date = datetime.utcfromtimestamp(since_timestamp).strftime('%Y-%m-%d') if since_timestamp else 'N/A'
    sol_data = get_sol_price_data()

    # Add analytics data
    analytics = {
        'avg_days_between_collections': get_average_redemption_frequency(),
        'daily_earnings': get_daily_earning_rates(),
        'collection_patterns': get_collection_patterns(),
        'redemption_frequency': get_redemption_frequency()
    }
    # # Debug prints
    # print("Analytics debug:")
    # for key, value in analytics.items():
    #     print(f"{key}: {type(value)} - {value}")

    return render_template('index.html', 
        sol=totals.get('SOL', 0),
        usdc=totals.get('USDC', 0),
        sol_price=sol_data['rate'],
        sol_img=sol_data['png64'],
        sol_deltas=sol_data['delta'],
        total_usd=(totals.get('SOL', 0) * sol_data['rate']) + totals.get('USDC', 0),
        since_date=since_date,
        analytics=analytics  # Pass analytics data to template
    )

def get_predictions(cursor, limit=5):
    cursor.execute(f"""
        SELECT timestamp, predicted_rate, actual_rate, error, mae
        FROM sol_predictions
        ORDER BY created_at DESC
        LIMIT {limit}
    """)
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

def get_bandits(cursor, limit=5):
    cursor.execute(f"""
        SELECT timestamp, action, reward, prediction_buy, prediction_sell, prediction_hold
        FROM bandit_logs
        ORDER BY created_at DESC
        LIMIT {limit}
    """)
    rows = cursor.fetchall()
    bandit_logs = []
    # Initially did this but reverted, keeping the code for reference.
    # Problem is in the chart.js module it somehow is mapping these to the wrong date.
    # utc = pytz.utc  # Use UTC for consistent timestamps
    #local_tz = pytz.timezone("America/New_York")  # Use local timezone for display - TODO: Make this dynamic, use tzlocal.get_localzone() or similar!
    # Convert timestamps to local timezone for display
    for row in rows:
        log_ts = datetime.fromisoformat(row[0]).strftime("%b %d, %I:%M %p")
        bandit_logs.append({
            "timestamp": log_ts,
            "action": row[1],
            "reward": round(row[2], 4),
            "prediction_buy": round(row[3], 4) if row[3] is not None else None,
            "prediction_sell": round(row[4], 4) if row[4] is not None else None,
            "prediction_hold": round(row[5], 4) if row[5] is not None else None
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


@app.route("/sol-tracker")
def sol_tracker():
    selected_range = request.args.get("range", "day")
    range_map = {
        "hour": 12,
        "day": 288,
        "week": 2016,
        "month": 8640,
        "year": 105120,
    }
    limit = range_map.get(selected_range, 288)

    fetcher = SOLPriceFetcher()
    all_data = fetcher.get_price_history(limit=limit)
    fetcher.close()

    timestamps = []
    prices = []
    for row in reversed(all_data):  # oldest to newest
        ts = row[1]
        price = row[2]
        dt = datetime.fromisoformat(ts)
        short_ts = dt.strftime("%b %d, %I:%M %p")
        timestamps.append(short_ts)
        prices.append(price)

    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()

    predictions = get_predictions(cursor, limit=limit)
    bandit_logs = get_bandits(cursor, limit=limit)  # match bandits to price points for charting

    conn.close()

    # Stats calculations (unchanged)
    def simple_moving_average(prices, window):
        if len(prices) < window:
            return None
        return round(mean(prices[-window:]), 4)

    sma_1h = simple_moving_average(prices, 12) if len(prices) >= 12 else None
    sma_4h = simple_moving_average(prices, 48) if len(prices) >= 48 else None
    sma_24h = simple_moving_average(prices, 288) if len(prices) >= 288 else None

    current_price = prices[-1]
    price_start = prices[0]
    percent_change = round(((current_price - price_start) / price_start) * 100, 2)

    high_price = round(max(prices), 4)
    low_price = round(min(prices), 4)
    price_range = round(high_price - low_price, 4)

    price_stdev = round(stdev(prices), 4) if len(prices) > 1 else 0
    avg_price_delta = round(mean([abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]), 4)

    stats = {
        "current_price": round(current_price, 4),
        "price_start": round(price_start, 4),
        "percent_change": percent_change,
        "high": high_price,
        "low": low_price,
        "range": price_range,
        "sma_1h": sma_1h,
        "sma_4h": sma_4h,
        "sma_24h": sma_24h,
        "std_dev": price_stdev,
        "avg_delta": avg_price_delta
    }

    # Load bandit state
    bandit_state = load_bandit_state()

    return render_template(
        "sol_tracker.html",
        timestamps=timestamps,
        prices=prices,
        stats=stats,
        predictions=predictions,
        bandit_logs=bandit_logs,
        selected_range=selected_range,
        bandit_state=bandit_state
    )

def setup_sol_price_fetcher():
    global price_fetcher, price_fetch_active

    try:
        price_fetcher = SOLPriceFetcher()

        credits = price_fetcher.get_credits()
        if credits:
            logger.info(f"Initial credits: {credits['dailyCreditsRemaining']}/{credits['dailyCreditsLimit']}")

        # price_data = price_fetcher.fetch_sol_price()
        #if price_data:
        #    logger.info(f"Current SOL price: ${price_data['rate']:.2f}")

        # Just always start price collection immediately with a safe default interval
        interval_minutes = 5
        logger.info(f"Starting SOL price collection immediately with {interval_minutes} minute interval")
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

def sol_price_fetch_loop(interval_minutes):
    """Main loop for SOL price fetching"""
    global price_fetcher, price_fetch_active

    interval_seconds = interval_minutes * 60

    while price_fetch_active:
        try:
            if price_fetcher:
                price_data = price_fetcher.fetch_sol_price()
                if price_data:
                    logger.info(f"SOL price: ${price_data['rate']:.2f}")
                else:
                    logger.warning("Failed to fetch SOL price")

            # Sleep for the specified interval
            time.sleep(interval_seconds)

        except Exception as e:
            logger.error(f"Error in SOL price fetch loop: {e}")
            time.sleep(60)  # Wait 1 minute before retrying


def background_fetch_loop():
    # Background loop to fetch transactions and insert COLLECT_FEES events
    # Configurable fetch interval (in seconds)
    fetch_interval = int(os.getenv("FETCH_INTERVAL_SECONDS", "7200"))  # Default 2 hours

    while True:
        try:
            transactions = fetch_helius_transactions(WALLET)
            for event in transactions:
                if event['type'] == 'COLLECT_FEES':
                    insert_collect_fee(event)
        except requests.RequestException as e:
            logger.info(f"Error fetching transactions: {e}")

        time.sleep(fetch_interval)

def start_background_fetch():
    # Initialize the database and seed tokens
    logger.info("Initializing database and seeding tokens...")
    init_db()
    seed_tokens()
    fetch_thread = threading.Thread(target=background_fetch_loop, daemon=True)
    fetch_thread.start()

if __name__ == "__main__":
    # Only run startup tasks if we're in the actual Flask process (not the reloader parent)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        logger.info("Starting background fetch thread...")
        start_background_fetch()

        logger.info("Setting up SOL price fetcher...")
        setup_sol_price_fetcher()
    else:
        logger.info("Skipping background tasks in parent process")

    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5030"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    app.run(host=host, port=port, debug=debug)