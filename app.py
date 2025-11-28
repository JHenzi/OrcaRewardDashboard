import requests
import time
import sqlite3
from threading import Thread
from flask import Flask, jsonify, render_template, request
import threading
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from sol_price_fetcher import SOLPriceFetcher
from statistics import mean, stdev
import logging
import traceback
import pytz
from river.tree import HoeffdingAdaptiveTreeRegressor
import pickle

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
        'redemption_frequency': get_redemption_frequency()
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
        'redemption_frequency': get_redemption_frequency()
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
        monthly_summary=monthly_summary_list
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

@app.route("/sol-tracker")
def sol_tracker():
    selected_range = request.args.get("range", "day")
    # Updated range_map to use timedelta for time-based filtering
    now = datetime.utcnow()
    range_map = {
        "hour": now - timedelta(days=1/24),  # 1 hour = 1/24 of a day
        "day": now - timedelta(days=1),
        "week": now - timedelta(weeks=1),
        "month": now - timedelta(days=30),  # Approx month
        "year": now - timedelta(days=365)   # Approx year
    }
    time_threshold = range_map.get(selected_range, now - timedelta(days=1))

    fetcher = SOLPriceFetcher()
    # Pass time_threshold to get_price_history
    # Optimized: Only fetch columns we need (timestamp, rate) and limit results for performance
    # For very long ranges, we could implement data sampling/downsampling
    max_data_points = 1000  # Limit to prevent chart overload
    all_data = fetcher.get_price_history(
        time_threshold=time_threshold.isoformat(),
        limit=max_data_points
    )
    fetcher.close()

    timestamps = []
    prices = []
    unix_timestamps = []  # Initialize Unix timestamps list
    
    if all_data: # Ensure all_data is not None or empty
        for row in all_data:  # oldest to newest
            # Optimized query now returns (timestamp, rate) tuple
            ts = row[0]  # timestamp column (first column)
            price = row[1]  # rate column (second column)
            # Ensure timestamp is valid before formatting
            if ts:
                try:
                    dt = datetime.fromisoformat(ts)
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

    if not prices: # Handle case with no price data for the selected range
        logger.warning(f"No price data found for range: {selected_range}. Returning empty data.")
        # Optionally, you could provide default empty values or render a specific message
        # For now, we'll let it proceed, and the template should handle empty lists gracefully.

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
    try:
        conn = sqlite3.connect("sol_prices.db")
        cursor = conn.cursor()
        cursor.execute("""
            SELECT rate, timestamp 
            FROM sol_prices 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        latest_row = cursor.fetchone()
        if latest_row:
            actual_current_price = round(latest_row[0], 4)
        conn.close()
    except Exception as e:
        logger.warning(f"Could not fetch latest price from database: {e}")
        # Fallback: use get_sol_price_data() for live price
        try:
            sol_data = get_sol_price_data()
            actual_current_price = round(sol_data.get('rate', 0), 4) if sol_data else None
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
    # Get current price from prices list
    current_price_for_indicators = prices[-1] if prices else None
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
        current_price = prices[-1]
        price_start = prices[0]
        stats["current_price"] = round(current_price, 4)
        stats["price_start"] = round(price_start, 4)
        
        # Update current_price for indicator calculations if not already set
        if stats.get("current_price") is None:
            stats["current_price"] = round(current_price, 4)

        if price_start != 0: # Avoid division by zero
            percent_change = round(((current_price - price_start) / price_start) * 100, 2)
            stats["percent_change"] = percent_change
        else:
            stats["percent_change"] = 0 # Or some other indicator like 'N/A' if preferred

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

    # Load bandit state
    # Removed bandit_state and bandit_stats - using RSI-based signals only
    fetcher.close()

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
        unix_timestamps=unix_timestamps  # Add Unix timestamps for TradingView
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

def sol_price_fetch_loop(initial_interval_minutes):
    global price_fetcher, price_fetch_active

    fast_mode = os.getenv("FAST_MODE", "Y").upper() != "N"
    interval_seconds = initial_interval_minutes * 60
    cycle_count = 0
    check_credits_every = 1000  # only used in fast mode

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