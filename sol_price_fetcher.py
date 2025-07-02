import requests
import sqlite3
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import os
# -- For Price Analysis Learning --
import pickle
from pathlib import Path
import numpy as np
from collections import deque

from river import linear_model, preprocessing, metrics

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle loading the model
MODEL_PATH = Path("sol_model.pkl")
METRIC_PATH = Path("sol_metric.pkl")
TRAIL = deque(maxlen=10)

# Load or initialize model and metric
if MODEL_PATH.exists() and METRIC_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(METRIC_PATH, "rb") as f:
        metric = pickle.load(f)
else:
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    metric = metrics.MAE()

def train_online_model(data):
    # Todo: Clean up that we are creating these connections every time!
    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()

    """Feed new price data into the online learning model."""
    global model, metric, TRAIL

    TRAIL.append(data)

    features = {
    "rate": data["rate"],             # Current price itself as a feature (can help regression)
    "volume": data["volume"],
    "liquidity": data["liquidity"],
    "market_cap": data["cap"],
    "delta_hour": data["delta"]["hour"],
    "delta_day": data["delta"]["day"],
    "delta_week": data["delta"]["week"],
    "delta_month": data["delta"]["month"],
    "delta_quarter": data["delta"]["quarter"],
    "delta_year": data["delta"]["year"],
    }

    # Add simple moving average if enough trailing data
    if len(TRAIL) >= 3:
        sma = np.mean([t["rate"] for t in TRAIL])
        features["sma"] = sma

    target = data["rate"]
    y_pred = model.predict_one(features) or 0.0
    model.learn_one(features, target)
    metric.update(target, y_pred)

    # Log the result
    print(f"Online Learning => Predicted: {y_pred:.2f}, Actual: {target:.2f}, Error: {abs(y_pred - target):.4f}, MAE: {metric.get():.4f}")
    cursor.execute('''
        INSERT INTO sol_predictions (timestamp, predicted_rate, actual_rate, error, mae)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        data["time"] if "time" in data else datetime.now().isoformat(),  # fallback if not present
        y_pred,
        target,
        abs(y_pred - target),
        metric.get()
    ))
    conn.commit()
    conn.close()

    # Persist model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(METRIC_PATH, "wb") as f:
        pickle.dump(metric, f)

class SOLPriceFetcher:
    def __init__(self):
        self.api_key = os.getenv('LIVECOINWATCH_API_KEY')
        if not self.api_key:
            raise ValueError("LIVECOINWATCH_API_KEY not found in environment variables")

        self.base_url = "https://api.livecoinwatch.com"
        self.headers = {
            'content-type': 'application/json',
            'x-api-key': self.api_key
        }

        # Initialize database
        self.init_database()

    def init_database(self):
        """Initialize SQLite database and create tables if they don't exist"""
        self.conn = sqlite3.connect('sol_prices.db', check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Create table for price data
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sol_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                rate REAL NOT NULL,
                volume REAL,
                market_cap REAL,
                liquidity REAL,
                delta_hour REAL,
                delta_day REAL,
                delta_week REAL,
                delta_month REAL,
                delta_quarter REAL,
                delta_year REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create table for API credits tracking
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_credits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                daily_credits_remaining INTEGER,
                daily_credits_limit INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        self.conn.commit()
        logger.info("Database initialized successfully")

    def get_credits(self):
        """Fetch remaining API credits"""
        try:
            response = requests.post(
                f"{self.base_url}/credits",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            # Store credits info in database
            timestamp = datetime.now().isoformat()
            self.cursor.execute('''
                INSERT INTO api_credits (timestamp, daily_credits_remaining, daily_credits_limit)
                VALUES (?, ?, ?)
            ''', (timestamp, data['dailyCreditsRemaining'], data['dailyCreditsLimit']))
            self.conn.commit()

            logger.info(f"Credits remaining: {data['dailyCreditsRemaining']}/{data['dailyCreditsLimit']}")
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching credits: {e}")
            return None

    def fetch_sol_price(self):
        """Fetch SOL price data"""
        payload = {
            "currency": "USD",
            "code": "SOL",
            "meta": False
        }

        try:
            response = requests.post(
                f"{self.base_url}/coins/single",
                headers=self.headers,
                data=json.dumps(payload)
            )
            response.raise_for_status()
            data = response.json()

            # Store price data in database
            timestamp = datetime.now().isoformat()
            self.cursor.execute('''
                INSERT INTO sol_prices (
                    timestamp, rate, volume, market_cap, liquidity,
                    delta_hour, delta_day, delta_week, delta_month, delta_quarter, delta_year
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                data['rate'],
                data['volume'],
                data['cap'],
                data['liquidity'],
                data['delta']['hour'],
                data['delta']['day'],
                data['delta']['week'],
                data['delta']['month'],
                data['delta']['quarter'],
                data['delta']['year']
            ))
            self.conn.commit()
            logger.info(f"SOL price: ${data['rate']:.2f} - Data stored successfully")
            train_online_model(data)
            return data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching SOL price: {e}")
            return None

    def run_collection_loop(self, interval_minutes=30, max_requests_per_day=None):
        """
        Run continuous price collection loop

        Args:
            interval_minutes: Minutes between price fetches
            max_requests_per_day: Maximum requests to make per day (None for unlimited)
        """
        logger.info(f"Starting SOL price collection loop (interval: {interval_minutes} minutes)")

        requests_made_today = 0
        last_date = datetime.now().date()

        try:
            while True:
                current_date = datetime.now().date()

                # Reset daily counter if it's a new day
                if current_date != last_date:
                    requests_made_today = 0
                    last_date = current_date
                    logger.info("New day - resetting request counter")

                # Check credits before making request
                credits_info = self.get_credits()
                if credits_info is None:
                    logger.warning("Could not fetch credits info, continuing anyway...")
                elif credits_info['dailyCreditsRemaining'] <= 0:
                    logger.warning("No credits remaining for today, waiting...")
                    time.sleep(3600)  # Wait 1 hour before checking again
                    continue

                # Check daily request limit if specified
                if max_requests_per_day and requests_made_today >= max_requests_per_day:
                    logger.info(f"Daily request limit ({max_requests_per_day}) reached, waiting until tomorrow...")
                    # Calculate seconds until midnight
                    now = datetime.now()
                    tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0)
                    tomorrow = tomorrow.replace(day=tomorrow.day + 1)
                    sleep_seconds = (tomorrow - now).total_seconds()
                    time.sleep(sleep_seconds)
                    continue

                # Fetch SOL price
                price_data = self.fetch_sol_price()
                if price_data:
                    requests_made_today += 1

                # Wait for next interval
                logger.info(f"Waiting {interval_minutes} minutes until next fetch...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Collection loop stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in collection loop: {e}")
        finally:
            self.close()

    def get_price_history(self, limit=None):
        """Retrieve price history from database"""
        query = "SELECT * FROM sol_prices ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_credits_history(self, limit=None):
        """Retrieve credits history from database"""
        query = "SELECT * FROM api_credits ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main function to run the price fetcher"""
    try:
        fetcher = SOLPriceFetcher()

        # Check initial credits
        credits = fetcher.get_credits()
        if credits:
            print(f"Initial credits: {credits['dailyCreditsRemaining']}/{credits['dailyCreditsLimit']}")

        # Fetch one sample to test
        print("\nFetching sample SOL price...")
        price_data = fetcher.fetch_sol_price()
        if price_data:
            print(f"Current SOL price: ${price_data['rate']:.2f}")

        # Ask user if they want to start continuous collection
        start_loop = input("\nStart continuous price collection? (y/n): ").lower().strip()
        if start_loop == 'y':
            # Calculate safe interval based on available credits
            if credits:
                remaining_credits = credits['dailyCreditsRemaining']
                # Leave some buffer and account for credit checks
                safe_requests = max(1, remaining_credits - 10)
                # Calculate interval for 24 hours
                interval_minutes = max(5, (24 * 60) // safe_requests)
                print(f"Suggested interval: {interval_minutes} minutes (based on {safe_requests} remaining credits)")

                custom_interval = input(f"Enter interval in minutes (default {interval_minutes}): ").strip()
                if custom_interval:
                    try:
                        interval_minutes = int(custom_interval)
                    except ValueError:
                        print("Invalid input, using default interval")
            else:
                interval_minutes = 30  # Default fallback

            fetcher.run_collection_loop(interval_minutes=interval_minutes)

    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()