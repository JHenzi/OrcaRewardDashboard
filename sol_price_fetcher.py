import requests
import sqlite3
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
# -- For Price Analysis Learning --
import pickle
from pathlib import Path
import numpy as np
from collections import deque
from statistics import mean, stdev
import traceback

from river import linear_model, preprocessing, metrics
import random

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle loading the price prediction in time model
MODEL_PATH = Path("sol_model.pkl")
METRIC_PATH = Path("sol_metric.pkl")
TRAIL = deque(maxlen=10)

# Handle loading the price horizon model
HORIZON_MODEL_PATH = Path("sol_horizon_model.pkl")
HORIZON_METRIC_PATH = Path("sol_horizon_metric.pkl")

actions = ["buy", "sell", "hold"]
def model_factory():
    return preprocessing.StandardScaler() | linear_model.LinearRegression()

def fetch_last_24h_prices(db_path="sol_prices.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Assuming your data is recorded every ~5 minutes, get last 288 entries
    cursor.execute("""
        SELECT rate FROM sol_prices
        ORDER BY timestamp DESC
        LIMIT 288
    """)
    rows = cursor.fetchall()
    conn.close()
    prices = [row[0] for row in reversed(rows)]  # Oldest → newest order
    return prices


def compute_price_features(prices):
    if not prices or len(prices) < 2:
        return {}

    def simple_moving_average(prices, window):
        if len(prices) < window:
            return None
        return round(mean(prices[-window:]), 4)

    sma_1h = simple_moving_average(prices, 12)     # assuming 5-min intervals → 12*5 = 60 mins
    sma_4h = simple_moving_average(prices, 48)     # 4 hours
    sma_24h = simple_moving_average(prices, 288)   # 24 hours

    current_price = prices[-1]
    price_start = prices[0]
    percent_change = round(((current_price - price_start) / price_start) * 100, 4)

    high_price = round(max(prices), 4)
    low_price = round(min(prices), 4)
    price_range = round(high_price - low_price, 4)

    price_stdev = round(stdev(prices), 4) if len(prices) > 1 else 0
    avg_price_delta = round(mean([abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]), 4)

    return {
        "sma_1h": sma_1h,
        "sma_4h": sma_4h,
        "sma_24h": sma_24h,
        "current_price": round(current_price, 4),
        "percent_change": percent_change,
        "high": high_price,
        "low": low_price,
        "range": price_range,
        "std_dev": price_stdev,
        "avg_delta": avg_price_delta,
    }

# This function flattens nested dictionary data for easier processing.
# It converts keys like "delta_hour" into "delta_hour" and "delta_day"
# to "delta_day" for easier access in the model.
# This is useful for the Contextual Bandit model to handle nested data structures.
# It allows us to create a flat representation of the data that can be easily used in machine learning models.
# For example, if we have a dictionary like {"delta": {"hour": 0.01, "day": 0.02}},
# it will be converted to {"delta_hour": 0.01, "delta_day": 0.02}.
# This makes it easier to work with the data in models that expect flat input.
def flatten_data(data: dict) -> dict:
    """Flatten nested dictionary data for easier processing."""
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_data[f"{key}_{sub_key}"] = sub_value
        else:
            flat_data[key] = value
    return flat_data

def compute_recent_volatility(prices):
    """
    prices: List or array of recent price points (ordered oldest → newest)
    returns: Standard deviation of % price changes
    """
    if len(prices) < 2:
        return 0.01  # fallback if not enough data

    # Use simple percent returns
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns)

    return max(volatility, 0.0001)  # avoid divide-by-zero

def fetch_recent_prices(db_path="sol_prices.db", limit=50):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the most recent `limit` prices, ordered by timestamp
    cursor.execute("""
        SELECT rate FROM sol_prices
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    # We need oldest → newest order for volatility calc
    prices = [row[0] for row in reversed(rows)]
    return prices

# Contextual Bandit - Onling Learning from "data" (price updates)
# This is a simple online learning model that updates with each new price data point.
class ContextualBandit:
    def __init__(self, model_factory, actions, epsilon=0.3, min_epsilon=0.05, decay=0.999):
        self.models = {a: model_factory() for a in actions}
        self.actions = actions
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.step = 0

    def pull(self, context):
        self.step += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            predictions = {a: self.models[a].predict_one(context) for a in self.actions}
            return max(predictions.items(), key=lambda item: item[1])[0]

    def update(self, x, action, reward):
        self.models[action].learn_one(x, reward)

    def reward_function(self, action, x, volatility):
        price = x.get("price_now")
        price_high = x.get("price_24h_high")
        price_low = x.get("price_24h_low")
        returns = x.get("returns_last_n", [])  # e.g., last 3-5 hourly returns
        rolling_mean = x.get("rolling_mean_price")
        logger.info(f"Price: {price}, High: {price_high}, Low: {price_low}, Returns: {returns}, Rolling Mean: {rolling_mean}")
        delta = x.get("delta", {}).get("day", 1.0) - 1.0
        delta = max(min(delta, 0.2), -0.2)  # Clip to ±20%

        # Normalize volatility to prevent division by zero
        risk_adjusted = delta / max(volatility, 0.01)

        # === Derived indicators ===
        price_rank = 0.5
        if price_high and price_low and price_high != price_low:
            price_rank = (price - price_low) / (price_high - price_low)  # 0 = low, 1 = high

        momentum = sum(returns) / len(returns) if returns else 0.0

        mean_reversion_score = 0.0
        if rolling_mean:
            mean_reversion_score = (rolling_mean - price) / rolling_mean  # > 0 = below average (dip)

        # === Decision logic ===
        reward = -0.2  # Default penalty

        if action == "sell" and price_rank > 0.85 and momentum > 0:
            # Selling near peak during uptrend
            reward = 1.0
        elif action == "buy" and price_rank < 0.15 and momentum < 0 and mean_reversion_score > 0.01:
            # Buying near low, mean-reverting conditions
            reward = 1.0
        elif action == "hold" and abs(delta) < 0.0025:
            reward = 0.5  # Flat market
        else:
            reward = -0.1  # Slight penalty for suboptimal moves

        # Optional logging
        logger.info(
            f"[Reward] Action: {action}, Δ: {delta:.4f}, RAΔ: {risk_adjusted:.2f}, "
            f"Rank: {price_rank:.2f}, Momentum: {momentum:.4f}, MeanRev: {mean_reversion_score:.4f}, "
            f"Reward: {reward:.2f}"
        )

        return reward


def get_agent():
    if HORIZON_MODEL_PATH.exists():
        try:
            with open(HORIZON_MODEL_PATH, "rb") as f:
                agent = pickle.load(f)
            logger.info("Loaded existing Contextual Bandit model from disk.")
        except (AttributeError, ImportError) as e:
            logger.info(f"Could not load existing model due to: {e}")
            logger.info("Creating new model")
            agent = ContextualBandit(model_factory, actions)
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            agent = ContextualBandit(model_factory, actions)
    else:
        agent = ContextualBandit(model_factory, actions)
        logger.info("Initialized new Contextual Bandit model.")
    return agent

def process_bandit_step(data, volatility):
    # Fetch last 24h prices from DB
    prices_24h = fetch_last_24h_prices()

    # Compute price-based features
    price_features = compute_price_features(prices_24h)

    # Assuming prices_24h = list of (timestamp, price)


    # Flatten raw data and merge with price features
    x = flatten_data(data)

    #prices = [p for (_, p) in prices_24h]
    prices = prices_24h
    price_now = prices[-1]
    price_high = max(prices)
    price_low = min(prices)
    rolling_mean = sum(prices[-10:]) / min(len(prices), 10)
    returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-1)]

    # Slice last 5 returns or fewer if not enough
    returns_last_n = returns[-5:] if len(returns) >= 5 else returns

    # Calculate aggregated stats safely
    returns_mean = sum(returns_last_n) / len(returns_last_n) if returns_last_n else 0.0
    returns_std = stdev(returns_last_n) if len(returns_last_n) > 1 else 0.0

    # Update features without passing the list itself
    x.update({
        "price_now": price_now,
        "price_24h_high": price_high,
        "price_24h_low": price_low,
        "rolling_mean_price": rolling_mean,
        "returns_mean": returns_mean,
        "returns_std": returns_std,
    })

    x.update(price_features)

    # Add volatility explicitly
    x["recent_volatility"] = volatility


    agent = get_agent()
    predictions = {a: agent.models[a].predict_one(x) for a in agent.actions}
    action = agent.pull(x)
    logger.info(f"Predictions: {predictions}, Chosen action: {action}")
    reward = agent.reward_function(action, x, volatility)
    agent.update(x, action, reward)

    logger.info(f"Chose action: {action}, Reward: {reward}")

    with open(HORIZON_MODEL_PATH, "wb") as f:
        pickle.dump(agent, f)

    timestamp = data.get("time") or datetime.now().isoformat()
    data_json = json.dumps(data)

    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO bandit_logs (
            timestamp, action, reward, prediction_buy, prediction_sell, prediction_hold, data_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        action,
        reward,
        predictions.get("buy"),
        predictions.get("sell"),
        predictions.get("hold"),
        data_json
    ))
    conn.commit()
    conn.close()


# Load or initialize model and metric
if MODEL_PATH.exists() and METRIC_PATH.exists():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info("Loaded existing online learning model from disk.")
    with open(METRIC_PATH, "rb") as f:
        metric = pickle.load(f)
    logger.info("Loaded existing metric from disk.")
else:
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    metric = metrics.MAE()

def train_online_model(data):
    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()

    global model, metric, TRAIL

    TRAIL.append(data)

    # Base features from current data
    features = {
        "rate": data["rate"],
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

    # Extract recent prices from TRAIL
    prices = [t["rate"] for t in TRAIL]

    # Fetch last 24h prices from DB and compute features
    prices_24h = fetch_last_24h_prices()
    price_features = compute_price_features(prices_24h)

    # Merge computed price features into your features dict
    features.update(price_features)

    # Optional: also add momentum and window size from your existing logic
    window_size = len(prices)
    if window_size >= 3:
        features["momentum"] = prices[-1] - prices[0]
        features["window_size"] = window_size

    target = data["rate"]
    y_pred = model.predict_one(features) or 0.0
    model.learn_one(features, target)
    metric.update(target, y_pred)

    logger.info(f"Online Learning => Predicted: {y_pred:.2f}, Actual: {target:.2f}, Error: {abs(y_pred - target):.4f}, MAE: {metric.get():.4f}")

    cursor.execute('''
        INSERT INTO sol_predictions (timestamp, predicted_rate, actual_rate, error, mae)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        data.get("time", datetime.now().isoformat()),
        y_pred,
        target,
        abs(y_pred - target),
        metric.get()
    ))

    conn.commit()
    conn.close()

    # Persist model and metric
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

        # Create table for Contextual Bandit predictions
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS bandit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                reward REAL NOT NULL,
                prediction_buy REAL,
                prediction_sell REAL,
                prediction_hold REAL,
                data_json TEXT,
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
            # Here is the place in the loop where we execute other actions on the "data" variable, which is our dictionary of price data.
            # Train the online model with the new data - it predicts the next price based on the current data.
            train_online_model(data)
            # Train the contextual bandit model with the new data
            # This will choose an action based on the current data and update the model with the reward
            volatility = compute_recent_volatility(fetch_recent_prices())
            #process_bandit_step(data, volatility)
            # Debugging float vs. tuple error.
            try:
                process_bandit_step(data, volatility)
            except Exception as e:
                logger.error(f"Bandit step failed: {e}")
                logger.error(traceback.format_exc())

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

def check_flask_app_running(host="127.0.0.1", port=5030):
    """Check if Flask app is running by trying to connect to it"""
    try:
        response = requests.get(f"http://{host}:{port}/", timeout=2)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

def main():
    """Main function to run the price fetcher"""
    try:
        # Check if app.py is running
        flask_host = os.getenv("FLASK_HOST", "127.0.0.1")
        flask_port = int(os.getenv("FLASK_PORT", "5030"))

        if check_flask_app_running(flask_host, flask_port):
            logger.info("Flask app (app.py) is already running!")
            user_choice = input("The main Flask app is running. Exit this script? It's already executing this script! (y/n): ").lower().strip()
            if user_choice == 'y':
                logger.info("Exiting - use the Flask app for SOL price collection")
                sys.exit(0)
            else:
                logger.info("Continuing with standalone mode...")

        fetcher = SOLPriceFetcher()

        # Check initial credits
        credits = fetcher.get_credits()
        if credits:
            logger.info(f"Initial credits: {credits['dailyCreditsRemaining']}/{credits['dailyCreditsLimit']}")

        # Fetch one sample to test
        logger.info("\nFetching sample SOL price...")
        price_data = fetcher.fetch_sol_price()
        if price_data:
            logger.info(f"Current SOL price: ${price_data['rate']:.2f}")

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
                logger.info(f"Suggested interval: {interval_minutes} minutes (based on {safe_requests} remaining credits)")

                custom_interval = input(f"Enter interval in minutes (default {interval_minutes}): ").strip()
                if custom_interval:
                    try:
                        interval_minutes = int(custom_interval)
                    except ValueError:
                        logger.info("Invalid input, using default interval")
            else:
                interval_minutes = 30  # Default fallback

            fetcher.run_collection_loop(interval_minutes=interval_minutes)

    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        fetcher.close()

if __name__ == "__main__":
    main()