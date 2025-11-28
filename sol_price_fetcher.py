import requests
import sqlite3
import json
import time
from datetime import timedelta, timezone
import logging
from datetime import datetime
from dotenv import load_dotenv
import sys
import os
import random
# -- For Price Analysis Learning --
import pickle
from pathlib import Path
import numpy as np
from collections import deque
from statistics import mean, stdev
import traceback
from river import linear_model, preprocessing, metrics, optim, tree
import pandas as pd

# # Load Jupiter Ultra Trading Bot!
# from trading_bot import JupiterTradingBot

# # Load bot
# bot = JupiterTradingBot()

if os.getenv("SOL_PRIVATE_KEY"):
    from trading_bot import JupiterTradingBot
    bot = JupiterTradingBot()
    trading_enabled = True
else:
    bot = None
    trading_enabled = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DEPRECATED: Price Prediction Model (Returns Irrational Values)
# ============================================================================
# Price prediction using contextual bandits has been disabled as it returns
# irrational values. The buy/sell/hold signal logic remains intact and functional.
# 
# MODEL_PATH = Path("sol_model.pkl")
# METRIC_PATH = Path("sol_metric.pkl") # We set this up but are we using it?
# TRAIL = deque(maxlen=10)

# Handle loading the Contextual Bandit Model Files - Function below to init these or create new.
HORIZON_MODEL_PATH = Path("sol_horizon_model.pkl")
HORIZON_METRIC_PATH = Path("sol_horizon_metric.pkl") # We aren't using this.

#################################################################
#                   LOAD THE MODEL FACTORY                      #
#################################################################
# Handle basic setup of the Contexual Bandit & It's Actions
actions = ["buy", "sell", "hold"]
# Old model - linear regression: Observations is that it can't handle price spikes
# def model_factory():
#     return preprocessing.StandardScaler() | linear_model.LinearRegression()
# Non-linear (bought a lot)
# def model_factory():
#     return linear_model.LinearRegression(
#         optimizer=optim.SGD(0.01),
#         l2=0.05
#     )
def model_factory():
    return tree.HoeffdingTreeRegressor()


# Critical function for price analysis, what has it done over 24 hours.
# TODO - Abstract db_path into environment variable, default to sol_price.db if missing from .env
def fetch_last_24h_prices(db_path="sol_prices.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Calculate cutoff datetime 24 hours ago in ISO format
    cutoff = (datetime.utcnow() - timedelta(hours=24)).isoformat()

    # Select prices where timestamp is within last 24 hours
    cursor.execute("""
        SELECT rate FROM sol_prices
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
    """, (cutoff,))

    rows = cursor.fetchall()
    conn.close()

    prices = [row[0] for row in rows]  # Already oldest → newest due to ASC order
    return prices

#########################################################################
#                    COMPUTE PRICE FEATURES, SMA ETC.                   #
#                    FOR PREDICTION ENGINES                             #
#########################################################################
def compute_price_features(prices):
    if not prices or len(prices) < 2:
        return {}

    def simple_moving_average(prices, window):
        if len(prices) < window:
            return None
        return round(mean(prices[-window:]), 4)

    sma_1h = simple_moving_average(prices, 12)     # assuming 5-min intervals → 12*5 = 60 mins
    sma_4h = simple_moving_average(prices, 48)     # 4 hours
    sma_24h = sum(prices) / len(prices) if prices else 0.0
    # 24 hours

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

##########################################################################
#                    COMPUTE RELATIVE STRENGTH INDED                     #
##########################################################################
def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    prices: List of prices, ordered oldest to newest.
    period: Lookback period for RSI calculation.
    """
    if len(prices) < period + 1:
        return None  # Not enough data

    deltas = np.diff(prices)
    seed = deltas[:period] # Use first 'period' deltas for initial average gain/loss

    # Calculate initial average gain and loss
    gains = seed[seed >= 0].sum() / period
    losses = -seed[seed < 0].sum() / period

    if losses == 0: # Avoid division by zero if no losses
        return 100.0
    if gains == 0: # Avoid division by zero if no gains
        return 0.0

    rs = gains / losses
    rsi_initial = 100.0 - (100.0 / (1.0 + rs))

    # Smooth RSI using Wilder's smoothing method
    # Note: For simplicity in an online setting without storing all historical RSIs,
    # this implementation recalculates based on available price history each time.
    # A more optimized version for very long series would update RSI incrementally.

    avg_gain = gains
    avg_loss = losses

    for i in range(period, len(deltas)):
        delta = deltas[i]
        if delta > 0:
            gain = delta
            loss = 0.0
        else:
            gain = 0.0
            loss = -delta

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0: # Avoid division by zero
            # If avg_loss is zero, it means all changes were gains or zero in the smoothing period.
            # RSI should be 100 in this case.
            return 100.0

    if avg_loss == 0: # Should be caught by above, but as a safeguard
        return 100.0

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi



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

#########################################################################
#                    TEMPORAL FEATURES FOR BANDIT                       #
#########################################################################
def add_temporal_features(x: dict, timestamp: str = None) -> dict:
    """
    Add temporal features to bandit context.
    
    Args:
        x: Feature dictionary to update
        timestamp: ISO format timestamp string (optional, uses current time if None)
    
    Returns:
        Updated feature dictionary with temporal features
    """
    # Use provided timestamp or current UTC time
    if timestamp:
        try:
            # Parse timestamp - handle both with and without timezone
            if 'Z' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            # Fallback to current time if parsing fails
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    # UTC features
    x["hour_of_day_utc"] = float(dt.hour)
    x["day_of_week"] = float(dt.weekday())  # 0=Monday, 6=Sunday
    x["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
    x["minutes_since_midnight"] = float(dt.hour * 60 + dt.minute)
    x["day_of_month"] = float(dt.day)
    x["month"] = float(dt.month)
    x["week_of_year"] = float(dt.isocalendar()[1])  # ISO week number
    
    # EST/EDT conversion for US market hours
    # EST is UTC-5, EDT is UTC-4
    # Simple approximation: EST = UTC - 5 hours (doesn't account for DST perfectly, but close enough)
    est_offset_hours = -5  # EST offset
    dt_est_hour = (dt.hour + est_offset_hours) % 24
    x["hour_of_day_est"] = float(dt_est_hour)
    
    # US market hours: 9:30 AM - 4:00 PM EST (14:30 - 21:00 UTC in winter, 13:30 - 20:00 UTC in summer)
    # Simplified: 9:30 AM EST = 14:30 UTC, 4:00 PM EST = 21:00 UTC
    # Check if within market hours (9:30-16:00 EST) and weekday
    est_hour_decimal = dt_est_hour + dt.minute / 60.0
    is_market_hours = (9.5 <= est_hour_decimal < 16.0) and dt.weekday() < 5
    x["is_us_market_hours"] = 1.0 if is_market_hours else 0.0
    
    # Asian market hours: 7:00 PM - 4:00 AM EST (next day) = 00:00 - 09:00 UTC (next day)
    # Simplified: 7:00 PM EST = 00:00 UTC, 4:00 AM EST = 09:00 UTC
    asian_start_utc = 0  # 7 PM EST = midnight UTC
    asian_end_utc = 9    # 4 AM EST = 9 AM UTC
    is_asian_hours = dt.hour >= asian_start_utc and dt.hour < asian_end_utc
    x["is_asian_market_hours"] = 1.0 if is_asian_hours else 0.0
    
    # European market hours: 3:00 AM - 12:00 PM EST = 8:00 - 17:00 UTC
    european_start_utc = 8
    european_end_utc = 17
    is_european_hours = european_start_utc <= dt.hour < european_end_utc
    x["is_european_market_hours"] = 1.0 if is_european_hours else 0.0
    
    # Month end/start (first/last 3 days)
    x["is_month_end"] = 1.0 if dt.day >= 28 else 0.0
    x["is_month_start"] = 1.0 if dt.day <= 3 else 0.0
    
    return x

#########################################################################
#                    TECHNICAL INDICATOR FUNCTIONS                       #
#########################################################################
def exponential_moving_average(prices, period):
    """Calculate Exponential Moving Average (EMA)"""
    if len(prices) < period:
        return None
    multiplier = 2.0 / (period + 1)
    ema = mean(prices[:period])  # Start with SMA
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    return round(ema, 4)

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    if len(prices) < slow + signal:
        return None, None, None, None
    
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None, None
    
    macd_line = ema_fast - ema_slow
    
    # For signal line, we need to calculate EMA of MACD values
    # Since we don't have historical MACD values, we'll use a simplified approach:
    # Calculate MACD for recent periods and then EMA of those values
    # For now, use a simplified signal line based on price momentum
    if len(prices) >= slow + signal:
        # Simplified signal line: use recent price change as proxy
        recent_price_change = (prices[-1] - prices[-signal-1]) / prices[-signal-1] if len(prices) > signal and prices[-signal-1] != 0 else 0.0
        # Scale it to approximate signal line magnitude
        signal_line = recent_price_change * abs(macd_line) * 0.1 if macd_line != 0 else 0.0
    else:
        signal_line = None
    
    histogram = macd_line - signal_line if signal_line is not None else None
    
    # Determine MACD signal: bullish if histogram > 0 and macd > signal, bearish if opposite
    if macd_line and signal_line:
        if macd_line > signal_line and histogram and histogram > 0:
            macd_signal = 1.0  # Bullish
        elif macd_line < signal_line and histogram and histogram < 0:
            macd_signal = -1.0  # Bearish
        else:
            macd_signal = 0.0  # Neutral
    else:
        macd_signal = 0.0
    
    return (
        round(macd_line, 4) if macd_line else None,
        round(signal_line, 4) if signal_line else None,
        round(histogram, 4) if histogram else None,
        macd_signal
    )

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None, None, None, None, None
    
    sma = mean(prices[-period:])
    std = stdev(prices[-period:])
    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)
    
    # Calculate additional BB features
    current_price = prices[-1]
    bb_width = (upper_band - lower_band) / sma if sma != 0 else 0.0
    bb_position = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) != 0 else 0.5
    
    # BB signal: -1=oversold (price near lower band), 1=overbought (price near upper band), 0=neutral
    if bb_position < 0.2:
        bb_signal = -1.0  # Oversold
    elif bb_position > 0.8:
        bb_signal = 1.0  # Overbought
    else:
        bb_signal = 0.0  # Neutral
    
    return (
        round(upper_band, 4),
        round(sma, 4),  # Middle (SMA)
        round(lower_band, 4),
        round(bb_width, 4),
        round(bb_position, 4),
        bb_signal
    )

def calculate_stochastic_oscillator(prices, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator (%K and %D)"""
    if len(prices) < k_period:
        return None, None, None
    
    # Get the last k_period prices
    recent_prices = prices[-k_period:]
    high = max(recent_prices)
    low = min(recent_prices)
    current = prices[-1]
    
    if high == low:
        stoch_k = 50.0  # Neutral if no range
    else:
        stoch_k = ((current - low) / (high - low)) * 100.0
    
    # Calculate %D (smoothed %K) - simple moving average of %K
    # For simplicity, we'll use the last d_period %K values
    # Since we only have one %K, we'll approximate %D
    if len(prices) >= k_period + d_period:
        # Calculate %K for last d_period windows
        stoch_k_values = []
        for i in range(d_period):
            window_prices = prices[-(k_period + d_period - i):-(d_period - i) if (d_period - i) > 0 else None]
            if len(window_prices) >= k_period:
                window_high = max(window_prices)
                window_low = min(window_prices)
                window_current = window_prices[-1]
                if window_high == window_low:
                    stoch_k_values.append(50.0)
                else:
                    stoch_k_values.append(((window_current - window_low) / (window_high - window_low)) * 100.0)
        stoch_d = mean(stoch_k_values) if stoch_k_values else stoch_k
    else:
        stoch_d = stoch_k
    
    # Stochastic signal: -1=oversold (%K < 20), 1=overbought (%K > 80), 0=neutral
    if stoch_k < 20:
        stoch_signal = -1.0  # Oversold
    elif stoch_k > 80:
        stoch_signal = 1.0  # Overbought
    else:
        stoch_signal = 0.0  # Neutral
    
    return (
        round(stoch_k, 4),
        round(stoch_d, 4),
        stoch_signal
    )

def calculate_atr(prices, period=14):
    """Calculate Average True Range (ATR) - volatility measure"""
    if len(prices) < period + 1:
        return None, None
    
    # True Range = max of:
    # 1. Current High - Current Low
    # 2. |Current High - Previous Close|
    # 3. |Current Low - Previous Close|
    # Since we only have prices (not OHLC), we'll approximate:
    # TR ≈ |price[i] - price[i-1]| (price change)
    
    true_ranges = []
    for i in range(1, len(prices)):
        tr = abs(prices[i] - prices[i-1])
        true_ranges.append(tr)
    
    if len(true_ranges) < period:
        return None, None
    
    # ATR is the average of true ranges over the period
    atr = mean(true_ranges[-period:])
    
    # Normalized ATR as percentage of current price
    current_price = prices[-1]
    atr_percent = (atr / current_price * 100.0) if current_price != 0 else 0.0
    
    return (
        round(atr, 4),
        round(atr_percent, 4)
    )

def calculate_momentum_indicators(prices):
    """Calculate multiple momentum indicators"""
    if len(prices) < 2:
        return None, None, None, None
    
    current_price = prices[-1]
    
    # Momentum 5-period
    momentum_5 = None
    if len(prices) >= 6:
        momentum_5 = current_price - prices[-6]
    
    # Momentum 10-period (already calculated elsewhere, but include here for consistency)
    momentum_10 = None
    if len(prices) >= 11:
        momentum_10 = current_price - prices[-11]
    
    # Momentum 30-period
    momentum_30 = None
    if len(prices) >= 31:
        momentum_30 = current_price - prices[-31]
    
    # Rate of Change (ROC) - percentage change over N periods
    rate_of_change = None
    if len(prices) >= 15:
        price_n_periods_ago = prices[-15]
        if price_n_periods_ago != 0:
            rate_of_change = ((current_price - price_n_periods_ago) / price_n_periods_ago) * 100.0
    
    return (
        round(momentum_5, 4) if momentum_5 is not None else None,
        round(momentum_10, 4) if momentum_10 is not None else None,
        round(momentum_30, 4) if momentum_30 is not None else None,
        round(rate_of_change, 4) if rate_of_change is not None else None
    )

def calculate_market_regime_features(prices, sma_period=20):
    """Calculate market regime indicators (trend vs range, volatility regime)"""
    if len(prices) < sma_period:
        return {}
    
    current_price = prices[-1]
    sma = mean(prices[-sma_period:])
    
    # Trend detection: check if price is consistently above or below SMA
    # Count how many of the last N prices are above/below SMA
    lookback = min(10, len(prices))
    prices_above_sma = sum(1 for p in prices[-lookback:] if p > sma)
    prices_below_sma = sum(1 for p in prices[-lookback:] if p < sma)
    
    # Determine trend
    if prices_above_sma >= lookback * 0.7:  # 70% of prices above SMA
        is_trending_up = 1.0
        is_trending_down = 0.0
        is_ranging = 0.0
    elif prices_below_sma >= lookback * 0.7:  # 70% of prices below SMA
        is_trending_up = 0.0
        is_trending_down = 1.0
        is_ranging = 0.0
    else:
        is_trending_up = 0.0
        is_trending_down = 0.0
        is_ranging = 1.0
    
    # Trend strength: how far is price from SMA (normalized)
    price_distance_from_sma = (current_price - sma) / sma if sma != 0 else 0.0
    trend_strength = abs(price_distance_from_sma)
    
    # Volatility regime: compare current volatility to historical
    if len(prices) >= sma_period * 2:
        # Calculate volatility for recent period and longer period
        recent_returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-sma_period, len(prices)-1)]
        historical_returns = [(prices[i+1] - prices[i]) / prices[i] for i in range(len(prices)-sma_period*2, len(prices)-sma_period-1)]
        
        if len(recent_returns) > 1 and len(historical_returns) > 1:
            recent_vol = stdev(recent_returns)
            historical_vol = stdev(historical_returns)
            
            # Volatility percentile (simplified: compare to historical)
            if historical_vol > 0:
                vol_ratio = recent_vol / historical_vol
                if vol_ratio > 1.5:
                    volatility_regime = 2.0  # High
                    is_high_volatility = 1.0
                    is_low_volatility = 0.0
                elif vol_ratio < 0.5:
                    volatility_regime = 0.0  # Low
                    is_high_volatility = 0.0
                    is_low_volatility = 1.0
                else:
                    volatility_regime = 1.0  # Medium
                    is_high_volatility = 0.0
                    is_low_volatility = 0.0
                volatility_percentile = min(1.0, max(0.0, vol_ratio / 2.0))  # Normalize to 0-1
            else:
                volatility_regime = 1.0
                volatility_percentile = 0.5
                is_high_volatility = 0.0
                is_low_volatility = 0.0
        else:
            volatility_regime = 1.0
            volatility_percentile = 0.5
            is_high_volatility = 0.0
            is_low_volatility = 0.0
    else:
        volatility_regime = 1.0
        volatility_percentile = 0.5
        is_high_volatility = 0.0
        is_low_volatility = 0.0
    
    return {
        "is_trending_up": is_trending_up,
        "is_trending_down": is_trending_down,
        "is_ranging": is_ranging,
        "trend_strength": round(trend_strength, 4),
        "price_distance_from_sma": round(price_distance_from_sma, 4),
        "volatility_regime": volatility_regime,
        "volatility_percentile": round(volatility_percentile, 4),
        "is_high_volatility": is_high_volatility,
        "is_low_volatility": is_low_volatility,
    }

def calculate_price_pattern_features(prices):
    """Calculate price pattern features (support/resistance, price action)"""
    if len(prices) < 2:
        return {}
    
    current_price = prices[-1]
    price_24h_high = max(prices)
    price_24h_low = min(prices)
    price_range = price_24h_high - price_24h_low
    
    # Distance to high/low
    distance_to_24h_high = (price_24h_high - current_price) / current_price if current_price != 0 else 0.0
    distance_to_24h_low = (current_price - price_24h_low) / current_price if current_price != 0 else 0.0
    
    # Price position in range (0-1)
    price_position_in_range = (current_price - price_24h_low) / price_range if price_range != 0 else 0.5
    
    # Near support/resistance (within 2% of high/low)
    is_near_support = 1.0 if distance_to_24h_low < 0.02 else 0.0
    is_near_resistance = 1.0 if distance_to_24h_high < 0.02 else 0.0
    
    # Count recent touches of high/low
    lookback = min(20, len(prices))
    recent_prices = prices[-lookback:]
    recent_high = max(recent_prices)
    recent_low = min(recent_prices)
    
    # Count how many times price was within 1% of high/low
    recent_high_count = sum(1 for p in recent_prices if abs(p - recent_high) / recent_high < 0.01) if recent_high != 0 else 0
    recent_low_count = sum(1 for p in recent_prices if abs(p - recent_low) / recent_low < 0.01) if recent_low != 0 else 0
    
    # Consecutive up/down periods
    consecutive_up = 0
    consecutive_down = 0
    for i in range(len(prices) - 1, 0, -1):
        if prices[i] > prices[i-1]:
            consecutive_up += 1
            if consecutive_down > 0:
                break
        elif prices[i] < prices[i-1]:
            consecutive_down += 1
            if consecutive_up > 0:
                break
        else:
            break
    
    # Price acceleration (rate of change of momentum)
    price_acceleration = 0.0
    if len(prices) >= 5:
        momentum_short = prices[-1] - prices[-3] if len(prices) >= 3 else 0
        momentum_long = prices[-3] - prices[-5] if len(prices) >= 5 else 0
        if momentum_long != 0:
            price_acceleration = (momentum_short - momentum_long) / abs(momentum_long) if momentum_long != 0 else 0.0
    
    # Reversal candidate: RSI extreme + momentum divergence
    # This is a simplified check - would need RSI to be calculated separately
    is_reversal_candidate = 0.0  # Placeholder - would need RSI value
    
    return {
        "distance_to_24h_high": round(distance_to_24h_high, 4),
        "distance_to_24h_low": round(distance_to_24h_low, 4),
        "price_position_in_range": round(price_position_in_range, 4),
        "is_near_support": is_near_support,
        "is_near_resistance": is_near_resistance,
        "recent_high_count": float(recent_high_count),
        "recent_low_count": float(recent_low_count),
        "consecutive_up_periods": float(consecutive_up),
        "consecutive_down_periods": float(consecutive_down),
        "price_acceleration": round(price_acceleration, 4),
        "is_reversal_candidate": is_reversal_candidate,
    }

def calculate_cross_timeframe_features(prices):
    """Calculate cross-timeframe features (multi-timeframe analysis)"""
    if len(prices) < 48:  # Need at least 48 periods for 4h SMA
        return {}
    
    current_price = prices[-1]
    
    # Calculate SMAs for different timeframes
    # Assuming 5-minute intervals: 12 = 1h, 48 = 4h, 288 = 24h
    sma_1h = mean(prices[-12:]) if len(prices) >= 12 else None
    sma_4h = mean(prices[-48:]) if len(prices) >= 48 else None
    sma_24h = mean(prices[-288:]) if len(prices) >= 288 else None
    
    # Price vs SMAs
    price_vs_1h_sma = (current_price / sma_1h - 1.0) * 100.0 if sma_1h and sma_1h != 0 else 0.0
    price_vs_4h_sma = (current_price / sma_4h - 1.0) * 100.0 if sma_4h and sma_4h != 0 else 0.0
    price_vs_24h_sma = (current_price / sma_24h - 1.0) * 100.0 if sma_24h and sma_24h != 0 else 0.0
    
    # SMA alignment: -1=bearish (SMA1 < SMA4 < SMA24), 1=bullish (SMA1 > SMA4 > SMA24), 0=mixed
    sma_alignment = 0.0
    if sma_1h and sma_4h and sma_24h:
        if sma_1h > sma_4h > sma_24h:
            sma_alignment = 1.0  # Bullish alignment
        elif sma_1h < sma_4h < sma_24h:
            sma_alignment = -1.0  # Bearish alignment
        else:
            sma_alignment = 0.0  # Mixed
    
    # RSI on different timeframes
    rsi_1h = calculate_rsi(prices[-12:], period=14) if len(prices) >= 26 else None  # Need 14+12=26
    rsi_4h = calculate_rsi(prices[-48:], period=14) if len(prices) >= 62 else None  # Need 14+48=62
    
    # Momentum on different timeframes
    momentum_1h = None
    if len(prices) >= 13:
        momentum_1h = current_price - prices[-13]
    
    momentum_4h = None
    if len(prices) >= 49:
        momentum_4h = current_price - prices[-49]
    
    return {
        "price_vs_1h_sma": round(price_vs_1h_sma, 4),
        "price_vs_4h_sma": round(price_vs_4h_sma, 4),
        "price_vs_24h_sma": round(price_vs_24h_sma, 4),
        "sma_alignment": sma_alignment,
        "rsi_1h": round(rsi_1h, 4) if rsi_1h is not None else None,
        "rsi_4h": round(rsi_4h, 4) if rsi_4h is not None else None,
        "momentum_1h": round(momentum_1h, 4) if momentum_1h is not None else None,
        "momentum_4h": round(momentum_4h, 4) if momentum_4h is not None else None,
    }

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


# Globals for portfolio tracking - needs better handling, we may want to include
# the bandit_state.json in the repo as an example and direct the user to update it to
# match the starting information. Then we can load the file and evaluate if it has defautls
# or handle defaulting a better way than in the code here.
# We may want to do different things too like start the model with different amounts of money
# to invest in, like a random amount between $650 and $1,500?
last_trade_action = None  # "buy" or "sell"
last_trade_price = 0.0
position_open = False

starting_cash = 1000

last_action = "hold"
entry_price = 0.0
# position_open = False
fee = 0.001  # realistic trading fee (0.1%)

portfolio = {
    "sol_balance": 0.0,
    "usd_balance": 1000.0,          # Starting budget
    "total_cost_basis": 0.0,        # Cost of SOL held
    "realized_pnl": 0.0,
    "entry_price": 0.0
    }

# Important file, however we are leaving it out of the repo. What do we do when it's missing? (We default - but should we look to .env?)
STATE_FILE = "bandit_state.json"


#########################################################################
#                    STATE HANDLING                                     #
#########################################################################
def save_state():
    state = {
        "last_action": last_action,
        "entry_price": entry_price,
        "position_open": position_open,
        "fee": fee,
        "portfolio": portfolio,
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_state():
    global last_action, entry_price, position_open, fee, portfolio
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
            last_action = state.get("last_action", last_action)
            entry_price = state.get("entry_price", entry_price)
            position_open = state.get("position_open", position_open)
            fee = state.get("fee", 0.001)
            portfolio = state.get("portfolio", {
                "sol_balance": 0.0,
                "usd_balance": starting_cash,
                "total_cost_basis": 0.0,
                "realized_pnl": 0.0,
                "entry_price": 0.0
            })
    else:
        logger.info("No saved state found, using defaults.")

load_state()

def log_trade_to_db(action, price_now, portfolio, fee,
                    price_24h_high, price_24h_low,
                    rolling_mean_price, returns_mean, returns_std, db_path="sol_prices.db"):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    sol_balance = portfolio["sol_balance"]
    usd_balance = portfolio["usd_balance"]
    total_cost_basis = portfolio["total_cost_basis"]
    realized_pnl = portfolio.get("realized_pnl", 0.0)
    portfolio_equity = usd_balance + sol_balance * price_now
    avg_entry_price = total_cost_basis / sol_balance if sol_balance > 0 else 0.0

    # Should these be added to the db?
    # price_pct_from_low = (price_now - price_24h_low) / (price_24h_high - price_24h_low + 1e-9) if price_24h_low and price_24h_high else 0.5
    # sharpe_ratio = (returns_mean / returns_std) if returns_std else 0

    # Fees in USD for 1 unit trade (simplify here for example)
    fee_usd = fee * price_now

    try:
        cursor.execute('''
            INSERT INTO trades (
                timestamp, action, price, amount, fee_pct, fee_usd, net_value_usd,
                portfolio_usd_balance, portfolio_sol_balance, portfolio_equity, avg_entry_price,
                realized_pnl, price_24h_high, price_24h_low, rolling_mean_price, returns_mean, returns_std
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            timestamp, action, price_now, 1, fee, fee_usd, price_now * (1 - fee),
            usd_balance, sol_balance, portfolio_equity, avg_entry_price,
            realized_pnl, price_24h_high, price_24h_low, rolling_mean_price, returns_mean, returns_std
        ))
        conn.commit()
        logger.info("Successfully logged trade to database.")
    except Exception as e:
        logger.error(f"Error logging trade to database: {e}")
    finally:
        conn.close()


#########################################################################
#          HELPER CALCULATIONS (for reward function)                    #
#########################################################################
def calculate_enhanced_timing_bonus(price_pct_from_low, pnl_percentage):
        """Enhanced timing bonus that scales with profitability"""
        # Base timing bonus (0-10 range)
        timing_bonus = (price_pct_from_low ** 2) * 10

        # Multiplicative bonus for profitable exits near highs
        if pnl_percentage > 0 and price_pct_from_low > 0.7:
            timing_bonus *= (1 + pnl_percentage * 2)

        return timing_bonus

def calculate_percentage_pnl_reward(net_pnl, position_value, max_reward=1.0):
    """Scale rewards based on percentage returns with dampened growth for small profits"""
    if position_value == 0:
        return 0

    pnl_percentage = net_pnl / position_value

    if pnl_percentage > 0:
        # Scaled exponential curve with diminishing returns
        scaled = min((pnl_percentage * 100) ** 0.7 / 5, max_reward)
        return scaled
    else:
        # Penalize losses linearly, capped
        return max(pnl_percentage * 10, -1.0)


def calculate_dynamic_hold_penalty(unrealized_pct, sharpe_ratio, margin_opportunity):
        """Dynamic penalties based on missed opportunities"""
        penalty = 0

        # Exponential penalty for not taking large profits
        if unrealized_pct > 0.05:  # 5%+ unrealized gains
            penalty += (unrealized_pct * 100) ** 1.2 * -0.1

        # Penalty for holding in strong downtrend
        if sharpe_ratio < -0.5 and unrealized_pct < -0.03:
            penalty += (abs(unrealized_pct) * 100) ** 1.1 * -0.1

        # Penalty for missing new buying opportunities while holding
        if margin_opportunity > 0.02:
            missed_opportunity_reward = ((margin_opportunity - 0.01) * 100) ** 1.5
            penalty += missed_opportunity_reward * -0.5  # Half the missed opportunity

        return penalty


#########################################################################
#                    REWARD FUNCTION FOR BANDIT                         #
#########################################################################
def calculate_reward(action, price_now, portfolio, fee=0.001,
                        price_24h_high=None, price_24h_low=None,
                        rolling_mean_price=None, returns_mean=None,
                        returns_std=None, prices_24h=None, rsi_value=None,price_momentum_15m=None):
    """
    Calculate the contextual reward for a given action in a bandit-based trading environment.

    This function evaluates the quality of a trading decision ("buy", "sell", or "hold") 
    by comparing the action taken with market conditions, volatility, recent performance,
    and historical price data. It is designed to encourage the bandit agent to:
    - Buy near local price lows or during dips with favorable reward-to-risk ratios.
    - Sell when positions are profitable and aligned with positive price momentum.
    - Hold when volatility is low or it's optimal to wait.
    - Penalize missed opportunities and poor trade timing.

    Returns:
    -------
    reward : float
        A scalar reward signal used to train the contextual bandit. Positive values
        represent profitable or wise actions, while negative values represent losses,
        missed opportunities, or risky behavior.
    """
    global last_trade_action, last_trade_price, position_open, last_action
    reward = 0.0
    net_pnl = 0.0  # Initialize net_pnl for logging purposes
    cost_with_fee = price_now * (1 + fee) # This is a good idea, it's unused - TODO
    sell_price_after_fee = price_now * (1 - fee) # This is a good idea, it's unused - TODO

    # Contextual indicators
    price_pct_from_low = (price_now - price_24h_low) / (price_24h_high - price_24h_low + 1e-9) if price_24h_low and price_24h_high else 0.01
    price_pct_from_mean = (price_now - rolling_mean_price) / rolling_mean_price if rolling_mean_price else 0
    sharpe_ratio = (returns_mean / returns_std) if returns_std else 0
    if prices_24h and len(prices_24h) >= 50:
        price_momentum = price_now - prices_24h[-50]
    elif prices_24h and len(prices_24h) >= 10:
        price_momentum = price_now - prices_24h[-10]
    elif prices_24h and len(prices_24h) >= 5:
        price_momentum = price_now - prices_24h[-5]
    else:
        price_momentum = 0

    sol_balance = portfolio["sol_balance"]
    usd_balance = portfolio["usd_balance"]
    total_cost_basis = portfolio["total_cost_basis"]

    ###################
    # Buy Logic
    ###################
    if action == "buy":
        if usd_balance >= price_now:
            # Signal strength encourages dip buying
            buy_signal_strength = (1 - price_pct_from_low) - price_pct_from_mean - sharpe_ratio
            dip_reward = max(min(buy_signal_strength, 1), -1)
            potential_margin = ((rolling_mean_price - price_now) / rolling_mean_price) * 2 if rolling_mean_price else 0
            if potential_margin >= 0.01:  # Only reward if 1%+ opportunity
                # Scale and shift so 1% = small reward, grows exponentially from there
                margin_reward = ((potential_margin - 0.01) * 100) ** 1.5
            else:
                margin_reward = 0
            # Normalize RSI (0-1)
            rsi_score = rsi_value / 100.0 if rsi_value is not None else 0.5

            # Bonus if oversold
            if rsi_value is not None:
                if rsi_value < 30:
                    rsi_bonus = (30 - rsi_value) / 30  # Scales from 0 to 1
                else:
                    rsi_bonus = -((rsi_value - 50) / 50) if rsi_value > 50 else 0.00001  # Mild penalty
            else:
                rsi_bonus = 0.00001
            # Momentum consideration (prefer to buy when momentum has flattened)
            momentum_penalty = -abs(price_momentum_15m or 0) * 0.2
            portfolio["sol_balance"] += 1
            portfolio["usd_balance"] -= price_now
            portfolio["total_cost_basis"] += price_now
            portfolio["entry_price"] = price_now
            reward = dip_reward + margin_reward + rsi_bonus + momentum_penalty # BIGGER  # combine both
            last_trade_action = "buy"
            last_trade_price = price_now
            position_open = True
            # log the trade
            log_trade_to_db(action, price_now, portfolio, fee,
                        price_24h_high, price_24h_low,
                        rolling_mean_price, returns_mean, returns_std)
        else:
            reward = -0.5  # can't afford

    ###################
    # Sell Logic
    ###################
    elif action == "sell":
        if sol_balance > 0:
            avg_entry = total_cost_basis / sol_balance
            # OLD WAY
            # gross_pnl = (price_now - avg_entry) * sol_balance
            #net_pnl = gross_pnl - fee # TODO: Make this better
            total_sale_value = sol_balance * price_now * (1 ) # - fee)
            total_entry_value = total_cost_basis
            net_pnl = total_sale_value - total_entry_value
            # Reward more if we sold near 24h high
            sell_bonus = price_pct_from_low  # higher is better
            # IMPROVED: Calculate position value for percentage-based rewards
            position_value = total_cost_basis  # Original investment amount
            # IMPROVED: Percentage-based PnL scaling instead of linear
            pnl_reward = calculate_percentage_pnl_reward(net_pnl, position_value)
            # IMPROVED: Enhanced timing bonus that scales with profitability
            pnl_percentage = net_pnl / position_value if position_value > 0 else 0
            timing_bonus = calculate_enhanced_timing_bonus(price_pct_from_low, pnl_percentage)
            profit_factor = max(pnl_percentage, 0)
            # Old ways;
            # reward = ((pnl_reward * 1.15) + (sell_bonus + timing_bonus)) * (1 + profit_factor)
            # Safety caps
            pnl_percentage = max(min(net_pnl / position_value, 1.0), -1.0)
            sell_bonus = min(price_pct_from_low, 0.05)
            timing_bonus = min(timing_bonus, 0.05)
            pnl_reward = max(min(pnl_reward, 1.0), -1.0)

            # Revised logic (flattened scaling)
            base = pnl_reward * 1.15
            if rsi_value is not None:
                if rsi_value > 70:
                    rsi_bonus = (rsi_value - 70) / 30  # Scales from 0 to 1
                else:
                    rsi_bonus = -((rsi_value - 50) / 50) if rsi_value > 50 else 0.00001  # Mild penalty
            else:
                rsi_bonus = 0.00001
            bonus = sell_bonus + timing_bonus + rsi_bonus
            reward = base + bonus * pnl_percentage

            # Final cap to avoid wild values
            reward = max(min(reward, 1.5), -1.0)

            portfolio["realized_pnl"] += net_pnl
            portfolio["entry_price"] = 0.0
            portfolio["usd_balance"] += sol_balance * price_now
            portfolio["sol_balance"] = 0
            portfolio["total_cost_basis"] = 0
            last_trade_action = "sell"
            last_trade_price = price_now
            position_open = False
            # log the trade
            log_trade_to_db(action, price_now, portfolio, fee,
                        price_24h_high, price_24h_low,
                        rolling_mean_price, returns_mean, returns_std)
        else:
            # Market momentum (e.g., price_now vs. past N values)
            # Assume this is passed in as `price_momentum` or computed earlier
            # You can also use `returns_mean` if cleaner
            momentum_factor = price_momentum if price_momentum is not None else 0
            # Normalize momentum to a reasonable range
            normalized_momentum = max(min(momentum_factor / price_now, 0.05), -0.05)
            # Base penalty
            base_penalty = -0.2
            # Reward modifier: if momentum is strongly positive, reduce penalty
            momentum_modifier = 0.5 * normalized_momentum  # half strength
            # Final reward: still negative, but less so in uptrends
            reward = base_penalty + momentum_modifier



    ###################
    # Hold Logic
    ###################
    elif action == "hold":
        if sol_balance > 0:
            avg_entry = total_cost_basis / sol_balance
            unrealized_pct = (price_now - avg_entry) / avg_entry
            # Encourage holding during uncertain/flat periods (low volatility, weak trend)
            safe_hold_zone = abs(sharpe_ratio) < 0.2 and abs(price_pct_from_mean) < 0.01
            # Penalize not selling during high unrealized profits in strong uptrends
            missed_exit = unrealized_pct > 0.03 and sharpe_ratio > 0.3
            # Penalize holding during drawdowns
            drawdown_risk = unrealized_pct < -0.01 and sharpe_ratio < -0.2
            rsi_hold_bonus = 0.0
            if rsi_value is not None:
                if 40 <= rsi_value <= 60:
                    rsi_hold_bonus = 0.2
                elif rsi_value < 30 and sharpe_ratio <0:
                    rsi_hold_bonus = -0.2
                elif rsi_value > 70 and sharpe_ratio > 0:
                    rsi_hold_bonus = -0.3
            if missed_exit:
                reward = -1.0 * min(unrealized_pct, 0.2)  # cap at -0.2
            elif drawdown_risk:
                reward = -0.5 * abs(unrealized_pct)  # proportional to drawdown
            elif safe_hold_zone:
                reward = 0.5 + 0.25 * (1 - abs(price_pct_from_mean) * 10)
            else:
                reward = rsi_hold_bonus + random.uniform(0.1, 0.25)  # mild default reward
        else:
            # Penalize for missing buying opportunity when price moves up from a recent low
            potential_margin = (rolling_mean_price - price_now) / rolling_mean_price if rolling_mean_price else 0
            # If price is below rolling mean, opportunity to buy
            if potential_margin > 0.01:
                missed_opportunity = ((potential_margin - 0.01) * 100) ** 1.5
                reward = min(missed_opportunity * 0.01, 0.05)  # much smaller value
            if rsi_value is not None and price_momentum is not None:
                if rsi_value > 60 and price_momentum > 0:
                    reward = -0.05
            # Additional penalty if price is rising and we are not in position
            # elif price_momentum > 0.01 and sharpe_ratio > 0.1:
            #     # Penalize missing upward momentum
            #     reward = min(price_momentum / price_now, 0.05)  # IN ABOVE REMOVING
            # Reward for correctly staying in cash during downtrend
            elif sharpe_ratio < -0.3 and price_momentum < 0:
                reward = random.uniform(0.2, 0.3)
            elif sharpe_ratio < 0.05:
                reward = random.uniform(0.15, 0.2)  # leaning defensive
            else:
                reward = random.uniform(0.1, 0.2)  # weak hold signal
    if abs(reward) > 1.0:
        # Only log net_pnl if it was calculated (i.e., for sell actions)
        pnl_str = f"pnl={net_pnl:.2f}" if action == "sell" else "pnl=N/A"
        logger.info(f"⚠️ High reward {reward:.2f} on {action}: {pnl_str}, rsi={rsi_value}, momentum={price_momentum_15m}")
    last_action = action
    save_state()
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

#########################################################################
#                    BANDIT STEP EXECUTION                              #
#########################################################################
def process_bandit_step(data, volatility):
    # Get globals.
    global last_action, last_price, entry_price, position_open, fee, portfolio
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

    # Define the lookback period for Sharpe Ratio calculation (e.g., 60 for 1 hour of 1-minute data)
    sharpe_lookback_period = 60

    # Ensure we have enough returns data (need lookback_period returns, so lookback_period + 1 prices)
    if len(returns) >= sharpe_lookback_period:
        returns_for_sharpe = returns[-sharpe_lookback_period:]
    else:
        returns_for_sharpe = returns # Use all available if not enough

    # Calculate aggregated stats safely for Sharpe Ratio
    if len(returns_for_sharpe) > 1: # Need at least 2 returns to calculate stdev
        returns_mean = mean(returns_for_sharpe)
        returns_std = stdev(returns_for_sharpe)
        if returns_std == 0: # Avoid division by zero in Sharpe Ratio
            returns_std = 1e-9 # A very small number instead of zero
    elif len(returns_for_sharpe) == 1:
        returns_mean = returns_for_sharpe[0]
        returns_std = 1e-9 # Cannot calculate stdev from a single point, use small epsilon
    else:
        returns_mean = 0.0
        returns_std = 1e-9

    # Calculate aggregated stats safely
    #returns_mean = sum(returns_last_n) / len(returns_last_n) if returns_last_n else 0.0
    #returns_std = stdev(returns_last_n) if len(returns_last_n) > 1 else 0.0

    # Update features without passing the list itself
    x.update({
        "price_now": price_now,
        "price_24h_high": price_high,
        "price_24h_low": price_low,
        "rolling_mean_price": rolling_mean,
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        # "returns_for_sharpe": returns_for_sharpe,
    })

    x.update(price_features)

    # Add Price vs SMA features
    price_vs_sma_1h = price_now / price_features["sma_1h"] if price_features.get("sma_1h") and price_features["sma_1h"] != 0 else 1.0
    price_vs_sma_4h = price_now / price_features["sma_4h"] if price_features.get("sma_4h") and price_features["sma_4h"] != 0 else 1.0


    # Calculate RSI
    # Ensure prices_24h has enough data for RSI (at least period + 1, e.g., 14+1=15 for 14-period RSI)
    rsi_period = 14
    rsi_value = calculate_rsi(prices, period=rsi_period) if len(prices) > rsi_period else None

    # Calculate Price Momentum (e.g., over last 15 periods)
    momentum_period = 15
    price_momentum_15m = None
    if len(prices) >= momentum_period + 1: # Need current price and price N periods ago
        price_momentum_15m = price_now - prices[-(momentum_period + 1)]

    x.update({
        "price_vs_sma_1h": price_vs_sma_1h,
        "price_vs_sma_4h": price_vs_sma_4h,
        "rsi_value": rsi_value,
        "price_momentum_15m": price_momentum_15m,
        "sharpe_ratio": (returns_mean / returns_std) if returns_std != 0 else 0, # Added Sharpe Ratio
    })

    if volatility is None:
        volatility = compute_recent_volatility(prices)
    else:
        volatility = max(volatility, 0.0001)  # avoid divide-by-zero

    # Add volatility explicitly
    x["recent_volatility"] = volatility

    if portfolio["sol_balance"] > 0:
        avg_entry = portfolio["total_cost_basis"] / portfolio["sol_balance"]
        unrealized_pnl = (price_now - avg_entry) * portfolio["sol_balance"]
    else:
        avg_entry = 0.0
        unrealized_pnl = 0.0

    # Explicit portfolio state features
    can_afford_buy = 1 if portfolio["usd_balance"] >= price_now else 0
    is_position_open = 1 if portfolio["sol_balance"] > 0 else 0

    # Ratio-based portfolio features
    current_equity = portfolio["usd_balance"] + portfolio["sol_balance"] * price_now
    usd_balance_ratio = portfolio["usd_balance"] / (current_equity + 1e-9) # Epsilon for safety
    sol_balance_value = portfolio["sol_balance"] * price_now
    sol_balance_ratio = sol_balance_value / (current_equity + 1e-9) # Epsilon for safety


    x.update({
        "portfolio_sol_balance": portfolio["sol_balance"],
        "portfolio_usd_balance": portfolio["usd_balance"],
        "portfolio_total_cost_basis": portfolio["total_cost_basis"],
        "portfolio_avg_entry_price": avg_entry,
        "portfolio_unrealized_pnl": unrealized_pnl,
        "portfolio_equity": current_equity,
        "can_afford_buy": can_afford_buy,
        "is_position_open": is_position_open,
        "usd_balance_ratio": usd_balance_ratio,
        "sol_balance_ratio": sol_balance_ratio,
    })
    
    # Add news sentiment features (only numeric features for River)
    try:
        from news_sentiment import NewsSentimentAnalyzer
        news_analyzer = NewsSentimentAnalyzer()
        news_features = news_analyzer.get_recent_news_features(hours=24, crypto_only=True)
        
        # Only add numeric features to x (River doesn't handle strings well)
        # Convert sentiment label to numeric: positive=1, negative=-1, neutral=0
        sentiment_label = news_features.get("news_sentiment_label", "neutral")
        sentiment_numeric = 1.0 if sentiment_label == "positive" else (-1.0 if sentiment_label == "negative" else 0.0)
        
        x.update({
            "news_sentiment_score": float(news_features.get("news_sentiment_score", 0.0)),
            "news_sentiment_numeric": float(sentiment_numeric),  # Numeric version of label
            "news_count": int(news_features.get("news_count", 0)),
            "news_positive_count": int(news_features.get("news_positive_count", 0)),
            "news_negative_count": int(news_features.get("news_negative_count", 0)),
            "news_crypto_count": int(news_features.get("news_crypto_count", 0)),
        })
        logger.info(f"Added news features: sentiment_score={news_features.get('news_sentiment_score', 0.0)}, count={news_features.get('news_count', 0)}")
    except Exception as e:
        logger.warning(f"Failed to add news features: {e}")
        # Add neutral news features as fallback (all numeric)
        x.update({
            "news_sentiment_score": 0.0,
            "news_sentiment_numeric": 0.0,
            "news_count": 0,
            "news_positive_count": 0,
            "news_negative_count": 0,
            "news_crypto_count": 0,
        })

    # Add temporal features (day of week, hour, timezone, market hours)
    try:
        timestamp = data.get("time") or datetime.now(timezone.utc).isoformat()
        add_temporal_features(x, timestamp)
        logger.debug(f"Added temporal features: hour_utc={x.get('hour_of_day_utc')}, day_of_week={x.get('day_of_week')}, is_weekend={x.get('is_weekend')}")
    except Exception as e:
        logger.warning(f"Failed to add temporal features: {e}")

    # Add technical indicators (MACD, Bollinger Bands, EMA)
    try:
        # Calculate MACD
        macd_line, macd_signal_line, macd_histogram, macd_cross_signal = calculate_macd(prices) if len(prices) >= 35 else (None, None, None, 0.0)
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower, bb_width, bb_position, bb_signal = calculate_bollinger_bands(prices) if len(prices) >= 20 else (None, None, None, None, None, 0.0)
        
        # Calculate EMA
        ema_12 = exponential_moving_average(prices, 12) if len(prices) >= 12 else None
        ema_26 = exponential_moving_average(prices, 26) if len(prices) >= 26 else None
        
        # Add MACD features
        if macd_line is not None:
            x.update({
                "macd_line": float(macd_line),
                "macd_signal": float(macd_signal_line) if macd_signal_line is not None else 0.0,
                "macd_histogram": float(macd_histogram) if macd_histogram is not None else 0.0,
                "macd_cross_signal": float(macd_cross_signal),
            })
        else:
            x.update({
                "macd_line": 0.0,
                "macd_signal": 0.0,
                "macd_histogram": 0.0,
                "macd_cross_signal": 0.0,
            })
        
        # Add Bollinger Bands features
        if bb_upper is not None and bb_lower is not None and bb_middle is not None:
            x.update({
                "bb_upper": float(bb_upper),
                "bb_middle": float(bb_middle),
                "bb_lower": float(bb_lower),
                "bb_width": float(bb_width) if bb_width is not None else 0.0,
                "bb_position": float(bb_position) if bb_position is not None else 0.5,
                "bb_signal": float(bb_signal),
            })
        else:
            x.update({
                "bb_upper": float(price_now),
                "bb_middle": float(price_now),
                "bb_lower": float(price_now),
                "bb_width": 0.0,
                "bb_position": 0.5,
                "bb_signal": 0.0,
            })
        
        # Add EMA features
        if ema_12 is not None and ema_26 is not None:
            price_vs_ema_12 = price_now / ema_12 if ema_12 != 0 else 1.0
            price_vs_ema_26 = price_now / ema_26 if ema_26 != 0 else 1.0
            ema_12_vs_ema_26 = ema_12 / ema_26 if ema_26 != 0 else 1.0
            
            x.update({
                "ema_12": float(ema_12),
                "ema_26": float(ema_26),
                "price_vs_ema_12": float(price_vs_ema_12),
                "price_vs_ema_26": float(price_vs_ema_26),
                "ema_12_vs_ema_26": float(ema_12_vs_ema_26),
            })
        else:
            x.update({
                "ema_12": float(price_now),
                "ema_26": float(price_now),
                "price_vs_ema_12": 1.0,
                "price_vs_ema_26": 1.0,
                "ema_12_vs_ema_26": 1.0,
            })
        
        logger.debug(f"Added technical indicators: MACD={macd_line}, BB_signal={bb_signal}, EMA_12={ema_12}")
    except Exception as e:
        logger.warning(f"Failed to add technical indicators: {e}")
        # Add default values as fallback
        x.update({
            "macd_line": 0.0,
            "macd_signal": 0.0,
            "macd_histogram": 0.0,
            "macd_cross_signal": 0.0,
            "bb_upper": float(price_now),
            "bb_middle": float(price_now),
            "bb_lower": float(price_now),
            "bb_width": 0.0,
            "bb_position": 0.5,
            "bb_signal": 0.0,
            "ema_12": float(price_now),
            "ema_26": float(price_now),
            "price_vs_ema_12": 1.0,
            "price_vs_ema_26": 1.0,
            "ema_12_vs_ema_26": 1.0,
        })

    # Add Stochastic Oscillator and ATR
    try:
        stoch_k, stoch_d, stoch_signal = calculate_stochastic_oscillator(prices) if len(prices) >= 14 else (None, None, 0.0)
        atr_14, atr_percent = calculate_atr(prices, period=14) if len(prices) >= 15 else (None, None)
        
        if stoch_k is not None:
            x.update({
                "stoch_k": float(stoch_k),
                "stoch_d": float(stoch_d) if stoch_d is not None else float(stoch_k),
                "stoch_signal": float(stoch_signal),
            })
        else:
            x.update({
                "stoch_k": 50.0,
                "stoch_d": 50.0,
                "stoch_signal": 0.0,
            })
        
        if atr_14 is not None:
            x.update({
                "atr_14": float(atr_14),
                "atr_percent": float(atr_percent),
            })
        else:
            x.update({
                "atr_14": 0.0,
                "atr_percent": 0.0,
            })
        
        logger.debug(f"Added Stochastic: K={stoch_k}, D={stoch_d}, ATR={atr_14}")
    except Exception as e:
        logger.warning(f"Failed to add Stochastic/ATR: {e}")
        x.update({
            "stoch_k": 50.0,
            "stoch_d": 50.0,
            "stoch_signal": 0.0,
            "atr_14": 0.0,
            "atr_percent": 0.0,
        })

    # Add additional momentum indicators
    try:
        momentum_5, momentum_10, momentum_30, rate_of_change = calculate_momentum_indicators(prices)
        
        x.update({
            "momentum_5": float(momentum_5) if momentum_5 is not None else 0.0,
            "momentum_10": float(momentum_10) if momentum_10 is not None else 0.0,
            "momentum_30": float(momentum_30) if momentum_30 is not None else 0.0,
            "rate_of_change": float(rate_of_change) if rate_of_change is not None else 0.0,
        })
        
        logger.debug(f"Added momentum indicators: momentum_5={momentum_5}, ROC={rate_of_change}")
    except Exception as e:
        logger.warning(f"Failed to add momentum indicators: {e}")
        x.update({
            "momentum_5": 0.0,
            "momentum_10": 0.0,
            "momentum_30": 0.0,
            "rate_of_change": 0.0,
        })

    # Add market regime features
    try:
        regime_features = calculate_market_regime_features(prices)
        x.update(regime_features)
        logger.debug(f"Added market regime: trending_up={regime_features.get('is_trending_up')}, volatility_regime={regime_features.get('volatility_regime')}")
    except Exception as e:
        logger.warning(f"Failed to add market regime features: {e}")
        x.update({
            "is_trending_up": 0.0,
            "is_trending_down": 0.0,
            "is_ranging": 0.0,
            "trend_strength": 0.0,
            "price_distance_from_sma": 0.0,
            "volatility_regime": 1.0,
            "volatility_percentile": 0.5,
            "is_high_volatility": 0.0,
            "is_low_volatility": 0.0,
        })

    # Add price pattern features
    try:
        pattern_features = calculate_price_pattern_features(prices)
        # Add RSI to pattern features for reversal candidate
        if rsi_value is not None:
            # Reversal candidate: RSI extreme (< 30 or > 70) + momentum divergence
            is_reversal = 0.0
            if (rsi_value < 30 and price_momentum_15m and price_momentum_15m < 0) or \
               (rsi_value > 70 and price_momentum_15m and price_momentum_15m > 0):
                is_reversal = 1.0
            pattern_features["is_reversal_candidate"] = is_reversal
        
        x.update(pattern_features)
        logger.debug(f"Added price patterns: near_support={pattern_features.get('is_near_support')}, consecutive_up={pattern_features.get('consecutive_up_periods')}")
    except Exception as e:
        logger.warning(f"Failed to add price pattern features: {e}")
        x.update({
            "distance_to_24h_high": 0.0,
            "distance_to_24h_low": 0.0,
            "price_position_in_range": 0.5,
            "is_near_support": 0.0,
            "is_near_resistance": 0.0,
            "recent_high_count": 0.0,
            "recent_low_count": 0.0,
            "consecutive_up_periods": 0.0,
            "consecutive_down_periods": 0.0,
            "price_acceleration": 0.0,
            "is_reversal_candidate": 0.0,
        })

    # Add cross-timeframe features
    try:
        cross_tf_features = calculate_cross_timeframe_features(prices)
        # Filter out None values and convert to floats
        cross_tf_features_clean = {}
        for k, v in cross_tf_features.items():
            if v is not None:
                cross_tf_features_clean[k] = float(v)
            else:
                cross_tf_features_clean[k] = 0.0
        
        x.update(cross_tf_features_clean)
        logger.debug(f"Added cross-timeframe: sma_alignment={cross_tf_features_clean.get('sma_alignment')}, rsi_1h={cross_tf_features_clean.get('rsi_1h')}")
    except Exception as e:
        logger.warning(f"Failed to add cross-timeframe features: {e}")
        x.update({
            "price_vs_1h_sma": 0.0,
            "price_vs_4h_sma": 0.0,
            "price_vs_24h_sma": 0.0,
            "sma_alignment": 0.0,
            "rsi_1h": 0.0,
            "rsi_4h": 0.0,
            "momentum_1h": 0.0,
            "momentum_4h": 0.0,
        })


    agent = get_agent()
    predictions = {a: agent.models[a].predict_one(x) for a in agent.actions}
    action = agent.pull(x)
    logger.info(f"Predictions: {predictions}, Chosen action: {action}")


    reward = calculate_reward(
        action=action,
        price_now=price_now,
        portfolio=portfolio,
        fee=fee,
        price_24h_high=price_high,
        price_24h_low=price_low,
        rolling_mean_price=rolling_mean,
        returns_mean=returns_mean,
        returns_std=returns_std,
        prices_24h=prices,
        rsi_value=rsi_value,
        price_momentum_15m=price_momentum_15m,
    )

    agent.update(x, action, reward)

    logger.info(f"Chose action: {action}, Reward: {reward}")

    # Save the model.
    with open(HORIZON_MODEL_PATH, "wb") as f:
        pickle.dump(agent, f)

    # Save the states to a JSON file
    #save_state()

    timestamp = data.get("time") or datetime.now().isoformat()
    data_json = json.dumps(x)

    conn = sqlite3.connect("sol_prices.db")
    cursor = conn.cursor()
    try:
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
    except Exception as e:
        logger.error(f"Error logging to database: {e}")
    conn.commit()
    conn.close()
    # ========================================================================
    # DEPRECATED: Live Trading Execution (Experimental - Not Recommended)
    # ========================================================================
    # Live trading via Jupiter Ultra API has been disabled until thoroughly
    # tested and validated. The buy/sell/hold signals are still generated for
    # informational purposes, but no actual trades are executed.
    #
    # --- Order Execution Logic (COMMENTED OUT) ---
    # enable_live_trading = os.getenv("ENABLE_LIVE_TRADING", "N").upper() == "Y"
    # if bot is not None:
    #     if enable_live_trading and action in ("buy", "sell"):
    #         wallet_address = str(bot.public_key)
    #         balances = bot.get_balances(wallet_address)
    #         trade_amount = 1  # or use a dynamic amount
    #
    #         # Check if you have enough balance
    #         if balances is not None:
    #             if (action == "buy" and balances["USDC"] >= trade_amount) or (action == "sell" and balances["SOL"] >= trade_amount):
    #                 order_response = bot.get_order(trade_amount, action)
    #                 if order_response and order_response.get("transaction") and order_response.get("requestId"):
    #                     # Optionally inspect order_response for slippage, price, etc.
    #                     # Is price within acceptaple range of what Livecoinwatch is telling us to execute on (check for abnormality)?
    #                     execute_response = bot.execute_order(
    #                         order_response["requestId"],
    #                         order_response["transaction"]
    #                     )
    #                     logger.info(f"Trade executed: {execute_response}")
    #                 else:
    #                     logger.warning("Order response missing transaction or requestId, skipping execution.")
    #             else:
    #                 logger.info("Insufficient balance for trade, skipping execution.")
    #         else:
    #             logger.warning("Failed to fetch balances, skipping execution.")
    #     elif action in ("buy", "sell"):
    #         logger.info("ENABLE_LIVE_TRADING is not set to 'Y'. Skipping live trade execution.")
    # else:
    #     logger.error("Bot not loaded, skipping live trade execution.")
    # --- End Order Execution Logic ---
    return action, reward




# ============================================================================
# DEPRECATED: Price Prediction Model Loading and Training
# ============================================================================
# Price prediction has been disabled due to irrational value returns.
# This code is kept for reference but is not executed.
#
# Load or initialize model and metric
# if MODEL_PATH.exists() and METRIC_PATH.exists():
#     with open(MODEL_PATH, "rb") as f:
#         model = pickle.load(f)
#     logger.info("Loaded existing online learning model from disk.")
#     with open(METRIC_PATH, "rb") as f:
#         metric = pickle.load(f)
#     logger.info("Loaded existing metric from disk.")
# else:
#     model = preprocessing.StandardScaler() | linear_model.LinearRegression()
#     metric = metrics.MAE()

# def train_online_model(data):
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
    #logger.info("Training model with features:")
    # for k, v in features.items():
    #     logger.info(f"{k}: {v} ({type(v)})")
    #     if v is None:
    #         logger.warning(f"Feature '{k}' is None!")
    #     elif isinstance(v, float) and (v != v):  # NaN
    #         logger.warning(f"Feature '{k}' is NaN!")

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
    # with open(MODEL_PATH, "wb") as f:
    #     pickle.dump(model, f)
    # with open(METRIC_PATH, "wb") as f:
    #     pickle.dump(metric, f)


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
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            fee_pct REAL,
            fee_usd REAL,
            net_value_usd REAL,
            portfolio_usd_balance REAL,
            portfolio_sol_balance REAL,
            portfolio_equity REAL,
            avg_entry_price REAL,
            realized_pnl REAL,
            price_24h_high REAL,
            price_24h_low REAL,
            rolling_mean_price REAL,
            returns_mean REAL,
            returns_std REAL
        )
        ''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sol_balance REAL,
            usd_balance REAL,
            total_cost_basis REAL,
            realized_pnl REAL,
            portfolio_equity REAL,
            price_now REAL
        )
        ''')

        # Create indexes for performance optimization
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_prices_timestamp ON sol_prices(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_prices_rate ON sol_prices(rate)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_prices_timestamp_rate ON sol_prices(timestamp, rate)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bandit_logs_timestamp ON bandit_logs(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bandit_logs_action ON bandit_logs(action)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bandit_logs_reward ON bandit_logs(reward)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_bandit_logs_timestamp_action ON bandit_logs(timestamp, action)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_price ON trades(price)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp_action ON trades(timestamp, action)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_predictions_timestamp ON sol_predictions(timestamp)')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_sol_predictions_created_at ON sol_predictions(created_at)')

        self.conn.commit()
        logger.info("Database initialized successfully with indexes")

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
#########################################################################
#                    LOOP                                               #
#########################################################################
            # Here is the place in the loop where we execute other actions on the "data" variable, which is our dictionary of price data.
            # ========================================================================
            # DEPRECATED: Price Prediction Training (Returns Irrational Values)
            # ========================================================================
            # Price prediction has been disabled. Buy/sell/hold signals remain active.
            # train_online_model(data)
            
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

    def get_price_history(self, time_threshold=None, limit=None):
        """
        Retrieve price history from database based on a time threshold.
        Optimized for performance with proper indexing and limiting.
        
        Args:
            time_threshold: ISO format timestamp string - only fetch records >= this time
            limit: Maximum number of records to return (for pagination/performance)
        """
        # Ensure the connection and cursor are available
        if not hasattr(self, 'conn') or not hasattr(self, 'cursor'):
            logger.error("Database connection not initialized in SOLPriceFetcher.")
            # Attempt to re-initialize, or handle error appropriately
            self.init_database() # Or return an empty list / raise an exception

        # Optimized query - only select columns we actually need for the chart
        # Using timestamp index for faster queries
        base_query = "SELECT timestamp, rate FROM sol_prices"
        conditions = []
        params = []
        
        if time_threshold:
            conditions.append("timestamp >= ?")
            params.append(time_threshold)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Order by timestamp for chronological display
        base_query += " ORDER BY timestamp ASC"
        
        # Add limit if specified (useful for large datasets)
        if limit:
            base_query += f" LIMIT {limit}"
        
        try:
            self.cursor.execute(base_query, params)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Database error in get_price_history: {e}")
            return []

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

    def get_bandit_stats(self, db_path="sol_prices.db", limit=5000):
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(f'''
                SELECT * FROM bandit_logs ORDER BY timestamp DESC LIMIT {limit}
            ''', conn)
            conn.close()
        except Exception as e:
            return {"error": f"Failed to load data: {str(e)}"}

        # Parse and enrich
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["features"] = df["data_json"].apply(json.loads)

        # Chosen prediction confidence
        df["chosen_prediction"] = df.apply(
            lambda row: row.get(f"prediction_{row['action']}", None), axis=1
        )

        # Optional: Calculate regret (how far the chosen prediction was from the best one)
        def regret(row: pd.Series) -> float:
            preds = [row["prediction_buy"], row["prediction_sell"], row["prediction_hold"]]
            chosen_pred = row.get(f"prediction_{row['action']}", None)
            if chosen_pred is None:
                return 0.0
            return max(preds) - chosen_pred


        df["regret"] = df.apply(regret, axis=1)

        # Summary stats
        summary = {
            "num_rows": len(df),
            "avg_reward": round(df["reward"].mean(), 4),
            "std_reward": round(df["reward"].std(), 4),
            "action_counts": df["action"].value_counts().to_dict(),
            "avg_reward_by_action": df.groupby("action")["reward"].mean().round(4).to_dict(),
            "avg_regret": round(df["regret"].mean(), 4),
        }

        # Optional: Top 5 best trades
        top_trades = df.sort_values("reward", ascending=False).head(5)
        summary["top_rewards"] = top_trades[["timestamp", "action", "reward"]].to_dict(orient="records")

        return summary

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