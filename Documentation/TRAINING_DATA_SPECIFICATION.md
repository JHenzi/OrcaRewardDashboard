# Training Data Specification

This document describes the exact features used to train the RL agent, with real-world examples.

## Overview

The RL agent receives a **state vector** encoding:
- **Price data**: Multi-resolution time-series (5min, 1hr, daily)
- **Technical indicators**: Stationary ratios and metrics
- **News data**: Embeddings, sentiment, and age
- **Position state**: Current holdings, P&L, time since last trade
- **Time features**: Hour, day of week, weekend flag
- **External markets**: BTC and S&P 500 prices (NEW)

**Total feature dimensions**: ~1,500+ features

---

## 1. Price Features (69 features)

### 1.1 Short-Term Log Returns (60 features)
- **What**: Log returns from last 60 price points (5-minute intervals = 5 hours)
- **Format**: `ln(price_t / price_{t-1})` for each of 60 points
- **Example**:
  ```python
  prices = [125.0, 125.2, 125.1, 125.5, ...]  # 60 prices
  returns = [0.0016, -0.0008, 0.0032, ...]     # 60 log returns
  ```
- **Why**: Stationary (no trend), better for neural networks than raw prices
- **Missing data**: Padded with zeros if < 60 points available

### 1.2 Technical Indicators (9 features)

#### Price/SMA Ratios (3 features)
- **What**: Current price divided by moving averages
- **Format**: `current_price / sma_period`
- **Example**:
  ```python
  current_price = 125.0
  sma_1h = 124.5   # 12-point average (1 hour)
  sma_4h = 124.0   # 48-point average (4 hours)
  sma_24h = 123.0  # 288-point average (24 hours)
  
  features = [
    125.0 / 124.5 = 1.004,  # Price/SMA_1h
    125.0 / 124.0 = 1.008,  # Price/SMA_4h
    125.0 / 123.0 = 1.016,  # Price/SMA_24h
  ]
  ```
- **Why**: Stationary (centered around 1.0), detects trend direction
- **Missing**: Defaults to 1.0 if SMA unavailable

#### SMA Ratios (2 features)
- **What**: Short-term vs long-term trend
- **Format**: `sma_short / sma_long`
- **Example**:
  ```python
  sma_1h / sma_4h = 124.5 / 124.0 = 1.004   # Short vs Medium
  sma_4h / sma_24h = 124.0 / 123.0 = 1.008  # Medium vs Long
  ```
- **Why**: Detects trend acceleration/deceleration

#### RSI (1 feature)
- **What**: Relative Strength Index (0-100), normalized to 0-1
- **Format**: `rsi / 100.0`
- **Example**:
  ```python
  rsi = 65.0  # Overbought territory
  feature = 65.0 / 100.0 = 0.65
  ```
- **Why**: Bounded indicator (0-1), detects overbought/oversold
- **Missing**: Defaults to 0.5 (neutral)

#### Volatility (1 feature)
- **What**: Standard deviation normalized by price
- **Format**: `std_dev / current_price`
- **Example**:
  ```python
  std_dev = 2.5
  current_price = 125.0
  feature = 2.5 / 125.0 = 0.02  # 2% volatility
  ```
- **Why**: Stationary (percentage-based), measures risk
- **Missing**: Defaults to 0.0

#### Momentum (1 feature)
- **What**: 15-minute price change percentage
- **Format**: `((price_now - price_15m_ago) / price_15m_ago) * 100`
- **Example**:
  ```python
  price_now = 125.0
  price_15m_ago = 124.5
  momentum = ((125.0 - 124.5) / 124.5) * 100 = 0.4%
  ```
- **Why**: Short-term momentum signal
- **Missing**: Defaults to 0.0

#### Percent Change (1 feature)
- **What**: Total change over price window, normalized
- **Format**: `percent_change / 100.0`
- **Example**:
  ```python
  percent_change = 2.5  # 2.5% increase
  feature = 2.5 / 100.0 = 0.025
  ```
- **Why**: Overall trend direction
- **Missing**: Defaults to 0.0

### ⚠️ Missing Features

#### Volume (NOT AVAILABLE)
- **Status**: We don't have volume data from our price source
- **Impact**: Missing volume-based signals:
  - Volume confirmation (price up + volume up = strong signal)
  - Volume divergence (price up + volume down = weak signal)
  - VWAP (Volume Weighted Average Price)
- **Workaround**: Currently defaults to 0/null
- **Future**: Could add if we switch to a data provider with volume

---

## 2. Multi-Resolution Price Data (90 features)

### 2.1 Medium-Term (60 features)
- **What**: Log returns from last 60 hourly prices (2.5 days)
- **Format**: Same as short-term but hourly intervals
- **Example**:
  ```python
  hourly_prices = [124.0, 124.5, 125.0, 125.2, ...]  # 60 hourly prices
  returns = [0.0040, 0.0040, 0.0016, ...]            # 60 hourly log returns
  ```
- **Why**: Captures medium-term trends (24h predictions)

### 2.2 Long-Term (30 features)
- **What**: Log returns from last 30 daily prices (1 month)
- **Format**: Same as short-term but daily intervals
- **Example**:
  ```python
  daily_prices = [120.0, 121.0, 122.0, 123.0, ...]  # 30 daily prices
  returns = [0.0083, 0.0082, 0.0081, ...]          # 30 daily log returns
  ```
- **Why**: Captures macro trends (24h+ predictions)

---

## 3. News Features (7,680 features)

### 3.1 News Embeddings (7,680 features = 20 headlines × 384 dims)
- **What**: Semantic embeddings from sentence-transformers (all-MiniLM-L6-v2)
- **Format**: 384-dimensional vectors per headline
- **Example**:
  ```python
  headline = "Solana Network Experiences Outage"
  embedding = [0.123, -0.456, 0.789, ...]  # 384 floats
  
  # 20 headlines = 20 × 384 = 7,680 features
  embeddings = [
    [0.123, -0.456, ...],  # Headline 1
    [0.234, -0.567, ...],  # Headline 2
    ...
    [0.000, 0.000, ...],   # Padding (if < 20 headlines)
  ]
  ```
- **Why**: Captures semantic meaning (similar headlines cluster)
- **Missing**: Padded with zeros if < 20 headlines

### 3.2 News Sentiment (20 features)
- **What**: Sentiment score per headline (-1 to +1)
- **Format**: One float per headline
- **Example**:
  ```python
  sentiment_scores = [
    0.8,   # "Solana Price Surges 10%" (bullish)
    -0.6,  # "Network Outage Causes Concern" (bearish)
    0.0,   # "Solana Foundation Announces Partnership" (neutral)
    ...
    0.0,   # Padding
  ]
  ```
- **Why**: Direct sentiment signal
- **Missing**: Defaults to 0.0 (neutral)

### 3.3 News Age (20 features)
- **What**: Age of each news item in normalized minutes (0=fresh, 1=1 day old)
- **Format**: One float per headline
- **Example**:
  ```python
  news_ages = [
    0.1,   # Published 2.4 hours ago (144 min / 1440 = 0.1)
    0.5,   # Published 12 hours ago
    0.9,   # Published 21.6 hours ago
    ...
    1.0,   # Padding (old, ignored)
  ]
  ```
- **Why**: Time decay - older news matters less
- **Missing**: Defaults to 0.5 (medium age)

### 3.4 News Clusters (20 features)
- **What**: Topic cluster ID per headline (-1 = no cluster)
- **Format**: Integer IDs
- **Example**:
  ```python
  cluster_ids = [
    5,   # "Network" cluster
    2,   # "Price" cluster
    -1,  # No cluster
    ...
    -1,  # Padding
  ]
  ```
- **Why**: Groups related news
- **Missing**: Defaults to -1

---

## 4. Position Features (5 features)

### 4.1 Position Ratio
- **What**: Position value / portfolio value
- **Format**: `(position_size * current_price) / portfolio_value`
- **Example**:
  ```python
  position_size = 10.0  # SOL
  current_price = 125.0
  portfolio_value = 10000.0
  
  position_ratio = (10.0 * 125.0) / 10000.0 = 0.125  # 12.5% of portfolio
  ```
- **Why**: Risk exposure

### 4.2 Position Value
- **What**: Absolute position value in thousands
- **Format**: `(position_size * current_price) / 1000.0`
- **Example**:
  ```python
  position_value = (10.0 * 125.0) / 1000.0 = 1.25  # $1,250
  ```
- **Why**: Position size signal

### 4.3 Entry Price Ratio
- **What**: Current price / entry price
- **Format**: `current_price / entry_price`
- **Example**:
  ```python
  entry_price = 120.0
  current_price = 125.0
  
  price_ratio = 125.0 / 120.0 = 1.042  # 4.2% profit
  ```
- **Why**: Profit/loss signal
- **Missing**: Defaults to 1.0 (no position)

### 4.4 Unrealized P&L
- **What**: Unrealized profit/loss normalized by portfolio
- **Format**: `unrealized_pnl / portfolio_value`
- **Example**:
  ```python
  unrealized_pnl = 500.0
  portfolio_value = 10000.0
  
  pnl_ratio = 500.0 / 10000.0 = 0.05  # 5% gain
  ```
- **Why**: Profit/loss signal

### 4.5 Time Since Last Trade
- **What**: Minutes since last trade, normalized to hours
- **Format**: `time_since_last_trade / 60.0`
- **Example**:
  ```python
  time_since_last_trade = 120  # 2 hours
  feature = 120 / 60.0 = 2.0
  ```
- **Why**: Trading frequency signal

---

## 5. Time Features (4 features)

### 5.1 Hour of Day
- **What**: Hour (0-23), normalized to 0-1
- **Format**: `hour / 24.0`
- **Example**:
  ```python
  hour = 14  # 2 PM
  feature = 14 / 24.0 = 0.583
  ```
- **Why**: Market hours patterns (US market opens at 9:30 AM ET)

### 5.2 Day of Week
- **What**: Weekday (0=Monday, 6=Sunday), normalized to 0-1
- **Format**: `weekday / 7.0`
- **Example**:
  ```python
  weekday = 2  # Wednesday
  feature = 2 / 7.0 = 0.286
  ```
- **Why**: Weekend/weekday patterns

### 5.3 Minute of Day
- **What**: Total minutes since midnight, normalized
- **Format**: `(hour * 60 + minute) / 1440.0`
- **Example**:
  ```python
  hour = 14
  minute = 30
  feature = (14 * 60 + 30) / 1440.0 = 0.604
  ```
- **Why**: Intraday patterns

### 5.4 Is Weekend
- **What**: Binary flag (1=weekend, 0=weekday)
- **Format**: `1.0 if weekday >= 5 else 0.0`
- **Example**:
  ```python
  weekday = 6  # Sunday
  feature = 1.0
  ```
- **Why**: Weekend trading patterns (lower volume)

---

## 6. External Market Data (60 features) - ⚠️ NOT USED IN TRAINING YET

### ⚠️ Current Status: Infrastructure Ready, But No Historical Data

**We do NOT have historical BTC and S&P 500 data aligned with our SOL price history.**

- State encoder supports these features (will pad with zeros)
- Training currently uses **zeros** for all external market features
- Model learns to ignore these features during training
- **Will be enabled when historical data is available**

### 6.1 BTC Price Returns (30 features) - Currently Zeros
- **What**: Log returns from last 30 BTC prices (5-minute intervals = 2.5 hours)
- **Format**: Same as SOL short-term returns
- **Example** (when available):
  ```python
  btc_prices = [45000.0, 45100.0, 45050.0, ...]  # 30 BTC prices
  btc_returns = [0.0022, -0.0011, ...]            # 30 log returns
  ```
- **Why**: BTC often leads SOL price movements (correlation ~0.7-0.9)
- **Current**: All zeros (no historical data available)

### 6.2 S&P 500 Price Returns (30 features) - Currently Zeros
- **What**: Log returns from last 30 S&P 500 prices (5-minute intervals = 2.5 hours)
- **Format**: Same as SOL short-term returns
- **Example** (when available):
  ```python
  sp500_prices = [4500.0, 4505.0, 4502.0, ...]  # 30 S&P 500 prices
  sp500_returns = [0.0011, -0.0007, ...]        # 30 log returns
  ```
- **Why**: Risk-on/risk-off sentiment affects crypto (correlation ~0.3-0.5)
- **Current**: All zeros (no historical data available)

---

## Complete Feature Summary

| Feature Group | Dimensions | Description |
|--------------|------------|-------------|
| SOL Short-Term Returns | 60 | 5-min log returns (5 hours) |
| SOL Technical Indicators | 9 | Price/SMA ratios, RSI, volatility, momentum |
| SOL Medium-Term Returns | 60 | Hourly log returns (2.5 days) |
| SOL Long-Term Returns | 30 | Daily log returns (1 month) |
| News Embeddings | 7,680 | 20 headlines × 384 dims |
| News Sentiment | 20 | Sentiment scores (-1 to +1) |
| News Age | 20 | Age in normalized minutes |
| News Clusters | 20 | Topic cluster IDs |
| Position Features | 5 | Position ratio, value, entry ratio, P&L, time |
| Time Features | 4 | Hour, weekday, minute, weekend flag |
| BTC Returns | 30 | BTC 5-min log returns (2.5 hours) - **Currently zeros** |
| S&P 500 Returns | 30 | S&P 500 5-min log returns (2.5 hours) - **Currently zeros** |
| **TOTAL** | **~1,992** | **Complete state vector** |

**Note**: External market features (BTC, S&P 500) are currently zeros during training because we don't have historical data aligned with SOL price history.

---

## Real-World Example

```python
# Timestamp: 2024-01-15 14:30:00 (Monday, 2:30 PM)

state = {
    # Price features (69)
    "price": np.array([
        # 60 log returns
        0.0016, -0.0008, 0.0032, ...,  # Last 5 hours of 5-min returns
        # 9 technical indicators
        1.004,   # Price/SMA_1h
        1.008,   # Price/SMA_4h
        1.016,   # Price/SMA_24h
        1.004,   # SMA_1h/SMA_4h
        1.008,   # SMA_4h/SMA_24h
        0.65,    # RSI (normalized)
        0.02,    # Volatility
        0.4,     # Momentum (%)
        0.025,   # Percent change
    ]),
    
    # Multi-resolution (90)
    "price_medium": np.array([...]),  # 60 hourly returns
    "price_long": np.array([...]),    # 30 daily returns
    
    # News (7,740)
    "news_embeddings": np.array([
        [0.123, -0.456, ...],  # Headline 1 (384 dims)
        [0.234, -0.567, ...],  # Headline 2 (384 dims)
        ...
    ]),  # Shape: (20, 384)
    "news_sentiment": np.array([0.8, -0.6, 0.0, ...]),  # 20 scores
    "news_age": np.array([0.1, 0.5, 0.9, ...]),        # 20 ages
    "news_clusters": np.array([5, 2, -1, ...]),         # 20 cluster IDs
    
    # Position (5)
    "position": np.array([
        0.125,  # Position ratio (12.5% of portfolio)
        1.25,   # Position value ($1,250)
        1.042,  # Entry price ratio (4.2% profit)
        0.05,   # Unrealized P&L (5% gain)
        2.0,    # Time since last trade (2 hours)
    ]),
    
    # Time (4)
    "time": np.array([
        0.604,  # Hour of day (14:30 = 0.604)
        0.286,  # Day of week (Wednesday = 0.286)
        0.604,  # Minute of day
        0.0,    # Is weekend (no)
    ]),
    
    # External markets (60)
    "btc_returns": np.array([0.0022, -0.0011, ...]),      # 30 BTC returns
    "sp500_returns": np.array([0.0011, -0.0007, ...]),    # 30 S&P 500 returns
}
```

---

## Missing Data Handling

### Volume
- **Status**: Not available from current data source
- **Default**: 0/null (not included in features)
- **Impact**: Missing volume confirmation signals
- **Future**: Add if switching data providers

### External Markets (BTC, S&P 500)
- **Status**: NEW - Being added
- **Default**: Zeros if unavailable
- **Impact**: Missing macro context
- **Solution**: Fetch from free APIs (CoinGecko, Alpha Vantage)

### News
- **Default**: Padded with zeros if < 20 headlines
- **Impact**: Model learns to ignore zero-padded news

### Price History
- **Default**: Padded with zeros if insufficient history
- **Impact**: Model learns to handle short history

---

## Notes

1. **All features are normalized** to prevent scale issues
2. **Stationary features only** (no raw prices) for better training
3. **Multi-resolution** captures short/medium/long-term trends
4. **Time decay** for news (older news matters less)
5. **External markets** provide macro context (BTC, S&P 500)
