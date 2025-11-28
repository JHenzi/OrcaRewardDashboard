# Deprecated Features & Historical Documentation

> **⚠️ This document contains information about features that have been deprecated or removed from the project.**
> 
> **Last Updated:** 2025-11-27

This document preserves historical information about deprecated features for reference purposes. These features are no longer actively maintained or supported.

---

## Table of Contents

1. [Price Prediction System](#price-prediction-system)
2. [Contextual Bandit Trading Agent](#contextual-bandit-trading-agent)
3. [Automated Trading Bot](#automated-trading-bot)
4. [Deprecated API Endpoints](#deprecated-api-endpoints)
5. [Historical Project Description](#historical-project-description)

---

## Price Prediction System

### Status: **DEPRECATED** (Disabled)

The online price prediction feature has been **deprecated and disabled** as it was returning irrational values. Historical predictions may still be visible in the database and UI, but no new predictions are being generated.

### What It Was

- **Online Price Prediction**: Utilized the `river` library for online machine learning
- A linear regression model (`river.linear_model.LinearRegression`) combined with a standard scaler (`river.preprocessing.StandardScaler`) was trained incrementally with new price data
- The model predicted the next SOL price point

### Why It Was Deprecated

The predictions were returning irrational values that did not align with actual market behavior. The model was not reliable enough for production use.

### Current Status

- Historical predictions remain in the database for reference
- No new predictions are generated
- The prediction code remains in the codebase but is not executed
- The `/api/latest-prediction` endpoint still exists but returns historical data only

### Migration Path

The project now focuses on **RSI-based trading signals** instead of price predictions. See the current README.md for information about the active RSI indicator system.

---

## Contextual Bandit Trading Agent

### Status: **REMOVED** from SOL Tracker UI

The contextual bandit algorithm has been **removed from the SOL Tracker page** and replaced with RSI-based signals. The bandit code may still exist in the codebase but is no longer displayed in the UI.

### What It Was

A reinforcement learning agent that learned to buy, sell, or hold SOL based on:
- Momentum and trend signals
- Statistical features (Sharpe ratio, rolling mean, price deviations)
- Profit and loss from past trades

### Implementation Details

- Used separate online learning models (typically Hoeffding Tree Regressor) for each potential action
- Learned to predict the expected reward based on the current market context
- The action with the highest predicted reward was chosen, with some randomness (epsilon-greedy strategy) to encourage exploration
- Bandit action logs (including chosen action, predicted rewards, and context) were stored in the `sol_prices.db` SQLite database

### Why It Was Removed

The SOL Tracker page was redesigned to focus on clear, interpretable RSI-based signals rather than the more complex bandit algorithm. RSI provides more transparent and widely-understood trading signals.

### Current Status

- Bandit code may still exist in `sol_price_fetcher.py` but is not displayed in the UI
- The `/api/latest-bandit-action` endpoint may still exist but is not actively used
- Historical bandit logs remain in the database

### Migration Path

The project now uses **RSI (Relative Strength Index)** for trading signals:
- **RSI < 30**: Buy signal (oversold conditions)
- **RSI > 70**: Sell signal (overbought conditions)
- **RSI 30-70**: Hold signal (neutral conditions)

---

## Automated Trading Bot

### Status: **DEPRECATED** (Permanently Disabled)

The Jupiter Ultra Trading Bot integration has been **deprecated and permanently disabled**. The framework for automated trading exists in the codebase but is currently commented out and will not execute trades, even if enabled in the environment variables.

### What It Was

An experimental automated trading system that could:
- Fetch quotes from Jupiter Ultra API
- Sign transactions locally using wallet keypair
- Execute swaps between SOL and USDC
- Log all trades to `sol_trades.db` database

### Trading Bot Features (DEPRECATED - Not Active)

> **⚠️ All trading bot features are currently disabled. The code exists but is commented out.**

- ~~**Automated Trading via Jupiter Ultra API:**~~
  ~~The system could fetch quotes, sign transactions, and execute swaps between SOL and USDC using the Jupiter Ultra API.~~ **(DISABLED)**
- ~~**Trade Logging:**~~
  ~~All attempted trades and execution results were logged to a dedicated `sol_trades.db` SQLite database for audit and analysis.~~ **(DISABLED - No trades are executed)**
- ~~**Balance Checks:**~~
  ~~The bot checked wallet balances before attempting trades.~~ **(DISABLED)**
- ~~**.env Controlled:**~~
  ~~Trading was only enabled if the required private key was present in `.env` and `ENABLE_LIVE_TRADING=Y` was set.~~ **(DISABLED - Trading code is commented out regardless of .env settings)**

### How It Worked (Framework - DISABLED)

> **⚠️ This entire workflow is currently disabled. The code is commented out and will not execute.**

~~1. **Balance Check:**~~
   ~~Before trading, the bot checked if there was enough SOL or USDC to execute the desired action.~~
~~2. **Quote Fetch:**~~
   ~~The bot fetched a quote/order from Jupiter Ultra for the intended swap.~~
~~3. **Transaction Signing:**~~
   ~~The unsigned transaction from the quote was signed locally using the wallet keypair.~~
~~4. **Order Execution:**~~
   ~~The signed transaction and request ID were sent to Jupiter Ultra's `/execute` endpoint.~~
~~5. **Logging:**~~
   ~~All order and execution details were stored in `sol_trades.db`.~~

### Enabling/Disabling Trading (DISABLED)

> **⚠️ Trading is permanently disabled. Even if you set `ENABLE_LIVE_TRADING=Y` in your `.env` file, no trades will be executed. The trading code has been commented out.**

- Trading is **permanently disabled** (code is commented out).
- ~~To enable live trading, set the following in your `.env` file:~~
  ~~```
  ENABLE_LIVE_TRADING=Y
  SOL_PRIVATE_KEY=your_private_key_here
  SOL_PUBLIC_KEY=your_public_key_here
  ```~~ **(This no longer has any effect)**
- The trading bot code remains in the repository for reference but will not execute trades regardless of environment variable settings.

### ⚠️ Disclaimer

- **Trading functionality is currently disabled and will not execute.**
- **The trading code remains in the repository for reference but is commented out.**
- **Do not attempt to re-enable trading without thorough testing and validation.**
- The authors are not responsible for any financial loss or unintended trades.

### Related Files

- `trading_bot.py`: Jupiter ULTRA Trading API function class (commented out)
- `bandit_state.json`: Stores the state of the bandit portfolio simulation (may still exist but unused)

---

## Deprecated API Endpoints

### Get Latest Price Prediction (DEPRECATED)

- **URL:** `/api/latest-prediction`
- **Method:** `GET`
- **Status:** Returns historical data only - no new predictions generated
- **Description:** Retrieves the most recent price prediction from historical data. **⚠️ Price prediction has been disabled - no new predictions are being generated. This endpoint returns historical data only.**

**Success Response (200 OK):**
```json
{
  "timestamp": "2023-10-27T10:30:00.123Z",
  "predicted_rate": 135.5,
  "actual_rate": 135.25,
  "error": 0.25,
  "mae": 0.15,
  "created_at": "2023-10-27T10:35:05.456Z",
  "deprecated": true,
  "message": "Price prediction has been disabled. This is historical data only."
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "No predictions found",
  "deprecated": true,
  "message": "Price prediction has been disabled. No historical data available."
}
```

### Get Latest Bandit Action (DEPRECATED)

- **URL:** `/api/latest-bandit-action`
- **Method:** `GET`
- **Status:** May still exist but not actively used
- **Description:** Retrieves the most recent action (buy/sell/hold) decided by the contextual bandit.

**Success Response (200 OK):**
```json
{
  "timestamp": "2023-10-27T10:30:00.123Z",
  "action": "buy",
  "reward": 0.05,
  "prediction_buy": 0.12,
  "prediction_sell": -0.05,
  "prediction_hold": 0.01,
  "data_json": {
    /* Contextual features used for this decision */
  },
  "created_at": "2023-10-27T10:35:05.789Z"
}
```

---

## Historical Project Description

### Original Project Vision

This project began as a simple Orca rewards tracker, but evolved into a Solana trading assistant, price predictor, and contextual bandit strategy simulator.

### Example Strategy (Historical)

_"Deposit a large amount into an [Orca.so](https://orca.so/) SOL/USDC pool. These pools are popular and generally pay consistent rewards. Use this application to track returns, and instead of reinvesting those rewards into the pool, allow the contextual bandit to reinvest them into SOL when conditions are favorable. For example, a liquidity range of $130 to $280 may be appropriate for 2024-2025. When SOL trades within that range, liquidity fees are generated, and this app helps determine whether to hold, buy, or exit into USDC."_

**Note:** This strategy referenced the contextual bandit, which has been replaced with RSI-based signals. The core concept of tracking Orca rewards and making trading decisions remains, but now uses RSI indicators instead.

### Historical Features

- ~~Real-time tracking of COLLECT_FEES events from your Solana wallet~~ (Now uses 2-hour refresh intervals on free API tier)
- ~~Dynamic credit usage of the LiveCoinWatch API - scaling credit usage dynamically as the day progresses~~ (Resolved)
- ~~Split price and bandit actions into two charts~~ (Resolved with TradingView Lightweight Charts)

---

## Migration Notes

### For Developers

If you're working with historical data or need to understand the deprecated features:

1. **Price Predictions**: Historical data remains in `sol_prices.db` in the `sol_predictions` table
2. **Bandit Logs**: Historical data remains in `sol_prices.db` in the `bandit_logs` table
3. **Trading Bot Code**: See `trading_bot.py` (commented out) for reference implementation
4. **Bandit State**: `bandit_state.json` may still exist but is not actively used

### For Users

- The current system uses **RSI-based signals** for trading recommendations
- All deprecated features have been replaced with more reliable, transparent alternatives
- Historical data is preserved but no longer actively generated

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-27  
**Maintained By:** Development Team

