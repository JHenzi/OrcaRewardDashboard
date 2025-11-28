# Deprecated Features & Historical Documentation

> **⚠️ This document contains information about features that have been deprecated or removed from the project.**
> 
> **Last Updated:** 2025-01-XX

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

### Status: **DEPRECATED** (Removed from UI, Replaced with RSI Signals)

The contextual bandit algorithm has been **deprecated and removed from the SOL Tracker page**. It was replaced with RSI-based signals, which provide more transparent and interpretable trading recommendations. The bandit code may still exist in the codebase but is no longer displayed in the UI or actively maintained.

---

### What It Really Did

The contextual bandit was a **multi-armed bandit** reinforcement learning system that attempted to learn optimal trading actions (buy, sell, or hold) by exploring different strategies and exploiting the ones that appeared most profitable.

#### Core Mechanism

1. **Three Separate Models**: The system maintained three independent online learning models (typically `HoeffdingAdaptiveTreeRegressor` from the `river` library), one for each action:
   - `model_buy`: Predicted expected reward for buying SOL
   - `model_sell`: Predicted expected reward for selling SOL
   - `model_hold`: Predicted expected reward for holding current position

2. **Contextual Features**: At each decision point, the bandit observed a feature vector including:
   - **Momentum signals**: Recent price changes, rate of change
   - **Trend indicators**: Moving averages, price vs. SMA ratios
   - **Statistical features**: Sharpe ratio, rolling standard deviation, price deviations from mean
   - **Portfolio state**: Current position size, unrealized P&L, time since last trade
   - **Market conditions**: Volatility metrics, recent price range

3. **Reward Prediction**: Each model predicted the expected reward (profit/loss) for its corresponding action given the current context

4. **Action Selection**: The system used an **epsilon-greedy** strategy:
   - **Exploitation (1-ε)**: Choose the action with the highest predicted reward
   - **Exploration (ε)**: Randomly select an action to discover new strategies
   - The epsilon value typically decayed over time to shift from exploration to exploitation

5. **Learning Loop**: After taking an action, the system:
   - Waited for a time period (e.g., 1 hour, 24 hours)
   - Calculated the actual reward based on price movement
   - Updated the corresponding model with the (context, reward) pair
   - Adjusted its predictions for future similar contexts

6. **Portfolio Simulation**: The bandit maintained a simulated portfolio in `bandit_state.json` to track:
   - Current position (SOL amount, entry price)
   - Portfolio value over time
   - Trade history and outcomes
   - Cumulative returns

#### Technical Implementation

- **Online Learning**: Models updated incrementally with each new data point (no batch retraining)
- **Adaptive Trees**: Used Hoeffding trees that could adapt to concept drift (changing market conditions)
- **Feature Engineering**: Computed rolling statistics, momentum indicators, and normalized features
- **Reward Calculation**: Rewards were based on log returns, transaction costs, and risk penalties
- **Logging**: All decisions, predicted rewards, and outcomes were stored in `sol_prices.db` for analysis

---

### Why It Was Fun to Build

The contextual bandit was an **intellectually stimulating project** that combined several interesting concepts:

1. **Reinforcement Learning in Practice**: It was a real-world application of RL concepts (exploration vs. exploitation, credit assignment, online learning) outside of academic exercises

2. **Adaptive Learning**: Watching the models learn and adapt in real-time was fascinating - you could see the predictions change as the system observed more market data

3. **Multi-Armed Bandit Problem**: The classic exploration/exploitation tradeoff is a fundamental problem in machine learning, and implementing it for trading was a natural fit

4. **Feature Engineering Challenge**: Designing contextual features that might predict profitable trades required thinking about market dynamics, technical analysis, and statistical properties

5. **Simulation & Experimentation**: The portfolio simulation allowed for safe experimentation - you could test strategies without risking real capital

6. **Data-Driven Decisions**: The idea of an algorithm that "learns" from market data and makes autonomous decisions felt like building a trading AI

7. **Online Learning Appeal**: The ability to update models incrementally without retraining from scratch made it feel like a "living" system that evolved with the market

8. **Transparency**: Unlike black-box neural networks, you could inspect the tree structures and understand (to some degree) what the models were learning

---

### Why It's Not a Good Trading Advice Model

Despite being an engaging technical project, the contextual bandit had **fundamental limitations** that made it unsuitable as a reliable trading advice system:

#### 1. **Overfitting to Noise**

- **Problem**: Financial markets are extremely noisy, and the bandit could easily learn spurious patterns that don't generalize
- **Example**: If SOL happened to rise after a specific combination of features occurred 3 times, the model might "learn" that pattern, even if it was just random chance
- **Impact**: The system could appear to perform well in backtesting but fail in live trading

#### 2. **Concept Drift & Non-Stationarity**

- **Problem**: Market conditions change constantly (bull markets, bear markets, volatility regimes, macro events). What worked yesterday may not work tomorrow
- **Example**: A strategy that worked during a trending market might fail during a ranging market, but the bandit would continue using it until it accumulated enough negative feedback
- **Impact**: The system could be slow to adapt to regime changes, leading to losses during transitions

#### 3. **Limited Context Window**

- **Problem**: The bandit only considered recent price data and simple technical indicators. It couldn't incorporate:
  - News events and sentiment
  - Macroeconomic factors
  - Market microstructure
  - Cross-asset correlations
  - Long-term trends
- **Impact**: The system made decisions based on incomplete information

#### 4. **Reward Signal Delay**

- **Problem**: The bandit learned from delayed rewards (waiting 1h/24h to see outcomes). This created several issues:
  - **Credit Assignment**: Hard to know which features actually caused the outcome
  - **Slow Learning**: Takes many trades to learn, but markets change faster
  - **Sparse Feedback**: Most of the time, you're waiting, not learning
- **Impact**: The system learned slowly and might miss short-term opportunities

#### 5. **Epsilon-Greedy Limitations**

- **Problem**: The exploration strategy was naive:
  - Random exploration doesn't prioritize promising strategies
  - Fixed epsilon decay might explore too much early (wasting capital) or too little later (missing new patterns)
  - No uncertainty quantification - couldn't distinguish between "confident" and "uncertain" predictions
- **Impact**: Inefficient use of capital and missed opportunities

#### 6. **No Risk Management**

- **Problem**: The bandit optimized for expected reward but didn't explicitly consider:
  - Drawdown limits
  - Position sizing based on confidence
  - Correlation with portfolio
  - Tail risk
- **Impact**: Could take large positions in uncertain situations, leading to catastrophic losses

#### 7. **Single-Asset Focus**

- **Problem**: The bandit only considered SOL price action, ignoring:
  - Market-wide conditions (crypto market cap, BTC correlation)
  - Sector trends
  - Liquidity conditions
- **Impact**: Missed important contextual information

#### 8. **Lack of Interpretability**

- **Problem**: While tree models are more interpretable than neural networks, understanding why the bandit made a specific decision required:
  - Inspecting tree structures (complex)
  - Analyzing feature importance (not always clear)
  - No human-readable "rules" or explanations
- **Impact**: Users couldn't understand or trust the recommendations

#### 9. **Sample Size Requirements**

- **Problem**: Online learning requires many examples to converge, but:
  - Each trade provides only one data point
  - Markets change before enough data accumulates
  - Need thousands of trades for statistical significance, but market conditions change weekly/monthly
- **Impact**: The system might never converge to optimal behavior

#### 10. **No Theoretical Guarantees**

- **Problem**: Unlike some RL algorithms with theoretical guarantees (regret bounds, convergence proofs), the contextual bandit had no guarantees about:
  - Convergence to optimal policy
  - Regret bounds
  - Performance in non-stationary environments
- **Impact**: No way to know if the system was working correctly or just lucky

#### 11. **Transaction Cost Ignorance**

- **Problem**: While transaction costs were included in reward calculation, the bandit didn't optimize for:
  - Trade frequency (too many small trades = high costs)
  - Optimal entry/exit timing to minimize slippage
  - Market impact of trades
- **Impact**: Could generate many small trades that look profitable but lose money after costs

#### 12. **Emotional/Behavioral Factors**

- **Problem**: Real trading involves human psychology, but the bandit:
  - Couldn't account for market sentiment shifts
  - Didn't consider FOMO, panic selling, or other behavioral factors
  - Made decisions purely on numerical features
- **Impact**: Missed important market dynamics driven by human behavior

---

### Why RSI Is Better (For Now)

RSI-based signals, while simpler, address many of the bandit's weaknesses:

1. **Interpretable**: Clear rules (RSI < 30 = oversold = buy signal) that anyone can understand
2. **Time-Tested**: RSI has been used by traders for decades with known strengths/weaknesses
3. **No Overfitting Risk**: Fixed rules don't adapt to noise
4. **Transparent**: You can see exactly why a signal was generated
5. **Fast**: No learning required, works immediately
6. **Reliable**: Consistent behavior across different market conditions

However, RSI also has limitations (it's a lagging indicator, can give false signals, etc.), which is why the project is moving toward a **full RL agent** (see [NewAgent.md](NewAgent.md)) that addresses the bandit's weaknesses while maintaining interpretability.

---

### Current Status

- Bandit code may still exist in `sol_price_fetcher.py` but is not displayed in the UI
- The `/api/latest-bandit-action` endpoint may still exist but is not actively used
- Historical bandit logs remain in the database for reference
- `bandit_state.json` may still exist but is not actively updated

### Migration Path

The project has moved to **RSI (Relative Strength Index)** for trading signals:
- **RSI < 30**: Buy signal (oversold conditions)
- **RSI > 70**: Sell signal (overbought conditions)
- **RSI 30-70**: Hold signal (neutral conditions)

**Future Direction**: The project is planning to implement a **full reinforcement learning agent** (see [NewAgent.md](NewAgent.md) and [SOL_TRACKER_IMPROVEMENT_PLAN.md](SOL_TRACKER_IMPROVEMENT_PLAN.md)) that will:
- Use proper RL algorithms (PPO/actor-critic) instead of bandits
- Integrate multi-modal features (price, technical indicators, news embeddings, sentiment)
- Provide explainable outputs (SHAP values, decision trees, attention mechanisms)
- Include proper risk management and safety constraints
- Learn from multi-horizon returns (1h, 24h) to accelerate learning
- Extract human-readable rules that users can understand and validate

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

**Document Version:** 2.0  
**Last Updated:** 2025-01-XX  
**Maintained By:** Development Team

