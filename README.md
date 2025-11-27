# 🧠 Solana Rewards & Trading Bot Tracker

A Flask web application that tracks and displays Solana rewards from [**Orca** liquidity pools](https://docs.orca.so/) using the [Helius API](https://docs.helius.dev/). **Now featuring an experimental automated trading bot for SOL/USDC swaps via the [Jupiter Ultra API](https://dev.jup.ag/docs/ultra-api/)!** In addition to passive liquidity tracking, the app now includes:

- 💸 **Real-time SOL/USDC Liquidity Pool Monitoring** - Tracks token redemptions and rewards from Orca's concentrated liquidity pools using CLMM tech [at Orca.so](https://docs.orca.so/).
- 🤖 **Automated SOL/USDC Trading Bot (Experimental)** - Framework for live trading and execution using the Jupiter Ultra API , with contextual bandit-driven buy/sell/hold decisions.
- 📈 **SOL Price Forecasting** - Uses historical price data to predict short-term Solana price movements.
- 🎯 **Contextual Bandit Trading Agent** - A reinforcement learning agent that learns to buy, sell, or hold SOL based on:
  - Momentum and trend signals
  - Statistical features (Sharpe ratio, rolling mean, price deviations)
  - Profit and loss from past trades
- 📊 **Interactive Web Dashboard** - Visualizes price trends, agent decisions, portfolio state, and trade logs.

This project blends DeFi analytics, machine learning, and intelligent trading automation for Solana.

![Screenshot](images/Screenshot.png)

## Features

- ~~Real-time tracking of COLLECT_FEES events from your Solana wallet~~
  - Using the free API tier we refresh every 2 hours!
- SQLite database storage for transaction history
- Web dashboard displaying total SOL and USDC rewards
- Live SOL price data from LiveCoinWatch
- Background fetching with configurable intervals
- Backfill functionality for historical data

### Bonus SOL Price Database/Scraper

We are using the LiveCoinWatch API to grab the price of SOL and stashing in a SQLite database for long term analysis and statistical display.

~~You can start the price fetcher, on a `screen` session, by executing this command(Note, this is NOT NEEDED if you are running app.py, it will dispatch the predictions and combined the logged output so you only have one script to run!):~~

```bash
python sol_price_fetcher.py
```

Intead launch `app.py` with python on a `screen` session. This will count the API calls that remain for the day and ask you how often it should fetch prices based on the math it does. I have it fetching the price of SOL every five minutes. You can then see a chart of the last 288 price fetches (24 hours @ 5 minutes between fetches), the standard deviation, 24 hour range, low, high and so forth at `http://localhost:5030/sol-tracker`.

![Chart.png](images/Chart.png)

### SOL Price Tracking & Trading Signals

![Bandits.png](images/Bandits.png)

> **⚠️ DEPRECATION NOTICE:**
> 
> **Price Prediction**: The online price prediction feature has been **deprecated and disabled** as it was returning irrational values. Historical predictions may still be visible in the database and UI, but no new predictions are being generated.
> 
> **Buy/Sell/Hold Signals**: The contextual bandit algorithm for generating buy/sell/hold action suggestions **remains active and functional**. This feature uses separate online learning models for each potential action, learning to predict the expected reward based on the current market context (price features, volatility, etc.). The action with the highest predicted reward is chosen, with some randomness (epsilon-greedy strategy) to encourage exploration.

**Recent Improvements:**
- ✅ Migrated to TradingView Lightweight Charts for professional-grade visualization
- ✅ Added RSI (Relative Strength Index) technical indicator
- ✅ Enhanced dark theme UI with improved readability
- ✅ Optimized data loading for faster page performance
- ✅ Interactive charts with crosshair synchronization
> 
> **Live Trading**: Automated trading execution via Jupiter Ultra API has been **deprecated and disabled** until thoroughly tested and validated. Buy/sell/hold signals are generated for informational purposes only.

The `sol_price_fetcher.py` script previously incorporated machine learning models for experimental SOL price prediction:

- ~~**Online Price Prediction**: Utilizes the `river` library for online machine learning. A linear regression model (`river.linear_model.LinearRegression`) combined with a standard scaler (`river.preprocessing.StandardScaler`) is trained incrementally with new price data. This model predicts the next SOL price point.~~ **(DEPRECATED)**
- **Contextual Bandit for Actions**: A contextual bandit algorithm is implemented to suggest "buy", "sell", or "hold" actions. It uses separate online learning models (typically Hoeffding Tree Regressor) for each potential action, learning to predict the expected reward based on the current market context (price features, volatility, etc.). The action with the highest predicted reward is chosen, with some randomness (epsilon-greedy strategy) to encourage exploration.
  
  Bandit action logs (including chosen action, predicted rewards, and context) are stored in the `sol_prices.db` SQLite database.

Visit `http://localhost:5030/sol-tracker` to view:

- Price chart with statistics
- ~~Recent price predictions vs actuals~~ **(Historical data only - no new predictions)**
- Latest bandit decision with buy/sell/hold action and reward

![Predictions.png](images/Predictions.png)

## Roadmap & Issues

This project began as a simple Orca rewards tracker, but has since evolved into something more complex: a Solana trading assistant, price predictor, and contextual bandit strategy simulator.

While we're still focused on tracking Orca liquidity pool redemptions, the application now includes intelligent trade decisions based on market signals. That brings new opportunities—and responsibilities—for improving the reliability, safety, and interpretability of the trading logic.

We're currently not collecting certain data points like the total sum of all fees, but for now the model is performing adequately without this information. Eventually, richer logging may improve transparency and downstream analytics.

### Issues/TODO:

~~We enabled dynamic credit usage of the LiveCoinWatch API - we scale credit usage dynamically as the day progresses. This means we can fetch prices every 15 seconds. However, some of the chart queries are based on five minute increments and have limits of "288" to signify one day.~~

~~I also am seeing the need to split the price and bandit actions into two charts - Chart.JS just doesn't merge the two datasets well. Or - we can pull "rate" from the bandit logs, it's there as a saved feature.~~

---

### Example Strategy in Scope

_“Deposit a large amount into an [Orca.so](https://orca.so/ "Orca - Solana Trading and Liquidity Provider Platform") SOL/USDC pool. These pools are popular and generally pay consistent rewards. Use this application to track returns, and instead of reinvesting those rewards into the pool, allow the contextual bandit to reinvest them into SOL when conditions are favorable. For example, a liquidity range of $130 to $280 may be appropriate for 2024-2025. When SOL trades within that range, liquidity fees are generated, and this app helps determine whether to hold, buy, or exit into USDC.”_

This strategy illustrates how both passive income and machine-assisted trading can coexist in a portfolio.

---

## 🚧 Trading Bot Integration (Experimental - DEPRECATED)

> **⚠️ DEPRECATION NOTICE:**
> 
> The Jupiter Ultra Trading Bot integration has been **deprecated and disabled**. The framework for automated trading exists in the codebase but is currently commented out and will not execute trades, even if enabled in the environment variables.
> 
> **Status**: Buy/sell/hold signals are still generated by the contextual bandit algorithm for informational purposes, but no actual trades are executed. The trading bot code remains in the repository for reference but should not be used with real funds until thoroughly tested and validated.

### Trading Bot Features (DEPRECATED - Not Active)

> **⚠️ All trading bot features are currently disabled. The code exists but is commented out.**

- ~~**Automated Trading via Jupiter Ultra API:**~~
  ~~The system can fetch quotes, sign transactions, and execute swaps between SOL and USDC using the Jupiter Ultra API.~~ **(DISABLED)**
- **Contextual Bandit Integration:**
  The contextual bandit model continues to generate buy/sell/hold recommendations, but these are for informational purposes only. No trades are executed.
- ~~**Trade Logging:**~~
  ~~All attempted trades and execution results are logged to a dedicated `sol_trades.db` SQLite database for audit and analysis.~~ **(DISABLED - No trades are executed)**
- ~~**Balance Checks:**~~
  ~~The bot checks wallet balances before attempting trades.~~ **(DISABLED)**
- ~~**.env Controlled:**~~
  ~~Trading is only enabled if the required private key is present in `.env` and `ENABLE_LIVE_TRADING=Y` is set.~~ **(DISABLED - Trading code is commented out regardless of .env settings)**

### How It Works (Framework - DISABLED)

> **⚠️ This entire workflow is currently disabled. The code is commented out and will not execute.**

~~1. **Balance Check:**~~
   ~~Before trading, the bot checks if there is enough SOL or USDC to execute the desired action.~~
~~2. **Quote Fetch:**~~
   ~~The bot fetches a quote/order from Jupiter Ultra for the intended swap.~~
~~3. **Transaction Signing:**~~
   ~~The unsigned transaction from the quote is signed locally using the wallet keypair.~~
~~4. **Order Execution:**~~
   ~~The signed transaction and request ID are sent to Jupiter Ultra's `/execute` endpoint.~~
~~5. **Logging:**~~
   ~~All order and execution details are stored in `sol_trades.db`.~~

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

---

**For more details, see the `trading_bot.py` and `sol_price_fetcher.py` files.**

## Tech Stack

- Python 3.10+
- Flask
- SQLite
- Tailwind CSS for UI styling
- Chart.js for charts
- Helius API for Solana blockchain data
- LiveCoinWatch API for SOL price data

## Setup

### 1\. Clone the repository

bash

```bash
git clone https://github.com/JHenzi/OrcaRewardDashboard
cd solana-rewards-tracker
```

### 2\. Install dependencies

bash

```bash
pip install -r requirements.txt
```

### 3\. Set up environment variables

Copy the `.env.example` file to `.env` and fill in your API keys and wallet address:

bash

```bash
cp .env.example .env
```

Edit the `.env` file with your actual values:

env

```env
HELIUS_API_KEY=your_helius_api_key_here
SOLANA_WALLET_ADDRESS=your_solana_wallet_address_here
LIVECOINWATCH_API_KEY=your_livecoinwatch_api_key_here
```

### 4\. Run the application

There are two scripts that must continously run. The Flask application on `app.py` and the price fetcher/predictor.

We suggest running each of these on separate `screen` sessions.

```bash
python app.py
python sol_price_fetcher.py
```

The application will be available at `http://localhost:5030`

## API Keys Required

### Helius API Key

- Sign up at [Helius](https://helius.xyz/)
- Create a new project
- Copy your API key to the `.env` file

### LiveCoinWatch API Key

- Sign up at [LiveCoinWatch](https://www.livecoinwatch.com/tools/api)
- Get your API key
- Copy it to the `.env` file

## Configuration

You can customize the application behavior through environment variables in the `.env` file:

- `HELIUS_API_KEY`: Your Helius API key (required)
- `SOLANA_WALLET_ADDRESS`: Your Solana wallet address to track (required)
- `LIVECOINWATCH_API_KEY`: Your LiveCoinWatch API key (required)
- `DATABASE_PATH`: Path to SQLite database file (default: rewards.db)
- `FLASK_HOST`: Flask server host (default: 0.0.0.0)
- `FLASK_PORT`: Flask server port (default: 5030)
- `FLASK_DEBUG`: Enable Flask debug mode (default: True)
- `FETCH_INTERVAL_SECONDS`: Background fetch interval in seconds (default: 7200 = 2 hours)
- `LAST_KNOWN_SIGNATURE`: Starting signature for backfill operations (this is the transaction ID from SolScan that you want to fetch since, i.e. the transaction you deposited your liquidity... this may eat into your API calls for the month - adjust fetch interval accordingly)

## Project Layout

A brief overview of the project structure:

- `app.py`: The main Flask web application. Handles HTTP requests, interacts with the `rewards.db` database, and renders HTML templates. This is the core controller for the rewards tracking dashboard.
- `sol_price_fetcher.py`: A script responsible for fetching SOL price data from the LiveCoinWatch API, storing it in `sol_prices.db`, and running the machine learning models for price prediction and bandit actions. It can be run independently or is triggered by `app.py`.
- `requirements.txt`: Lists the Python dependencies for the project.
- `.env.example`: A template for the environment variables file (`.env`). You need to copy this to `.env` and fill in your API keys and other configurations.
- `templates/`: Contains the HTML templates used by Flask to render the web pages (e.g., `index.html`, `sol_tracker.html`).
- `rewards.db`: (Created at runtime) SQLite database for storing Solana reward transaction data.
- `sol_prices.db`: (Created at runtime) SQLite database for storing SOL price history, predictions, and bandit logs.
- `sol_model.pkl`, `sol_metric.pkl`, `sol_horizon_model.pkl`: (Created at runtime) Saved machine learning models and metrics.
- `bandit_state.json`: (Created at runtime) Stores the state of the bandit portfolio simulation.
- `trading_bot.py`: Jupiter ULTRA Trading API function class. Executes orders (quotes) and transactions on Jupiter's swap API. Goal being capital appreciation.

## Usage

### Web Dashboard

Visit `http://localhost:5030` to view:

- Total SOL and USDC rewards collected
- Current SOL price and USD value of rewards
- Collection analytics (frequency, sessions, daily rate, patterns)

### SOL Price & Trading Signals

Visit `http://localhost:5030/sol-tracker` to view:

- **Professional Trading Charts**: Interactive price charts using TradingView Lightweight Charts
- **RSI Indicator**: Relative Strength Index (14-period) with overbought/oversold levels
- **Trading Signals**: Real-time buy/sell/hold recommendations from contextual bandit algorithm
- **Statistical Analysis**: SMA (1h, 4h, 24h), standard deviation, price ranges
- **Time Range Selection**: View data for 1 hour, 1 day, 1 week, 1 month, or 1 year
- **Bandit Strategy State**: Portfolio tracking, entry prices, and performance metrics

> **Note**: Price prediction has been deprecated. The page now focuses on trading signals and technical analysis.

### Backfill Historical Data

Visit `http://localhost:5030/backfill_newer` to manually trigger fetching newer transactions.

### API Endpoints

The application provides the following API endpoints for programmatic access to data:

- **Get Latest Price Prediction (DEPRECATED):**

  - **URL:** `/api/latest-prediction`
  - **Method:** `GET`
  - **Description:** Retrieves the most recent price prediction from historical data. **⚠️ Price prediction has been disabled - no new predictions are being generated. This endpoint returns historical data only.**
  - **Success Response (200 OK):**
    ```json
    {
      "timestamp": "2023-10-27T10:30:00.123Z", // ISO format timestamp of the data point used for prediction
      "predicted_rate": 135.5,
      "actual_rate": 135.25,
      "error": 0.25,
      "mae": 0.15, // Mean Absolute Error at the time of this prediction
      "created_at": "2023-10-27T10:35:05.456Z", // ISO format timestamp when the prediction record was created
      "deprecated": true,
      "message": "Price prediction has been disabled. This is historical data only."
    }
    ```
  - **Error Response (404 Not Found):** If no predictions are found.
    ```json
    {
      "error": "No predictions found",
      "deprecated": true,
      "message": "Price prediction has been disabled. No historical data available."
    }
    ```
  - **Error Response (500 Internal Server Error):** For database or other server-side issues.
    ```json
    {
      "error": "Database error"
    }
    ```

- **Get Latest Bandit Action:**

  - **URL:** `/api/latest-bandit-action`
  - **Method:** `GET`
  - **Description:** Retrieves the most recent action (buy/sell/hold) decided by the contextual bandit.
  - **Success Response (200 OK):**
    ```json
    {
      "timestamp": "2023-10-27T10:30:00.123Z", // ISO format timestamp of the data point used for the action
      "action": "buy",
      "reward": 0.05,
      "prediction_buy": 0.12,
      "prediction_sell": -0.05,
      "prediction_hold": 0.01,
      "data_json": {
        /* Contextual features used for this decision */
      },
      "created_at": "2023-10-27T10:35:05.789Z" // ISO format timestamp when the bandit log record was created
    }
    ```
  - **Error Response (404 Not Found):** If no bandit actions are found.
    ```json
    {
      "error": "No bandit actions found"
    }
    ```
  - **Error Response (500 Internal Server Error):** For database or other server-side issues.
    ```json
    {
      "error": "Database error"
    }
    ```

## Security Notes

- Never commit your `.env` file to version control
- The `.gitignore` file is configured to exclude sensitive files
- Keep your API keys secure and rotate them regularly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

---

## Disclaimer & Author Note

This project is the experimental work of **Joseph Henzi** and is intended as a technical and educational exercise in Python programming, data processing, and algorithmic trading logic.

It is **not affiliated with The Henzi Foundation/The Frankie Fund**, a charitable organization dedicated to providing financial support to families facing the unexpected loss of a child.

If you are looking for more information on the foundation or would like to support its mission of covering funeral and final expenses for children, please visit: [https://henzi.org](https://henzi.org)
