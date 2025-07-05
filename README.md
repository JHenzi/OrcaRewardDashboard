### 🧠 Solana Rewards Tracker

A Flask web application that tracks and displays Solana rewards from Orca liquidity pools using the Helius API. In addition to passive liquidity tracking, the app now includes:

- 💸 **Real-time SOL/USDC Liquidity Pool Monitoring** – Tracks token redemptions and rewards from Orca.
- 📈 **SOL Price Forecasting** – Uses historical price data to predict short-term Solana price movements.
- 🎯 **Contextual Bandit Trading Agent** – A reinforcement learning agent that learns to buy, sell, or hold SOL based on:

  - Momentum and trend signals
  - Statistical features like Sharpe ratio, rolling mean, and price deviations
  - Profit and loss outcomes from past trades

- 📊 **Interactive Web Dashboard** – Visualizes price trends, agent decisions, portfolio state, and trade logs.

This project blends DeFi analytics with intelligent trading automation.

![Screenshot](images/Screenshot.png)

## Features

- ~~Real-time tracking of COLLECT_FEES events from your Solana wallet~~ Using the free API tier we refresh every 2 hours!
- SQLite database storage for transaction history
- Web dashboard displaying total SOL and USDC rewards
- Live SOL price data from LiveCoinWatch
- Background fetching with configurable intervals
- Backfill functionality for historical data

### Bonus SOL Price Database/Scraper

We are using the LiveCoinWatch API to grab the price of SOL and stashing in a SQLite database for long term analysis and statistical display.

You can start the price fetcher, on a `screen` session, by executing this command(Note, this is NOT NEEDED if you are running app.py, it will dispatch the predictions and combined the logged output so you only have one script to run!):

```bash
python sol_price_fetcher.py
```

This will count the API calls that remain for the day and ask you how often it should fetch prices based on the math it does. I have it fetching the price of SOL every five minutes. You can then see a chart of the last 288 price fetches (24 hours @ 5 minutes between fetches), the standard deviation, 24 hour range, low, high and so forth at `http://localhost:5030/sol-tracker`.

![Chart.png](images/Chart.png)

### Bonus SOL Price Prediction and Bandit Actions

![Bandits.png](images/Bandits.png)

The `sol_price_fetcher.py` script incorporates machine learning models for experimental SOL price prediction and automated action suggestions:

- **Online Price Prediction**: Utilizes the `river` library for online machine learning. A linear regression model (`river.linear_model.LinearRegression`) combined with a standard scaler (`river.preprocessing.StandardScaler`) is trained incrementally with new price data. This model predicts the next SOL price point.
- **Contextual Bandit for Actions**: A contextual bandit algorithm is implemented to suggest "buy", "sell", or "hold" actions. It uses separate online learning models (again, typically linear regression) for each potential action, learning to predict the expected reward based on the current market context (price features, volatility, etc.). The action with the highest predicted reward is chosen, with some randomness (epsilon-greedy strategy) to encourage exploration.
  Predictions, model metrics (like Mean Absolute Error for price prediction), and bandit action logs (including chosen action, predicted rewards, and context) are stored in the `sol_prices.db` SQLite database.

Visit `http://localhost:5030/sol-tracker` to view:

- Price chart with statistics
- Recent price predictions vs actuals
- Latest bandit decision with buy/sell/hold action and reward

![Predictions.png](images/Predictions.png)

## Roadmap & Issues

This project began as a simple Orca rewards tracker, but has since evolved into something more complex: a Solana trading assistant, price predictor, and contextual bandit strategy simulator.

While we're still focused on tracking Orca liquidity pool redemptions, the application now includes intelligent trade decisions based on market signals. That brings new opportunities—and responsibilities—for improving the reliability, safety, and interpretability of the trading logic.

We're currently not collecting certain data points like the total sum of all fees, but for now the model is performing adequately without this information. Eventually, richer logging may improve transparency and downstream analytics.

---

### Example Strategy in Scope

_“Deposit a large amount into an [Orca.so](https://orca.so/ "Orca - Solana Trading and Liquidity Provider Platform") SOL/USDC pool. These pools are popular and generally pay consistent rewards. Use this application to track returns, and instead of reinvesting those rewards into the pool, allow the contextual bandit to reinvest them into SOL when conditions are favorable. For example, a liquidity range of $130 to $280 may be appropriate for 2024-2025. When SOL trades within that range, liquidity fees are generated, and this app helps determine whether to hold, buy, or exit into USDC.”_

This strategy illustrates how both passive income and machine-assisted trading can coexist in a portfolio.

---

### In Progress / Open Questions

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

## Usage

### Web Dashboard

Visit `http://localhost:5030` to view:

- Total SOL and USDC rewards collected
- Current SOL price and USD value of rewards
- Collection analytics (frequency, sessions, daily rate, patterns)

### SOL Price & Predictions

Visit `http://localhost:5030/sol-tracker` to view:

- SOL price chart with recent 24-hour data
- Statistical summaries (SMA, range, std dev, etc.)
- Latest model price predictions vs actuals
- Bandit buy/sell/hold actions and rewards

### Backfill Historical Data

Visit `http://localhost:5030/backfill_newer` to manually trigger fetching newer transactions.

## Database Schema

The application uses SQLite, with new tables being added (i.e. right now it is missing the "trades" schema from bandit trading.):

### collect_fees

Stores reward collection events:

- `signature`: Transaction signature
- `timestamp`: Timestamp
- `fee_payer`: Fee payer address
- `token_mint`: Token mint address
- `token_amount`: Amount transferred
- `from_token_account`, `to_token_account`: Token accounts
- `from_user_account`, `to_user_account`: User accounts

### tokens

Stores token metadata:

- `mint`: Token mint address (primary key)
- `symbol`: Token symbol
- `name`: Token name
- `decimals`: Token decimal precision

### sol_predictions

Stores model price predictions:

- `timestamp`, `predicted_rate`, `actual_rate`, `error`, `mae`, `created_at`

### bandit_logs

Stores contextual bandit decision logs:

- `timestamp`, `action`, `reward`, `prediction_buy`, `prediction_sell`, `prediction_hold`, `created_at`

### trades

Stores each buy, sell, or hold decision made by the contextual bandit, along with market context and portfolio state:

- `id`: Auto-incrementing primary key
- `timestamp`: ISO 8601 timestamp of the trade
- `action`: `"buy"`, `"sell"`, or `"hold"`
- `price`: SOL price at time of action
- `amount`: Quantity of SOL traded (typically 1.0)
- `fee_pct`: Trade fee as a decimal (e.g., 0.001)
- `fee_usd`: Fee amount in USD
- `net_value_usd`: Net USD value after fee
- `portfolio_usd_balance`: USD balance after trade
- `portfolio_sol_balance`: SOL balance after trade
- `portfolio_equity`: Total portfolio value in USD
- `avg_entry_price`: Average entry price for SOL position
- `realized_pnl`: Realized profit/loss from trade in USD
- `price_24h_high`: 24-hour high price of SOL
- `price_24h_low`: 24-hour low price of SOL
- `rolling_mean_price`: Rolling mean of recent prices
- `returns_mean`: Mean of recent returns
- `returns_std`: Standard deviation of recent returns

---

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
