# 🧠 Solana Rewards & Trading Bot Tracker

> **📢 AI-Assisted Development Disclosure:** This project was initially created as a personal pet project and has been significantly enhanced through the use of **Cursor**, an AI-powered coding assistant. We believe in full transparency about our development process. [Read our full disclosure →](VibeCoded.md)

A Flask web application that tracks and displays Solana rewards from [**Orca** liquidity pools](https://docs.orca.so/) using the [Helius API](https://docs.helius.dev/). The application provides comprehensive analytics and trading insights for Solana DeFi activities.

**Current Features:**

- 💸 **SOL/USDC Liquidity Pool Monitoring** - Tracks token redemptions and rewards from Orca's concentrated liquidity pools using CLMM tech [at Orca.so](https://docs.orca.so/)
- 📊 **Interactive Web Dashboard** - Modern, responsive dashboard with real-time metrics and analytics
- 📈 **SOL Price Tracking** - Professional trading charts with technical indicators using TradingView Lightweight Charts
- 🎯 **RSI-Based Trading Signals** - Clear buy/sell/hold recommendations based on Relative Strength Index (RSI)
- 📉 **Technical Analysis** - Moving averages (SMA), MACD, Bollinger Bands, momentum indicators, and volatility metrics
- 📊 **Signal Performance Tracker** - **NEW!** Track the reliability of each trading signal over time. See win rates, average returns, and historical performance for RSI signals to make informed decisions
- 💰 **Rewards Analytics** - Daily and monthly breakdowns, collection patterns, and performance metrics

> **Note:** Some features have been deprecated. See [DEPRECATED.md](DEPRECATED.md) for historical information about price prediction, contextual bandit algorithms, and automated trading bots.

This project blends DeFi analytics, technical analysis, and intelligent trading insights for Solana.

![Screenshot](images/ModernScreenshot.png)

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

### SOL Price Tracking & Trading Signals

The SOL Tracker provides professional-grade price analysis with clear, actionable trading signals and **signal reliability tracking** to help you make informed decisions.

**Key Features:**
- ✅ **TradingView Lightweight Charts** - Professional-grade interactive price charts
- ✅ **RSI Indicator** - Relative Strength Index (14-period) with overbought/oversold levels
- ✅ **RSI-Based Trading Signals** - Clear buy/sell/hold recommendations:
  - **Buy Signal**: RSI < 30 (oversold conditions)
  - **Sell Signal**: RSI > 70 (overbought conditions)
  - **Hold Signal**: RSI 30-70 (neutral conditions)
- ✅ **Technical Indicators** - Moving averages (SMA 1h, 4h, 24h), MACD, Bollinger Bands, momentum indicators, and volatility metrics
- ✅ **Signal Performance Tracker** - **Track signal reliability!** See which signals have been profitable:
  - **Win Rate**: Percentage of signals that resulted in profitable outcomes
  - **Average Return**: Expected performance for each signal type (24-hour horizon)
  - **Best/Worst Cases**: Historical performance ranges
  - **Color-Coded Reliability**: Green (≥60% win rate), Yellow (≥50%), Red (<50%)
  - Tracks: RSI Buy/Sell/Hold signals
- ✅ **Time Range Selection** - View data for 1 hour, 1 day, 1 week, 1 month, or 1 year
- ✅ **Dark Theme UI** - Modern, professional trading platform aesthetic

Visit `http://localhost:5030/sol-tracker` to view:
- Interactive price charts with RSI overlay
- Real-time RSI-based trading signals
- **Signal reliability metrics** - Know which signals to trust!
- Comprehensive technical indicator panel (RSI, MACD, Bollinger Bands, Momentum, SMA)
- Price statistics and volatility metrics

### Why Signal Performance Tracking Matters

**The Problem:** Trading signals are only as good as their track record. Without performance tracking, you can't know if a "BUY" signal is actually profitable or if it's just noise.

**The Solution:** The Signal Performance Tracker automatically:
1. **Logs every signal** when it's generated (RSI buy/sell/hold)
2. **Tracks outcomes** by checking price movements 1 hour, 4 hours, 24 hours, and 7 days later
3. **Calculates reliability metrics** including win rate, average return, and best/worst cases
4. **Displays results** in an easy-to-understand format on the SOL Tracker page

**What This Means for You:**
- **Make Informed Decisions**: See which signals have historically been profitable
- **Trust the Data**: Know if RSI buy signals actually lead to price increases
- **Continuous Improvement**: As more signals are tracked, reliability metrics become more accurate

**Example:** If RSI Buy signals show a 65% win rate with an average +2.3% return over 24 hours, you can confidently act on those signals. If RSI Sell signals show only 40% win rate, you might want to be more cautious.

> **Note:** Price prediction and contextual bandit features have been deprecated. See [DEPRECATED.md](DEPRECATED.md) for historical information.

### Signal Performance Tracker - Why It Matters

**The Challenge:** Trading signals are everywhere, but how do you know which ones to trust? A "BUY" signal might look promising, but without historical data, you're flying blind.

**The Solution:** Our Signal Performance Tracker automatically monitors every trading signal and tracks its outcome over time. This gives you **data-driven confidence** in your trading decisions.

**What Gets Tracked:**
- **RSI Signals**: Every time RSI generates a buy/sell/hold signal, we log it
- **Outcomes**: We check prices 1 hour, 4 hours, 24 hours, and 7 days later
- **Metrics**: Win rate, average return, best/worst cases are calculated automatically

**Why This Matters:**
1. **Trust the Data**: See which signals have actually been profitable historically
2. **Risk Assessment**: Know the best and worst case scenarios for each signal type
3. **Continuous Learning**: As more signals are tracked, reliability metrics become more accurate
4. **Informed Decisions**: Make trading decisions based on proven track records, not just current signals

**Example Use Case:**
- RSI Buy signals show **65% win rate** with **+2.3% average return** → High confidence to act
- RSI Sell signals show **40% win rate** with **-0.5% average return** → Be cautious, maybe wait for stronger signals
- RSI Hold signals show **55% win rate** → Neutral, confirms it's a good time to wait

This feature transforms the SOL Tracker from a simple signal generator into a **data-driven trading intelligence platform**.

## Project Roadmap

This project began as a simple Orca rewards tracker and has evolved into a comprehensive DeFi analytics platform. We're continuously improving the application with better UI/UX, more reliable trading signals, and enhanced analytics.

### Current Focus Areas

- **UI/UX Modernization** - Modern, responsive design with dark theme
- **RSI-Based Trading Signals** - Clear, interpretable trading recommendations
- **Signal Performance Tracking** - Track and display reliability metrics for all trading signals
- **Performance Optimization** - Faster page loads and better data handling
- **Database Improvements** - Better indexing and data capture

### Future Enhancements

See [SOL_TRACKER_IMPROVEMENT_PLAN.md](SOL_TRACKER_IMPROVEMENT_PLAN.md) for detailed roadmap including:
- **RL-Based Trading Agent** - Full reinforcement learning system with explainable rule discovery (see [NewAgent.md](NewAgent.md))
- Advanced AI features (multi-horizon predictions, news sentiment integration, attention mechanisms)
- Explainability features (SHAP values, decision tree rules, attention visualization)
- Production deployment strategies
- Security hardening
- Additional technical indicators

---

### Example Use Case

_"Deposit a large amount into an [Orca.so](https://orca.so/) SOL/USDC pool. These pools are popular and generally pay consistent rewards. Use this application to track returns, and use the RSI-based trading signals to help determine when to reinvest rewards into SOL or hold USDC. For example, a liquidity range of $130 to $280 may be appropriate for 2024-2025. When SOL trades within that range, liquidity fees are generated, and this app provides clear trading signals based on technical indicators."_

This strategy illustrates how both passive income and technical analysis can coexist in a portfolio.

> **Note:** Historical features (price prediction, contextual bandit, automated trading) have been deprecated. See [DEPRECATED.md](DEPRECATED.md) for details.

## Tech Stack

- Python 3.10+
- Flask
- SQLite
- Tailwind CSS for UI styling
- TradingView Lightweight Charts for professional trading charts
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
- `sol_price_fetcher.py`: A script responsible for fetching SOL price data from the LiveCoinWatch API and storing it in `sol_prices.db`. It can be run independently or is triggered by `app.py`.
- `requirements.txt`: Lists the Python dependencies for the project.
- `.env.example`: A template for the environment variables file (`.env`). You need to copy this to `.env` and fill in your API keys and other configurations.
- `templates/`: Contains the HTML templates used by Flask to render the web pages (e.g., `home.html`, `index.html`, `sol_tracker.html`).
- `rewards.db`: (Created at runtime) SQLite database for storing Solana reward transaction data.
- `sol_prices.db`: (Created at runtime) SQLite database for storing SOL price history, RSI calculations, and signal performance data.
- `signal_performance_tracker.py`: Module for tracking and analyzing the performance of trading signals over time.
- `DEPRECATED.md`: Documentation of deprecated features (price prediction, contextual bandit, automated trading).
- `trading_bot.py`: (Deprecated) Jupiter ULTRA Trading API function class. Code exists but is disabled. See [DEPRECATED.md](DEPRECATED.md) for details.

## Usage

### Web Dashboard

Visit `http://localhost:5030` to view the **Home Dashboard** with:
- Summary metrics and quick links
- Total value collected overview
- Recent activity timeline

Visit `http://localhost:5030/orca` to view the **Detailed Rewards Dashboard** with:
- Total SOL and USDC rewards collected
- Current SOL price and USD value of rewards
- Collection analytics (frequency, sessions, daily rate, patterns)
- Daily and monthly breakdowns

### SOL Price & Trading Signals

Visit `http://localhost:5030/sol-tracker` to view:

- **Professional Trading Charts**: Interactive price charts using TradingView Lightweight Charts
- **RSI Indicator**: Relative Strength Index (14-period) with overbought/oversold levels
- **RSI-Based Trading Signals**: Clear buy/sell/hold recommendations based on RSI levels
- **Signal Performance Tracker**: See win rates, average returns, and reliability metrics for each signal type
- **Technical Indicators Summary**: Moving averages (SMA 1h, 4h, 24h), MACD, Bollinger Bands, momentum indicators
- **Time Range Selection**: View data for 1 hour, 1 day, 1 week, 1 month, or 1 year
- **Price Statistics**: Current price, 24h change, high/low ranges, standard deviation

### Backfill Historical Data

Visit `http://localhost:5030/backfill_newer` to manually trigger fetching newer transactions.

### API Endpoints

The application provides API endpoints for programmatic access to data. Some endpoints may be deprecated - see [DEPRECATED.md](DEPRECATED.md) for details on deprecated endpoints.

**Note:** API documentation for deprecated endpoints (price prediction, bandit actions) has been moved to [DEPRECATED.md](DEPRECATED.md).

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

This project is open source and available under the [GNU GPL](LICENSE).

```bash
curl -o LICENSE https://www.gnu.org/licenses/gpl-3.0.txt
```

## 🙏 KUDOS: Thank You to RSS Feed Providers

We are deeply grateful to the news organizations and content providers who continue to support **RSS (Really Simple Syndication)**, an open Internet standard that enables free and open access to information.

**Information should be free.** RSS feeds allow anyone to access news and content without proprietary APIs, paywalls, or complex authentication systems. They represent the best of the open web—a decentralized, standards-based approach to content distribution.

### News Sources We're Grateful For

This project relies on RSS feeds from the following organizations:

**Crypto News:**
- CoinTelegraph, Decrypt
- *(CoinDesk is currently rate-limited)*

**Finance News:**
- Bloomberg
- *(Financial Times and Reuters feeds are not currently accessible)*

**Technology News:**
- TechCrunch, The Verge, Ars Technica

**General & US National News:**
- BBC News, ABC News, NPR, New York Times, NBC News, Fox News

These organizations maintain RSS feeds that allow developers, researchers, and users to access news content programmatically, supporting innovation, transparency, and the free flow of information.

**Thank you for keeping the open web alive!** 🌐

> **Note:** If you're a news organization reading this and considering removing RSS support, please reconsider. RSS feeds are a cornerstone of the open web and enable countless legitimate use cases, from accessibility tools to research projects to personal news aggregation. They represent a commitment to open standards and information accessibility.

## AI-Assisted Development

This project uses AI coding assistants (primarily Cursor) for implementation while maintaining human oversight for planning, architecture, and code review. We are committed to transparency about our development process. [Read our full disclosure →](VibeCoded.md)

---

## Disclaimer & Author Note

This project is the experimental work of **Joseph Henzi** and is intended as a technical and educational exercise in Python programming, data processing, and algorithmic trading logic.

It is **not affiliated with The Henzi Foundation/The Frankie Fund**, a charitable organization dedicated to providing financial support to families facing the unexpected loss of a child.

If you are looking for more information on the foundation or would like to support its mission of covering funeral and final expenses for children, please visit: [https://henzi.org](https://henzi.org)
