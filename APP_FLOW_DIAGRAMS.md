# Application Flow Diagrams

This document provides visual representations of key application flows.

## 1. Application Startup Sequence

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Startup                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 1. Import Dependencies            │
        │    - Flask, SQLite, Threading     │
        │    - Optional: RL Agent, News     │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 2. Initialize Logging             │
        │    - Configure logger             │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 3. Validate Environment           │
        │    - Check .env exists            │
        │    - Validate required vars       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 4. Create Flask App                │
        │    - app = Flask(__name__)        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 5. Check WERKZEUG_RUN_MAIN       │
        │    (Skip if reloader parent)      │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 6. Start Background Threads      │
        │    ├─ Background Fetch            │
        │    ├─ SOL Price Fetch             │
        │    ├─ News Fetch                  │
        │    └─ RL Agent Init               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 7. Start Flask Server             │
        │    - app.run(host, port, debug)   │
        └───────────────────────────────────┘
```

## 2. Request Flow - SOL Tracker Page

```
┌─────────────────────────────────────────────────────────────┐
│              GET /sol-tracker?range=day                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 1. Parse Query Parameters      │
        │    - range (hour/day/week/etc) │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 2. Create SOLPriceFetcher         │
        │    - Connect to sol_prices.db    │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 3. Fetch Price History            │
        │    - Filter by time range         │
        │    - Limit to 1000 points         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 4. Process Price Data             │
        │    - Parse timestamps             │
        │    - Extract prices              │
        │    - Convert to Unix timestamps  │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 5. Calculate Technical Indicators  │
        │    - RSI (14 period)              │
        │    - MACD (12, 26, 9)             │
        │    - Bollinger Bands (20, 2)      │
        │    - SMAs (1h, 4h, 24h)          │
        │    - EMAs (12, 26)                │
        │    - Momentum                     │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 6. Get Latest Price               │
        │    - Query most recent price     │
        │    - Calculate 24h change        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 7. Initialize Signal Tracker      │
        │    - Create SignalPerformanceTracker│
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 8. Log RSI Signals                │
        │    - Check if signal changed      │
        │    - Log if new/different         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 9. Log Bandit Signals              │
        │    - Get latest bandit action     │
        │    - Log if new/different         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 10. Update Performance Metrics    │
        │     - Calculate win rates          │
        │     - Get performance stats       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 11. Get News Features              │
        │     - Fetch recent news            │
        │     - Get sentiment scores        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 12. Render Template               │
        │     - sol_tracker.html            │
        │     - Pass all calculated data    │
        └───────────────────────────────────┘
```

## 3. Background Thread - SOL Price Fetch Loop

```
┌─────────────────────────────────────────────────────────────┐
│         Background Thread: sol_price_fetch_loop             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Initialize                        │
        │ - Check FAST_MODE                 │
        │ - Set initial interval            │
        │ - cycle_count = 0                 │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Loop Forever │
                    └───────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Fetch SOL Price                   │
        │ - price_fetcher.fetch_sol_price()  │
        │ - Log price                       │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  FAST_MODE?   │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌───────────────┐
    │   YES         │              │      NO      │
    │               │              │               │
    │ cycle_count++ │              │ Use fixed     │
    │               │              │ 1 min interval│
    └───────┬───────┘              └───────┬───────┘
            │                               │
            ▼                               │
    ┌───────────────┐                      │
    │ cycle_count   │                      │
    │ % 1000 == 0?  │                      │
    └───────┬───────┘                      │
            │                               │
        ┌───┴───┐                          │
        │       │                          │
        YES     NO                         │
        │       │                          │
        ▼       │                          │
    ┌───────────────┐                      │
    │ Check Credits  │                      │
    │ - Get remaining│                      │
    │ - Calculate    │                      │
    │   safe credits │                      │
    │ - Adjust       │                      │
    │   interval     │                      │
    └───────┬───────┘                      │
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Sleep for      │
                    │ interval       │
                    └───────────────┘
                            │
                            └───────┐
                                    │
                                    ▼
                            (Loop continues)
```

## 4. Background Thread - News Fetch Loop

```
┌─────────────────────────────────────────────────────────────┐
│         Background Thread: news_fetch_loop                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Initialize                        │
        │ - Create NewsSentimentAnalyzer    │
        │ - fetch_interval = 6 hours        │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Loop Forever │
                    └───────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Get Current SOL Price             │
        │ - From price_fetcher or DB         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Fetch News                        │
        │ - news_analyzer.fetch_news()      │
        │ - Respects internal cooldown      │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Articles      │
                    │ Returned?     │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            YES                             NO
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌───────────────┐
    │ Process News  │              │ Skip (cooldown│
    │ - Generate    │              │  or no new)   │
    │   embeddings  │              │               │
    │ - Sentiment   │              │               │
    │ - Store in DB │              │               │
    └───────┬───────┘              └───────┬───────┘
            │                               │
            └───────────────┬───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Sleep for      │
                    │ 6 hours        │
                    │ (inefficient   │
                    │  loop)         │
                    └───────────────┘
                            │
                            └───────┐
                                    │
                                    ▼
                            (Loop continues)
```

## 5. RL Agent Decision Flow

```
┌─────────────────────────────────────────────────────────────┐
│         POST /api/rl-agent/decision                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Check RL Agent Available           │
        │ - RL_AGENT_AVAILABLE flag          │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Check Model Loaded                 │
        │ - rl_agent_integration != None    │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ RLAgentIntegration.make_decision()│
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 1. Fetch Current Price             │
        │    - From sol_price_fetcher        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 2. Fetch Recent News               │
        │    - From news_sentiment           │
        │    - Get embeddings & sentiment   │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 3. Encode State                    │
        │    - Price features                │
        │    - Technical indicators          │
        │    - News embeddings               │
        │    - News sentiment                │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 4. Get Model Prediction            │
        │    - Forward pass through model    │
        │    - Get action probabilities     │
        │    - Get value estimate           │
        │    - Get return predictions       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 5. Apply Risk Management            │
        │    - Check position limits         │
        │    - Check trade frequency         │
        │    - Check daily loss cap         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 6. Log Decision                    │
        │    - Store in rl_agent_decisions   │
        │    - Log attention weights         │
        │    - Store predictions             │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ 7. Return Decision                 │
        │    - Action (BUY/SELL/HOLD)       │
        │    - Confidence                    │
        │    - Predictions                   │
        └───────────────────────────────────┘
```

## 6. Database Connection Pattern (Current - Problematic)

```
┌─────────────────────────────────────────────────────────────┐
│              Request Handler                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Function 1                        │
        │ conn = sqlite3.connect(db)        │
        │ cursor = conn.cursor()             │
        │ ... query ...                      │
        │ conn.close()                      │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Function 2                        │
        │ conn = sqlite3.connect(db)        │
        │ cursor = conn.cursor()            │
        │ ... query ...                     │
        │ conn.close()                      │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Function 3                        │
        │ conn = sqlite3.connect(db)        │
        │ cursor = conn.cursor()            │
        │ ... query ...                     │
        │ conn.close()                      │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Response     │
                    └───────────────┘

ISSUE: Multiple connections opened/closed per request
```

## 7. Database Connection Pattern (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│              Request Handler                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ with get_db_connection() as conn: │
        │     cursor = conn.cursor()        │
        │     ... all queries ...           │
        │     conn.commit()                │
        └───────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Response     │
                    └───────────────┘

BENEFIT: Single connection per request, automatic cleanup
```

## 8. Error Handling Flow (Current)

```
┌─────────────────────────────────────────────────────────────┐
│                    Operation                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Try Block    │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            Success                         Exception
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌───────────────┐
    │ Return Result │              │ Log Error     │
    │               │              │ Continue      │
    └───────────────┘              └───────────────┘

ISSUE: No recovery, no retry, no circuit breaker
```

## 9. Error Handling Flow (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    Operation                                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  Try Block    │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            Success                         Exception
            │                               │
            ▼                               ▼
    ┌───────────────┐              ┌───────────────┐
    │ Return Result │              │ Check Circuit │
    │               │              │   Breaker     │
    └───────────────┘              └───────┬───────┘
                                           │
                            ┌──────────────┴───────────────┐
                            │                               │
                    Circuit Open                    Circuit Closed
                            │                               │
                            ▼                               ▼
                    ┌───────────────┐              ┌───────────────┐
                    │ Return Error  │              │ Retry with    │
                    │ (Fast Fail)   │              │ Backoff       │
                    └───────────────┘              └───────┬───────┘
                                                           │
                                                           ▼
                                                    ┌───────────────┐
                                                    │ Success?      │
                                                    └───────┬───────┘
                                                            │
                                            ┌───────────────┴───────────────┐
                                            │                               │
                                            YES                             NO
                                            │                               │
                                            ▼                               ▼
                                    ┌───────────────┐              ┌───────────────┐
                                    │ Return Result │              │ Update Circuit│
                                    │               │              │ Log & Return │
                                    └───────────────┘              └───────────────┘
```

## 10. Module Dependency Graph

```
                    app.py
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
sol_price_fetcher  news_sentiment  signal_performance_tracker
        │              │                      │
        │              │                      │
        ▼              ▼                      ▼
  trading_bot    sentence_transformers    sqlite3
  (optional)     scikit-learn
        │
        ▼
    solana SDK
```

## 11. Data Flow - Collection Statistics

```
┌─────────────────────────────────────────────────────────────┐
│         GET /orca                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ get_collection_statistics()       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Open DB Connection                │
        │ (rewards.db)                      │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 1: Best Day                 │
        │ - GROUP BY date                   │
        │ - COUNT collections                │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 2: Largest Collection       │
        │ - ORDER BY amount DESC            │
        │ - LIMIT 1                         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 3: Average Collection       │
        │ - AVG by token type               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 4: Current Streak           │
        │ - Get distinct dates               │
        │ - Calculate consecutive days       │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 5: Token Ratio              │
        │ - SUM USD values by token          │
        │ - Calculate percentages            │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 6: Weekly Patterns          │
        │ - Convert timestamps to local TZ   │
        │ - Count by day of week            │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Query 7: Last 7 Days              │
        │ - Filter by timestamp             │
        │ - SUM USD values                  │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Close DB Connection               │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │ Return Statistics Dict            │
        └───────────────────────────────────┘

ISSUE: 7 separate queries, 7 separate connections
```

---

## Legend

- **Solid arrows**: Direct flow
- **Dashed arrows**: Optional/conditional flow
- **Boxes**: Functions/operations
- **Diamonds**: Decision points
- **Ovals**: Start/End points

---

## Notes

1. These diagrams represent the current implementation
2. Some flows show problematic patterns (marked with "ISSUE")
3. Recommended patterns are shown in separate diagrams
4. See `APP_FLOW_AUDIT.md` for detailed analysis and fixes

