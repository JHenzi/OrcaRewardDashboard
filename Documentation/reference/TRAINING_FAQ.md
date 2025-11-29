# RL Agent Training FAQ

## What About Missing News Data?

**Question**: I only have 2 days of news data but 5 months of price data. Can I still train?

**Answer**: ✅ **YES!** The system handles missing news gracefully.

### How It Works

**Current Data:**
- Price: July 2025 - November 2025 (5 months, 202K+ points)
- News: November 27-28, 2025 only (2 days, 550 articles)

**What Happens During Training:**

1. **For timestamps with news** (Nov 27-28):
   - Uses actual news embeddings and sentiment
   - Model learns from news + price

2. **For timestamps without news** (July - Nov 26):
   - `get_news_at_time()` returns empty list `[]`
   - State encoder pads with zeros:
     - 20 headlines × 384-dim embeddings = all zeros
     - Sentiment scores = all zeros
     - Cluster IDs = all -1
   - Model receives zero-padded news (learns to ignore it)

3. **Model Behavior**:
   - Learns primarily from price patterns (available everywhere)
   - News attention mechanism learns to ignore zero-padded inputs
   - As news accumulates, retraining incorporates it

### Training Phases

**Phase 1 (Now)**: Train on price data + empty news
- Model learns price patterns and technical indicators
- ~99% of training data has no news (padded zeros)
- Model still trains successfully

**Phase 2 (Later)**: Retrain with accumulated news
- After weeks/months, more news data available
- Retraining incorporates news for recent periods
- Model learns to use news when available

### Is This a Problem?

**No!** This is actually fine because:
- ✅ Price data is the primary signal (available everywhere)
- ✅ News is supplementary (enhances decisions when available)
- ✅ Model architecture supports missing news (attention mask)
- ✅ Training works with any mix of news/no-news data

---

## Do We Need to Wait 24 Hours to Train?

**Short Answer: NO** ✅

### Historical Training (What We're Doing)

We're training on **historical data** where we already know what happened 24 hours later:

```
Timeline for Historical Training:
┌─────────────────────────────────────────────────────────┐
│ Historical Data (July - November 2025)                  │
│                                                          │
│  T=0: 2025-11-01 10:00:00  Price: $150                 │
│  T+1h: 2025-11-01 11:00:00  Price: $151  ✅ Known       │
│  T+24h: 2025-11-02 10:00:00 Price: $155  ✅ Known       │
│                                                          │
│  We can calculate reward immediately!                   │
│  Reward = (155 - 150) / 150 = +3.3% (if BUY)           │
└─────────────────────────────────────────────────────────┘
```

**Why this works:**
- All price data is already in `sol_prices.db`
- For any timestamp T, we can look up price at T+24h
- No waiting needed - the future is already in the database!

### Real-Time Training (Not What We're Doing)

If we were training in real-time as decisions are made:

```
Timeline for Real-Time Training (NOT IMPLEMENTED):
┌─────────────────────────────────────────────────────────┐
│  T=0: 2025-11-28 10:00:00  Make decision (BUY)          │
│  T+1h: 2025-11-28 11:00:00  Price: $151  ✅ Known       │
│  T+24h: 2025-11-29 10:00:00 Price: ???  ❌ Unknown      │
│                                                          │
│  Would need to WAIT 24 hours to calculate reward         │
│  Then update model with delayed feedback                 │
└─────────────────────────────────────────────────────────┘
```

**This is slower and more complex** - we're not doing this.

---

## How Does Reward Calculation Work?

### During Historical Training

```python
# In train_rl_agent.py
for step in range(len(episode["prices"])):
    current_price = episode["prices"][step]  # Price at time T
    future_price_24h = episode["future_prices_24h"][step]  # Price at T+24h (already known!)
    
    # Calculate reward immediately
    return_24h = (future_price_24h - current_price) / current_price
    if action == BUY:
        reward = return_24h - transaction_cost
```

**Key**: `future_prices_24h` is pre-calculated when creating episodes from historical data.

### During Real-Time Decision Making (After Training)

Once the model is trained and making real decisions:

```python
# Model makes prediction (no waiting)
predicted_return_24h = model.predict(state)  # Predicts future return
action = model.get_action(state)  # Makes decision now

# Reward calculation happens later (for future retraining)
# But decision is made immediately
```

**Key**: Model predicts future returns, doesn't wait for them.

---

## What About Retraining?

### Periodic Retraining (Recommended)

```bash
# Run weekly
python retrain_rl_agent.py --mode weekly
```

**How it works:**
1. New data accumulates over the week
2. After 1 week, we have 24h outcomes for decisions from 1 week ago
3. Retraining uses this "new historical data"
4. **Still no waiting** - we only retrain on data where 24h outcomes exist

**Example:**
- Week 1: Train on data from July–October (all have 24h outcomes)
- Week 2: Retrain on data from July–November (new November data now has 24h outcomes)
- Week 3: Retrain on data from July–December (new December data now has 24h outcomes)

### Why This Works

- We always train on data that's at least 24 hours old
- By the time we retrain, 24h outcomes are available
- No waiting during training - outcomes are already known

---

## Summary

| Scenario | Need to Wait? | Why? |
|----------|---------------|------|
| **Historical Training** | ❌ NO | Future prices already in database |
| **Real-Time Training** | ✅ YES (24h) | Need to wait for outcomes |
| **Real-Time Decisions** | ❌ NO | Model predicts, doesn't wait |
| **Periodic Retraining** | ❌ NO | Only uses data with known outcomes |

**Our Approach**: Historical training + periodic retraining = No waiting needed! ✅

