# AI Features Implementation Plan
## Prioritized Roadmap for Advanced AI Features

**Last Updated:** 2025-11-27

---

## 🎯 Recommended Implementation Order

### Phase 1: Performance Evaluation Framework (EASIEST - Start Here)
**Estimated Time:** 2-4 hours  
**Dependencies:** None (uses existing data)

**Why Start Here:**
- ✅ No new dependencies required
- ✅ Foundation for evaluating other AI features
- ✅ Immediate value - understand current system performance
- ✅ Can be built incrementally

**What to Build:**
1. **Signal Performance Tracker**
   - Track when each signal (RSI, Bandit, etc.) was generated
   - Record price at signal time
   - Track price X hours/days later
   - Calculate: accuracy, average return, win rate

2. **Performance Metrics Dashboard**
   - Add to SOL Tracker page
   - Show: "RSI Buy signals: 65% profitable, avg +2.3%"
   - Historical performance charts
   - Best/worst performing signals

3. **Database Schema**
   ```sql
   CREATE TABLE IF NOT EXISTS signal_performance (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       signal_type TEXT NOT NULL,  -- 'rsi_buy', 'rsi_sell', 'bandit_buy', etc.
       signal_timestamp TEXT NOT NULL,
       price_at_signal REAL NOT NULL,
       price_1h_later REAL,
       price_4h_later REAL,
       price_24h_later REAL,
       price_7d_later REAL,
       return_1h REAL,
       return_4h REAL,
       return_24h REAL,
       return_7d REAL,
       was_profitable BOOLEAN,
       created_at DATETIME DEFAULT CURRENT_TIMESTAMP
   );
   ```

**Implementation Steps:**
1. Create `signal_performance_tracker.py` module
2. Add signal logging when RSI/Bandit signals are generated
3. Create background job to update performance metrics
4. Add performance display to SOL Tracker page
5. Create API endpoint for performance data

---

### Phase 2: News Sentiment Analysis (MEDIUM - Good Next Step)
**Estimated Time:** 4-8 hours  
**Dependencies:** `feedparser`, `transformers` or `textblob` (lighter option)

**Why This Next:**
- ✅ Moderate complexity
- ✅ Can use lightweight libraries (TextBlob) for quick start
- ✅ Immediate value - news impacts crypto prices
- ✅ Can integrate with existing bandit system

**What to Build:**
1. **News Fetcher**
   - RSS feed parser for crypto news
   - Filter for Solana-related articles
   - Store articles in database

2. **Sentiment Analyzer**
   - Option A (Lightweight): Use TextBlob for sentiment scores
   - Option B (Advanced): Use sentence-transformers for embeddings
   - Calculate sentiment score per article
   - Aggregate sentiment over time windows

3. **Integration**
   - Add sentiment features to bandit model
   - Display sentiment on SOL Tracker page
   - Use sentiment as additional signal

**Dependencies to Add:**
```txt
feedparser==6.0.10
textblob==0.17.1
# OR for advanced:
# sentence-transformers==2.2.2
# torch (if using sentence-transformers)
```

**Implementation Steps:**
1. Create `news_sentiment.py` module
2. Set up RSS feed sources (CoinDesk, CryptoSlate, etc.)
3. Implement sentiment analysis
4. Store results in database
5. Add sentiment display to UI
6. Integrate with bandit features

---

### Phase 3: LSTM Price Prediction (MOST COMPLEX - Do Last)
**Estimated Time:** 8-16 hours  
**Dependencies:** `tensorflow` or `pytorch`, `scikit-learn`

**Why Last:**
- ⚠️ Most complex implementation
- ⚠️ Requires significant data preparation
- ⚠️ Needs model training infrastructure
- ⚠️ Heavy dependencies (TensorFlow ~500MB)
- ⚠️ Previous price prediction was deprecated for "irrational values"

**What to Build:**
1. **Data Preparation Pipeline**
   - Sequence generation (60-minute windows)
   - Data normalization
   - Train/validation/test splits
   - Feature engineering

2. **LSTM Model**
   - Model architecture (3-layer LSTM)
   - Training loop with early stopping
   - Model checkpointing
   - Hyperparameter tuning

3. **Prediction System**
   - Real-time prediction generation
   - Confidence intervals
   - Integration with existing signals
   - Performance tracking

**Dependencies to Add:**
```txt
tensorflow==2.15.0
# OR
# torch==2.1.0
scikit-learn==1.3.2
```

**Implementation Steps:**
1. Create `lstm_price_predictor.py` module
2. Build data preparation pipeline
3. Design and train LSTM model
4. Implement prediction generation
5. Add to SOL Tracker page
6. Track prediction accuracy

**⚠️ Important Note:**
Previous price prediction was deprecated. Before implementing LSTM:
- Analyze why previous predictions were "irrational"
- Ensure proper data normalization
- Add confidence intervals
- Validate predictions against actuals
- Start with simple model, iterate

---

### Phase 4: Enhanced Trading System (ENSEMBLE)
**Estimated Time:** 4-8 hours  
**Dependencies:** All of the above

**What to Build:**
1. **Ensemble Model**
   - Combine LSTM predictions + Bandit signals + News sentiment
   - Weighted voting system
   - Confidence scoring

2. **Decision Engine**
   - Risk management rules
   - Position sizing logic
   - Signal filtering (only act on high-confidence signals)

3. **Performance Tracking**
   - Track ensemble vs individual model performance
   - A/B testing framework
   - Continuous improvement loop

---

## 📊 Quick Start Recommendation

### Start with Performance Evaluation (Today)

**File to Create:** `signal_performance_tracker.py`

```python
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List

class SignalPerformanceTracker:
    def __init__(self, db_path="sol_prices.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create signal_performance table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                signal_timestamp TEXT NOT NULL,
                price_at_signal REAL NOT NULL,
                price_1h_later REAL,
                price_4h_later REAL,
                price_24h_later REAL,
                return_1h REAL,
                return_4h REAL,
                return_24h REAL,
                was_profitable BOOLEAN,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_signal(self, signal_type: str, price: float):
        """Log a new signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO signal_performance 
            (signal_type, signal_timestamp, price_at_signal)
            VALUES (?, ?, ?)
        ''', (signal_type, datetime.utcnow().isoformat(), price))
        conn.commit()
        conn.close()
    
    def update_performance(self, hours_later: int = 24):
        """Update performance metrics for signals older than X hours"""
        # Implementation to fetch prices and calculate returns
        pass
    
    def get_performance_stats(self, signal_type: Optional[str] = None) -> Dict:
        """Get performance statistics for signals"""
        # Implementation to calculate win rate, avg return, etc.
        pass
```

**Integration Points:**
1. Call `log_signal()` when RSI signals are generated
2. Call `log_signal()` when Bandit recommendations are made
3. Add background job to update performance metrics
4. Display stats on SOL Tracker page

---

## 🎯 Success Criteria

### Performance Evaluation
- [ ] Track all signal types (RSI, Bandit, MACD, etc.)
- [ ] Calculate win rate, average return, Sharpe ratio
- [ ] Display performance metrics on dashboard
- [ ] Historical performance charts

### News Sentiment
- [ ] Fetch news from 3+ RSS feeds
- [ ] Calculate sentiment scores
- [ ] Display sentiment on SOL Tracker
- [ ] Integrate with bandit features

### LSTM Prediction
- [ ] Train model on historical data
- [ ] Generate predictions with confidence intervals
- [ ] Track prediction accuracy
- [ ] Display on SOL Tracker page

### Ensemble System
- [ ] Combine all signals
- [ ] Weighted decision making
- [ ] Performance tracking
- [ ] A/B testing framework

---

## 📦 Dependencies Summary

**Current:**
- ✅ numpy, pandas (already installed)
- ✅ river (online ML - already installed)

**To Add:**
- Performance Evaluation: None
- News Sentiment: `feedparser`, `textblob` (or `sentence-transformers`)
- LSTM: `tensorflow` (or `torch`), `scikit-learn`
- Ensemble: All of the above

---

## 🚀 Recommended Next Steps

1. **This Week:** Implement Performance Evaluation Framework
2. **Next Week:** Add News Sentiment Analysis (start with TextBlob for simplicity)
3. **Following Weeks:** Evaluate LSTM approach, start with simple model
4. **Future:** Build ensemble system combining all signals

---

**Note:** Start simple, iterate, and measure everything. Performance evaluation should be the foundation before adding more complex models.

