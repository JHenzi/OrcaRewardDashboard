# SOL Tracker Page Improvement Plan

## Current State Analysis

### Issues/Defects Identified
1. **Chart Data Loading**: Inefficient loading and processing of price history from cache
2. **Price Prediction Deprecated**: Price prediction feature disabled (returns irrational values)
3. **Limited Technical Analysis**: Basic statistics only (SMA, std dev) - missing comprehensive indicators
4. **No Historical Signal Analysis**: Can't compare past buy/sell/hold signals to actual outcomes
5. **Data Visualization Gaps**: Chart could better integrate technical indicators and signals

### What's Working Well
- ✅ Buy/sell/hold signals from contextual bandit are functional
- ✅ Price data collection and caching is working
- ✅ Basic price statistics (SMA, std dev, range)
- ✅ Bandit logs and portfolio state tracking

---

## Improvement Goals

### Primary Goal
**Transform the page into an intelligent price analysis dashboard that answers:**
- "What is the price of SOL doing?"
- "What are the intelligent signals saying about it?"
- "How did past signals perform?"

---

## Phase 1: Streamline Data Loading & Chart Infrastructure

### 1.1 Optimize Price History Loading
**Problem**: Chart data loading is inefficient, may have issues with timestamp alignment

**Solution**:
- [ ] Refactor `get_price_history()` to use efficient SQL queries with proper indexing
- [ ] Implement data pagination/chunking for large datasets
- [ ] Add caching layer for computed statistics (SMA, indicators)
- [ ] Standardize timestamp handling across all data sources
- [ ] Add data validation and error handling for missing/invalid price points

**Files to Modify**:
- `app.py`: `sol_tracker()` route, `get_price_history()` method - **COMPLETED**
- `sol_price_fetcher.py`: Database query optimization - **COMPLETED**

**Completed Optimizations**:
- ✅ Optimized SQL query to only fetch `timestamp` and `rate` columns (reduced data transfer)
- ✅ Added `limit` parameter to prevent loading excessive data points (max 1000 points)
- ✅ Improved error handling for invalid data
- ✅ Fixed data type conversion (ensure prices are floats)

**Expected Outcome**: Faster page loads, more reliable chart rendering - **ACHIEVED**

**Future Optimization**: Add database index on `timestamp` column for even faster queries:
```sql
CREATE INDEX IF NOT EXISTS idx_sol_prices_timestamp ON sol_prices(timestamp);
```

---

### 1.2 Enhance Chart Visualization
**Current**: Basic line chart with buy/sell/hold scatter points

**Improvements**:
- [x] Add multiple Y-axes for price vs. indicators (RSI, volume, etc.) - **COMPLETED** (RSI on secondary Y-axis)
- [ ] Implement zoom/pan functionality for detailed analysis
- [ ] Add technical indicator overlays (SMA lines, Bollinger Bands)
- [x] Improve signal visualization (larger markers, tooltips with context) - **ENHANCED** (RSI tooltips with overbought/oversold status)
- [ ] Add price action patterns (support/resistance levels)

**Technology**: Enhance Chart.js configuration or consider alternative (TradingView Lightweight Charts)

**Completed Chart Enhancements**:
- ✅ Added RSI line on secondary Y-axis (right side, 0-100 scale)
- ✅ RSI tooltips show overbought (>70) / oversold (<30) status
- ✅ Proper axis labels for both price and RSI
- ✅ Chart order optimized (signals on top, price line, then RSI)

---

## Phase 2: Technical Indicators & Analysis

### 2.1 Core Technical Indicators
**Goal**: Add comprehensive technical analysis tools

**Indicators to Implement**:

1. **Momentum Indicators**:
   - [ ] RSI (Relative Strength Index) - Already calculated, need to display
   - [ ] MACD (Moving Average Convergence Divergence)
   - [ ] Stochastic Oscillator
   - [ ] Williams %R

2. **Trend Indicators**:
   - [ ] Multiple SMA/EMA lines (already have SMA 1h, 4h, 24h - expand)
   - [ ] Bollinger Bands
   - [ ] ADX (Average Directional Index)
   - [ ] Parabolic SAR

3. **Volume Indicators**:
   - [ ] Volume bars (if volume data available)
   - [ ] On-Balance Volume (OBV)
   - [ ] Volume Weighted Average Price (VWAP)

4. **Volatility Indicators**:
   - [ ] ATR (Average True Range)
   - [ ] Bollinger Band width
   - [ ] Historical volatility (already have std dev)

**Implementation**:
- [ ] Create `technical_indicators.py` module with indicator calculations
- [ ] Add indicator computation to `compute_price_features()` or new function
- [ ] Store computed indicators in database or cache for performance
- [ ] Add indicator selection UI (toggle on/off)

**Files to Create/Modify**:
- `technical_indicators.py` (new)
- `sol_price_fetcher.py`: Add indicator calculations
- `app.py`: Pass indicators to template
- `templates/sol_tracker.html`: Display indicators

---

## Phase 3: Intelligent Signal Analysis

### 3.1 Enhanced Signal Display
**Current**: Shows latest buy/sell/hold action with scores

**Improvements**:
- [ ] Signal strength indicator (visual gauge/percentage)
- [ ] Signal confidence level based on historical accuracy
- [ ] Contextual information (why this signal? - show key features)
- [ ] Signal history timeline (when signals changed)
- [ ] Signal frequency analysis (how often does bandit change mind?)

**UI Components**:
- [ ] Signal strength meter (0-100%)
- [ ] Feature importance display (what drove the decision?)
- [ ] Signal change notifications (alert when signal flips)

---

### 3.2 Historical Signal Performance Analysis
**Goal**: "Previously we predicted a buy event - but was it correct?"

**Features**:
- [ ] **Signal Outcome Tracking**:
  - [ ] For each buy signal: Track price X hours/days later
  - [ ] For each sell signal: Track if it was profitable
  - [ ] For each hold signal: Track opportunity cost

- [ ] **Performance Metrics**:
  - [ ] Signal accuracy rate (% of profitable signals)
  - [ ] Average return per signal type
  - [ ] Best/worst performing signals
  - [ ] Reward distribution analysis

- [ ] **Retrospective Analysis**:
  - [ ] "If you followed all buy signals, what would your return be?"
  - [ ] "If you followed all sell signals, what would you have avoided?"
  - [ ] Compare bandit rewards to actual outcomes

**Implementation**:
- [ ] Create `signal_analyzer.py` module
- [ ] Add function to compute signal outcomes using reward logic
- [ ] Store signal performance in database
- [ ] Create performance dashboard section in template

**Database Schema Addition**:
```sql
CREATE TABLE IF NOT EXISTS signal_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_timestamp TEXT NOT NULL,
    signal_action TEXT NOT NULL,
    signal_reward REAL,
    price_at_signal REAL,
    price_after_1h REAL,
    price_after_4h REAL,
    price_after_24h REAL,
    actual_outcome REAL,  -- Computed using reward logic
    performance_score REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

**Files to Create/Modify**:
- `signal_analyzer.py` (new)
- `sol_price_fetcher.py`: Add signal outcome tracking
- `app.py`: Add signal performance analysis route
- `templates/sol_tracker.html`: Add performance dashboard section

---

## Phase 4: Advanced Features

### 4.1 Pattern Recognition
- [ ] Identify common price patterns (head & shoulders, triangles, etc.)
- [ ] Match current price action to historical patterns
- [ ] Pattern-based signal confidence adjustment

### 4.2 Multi-Timeframe Analysis
- [ ] Display signals across different timeframes (1h, 4h, 24h)
- [ ] Show signal alignment/conflicts across timeframes
- [ ] Weighted signal strength based on timeframe agreement

### 4.3 Alert System
- [ ] Email/push notifications for significant signal changes
- [ ] Price threshold alerts
- [ ] Custom alert conditions

### 4.4 Export & Reporting
- [ ] Export price history and signals to CSV
- [ ] Generate PDF reports with charts and analysis
- [ ] API endpoints for programmatic access

---

## Implementation Priority

### High Priority (Do First)
1. ✅ Streamline chart data loading (Phase 1.1) - **COMPLETED**
2. ✅ Add RSI display (already calculated, just need to show) - **COMPLETED**
3. ✅ Enhance chart visualization (Phase 1.2) - **PARTIALLY COMPLETED** (RSI added to chart)
4. ⏳ Historical signal performance analysis (Phase 3.2) - **IN PROGRESS**

### Medium Priority
5. Additional technical indicators (Phase 2.1)
6. Enhanced signal display (Phase 3.1)
7. Multi-timeframe analysis (Phase 4.2)

### Low Priority (Future)
8. Pattern recognition (Phase 4.1)
9. Alert system (Phase 4.3)
10. Export & reporting (Phase 4.4)

---

## Technical Considerations

### Performance
- Cache computed indicators (don't recalculate on every page load)
- Use database indexes on timestamp columns
- Implement lazy loading for historical data
- Consider using a time-series database (InfluxDB) if data grows large

### Data Quality
- Handle missing price data gracefully
- Validate indicator calculations (edge cases, division by zero)
- Add data quality checks and warnings

### User Experience
- Progressive enhancement (show basic chart first, load indicators after)
- Responsive design for mobile devices
- Loading states and error messages
- Tooltips and help text for technical terms

---

## Success Metrics

1. **Page Load Time**: < 2 seconds for 24h view
2. **Chart Responsiveness**: Smooth zoom/pan, no lag
3. **Signal Accuracy**: Track and display historical accuracy
4. **User Engagement**: Users can answer "What should I do?" from the page

---

## Next Steps

1. **Immediate**: Fix chart data loading issues
2. **Week 1**: Add RSI and additional SMAs to chart
3. **Week 2**: Implement signal performance tracking
4. **Week 3**: Add more technical indicators
5. **Ongoing**: Iterate based on usage and feedback

---

## Notes

- Keep buy/sell/hold signals as the core feature - they're working well
- Price prediction is deprecated - focus on signal quality, not price prediction
- Use reward logic from `calculate_reward()` to evaluate signal outcomes
- Consider A/B testing different signal display formats

