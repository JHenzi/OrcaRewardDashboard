# SOL Tracker Page Improvement Plan

> **Updated**: This plan now includes integration of the NewAgent RL-based trading system as described in `NewAgent.md`. The agent will autonomously discover trading rules, provide explainable decisions, and integrate multi-modal features (price, technical indicators, news embeddings, sentiment).

---

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

## Phase 4: RL Agent Implementation (NewAgent Integration)

### 4.1 RL Agent Architecture Setup
**Goal**: Replace/enhance contextual bandit with full RL agent (PPO/actor-critic) that learns sequential trading strategies

**Components to Implement**:

1. **Model Architecture**:
   - [ ] Price branch: 1D-CNN or small LSTM → latent_p (128 dim)
   - [ ] News branch: Multi-head attention over embeddings [M x E] → latent_n (128 dim)
   - [ ] Position/time branch → latent_s (32 dim)
   - [ ] Concatenate → shared latent (256 dim)
   - [ ] Actor head (policy) → softmax over {SELL, HOLD, BUY}
   - [ ] Critic head → value scalar
   - [ ] Auxiliary heads → predict `return_1h` and `return_24h` (regression)

2. **Environment/Gym Setup**:
   - [ ] Create trading environment with discrete actions
   - [ ] Implement realistic transaction cost model
   - [ ] Paper-trade logging and state tracking
   - [ ] Reward function: `r_t = log(p_{t+1}/p_t) - λ_cost * transaction_cost - λ_risk * pos_t^2`

3. **Training Infrastructure**:
   - [ ] PPO training loop with GAE (Generalized Advantage Estimation)
   - [ ] N-step returns for multi-horizon learning
   - [ ] Experience replay buffer
   - [ ] Online training on paper-trade logs
   - [ ] Periodic updates with complete 1h/24h return data

**Files to Create/Modify**:
- `rl_agent/` (new directory)
  - `rl_agent/model.py` - PyTorch actor-critic model with attention
  - `rl_agent/environment.py` - Trading environment (Gym-style)
  - `rl_agent/trainer.py` - PPO training loop
  - `rl_agent/state_encoder.py` - Feature encoding (price, news, position)
- `app.py`: Integrate RL agent alongside/after bandit
- `sol_price_fetcher.py`: Add state encoding functions

**Dependencies to Add**:
- `torch` (PyTorch)
- `stable-baselines3` or custom PPO implementation
- `gym` or `gymnasium` (for environment interface)

---

### 4.2 Multi-Horizon Return Predictions
**Goal**: Display agent's predictions for 1h and 24h returns alongside current signals

**Features**:
- [ ] Display predicted 1h return (from auxiliary head)
- [ ] Display predicted 24h return (from auxiliary head)
- [ ] Show prediction confidence/uncertainty
- [ ] Track prediction accuracy over time
- [ ] Visualize predicted vs actual returns on chart

**UI Components**:
- [ ] Prediction cards showing 1h/24h forecasts
- [ ] Confidence intervals or uncertainty bands
- [ ] Historical prediction accuracy metrics
- [ ] Chart overlay showing predicted price paths

**Files to Modify**:
- `app.py`: Add prediction endpoints
- `templates/sol_tracker.html`: Add prediction display section
- `rl_agent/model.py`: Ensure aux heads output predictions

---

### 4.3 News Embedding & Attention Integration
**Goal**: Integrate news embeddings with attention mechanism to highlight influential headlines

**Features**:
- [ ] Multi-head attention over latest M headlines (e.g., M=20)
- [ ] Store attention weights for each decision
- [ ] Display top-k headlines that influenced current signal
- [ ] Attention heatmap visualization
- [ ] News topic clustering (HDBSCAN/KMeans) for interpretability

**Implementation**:
- [ ] Enhance news branch in RL model with attention mechanism
- [ ] Log attention weights with each decision
- [ ] Create topic clusters from embeddings (daily/weekly)
- [ ] Map clusters to human-readable labels (e.g., "politics", "earnings", "regulation")

**Files to Create/Modify**:
- `rl_agent/model.py`: Add attention mechanism to news branch
- `rl_agent/attention_logger.py` (new): Log attention weights
- `news_sentiment.py`: Add topic clustering functionality
- `app.py`: Fetch and display attention-highlighted headlines
- `templates/sol_tracker.html`: Add attention visualization section

---

### 4.4 Safe Exploration & Risk Management
**Goal**: Implement conservative exploration with hard safety constraints

**Safety Features**:
- [ ] Max position size limit (< 0.1% of capital initially)
- [ ] Trade frequency limit (≤ 5 trades/hour)
- [ ] Daily loss cap (auto-disable if exceeded)
- [ ] Entropy-driven exploration (high initially, decay slowly)
- [ ] Thompson-like seeding for early decisions
- [ ] Uncertainty-aware action selection (force HOLD if uncertainty high)

**Implementation**:
- [ ] Add safety constraints to environment
- [ ] Implement uncertainty estimation (ensemble/bootstrap)
- [ ] Add risk monitoring dashboard
- [ ] Paper-trade mode with full logging

**Files to Create/Modify**:
- `rl_agent/environment.py`: Add safety constraints
- `rl_agent/risk_manager.py` (new): Risk monitoring and limits
- `app.py`: Display risk metrics
- `templates/sol_tracker.html`: Add risk dashboard section

---

## Phase 5: Explainability & Theory Discovery

### 5.1 Rule Extraction Pipeline
**Goal**: Extract human-readable rules/theories from the RL agent's learned behavior

**Features**:
- [ ] Train shallow decision tree surrogate on agent's decisions
- [ ] Extract top rules (e.g., "If [cluster='politics'] & sentiment < -0.3 & RSI < 40 → BUY")
- [ ] Compute rule performance metrics (win rate, avg return, sample size)
- [ ] Bootstrap confidence intervals for rule statistics
- [ ] Rule validation against historical outcomes

**Implementation**:
- [ ] Periodically (daily/weekly) sample labeled dataset: state → action/outcome
- [ ] Train decision tree or RuleFit on (state features → action success)
- [ ] Export rules in human-readable format
- [ ] Store rules in database with performance stats

**Files to Create/Modify**:
- `rl_agent/explainability.py` (new): Rule extraction pipeline
- `app.py`: Endpoint to fetch and display discovered rules
- `templates/sol_tracker.html`: Add "Discovered Rules" section
- Database: Add `discovered_rules` table

**Database Schema**:
```sql
CREATE TABLE IF NOT EXISTS discovered_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_text TEXT NOT NULL,
    rule_conditions TEXT,  -- JSON of conditions
    action TEXT NOT NULL,
    win_rate REAL,
    avg_return_1h REAL,
    avg_return_24h REAL,
    sample_size INTEGER,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_validated_at DATETIME
);
```

---

### 5.2 SHAP Feature Importance
**Goal**: Rank features and news embeddings by importance using SHAP values

**Features**:
- [ ] Compute SHAP values for policy logits and auxiliary heads
- [ ] Rank features by importance (indicators, sentiment, embedding clusters)
- [ ] Visualize feature importance charts
- [ ] Show how features contribute to current decision
- [ ] Track feature importance over time

**Implementation**:
- [ ] Use SHAP library (TreeExplainer or KernelExplainer)
- [ ] Compute SHAP values for recent decisions
- [ ] Aggregate importance scores
- [ ] Display in dashboard

**Files to Create/Modify**:
- `rl_agent/explainability.py`: Add SHAP computation
- `app.py`: Endpoint for feature importance
- `templates/sol_tracker.html`: Add SHAP visualization (bar charts, waterfall plots)

**Dependencies**:
- `shap` library

---

### 5.3 Attention-Based Saliency
**Goal**: Use attention weights to highlight which headlines influenced decisions

**Features**:
- [ ] Display top-k headlines with highest attention weights for current signal
- [ ] Attention heatmap showing headline importance over time
- [ ] Link headlines to discovered rules (show headlines when rule fired)
- [ ] Representative headline sampling per topic cluster

**UI Components**:
- [ ] "Influential Headlines" section showing top-k headlines
- [ ] Attention weight visualization (bar chart or heatmap)
- [ ] Clickable headlines linking to full articles
- [ ] Timeline showing attention weights over recent decisions

**Files to Modify**:
- `app.py`: Fetch attention logs and top headlines
- `templates/sol_tracker.html`: Add attention visualization section
- `rl_agent/attention_logger.py`: Store attention weights with decisions

---

### 5.4 Topic Discovery & Cluster Narratives
**Goal**: Discover news topics via clustering and create human-readable narratives

**Features**:
- [ ] Run lightweight clustering (HDBSCAN/KMeans) on recent embeddings
- [ ] Map clusters to human-readable labels by sampling representative headlines
- [ ] Use cluster ID as categorical feature in rules
- [ ] Display cluster narratives (e.g., "politics_trump", "earnings", "regulation")

**Implementation**:
- [ ] Daily/weekly clustering on recent news embeddings
- [ ] Label clusters by sampling top headlines
- [ ] Store cluster mappings in database
- [ ] Include cluster features in rule extraction

**Files to Create/Modify**:
- `news_sentiment.py`: Add clustering functionality
- `rl_agent/state_encoder.py`: Include cluster IDs as features
- `app.py`: Display cluster information
- `templates/sol_tracker.html`: Add topic/cluster display

**Database Schema**:
```sql
CREATE TABLE IF NOT EXISTS news_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    representative_headlines TEXT,  -- JSON array
    embedding_centroid BLOB,  -- Store cluster center
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Phase 6: Dashboard Integration & Visualization

### 6.1 RL Agent Dashboard Section
**Goal**: Add comprehensive dashboard section for RL agent status and outputs

**Dashboard Components**:
- [ ] **Agent Status Panel**:
  - [ ] Current policy version/training step
  - [ ] Portfolio value (paper-trade)
  - [ ] Recent action history
  - [ ] Training metrics (loss, entropy, value estimates)

- [ ] **Prediction Panel**:
  - [ ] 1h return prediction with confidence
  - [ ] 24h return prediction with confidence
  - [ ] Historical prediction accuracy chart

- [ ] **Explainability Panel**:
  - [ ] Top discovered rules with performance stats
  - [ ] Feature importance (SHAP) visualization
  - [ ] Attention-highlighted headlines
  - [ ] Topic clusters and narratives

- [ ] **Rule Performance Table**:
  - [ ] List of discovered rules
  - [ ] Win rate, avg return, sample size
  - [ ] Confidence intervals
  - [ ] Filter/sort capabilities

**Files to Modify**:
- `templates/sol_tracker.html`: Add comprehensive RL agent dashboard section
- `app.py`: Add endpoints for all dashboard data
- `static/` (if exists): Add CSS/JS for new visualizations

---

### 6.2 Enhanced Chart Visualizations
**Goal**: Integrate RL agent outputs into existing chart

**Chart Enhancements**:
- [ ] Overlay predicted 1h/24h price paths
- [ ] Highlight time periods when specific rules fired
- [ ] Show attention-weighted news events on timeline
- [ ] Display agent actions (BUY/SELL/HOLD) with confidence scores
- [ ] Multi-timeframe view (1h, 4h, 24h predictions)

**Files to Modify**:
- `templates/sol_tracker.html`: Enhance Chart.js configuration
- `app.py`: Provide prediction and rule data to chart

---

### 6.3 Theory Discovery Report
**Goal**: Generate periodic reports summarizing discovered theories and their performance

**Report Contents**:
- [ ] Top 10 most successful rules
- [ ] Feature importance rankings
- [ ] News topic clusters and their impact
- [ ] Prediction accuracy metrics
- [ ] Portfolio performance vs baseline strategies

**Implementation**:
- [ ] Daily/weekly report generation
- [ ] Export to HTML/PDF
- [ ] Email or in-app notification

**Files to Create/Modify**:
- `rl_agent/report_generator.py` (new): Generate theory discovery reports
- `app.py`: Add report endpoint
- `templates/theory_report.html` (new): Report template

---

## Phase 7: Advanced Features (Original)

### 7.1 Pattern Recognition
- [ ] Identify common price patterns (head & shoulders, triangles, etc.)
- [ ] Match current price action to historical patterns
- [ ] Pattern-based signal confidence adjustment

### 7.2 Multi-Timeframe Analysis
- [ ] Display signals across different timeframes (1h, 4h, 24h)
- [ ] Show signal alignment/conflicts across timeframes
- [ ] Weighted signal strength based on timeframe agreement

### 7.3 Alert System
- [ ] Email/push notifications for significant signal changes
- [ ] Price threshold alerts
- [ ] Custom alert conditions
- [ ] Rule discovery notifications (new high-performing rule found)

### 7.4 Export & Reporting
- [ ] Export price history and signals to CSV
- [ ] Generate PDF reports with charts and analysis
- [ ] API endpoints for programmatic access
- [ ] Export discovered rules and theories

---

## Implementation Priority

### High Priority (Do First - Foundation)
1. ✅ Streamline chart data loading (Phase 1.1) - **COMPLETED**
2. ✅ Add RSI display (already calculated, just need to show) - **COMPLETED**
3. ✅ Enhance chart visualization (Phase 1.2) - **PARTIALLY COMPLETED** (RSI added to chart)
4. ⏳ Historical signal performance analysis (Phase 3.2) - **IN PROGRESS**

### High Priority (NewAgent Integration - Phase 1)
5. **RL Agent Architecture Setup** (Phase 4.1)
   - Create model architecture (actor-critic with attention)
   - Set up trading environment
   - Implement basic PPO training loop
   - **Estimated effort**: 3-5 days

6. **Multi-Horizon Return Predictions** (Phase 4.2)
   - Add auxiliary heads for 1h/24h predictions
   - Display predictions in dashboard
   - **Estimated effort**: 2-3 days

7. **News Embedding & Attention Integration** (Phase 4.3)
   - Integrate attention mechanism into model
   - Log attention weights
   - Display influential headlines
   - **Estimated effort**: 2-3 days

### Medium Priority (NewAgent Integration - Phase 2)
8. **Rule Extraction Pipeline** (Phase 5.1)
   - Implement decision tree surrogate
   - Extract and store rules
   - Display rules in dashboard
   - **Estimated effort**: 2-3 days

9. **SHAP Feature Importance** (Phase 5.2)
   - Integrate SHAP library
   - Compute and display feature importance
   - **Estimated effort**: 1-2 days

10. **Attention-Based Saliency** (Phase 5.3)
    - Visualize attention weights
    - Link headlines to decisions
    - **Estimated effort**: 1-2 days

11. **Topic Discovery & Clustering** (Phase 5.4)
    - Implement news clustering
    - Create cluster narratives
    - **Estimated effort**: 2-3 days

### Medium Priority (Original Features)
12. Additional technical indicators (Phase 2.1)
13. Enhanced signal display (Phase 3.1)
14. Dashboard Integration (Phase 6.1, 6.2)
15. Multi-timeframe analysis (Phase 7.2)

### Low Priority (Future)
16. Pattern recognition (Phase 7.1)
17. Alert system (Phase 7.3)
18. Export & reporting (Phase 7.4)
19. Theory Discovery Reports (Phase 6.3)

---

## Database Schema Additions

### RL Agent Tables

#### `rl_agent_decisions`
Store each decision made by the RL agent:
```sql
CREATE TABLE IF NOT EXISTS rl_agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    action TEXT NOT NULL,  -- BUY, SELL, HOLD
    action_probabilities TEXT,  -- JSON: {BUY: 0.3, SELL: 0.1, HOLD: 0.6}
    state_features TEXT,  -- JSON of state at decision time
    predicted_return_1h REAL,
    predicted_return_24h REAL,
    predicted_return_1h_confidence REAL,
    predicted_return_24h_confidence REAL,
    value_estimate REAL,
    entropy REAL,
    portfolio_value REAL,
    position_size REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_rl_decisions_timestamp ON rl_agent_decisions(timestamp);
```

#### `rl_attention_logs`
Store attention weights for each decision:
```sql
CREATE TABLE IF NOT EXISTS rl_attention_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id INTEGER NOT NULL,
    headline_id INTEGER,  -- Reference to news_sentiment table
    headline_text TEXT,
    attention_weight REAL,
    cluster_id INTEGER,
    FOREIGN KEY (decision_id) REFERENCES rl_agent_decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_attention_decision ON rl_attention_logs(decision_id);
```

#### `discovered_rules`
Store extracted rules (already defined in Phase 5.1):
```sql
CREATE TABLE IF NOT EXISTS discovered_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_text TEXT NOT NULL,
    rule_conditions TEXT,  -- JSON of conditions
    action TEXT NOT NULL,
    win_rate REAL,
    avg_return_1h REAL,
    avg_return_24h REAL,
    sample_size INTEGER,
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_validated_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_rules_action ON discovered_rules(action);
CREATE INDEX IF NOT EXISTS idx_rules_win_rate ON discovered_rules(win_rate DESC);
```

#### `news_clusters`
Store topic clusters (already defined in Phase 5.4):
```sql
CREATE TABLE IF NOT EXISTS news_clusters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id INTEGER NOT NULL,
    cluster_label TEXT,
    representative_headlines TEXT,  -- JSON array
    embedding_centroid BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_clusters_id ON news_clusters(cluster_id);
```

#### `rl_training_metrics`
Store training metrics for monitoring:
```sql
CREATE TABLE IF NOT EXISTS rl_training_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    training_step INTEGER NOT NULL,
    policy_loss REAL,
    value_loss REAL,
    entropy REAL,
    total_loss REAL,
    learning_rate REAL,
    clip_fraction REAL,
    explained_variance REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_step ON rl_training_metrics(training_step);
```

#### `rl_rule_firings`
Track when rules fire in production:
```sql
CREATE TABLE IF NOT EXISTS rl_rule_firings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    decision_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    outcome_1h REAL,  -- Filled after 1h
    outcome_24h REAL,  -- Filled after 24h
    FOREIGN KEY (rule_id) REFERENCES discovered_rules(id),
    FOREIGN KEY (decision_id) REFERENCES rl_agent_decisions(id)
);

CREATE INDEX IF NOT EXISTS idx_rule_firings_rule ON rl_rule_firings(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_firings_timestamp ON rl_rule_firings(timestamp);
```

---

## Technical Considerations

### Performance
- Cache computed indicators (don't recalculate on every page load)
- Use database indexes on timestamp columns
- Implement lazy loading for historical data
- Consider using a time-series database (InfluxDB) if data grows large
- **NEW**: Cache SHAP values (expensive computation) - recompute daily, not per-request
- **NEW**: Batch attention weight storage (don't write on every decision)
- **NEW**: Use model checkpoints for RL agent (save/load trained models)
- **NEW**: Implement async/background processing for rule extraction (daily job)

### Data Quality
- Handle missing price data gracefully
- Validate indicator calculations (edge cases, division by zero)
- Add data quality checks and warnings
- **NEW**: Validate news embeddings (check for NaN, infinite values)
- **NEW**: Monitor prediction accuracy and flag if accuracy drops
- **NEW**: Validate rule extraction (ensure rules are statistically significant)

### User Experience
- Progressive enhancement (show basic chart first, load indicators after)
- Responsive design for mobile devices
- Loading states and error messages
- Tooltips and help text for technical terms
- **NEW**: Progressive loading for explainability sections (load rules first, then SHAP)
- **NEW**: Interactive tooltips for rules (show rule details on hover)
- **NEW**: Collapsible sections for mobile view
- **NEW**: Real-time updates via polling or WebSocket for agent status

### Model Management
- **NEW**: Version control for RL agent models (save checkpoints with version numbers)
- **NEW**: A/B testing framework (run multiple model versions in parallel)
- **NEW**: Model rollback capability (revert to previous model if performance degrades)
- **NEW**: Model monitoring (track prediction drift, accuracy over time)

### Security & Safety
- **NEW**: Hard limits on position sizing (enforced in environment)
- **NEW**: Daily loss caps (auto-disable trading if exceeded)
- **NEW**: Trade frequency limits (prevent overtrading)
- **NEW**: Paper-trade mode by default (require explicit flag for live trading)
- **NEW**: Audit logging for all agent decisions and rule extractions

---

## Success Metrics

1. **Page Load Time**: < 2 seconds for 24h view
2. **Chart Responsiveness**: Smooth zoom/pan, no lag
3. **Signal Accuracy**: Track and display historical accuracy
4. **User Engagement**: Users can answer "What should I do?" from the page

---

## Template Changes Required

### `templates/sol_tracker.html` - New Sections to Add

#### 1. RL Agent Status Panel
**Location**: Add new section after current signal display
**Components**:
- [ ] Agent status card showing:
  - Current policy version/training step
  - Paper-trade portfolio value
  - Recent action history (last 10 actions)
  - Training metrics (loss, entropy, value estimates)
- [ ] Real-time update via AJAX/WebSocket

**HTML Structure**:
```html
<div class="rl-agent-status-panel">
  <h3>RL Agent Status</h3>
  <div class="status-grid">
    <div class="status-card">Policy Version: <span id="policy-version"></span></div>
    <div class="status-card">Portfolio Value: <span id="portfolio-value"></span></div>
    <div class="status-card">Training Step: <span id="training-step"></span></div>
  </div>
  <div class="recent-actions">
    <h4>Recent Actions</h4>
    <ul id="recent-actions-list"></ul>
  </div>
</div>
```

---

#### 2. Prediction Panel
**Location**: New section for multi-horizon predictions
**Components**:
- [ ] 1h return prediction card with:
  - Predicted return percentage
  - Confidence interval
  - Prediction timestamp
- [ ] 24h return prediction card (same structure)
- [ ] Historical accuracy chart (Chart.js)
- [ ] Prediction vs actual comparison table

**HTML Structure**:
```html
<div class="prediction-panel">
  <h3>Multi-Horizon Predictions</h3>
  <div class="prediction-cards">
    <div class="prediction-card">
      <h4>1 Hour Prediction</h4>
      <div class="prediction-value" id="pred-1h"></div>
      <div class="confidence-interval" id="ci-1h"></div>
    </div>
    <div class="prediction-card">
      <h4>24 Hour Prediction</h4>
      <div class="prediction-value" id="pred-24h"></div>
      <div class="confidence-interval" id="ci-24h"></div>
    </div>
  </div>
  <canvas id="prediction-accuracy-chart"></canvas>
</div>
```

---

#### 3. Explainability Panel
**Location**: New comprehensive explainability section
**Components**:
- [ ] **Discovered Rules Section**:
  - Table of top rules with:
    - Rule text (human-readable conditions)
    - Action (BUY/SELL/HOLD)
    - Win rate
    - Avg return (1h, 24h)
    - Sample size
    - Confidence intervals
  - Filter/sort functionality
  - Click to see rule details

- [ ] **Feature Importance Section**:
  - SHAP bar chart (horizontal bar chart)
  - Feature names with importance scores
  - Color-coded by positive/negative impact

- [ ] **Attention Headlines Section**:
  - List of top-k headlines with attention weights
  - Visual attention weight bars
  - Clickable headlines (link to source)
  - Timeline view of attention over recent decisions

- [ ] **Topic Clusters Section**:
  - List of discovered topic clusters
  - Cluster labels and representative headlines
  - Cluster impact on decisions

**HTML Structure**:
```html
<div class="explainability-panel">
  <h3>Agent Explainability</h3>
  
  <!-- Discovered Rules -->
  <div class="rules-section">
    <h4>Discovered Trading Rules</h4>
    <table id="rules-table" class="rules-table">
      <thead>
        <tr>
          <th>Rule</th>
          <th>Action</th>
          <th>Win Rate</th>
          <th>Avg Return (1h)</th>
          <th>Avg Return (24h)</th>
          <th>Sample Size</th>
        </tr>
      </thead>
      <tbody id="rules-tbody"></tbody>
    </table>
  </div>
  
  <!-- Feature Importance -->
  <div class="feature-importance-section">
    <h4>Feature Importance (SHAP)</h4>
    <canvas id="shap-chart"></canvas>
  </div>
  
  <!-- Attention Headlines -->
  <div class="attention-headlines-section">
    <h4>Influential Headlines (Current Decision)</h4>
    <div id="attention-headlines-list"></div>
  </div>
  
  <!-- Topic Clusters -->
  <div class="topic-clusters-section">
    <h4>News Topic Clusters</h4>
    <div id="topic-clusters-list"></div>
  </div>
</div>
```

---

#### 4. Enhanced Chart Integration
**Location**: Update existing chart section
**Components**:
- [ ] Overlay predicted price paths (1h, 24h) as dashed lines
- [ ] Markers for rule firing events
- [ ] News event markers with attention weights
- [ ] Agent action markers with confidence scores
- [ ] Toggle switches for each overlay

**Chart.js Updates**:
- [ ] Add new datasets for predictions
- [ ] Add scatter points for rule events
- [ ] Add annotation plugin for news events
- [ ] Add legend with toggles

---

#### 5. Risk Dashboard Section
**Location**: New section for risk monitoring
**Components**:
- [ ] Current position size
- [ ] Trade frequency (trades/hour)
- [ ] Daily P&L
- [ ] Risk limits status (visual indicators)
- [ ] Uncertainty metrics

**HTML Structure**:
```html
<div class="risk-dashboard">
  <h3>Risk Management</h3>
  <div class="risk-metrics">
    <div class="metric-card">
      <label>Position Size</label>
      <div id="position-size"></div>
      <div class="limit-indicator" id="position-limit"></div>
    </div>
    <div class="metric-card">
      <label>Trade Frequency</label>
      <div id="trade-frequency"></div>
      <div class="limit-indicator" id="frequency-limit"></div>
    </div>
    <div class="metric-card">
      <label>Daily P&L</label>
      <div id="daily-pnl"></div>
    </div>
    <div class="metric-card">
      <label>Uncertainty</label>
      <div id="uncertainty-metric"></div>
    </div>
  </div>
</div>
```

---

### JavaScript/API Integration Requirements

#### New API Endpoints Needed in `app.py`:

1. **`/api/rl-agent/status`** - Get agent status
   - Returns: policy version, portfolio value, training step, recent actions

2. **`/api/rl-agent/predictions`** - Get multi-horizon predictions
   - Returns: 1h prediction, 24h prediction, confidence intervals

3. **`/api/rl-agent/rules`** - Get discovered rules
   - Returns: List of rules with performance metrics
   - Query params: `limit`, `sort_by`, `filter_action`

4. **`/api/rl-agent/feature-importance`** - Get SHAP values
   - Returns: Feature names and importance scores

5. **`/api/rl-agent/attention`** - Get attention weights and headlines
   - Returns: Top-k headlines with attention weights for recent decisions

6. **`/api/rl-agent/topics`** - Get topic clusters
   - Returns: List of clusters with labels and representative headlines

7. **`/api/rl-agent/risk`** - Get risk metrics
   - Returns: Position size, trade frequency, daily P&L, uncertainty

#### JavaScript Functions to Add:

```javascript
// Update functions for each section
function updateRLAgentStatus() { /* Fetch and update status */ }
function updatePredictions() { /* Fetch and update predictions */ }
function updateDiscoveredRules() { /* Fetch and render rules table */ }
function updateFeatureImportance() { /* Fetch and render SHAP chart */ }
function updateAttentionHeadlines() { /* Fetch and render headlines */ }
function updateTopicClusters() { /* Fetch and render clusters */ }
function updateRiskDashboard() { /* Fetch and update risk metrics */ }

// Chart enhancement functions
function addPredictionOverlays(chart) { /* Add prediction lines */ }
function addRuleEventMarkers(chart) { /* Add rule firing markers */ }
function addNewsEventMarkers(chart) { /* Add news event markers */ }
```

---

### CSS Styling Requirements

**New CSS Classes Needed**:
- `.rl-agent-status-panel` - Status panel container
- `.prediction-panel` - Prediction cards container
- `.explainability-panel` - Explainability section
- `.rules-table` - Rules table styling
- `.attention-headline-item` - Individual headline with attention bar
- `.topic-cluster-card` - Topic cluster display
- `.risk-dashboard` - Risk metrics grid
- `.limit-indicator` - Visual indicator for risk limits (green/yellow/red)

**Responsive Design**:
- Ensure all new sections are mobile-friendly
- Use CSS Grid/Flexbox for responsive layouts
- Collapsible sections for mobile view

---

## Implementation Status Summary

### ✅ Completed Phases

**Phase 1**: Streamline Data Loading & Chart Infrastructure - **COMPLETED**
- ✅ Optimized price history loading
- ✅ Enhanced chart visualization with RSI

**Phase 2**: Technical Indicators & Analysis - **COMPLETED**
- ✅ RSI calculation and display
- ✅ Multiple SMA lines
- ✅ MACD, Bollinger Bands, momentum indicators

**Phase 3**: Intelligent Signal Analysis - **COMPLETED**
- ✅ Signal performance tracking
- ✅ Historical signal analysis
- ✅ Win rate and return metrics

**Phase 4**: RL Agent Implementation - **COMPLETED**
- ✅ 4.1: RL Agent Architecture Setup
- ✅ 4.2: Multi-Horizon Return Predictions
- ✅ 4.3: News Embedding & Attention Integration
- ✅ 4.4: Safe Exploration & Risk Management

**Phase 5**: Explainability & Theory Discovery - **COMPLETED**
- ✅ 5.1: Rule Extraction Pipeline
- ✅ 5.2: SHAP Feature Importance
- ✅ 5.3: Attention-Based Saliency
- ✅ 5.4: Topic Discovery & Cluster Narratives

**Phase 6**: Dashboard Integration - **COMPLETED**
- ✅ Prediction panels
- ✅ Attention visualization
- ✅ Risk dashboard
- ✅ Rules table
- ✅ Feature importance display

**Integration**: System Integration - **COMPLETED**
- ✅ Price data integration
- ✅ News data integration
- ✅ Decision-making pipeline
- ✅ All API endpoints

### 🎯 Remaining Work

See [WHAT_REMAINS.md](WHAT_REMAINS.md) for detailed next steps. Main items:

1. **Model Training** - Train RL agent on historical data
2. **Paper Trading** - Validate decisions in simulation (1-2 weeks)
3. **Optional Enhancements** - Additional UI features, optimizations
4. **Production Deployment** - Only after successful validation

## Next Steps

### Immediate (Week 1-2)
1. ✅ Streamline chart data loading - **COMPLETED**
2. ✅ Add RSI display - **COMPLETED**
3. ✅ Complete RL agent infrastructure - **COMPLETED**
3. ⏳ Historical signal performance analysis - **IN PROGRESS**
4. **NEW**: Set up RL agent project structure (Phase 4.1 foundation)

### Short-term (Week 3-4)
5. Implement RL agent model architecture (Phase 4.1)
6. Add multi-horizon prediction display (Phase 4.2)
7. Integrate attention mechanism (Phase 4.3)
8. Create basic explainability dashboard (Phase 5.1, 5.2)

### Medium-term (Week 5-8)
9. Complete rule extraction pipeline (Phase 5.1)
10. Add SHAP visualizations (Phase 5.2)
11. Implement attention saliency display (Phase 5.3)
12. Add topic clustering (Phase 5.4)
13. Complete dashboard integration (Phase 6)

### Long-term (Ongoing)
14. Additional technical indicators (Phase 2.1)
15. Pattern recognition (Phase 7.1)
16. Alert system (Phase 7.3)
17. Export & reporting (Phase 7.4)

---

## Dependencies & Requirements

### New Python Packages Required

#### Core RL Agent
- `torch` (PyTorch) - Deep learning framework
- `stable-baselines3` or custom PPO implementation - RL algorithms
- `gym` or `gymnasium` - Environment interface
- `numpy` - Numerical operations (likely already installed)

#### Explainability
- `shap` - SHAP values for feature importance
- `scikit-learn` - Decision trees, clustering (likely already installed)
- `hdbscan` or `scikit-learn.cluster` - Topic clustering

#### News & Embeddings
- `sentence-transformers` - News embeddings (likely already installed)
- `feedparser` - RSS feed parsing (likely already installed)

#### Utilities
- `pandas` - Data manipulation (for rule extraction)
- `matplotlib` or `plotly` - Additional visualizations (optional)

### Installation Command
```bash
pip install torch stable-baselines3 gymnasium shap hdbscan pandas
```

### System Requirements
- **GPU**: Optional but recommended for faster training (CUDA-compatible GPU)
- **RAM**: Minimum 8GB, recommended 16GB+ for model training
- **Storage**: Additional space for model checkpoints (~100MB-1GB per checkpoint)

---

## Implementation Checklist

### Phase 4: RL Agent (Foundation) - ✅ COMPLETED
- [x] Create `rl_agent/` directory structure
- [x] Implement `rl_agent/model.py` (actor-critic with attention)
- [x] Implement `rl_agent/environment.py` (trading environment)
- [x] Implement `rl_agent/state_encoder.py` (feature encoding)
- [x] Implement `rl_agent/trainer.py` (PPO training loop)
- [x] Add database tables for RL agent data
- [x] Create model checkpoint system
- [x] Test basic training loop

### Phase 4: Multi-Horizon Predictions - ✅ COMPLETED
- [x] Add auxiliary heads to model
- [x] Implement prediction storage
- [x] Create API endpoint `/api/rl-agent/predictions`
- [x] Add prediction display to template
- [x] Add prediction accuracy tracking

### Phase 4: News & Attention - ✅ COMPLETED
- [x] Integrate attention mechanism into model
- [x] Implement `rl_agent/attention_logger.py`
- [x] Create API endpoint `/api/rl-agent/attention`
- [x] Add attention visualization to template
- [x] Test attention weight logging

### Phase 4: Risk Management - ✅ COMPLETED
- [x] Implement `rl_agent/risk_manager.py`
- [x] Create API endpoint `/api/rl-agent/risk`
- [x] Add risk dashboard to template
- [x] Implement all safety constraints

### Phase 5: Explainability - ✅ COMPLETED
- [x] Implement `rl_agent/explainability.py`
- [x] Add rule extraction (decision tree surrogate)
- [x] Add SHAP computation
- [x] Create API endpoints for explainability data
- [x] Add explainability sections to template
- [x] Test rule extraction pipeline

### Phase 5: Topic Clustering - ✅ COMPLETED
- [x] Add clustering to `news_sentiment.py`
- [x] Implement cluster labeling
- [x] Store clusters in database
- [x] Integrate with state encoder
- [ ] Create API endpoint `/api/rl-agent/topics` (optional)
- [ ] Add topic display to template (optional)

### Phase 6: Dashboard - ✅ COMPLETED
- [x] Create prediction panel
- [x] Create explainability panel
- [x] Create risk dashboard
- [x] Add attention visualization
- [x] Add JavaScript update functions
- [x] Style with CSS
- [ ] Create RL agent status panel (optional - can show training metrics)

### Integration - ✅ COMPLETED
- [x] Create `rl_agent/integration.py`
- [x] Connect to price fetcher
- [x] Connect to news sentiment
- [x] Create decision API endpoint
- [x] Full integration pipeline

### Testing & Validation
- [ ] Unit tests for model architecture
- [ ] Integration tests for training loop
- [ ] Validation tests for rule extraction
- [ ] Performance tests for dashboard loading
- [ ] Paper-trade validation (run for 1 week minimum)

---

## Notes

- Keep buy/sell/hold signals as the core feature - they're working well
- Price prediction is deprecated - focus on signal quality, not price prediction
- Use reward logic from `calculate_reward()` to evaluate signal outcomes
- Consider A/B testing different signal display formats
- **NEW**: Start with paper-trading only - never enable live trading without extensive validation
- **NEW**: The RL agent should complement, not replace, the existing contextual bandit initially
- **NEW**: Run both systems in parallel and compare performance before full migration
- **NEW**: Rule extraction should run as a background job (daily/weekly) to avoid blocking requests
- **NEW**: SHAP computation is expensive - cache results and recompute periodically
- **NEW**: Monitor model performance continuously - set up alerts for accuracy degradation

