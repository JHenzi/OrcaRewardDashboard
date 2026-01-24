Here is a technical audit of the model, focusing on architectural risks, causality violations, and data sufficiency.

### 1\. Critical Architectural Flaw: Causal Leakage in Shared Latent

The most significant issue in the design is the structure of the **Shared Latent Layer**.

-   **The Problem:** The `Position Branch` (Position Size, Unrealized P&L, Entry Price) feeds into the `Shared Latent`, which then feeds into the `Auxiliary Prediction Heads` (15m/1h/24h Return).
    
-   **Why it fails:** This implies that **your entry price influences the market price of Solana**. The model will learn spurious correlations, such as _"When I am losing money (Negative P&L), the price is likely to go up"_ (gambler's fallacy) or _"When I have a large position, volatility decreases."_
    
-   **Consequence:** The model will overfit to its own trading history rather than market dynamics. The `Aux Heads` must only see market data, not account data.
    
-   **Fix:** Split the architecture. Create a `Market Latent` (Price + News + Time) and a separate `Account Latent` (Position).
    
    -   **Aux Heads** (Price Prediction) take _only_ `Market Latent`.
        
    -   **Actor/Critic Heads** (Trading Decision) take `concatenation(Market Latent, Account Latent)`.
        

### 2\. Temporal Horizon Mismatch (The "Pinhole" Problem)

The model attempts to predict **24-hour returns** based on **5 hours of input history**.

-   **The Problem:** The `Price Branch` input is "60 points at 5-minute intervals" (5 hours total). The model has zero visibility into daily trends, weekly support levels, or macro shifts.
    
-   **Why it fails:** A 24-hour price movement is often driven by factors established over the previous days or weeks. Predicting it from a 5-hour window is mathematically impossible (insufficient state information).
    
-   **Fix:** Implement a **Multi-Resolution Input**:
    
    -   **Head A:** 60 points @ 5-min intervals (Short term / microstructure).
        
    -   **Head B:** 60 points @ 1-hour intervals (Medium term / 2.5 days).
        
    -   **Head C:** 30 points @ 1-day intervals (Long term / 1 month).
        

### 3\. Feature Stationarity & Normalization Risks

The document mentions: `current_price - Current SOL price (normalized by /100)`.

-   **The Problem:** This normalization is **non-stationary**. If SOL moves from $150 to $300, the input value doubles. If it drops to $75, it halves. Neural networks struggle when test data (future prices) drifts significantly from the range of training data.
    
-   **Risk:** If the model is trained on 2023 data (SOL ~$20) and deployed in 2026 (SOL ~$150), the `current_price` feature will be strictly Out-Of-Distribution (OOD), rendering the weights attached to it useless or harmful.
    
-   **Fix:** **Never input raw prices.** Use only:
    
    -   Log-returns.
        
    -   Price relative to Moving Averages (which you already have: `price/sma_1h`).
        
    -   Price relative to All-Time-High or local Min/Max.
        
    -   Remove `current_price` (normalized) entirely; the model can infer the "state" from the SMA ratios.
        

### 4\. News Attention Deficit (Missing "Time Decay")

The `News Branch` takes 20 headlines but lacks explicit **temporal embedding**.

-   **The Problem:** The attention mechanism sees a bag of 20 headlines. It does not know if "Solana Hacked" happened 5 minutes ago or 12 hours ago (if the news volume is low). The `Time Branch` encodes the _current_ wall-clock time, but not the _age_ of the news items.
    
-   **Fix:** Add a **"Time Delta" feature** to the News Embeddings.
    
    -   Input: `[Embedding (384) + Sentiment (1) + Age_in_Minutes (1)]`.
        
    -   This allows the Attention mechanism to learn that "Old bad news" matters less than "Fresh bad news."
        

### 5\. Prediction Target Stability

The model attempts to predict a specific scalar return (e.g., `0.015`).

-   **The Problem:** Financial time series are stochastic. The "Mean Squared Error" (MSE) loss on a scalar target encourages the model to predict the **mean**, which is often close to 0 in efficient markets. This leads to "flat" predictions that are safe but useless.
    
-   **Fix:** Switch to **Classification** or **Distributional Regression**:
    
    -   Instead of predicting `0.015`, predict the probability of 3 classes: `[Bearish (< -1%), Neutral, Bullish (> 1%)]`.
        
    -   This is often easier for an RL agent to learn and more actionable for the Policy head.
        

### Summary of Recommended Changes

Component

Current Implementation

Recommended Change

**Latent Structure**

Shared (Position + Price → Prediction)

**Disentangled** (Price → Prediction; Price + Position → Action)

**Input Window**

5 Hours (60 x 5m)

**Multi-Scale** (5m, 1h, and Daily candles)

**Price Feature**

Normalized Raw Price (`price/100`)

**Remove Raw Price** (Use only Returns/SMA ratios)

**News Input**

Embedding + Sentiment

Embedding + Sentiment + **Time Since Publication**

**Prediction Head**

Scalar Regression (MSE)

**3-Class Classification** (Bull/Bear/Neutral)



