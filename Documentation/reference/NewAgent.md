### **Mission Statement**

> Build an AI agent that continuously learns to generate actionable trading strategies (buy/sell/hold) by observing price, technical indicators, news embeddings, and sentiment. The agent should invent hypotheses autonomously, test them against observed outcomes (1h, 24h returns), and surface interpretable rules that humans can understand and act upon.

### **Primary Objectives**

1. **Autonomous Hypothesis Discovery**
    
    - The agent should _generate its own rules/theories_ about which market conditions predict positive returns, without being told in advance what these rules are.
        
    - It should integrate multiple modalities: numeric indicators + textual embeddings + sentiment.
        
2. **Sequential Decision Learning**
    
    - Use reinforcement learning principles (e.g., PPO or actor-critic) to optimize for portfolio outcomes while safely exploring new strategies.
        
    - Include auxiliary predictions for multi-horizon returns (1h, 24h) to accelerate learning.
        
3. **Explainability / Interpretability**
    
    - The agent must provide interpretable outputs:
        
        - Extract “human-readable rules” via surrogate models (decision trees / RuleFit).
            
        - Rank features or news embeddings by importance (SHAP or attention-based).
            
        - Highlight representative news headlines that influenced decisions.
            
4. **Safe Experimentation**
    
    - Ensure exploration is conservative: limit max position size, trade frequency, and risk exposure.
        
    - Provide logging and metrics for all actions and outcomes, even in paper-trade mode.
        
5. **Continuous Improvement**
    
    - The agent should iteratively update its internal understanding of which features and embeddings correlate with positive returns.
        
    - Maintain a replay or experience buffer to accelerate learning from new data without discarding old insights.
        
6. **Modular & Extensible Design**
    
    - The code should be modular:
        
        - Feature ingestion (indicators, news embeddings, sentiment)
            
        - Agent model (RL + auxiliary heads)
            
        - Environment / paper-trade simulation
            
        - Explainability pipeline (surrogate model, SHAP/attention)
            
        - Reporting / dashboard output
            
    - Future improvements should allow adding new features, embeddings, or horizons easily.
        

---

### **High-Level Success Criteria**

- The agent autonomously generates hypotheses that _improve portfolio outcomes_ over random or naive strategies.
    
- Outputs interpretable rules linking features and news clusters to actionable insights.
    
- Maintains safety limits and can be paper-traded without catastrophic loss.
    
- Produces dashboards or reports summarizing top hypotheses, their win rates, and feature importance.
    
- Can continue learning incrementally as new data arrives without re-engineering the system.

---


# TL;DR — recommended approach (one-sentence)

Use a **single online RL agent (PPO / recurrent actor-critic)** that is _warm-started_ by auxiliary supervised heads predicting multi-horizon returns (1h, 24h), uses attention over news embeddings so the model can "point" to important headlines, explores with conservative Thompson/entropy-driven exploration, and periodically **extracts human-readable rules** via a decision-tree surrogate and SHAP/attention saliency on the features/embeddings.

---

# Why this combo?

- **RL (PPO)** handles sequential decisions and long-term outcomes (positions, compounding).
    
- **Auxiliary supervised heads** predicting next-1h and next-24h returns dramatically accelerate learning (they supply training signal before long-term rewards accumulate). This is like teaching the agent candidate theories.
    
- **Attention over embeddings** makes the model naturally highlight which headlines/topics influenced a decision.
    
- **Decision-tree surrogate + SHAP** extracts crisp, human-readable “theories” the agent discovered.
    
- **Conservative exploration (entropy + Bayesian bandit-ish seeding)** lets the agent try hypotheses while limiting risk.
    

---

# System components (compact)

1. **Data & state**
    
    - Price/time-series window (last N minutes) + technical indicators (SMA, EMA, RSI, ATR, vol, returns)
        
    - Aggregated news: latest M headlines embedded (e.g., 768-dim each from an embedding model), each with sentiment and rank
        
    - Position state: current position size, P&L, time-since-last-trade
        
    - Time features (minute-of-day, day-of-week)
        
    - Concatenate or use separate branches for sequence + news
        
2. **Model architecture (small, effective)**
    
    - Price branch: 1D-CNN or small LSTM → latent\_p (128)
        
    - News branch: embeddings \[M x E\] → multi-head attention pooling → latent\_n (128). Keep the attention matrix — it’s your saliency map for headlines.
        
    - Position/time branch → latent\_s (32)
        
    - Concat → shared latent (256) →
        
        - **Actor head** (policy) → softmax over discrete actions: {SELL, HOLD, BUY} or discrete sizes.
            
        - **Critic head** → value scalar.
            
        - **Aux heads** → predict `return_1h` and `return_24h` (regression).
            
    - Use recurrence (LSTM) or include last K actions to capture sequential dependencies.
        
3. **Reward & multi-horizon objective**
    
    - Use primary RL reward = log portfolio value increase at each step, minus transaction cost and a risk penalty.
        
    - Also train auxiliary heads with supervised MSE on target returns (observed after 1h and 24h) using transitions when those horizons complete (or using bootstrapped n-step returns).
        
    - Combined loss = PPO loss + c1 \* critic\_loss + c2 \* aux1\_loss + c3 \* aux2\_loss.
        
4. **Exploration strategy**
    
    - Start with higher entropy bonus in PPO; decay slowly.
        
    - Use **Thompson-like seeding** for early decisions: pick actions proportionally to posterior on predicted return (via ensembles or bootstrapped heads) so you hedge exploration across plausible theories.
        
    - Enforce hard constraints: max position size, max trades per hour, daily loss cap.
        
5. **Credit assignment (delayed feedback)**
    
    - Use **n-step returns** (e.g., 1h and 24h targets as auxiliary supervision) and GAE for policy updates.
        
    - The agent still optimizes immediate-step reward (portfolio delta), but auxiliary losses let it learn predictive signals that correlate with longer horizons faster.
        
6. **Interpretable theory extraction**
    
    - Periodically (daily or weekly), sample labeled dataset: state → action / outcome (1h/24h return).
        
    - Train a shallow decision tree or RuleFit to predict whether action BUY would have produced >threshold return. Extract the top rules as human-readable "theories".
        
    - Also compute SHAP values (or use attention weights aggregated by headline topic) to rank importance among features (indicators, cluster/topic of embedding, sentiment).
        
    - Show representative headlines that got high attention at times the rule fired.
        
7. **Topic discovery on embeddings**
    
    - Run lightweight clustering (HDBSCAN / KMeans with small k) on recent embeddings to discover topical clusters (e.g., "earnings", "lawsuit", "election", "supply-chain").
        
    - Use cluster id as a categorical feature and include cluster-level signals in the surrogate rule extraction — it makes theories like “when cluster = X and sentiment < -0.2 then SELL” readable.
        
8. **Safe rollout**
    
    - Paper trade first; enforce conservative position sizing, daily loss kill, max trade frequency. Use paper logs to train.
        
    - After stable outperformance vs baseline and safety tests, consider small, controlled live tests.
        

---

# Concrete reward formula (starter)

At time t (minute-step), let p\_t be portfolio value, pos\_t position. Use per-step reward:
r_t = log(p_{t+1}/p_t)  -  λ_cost * transaction_cost  - λ_risk * pos_t^2

Aux losses:
L_aux1 = MSE(pred_1h, realized_return_1h)
L_aux2 = MSE(pred_24h, realized_return_24h)

Total loss: PPO\_loss + c\_v \* value\_loss + c\_e \* entropy\_bonus + α1 \* L\_aux1 + α2 \* L\_aux2

Tune αs so auxiliary tasks meaningfully accelerate learning but don’t dominate.

---

# How the agent “invents” theories

- The **aux heads** learn correlations between current state (incl. news embeddings) and multi-horizon returns → these are implicit hypotheses.
    
- The **policy** learns which of these predicted signals translate into better portfolio outcomes after considering costs and risk.
    
- The **decision-tree surrogate** and SHAP convert the neural policy/aux predictions into explicit rules (theories) that you can read and test.
    

Example output you’ll get:

- “Rule: If \[embedding cluster = ‘politics\_trump’\] & sentiment < -0.3 & RSI < 40 → BUY, avg 1h return +0.4% (N=37)”
    
- Attention heatmap showing which headlines the model looked at when making that decision.
    
- SHAP ranking: embeddings cluster importance = 0.34, sentiment = 0.22, RSI = 0.05, MACD = 0.01.
    

---

# Practicalities — right-sized implementation plan (6 steps, minimal)

1. **Feature + ingestion**: collect minute bars, indicators, M latest embeddings & sentiment. Store rows for each minute. (You already have this.)
    
2. **Small Gym env**: discrete actions; keep realistic transaction cost model; paper-trade logging.
    
3. **Model code**: lightweight PyTorch actor-critic with attention-based news branch and two aux heads (1h,24h). Keep model small (few layers) for speed.
    
4. **Training loop**: PPO with GAE, n-step returns. Train online on paper-trade logs, and periodically append complete 1h/24h returns to the dataset for aux supervision.
    
5. **Explainability pipeline**: daily run of surrogate decision-tree + SHAP on last Xk transitions to extract readable rules and feature importance.
    
6. **Dashboard**: surface discovered rules, attention-highlighted headlines, rule performance (win rate, avg return, count), and uncertainty metrics.
    

You can get a minimal working system in a few days of focused work if you reuse standard libraries (Stable-Baselines3, PyTorch, SHAP, scikit-learn).

---

# Quick notes on contextual bandits vs RL for your case

- A pure contextual bandit is attractive for simplicity, but because decisions create positions and influence future states, **full RL** is more appropriate.
    
- However, contextual bandit ideas (Thompson sampling, uncertainty-aware exploration) are useful _inside_ the RL system (for seeding actions and maintaining uncertainty-aware behavior).
    

---

# Explainability recipes (how to turn model into “theories”)

1. **Attention logs**: for every decision, save attention weights over M headlines → link top-k headlines to decision.
    
2. **Surrogate tree**: train a shallow tree on (state features → action or action-success) and export rules.
    
3. **SHAP**: compute feature SHAP values for the aux heads or policy logits to rank importance.
    
4. **Cluster narratives**: map embedding clusters to human-readable labels by sampling representative headlines per cluster.
    
5. **Rule validation**: for each extracted rule, compute bootstrap CI on its avg 1h/24h return and N to show statistical strength.
    

These deliver exactly the “theories” you asked for: human-readable conditions with performance stats.

---

# Safety & sampling budget (practical)

- Keep initial trade size < 0.1% of target capital; trade frequency limit e.g., ≤ 5 trades/hour.
    
- Use heavy regularization & ensemble/bootstraps on aux heads so the model’s uncertainty is measurable. When uncertainty high, force HOLD or tiny test trades.
    
- Auto-disable exploration in live mode if drawdown > threshold.

