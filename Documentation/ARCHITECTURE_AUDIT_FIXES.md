# Architecture Audit Fixes

**Date**: 2026-01-23  
**Status**: Implemented

This document summarizes the architectural fixes implemented based on the technical audit in `Audit.md`.

---

## Summary of Issues Fixed

| Issue | Problem | Solution | Files Changed |
|-------|---------|----------|---------------|
| **#1 Causal Leakage** | Position data fed into price predictions | Disentangled Market/Account latents | `model.py` |
| **#2 Temporal Mismatch** | 5-hour input for 24-hour predictions | Multi-resolution input (5m, 1h, daily) | `model.py`, `state_encoder.py` |
| **#3 Non-Stationary Features** | Raw price normalized by /100 | Use only returns and SMA ratios | `state_encoder.py` |
| **#4 News Time Decay** | No age information for headlines | Added `news_age` feature | `state_encoder.py`, `model.py` |
| **#5 Prediction Target** | Scalar regression (MSE) | 3-class classification (CrossEntropy) | `model.py`, `trainer.py`, `prediction_generator.py` |

---

## Detailed Changes

### Fix #1: Causal Leakage Prevention

**Problem**: Position data (entry price, unrealized P&L) was fed into the shared latent, which then fed into price prediction heads. This caused the model to learn spurious correlations like "when I'm losing money, price goes up."

**Solution**: Split the architecture into two separate latent spaces:

```
BEFORE:
  [Price + News + Position + Time] → Shared Latent → All Heads

AFTER:
  [Price + News + Time] → Market Latent → Aux Prediction Heads (15m/1h/24h)
  [Market Latent + Position] → Full Latent → Actor/Critic Heads
```

**Key Code Changes** (`model.py`):
- Added `self.market_latent` layer (256-dim) from Price + News + Time
- Added `self.full_latent` layer (256-dim) from Market + Position
- Aux heads (`aux_15m`, `aux_1h`, `aux_24h`) now use `market_latent` only
- Actor/Critic heads use `full_latent` (includes position for trading decisions)

---

### Fix #2: Multi-Resolution Input

**Problem**: Model only saw 5 hours of 5-minute data (60 points). This is insufficient to predict 24-hour price movements, which depend on daily/weekly trends.

**Solution**: Three-scale input architecture:

| Scale | Resolution | Window | Total Span | Purpose |
|-------|------------|--------|------------|---------|
| Short | 5 minutes | 60 points | 5 hours | 15m/1h predictions |
| Medium | 1 hour | 60 points | 2.5 days | 24h predictions |
| Long | 1 day | 30 points | 1 month | Macro context |

**Key Code Changes**:

`model.py`:
- Added `PriceBranchSingleScale` class for each resolution
- `PriceBranch` now has `short_branch`, `medium_branch`, `long_branch`
- Forward accepts `price_medium` and `price_long` tensors

`state_encoder.py`:
- Added `encode_price_medium()` for hourly returns
- Added `encode_price_long()` for daily returns
- `encode_full_state()` now returns `price_medium` and `price_long`

---

### Fix #3: Feature Stationarity

**Problem**: Raw price divided by 100 is non-stationary. If SOL moves from $20 to $150, the feature value changes 7.5x, making old training data incompatible with new data.

**Solution**: Use only stationary features:

**REMOVED** (non-stationary):
- `current_price / 100`
- `sma_1h / 100`
- `sma_4h / 100`
- `sma_24h / 100`

**ADDED/KEPT** (stationary):
- `current_price / sma_1h` (ratio, centered ~1.0)
- `current_price / sma_4h` (ratio, centered ~1.0)
- `current_price / sma_24h` (ratio, centered ~1.0)
- `sma_1h / sma_4h` (short vs medium trend)
- `sma_4h / sma_24h` (medium vs long trend)
- `rsi / 100` (bounded 0-1)
- `volatility / price` (relative volatility)
- `momentum_15m` (already a return)
- `percent_change / 100` (already a return)

**Result**: 60 log-returns + 9 stationary indicators = 69 features (down from 70)

---

### Fix #4: News Time Decay

**Problem**: Attention mechanism couldn't distinguish "Solana hacked 5 minutes ago" from "Solana hacked 12 hours ago" if both were in the news window.

**Solution**: Added `news_age` feature for each headline.

**Key Code Changes**:

`state_encoder.py`:
- `encode_news_features()` now returns 4 arrays: embeddings, sentiment, clusters, **ages**
- Age normalized: 0 = fresh (just published), 1 = 1 day old
- Uses `published_at` timestamp from news data

`model.py` (`NewsBranch`):
- Added `self.time_decay` layer that learns decay weights
- News embeddings are multiplied by learned decay: `x = x * decay_weights`
- Fresh news gets high weight, old news gets low weight

---

### Fix #5: Classification Instead of Regression

**Problem**: MSE loss on scalar return predictions encourages the model to predict the mean (often ~0 in efficient markets), leading to useless flat predictions.

**Solution**: 3-class classification with CrossEntropy loss:

| Class | Label | Return Range | Representative Value |
|-------|-------|--------------|---------------------|
| 0 | Bearish | < -1% | -2% |
| 1 | Neutral | -1% to +1% | 0% |
| 2 | Bullish | > +1% | +2% |

**Key Code Changes**:

`model.py`:
- Aux heads now output 3 logits instead of 1 scalar
- Added `return_to_class()` static method for conversion
- Added `class_to_return_estimate()` for backward compatibility
- Output includes both `pred_*_logits` (for training) and `pred_*` (expected return)

`trainer.py`:
- Changed from `mse_loss(pred, target)` to `cross_entropy(logits, class_labels)`
- Added `return_to_class_batch()` helper function
- Logs classification accuracy periodically

`prediction_generator.py`:
- Handles both classification (new) and regression (old) models
- Returns class probabilities for interpretability
- Computes expected return from class probabilities

---

## Backward Compatibility

The changes maintain backward compatibility:

1. **Model Loading**: Old checkpoints can be loaded (missing keys use random init)
2. **API Responses**: `pred_1h`, `pred_24h` still returned as expected returns
3. **Training**: Falls back to MSE if logits not in output
4. **State Encoding**: Multi-resolution data optional (defaults to zeros)

---

## Training Impact

After these fixes:

1. **Predictions are disentangled from trading state** - no more spurious correlations
2. **Long-term predictions have long-term context** - can see weekly trends
3. **Features are OOD-resistant** - model works across price regimes
4. **News freshness matters** - recent news weighted more heavily
5. **No more flat predictions** - classification forces directional opinions

---

## Files Modified

| File | Changes |
|------|---------|
| `rl_agent/model.py` | Disentangled architecture, multi-resolution, classification heads |
| `rl_agent/state_encoder.py` | Multi-resolution encoding, news age, stationary features |
| `rl_agent/trainer.py` | CrossEntropy loss, classification metrics |
| `rl_agent/prediction_generator.py` | Classification output handling |

---

## Usage

### Training with New Architecture

```bash
python scripts/train_rl_agent.py \
    --epochs 10 \
    --enable-auxiliary \
    --use-prediction-outcomes
```

### API Prediction (includes class probabilities)

```bash
curl http://localhost:5030/api/sol-price/predict-15m
```

Response now includes:
```json
{
  "predicted_return_15m": 0.005,
  "predicted_price_15m": 150.75,
  "confidence": 0.72,
  "class_info": {
    "class_probs": {
      "bearish": 0.15,
      "neutral": 0.28,
      "bullish": 0.57
    },
    "predicted_class": "Bullish (>+1%)"
  }
}
```

---

## References

- Original audit: `Documentation/Audit.md`
- Model documentation: `Documentation/RL_MODEL_DOCUMENTATION.md`
