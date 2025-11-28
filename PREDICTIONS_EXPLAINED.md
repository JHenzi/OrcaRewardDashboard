# RL Agent Predictions Explained

## Overview

The RL agent automatically generates **multi-horizon return predictions** every time a decision is made. These predictions estimate how SOL price will change over 1 hour and 24 hours.

## How Predictions Work

### When Predictions Are Generated

Predictions are automatically created when:
1. You call `/api/rl-agent/decision` (POST)
2. The `RLAgentIntegration.make_decision()` method is called
3. The model is loaded and ready

### What Gets Predicted

The model uses its **auxiliary prediction heads** to forecast:

1. **1-Hour Return Prediction** (`pred_1h`)
   - Predicted percentage change in SOL price over the next hour
   - Format: Decimal (0.01 = +1%, -0.02 = -2%)
   - Example: `0.015` means predicted +1.5% price increase

2. **24-Hour Return Prediction** (`pred_24h`)
   - Predicted percentage change in SOL price over the next 24 hours
   - Format: Decimal (0.05 = +5%, -0.03 = -3%)
   - Example: `0.025` means predicted +2.5% price increase

3. **Confidence Scores** (`conf_1h`, `conf_24h`)
   - Confidence level for each prediction (0.0 to 1.0)
   - Higher confidence = more reliable prediction
   - Currently uses heuristic based on value estimate

### How Predictions Are Generated

```python
# In rl_agent/integration.py - make_decision()
pred_1h, pred_24h, conf_1h, conf_24h = generate_prediction(
    self.model,
    self.state_encoder,
    prices,
    price_features,
    news_data,
    position_state,
    current_price,
    datetime.now(),
    self.device,
)
```

The `generate_prediction()` function:
1. Encodes current market state (price, indicators, news, position)
2. Runs the model forward pass
3. Extracts predictions from model's auxiliary heads (`pred_1h`, `pred_24h`)
4. Calculates confidence scores
5. Returns predictions and confidence

### Model Architecture

The `TradingActorCritic` model has **auxiliary prediction heads**:
- Separate neural network layers that predict returns
- Trained alongside the main policy/value functions
- Use the same state representation as the trading policy

```python
# In rl_agent/model.py
self.pred_1h_head = nn.Linear(hidden_dim, 1)  # 1h return prediction
self.pred_24h_head = nn.Linear(hidden_dim, 1)  # 24h return prediction
```

## Where Predictions Are Stored

### Database Storage

Predictions are stored in the `rl_agent_decisions` table:
- `predicted_return_1h` - 1-hour prediction
- `predicted_return_24h` - 24-hour prediction
- `confidence_1h` - Confidence for 1h prediction
- `confidence_24h` - Confidence for 24h prediction

### API Access

**Get Latest Predictions:**
```bash
GET /api/rl-agent/predictions
```

Returns:
```json
{
  "success": true,
  "current_prediction": {
    "timestamp": "2025-11-28T15:30:00",
    "pred_1h": 0.015,
    "pred_24h": 0.025,
    "confidence_1h": 0.75,
    "confidence_24h": 0.65,
    "price_at_prediction": 150.50
  },
  "recent_predictions": [...],
  "accuracy_stats": {
    "1h": {
      "mae": 0.023,
      "rmse": 0.031,
      "samples": 100
    },
    "24h": {
      "mae": 0.045,
      "rmse": 0.062,
      "samples": 95
    }
  }
}
```

## Prediction Accuracy Tracking

The system automatically tracks prediction accuracy:

1. **When predictions are made**: Stored with timestamp and price
2. **When outcomes are known**: Actual returns are calculated
3. **Accuracy metrics**: MAE, RMSE computed automatically
4. **Available via API**: `/api/rl-agent/predictions` shows accuracy stats

### Accuracy Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual returns
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more
- **Sample Count**: Number of predictions with known outcomes

## Using Predictions

### In Trading Decisions

The model uses predictions internally:
- Predictions help inform the trading action (BUY/SELL/HOLD)
- Higher predicted returns → more likely to BUY
- Lower predicted returns → more likely to SELL

### For Analysis

You can use predictions to:
- **Monitor model performance**: Track accuracy over time
- **Understand model confidence**: See when model is uncertain
- **Compare predictions vs. reality**: Validate model quality
- **Make informed decisions**: Use predictions as additional signal

## Current Status

✅ **Predictions are automatically generated** when model makes decisions
✅ **Predictions are stored** in database
✅ **Accuracy is tracked** automatically
✅ **API endpoints** provide access to predictions
✅ **Dashboard displays** predictions (if implemented)

⚠️ **Note**: Predictions require a **trained model**. Until training is complete, predictions won't be available.

## Next Steps

1. **Train the model**: `python train_rl_agent.py --epochs 10`
2. **Model will auto-load**: On next app restart
3. **Make decisions**: Call `/api/rl-agent/decision` to generate predictions
4. **Monitor accuracy**: Check `/api/rl-agent/predictions` for accuracy stats

---

**Summary**: The RL agent automatically generates 1h and 24h return predictions every time it makes a decision. These are stored, tracked, and accessible via API. Predictions help inform trading decisions and can be used to monitor model performance.

