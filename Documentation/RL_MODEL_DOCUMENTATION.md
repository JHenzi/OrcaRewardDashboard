# RL Agent Model - Complete Documentation

**Last Updated**: 2026-01-23  
**Model Version**: TradingActorCritic with Multi-Horizon Predictions

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Input Features](#input-features)
4. [Outputs](#outputs)
5. [How It Learns](#how-it-learns)
6. [Training Process](#training-process)
7. [Prediction Generation](#prediction-generation)
8. [Example Data Structures](#example-data-structures)
9. [Feedback Loop](#feedback-loop)

---

## Overview

The RL Agent is a **Proximal Policy Optimization (PPO)** reinforcement learning model designed to:
- Make trading decisions (BUY/SELL/HOLD) for SOL
- Predict price returns at multiple horizons (15 minutes, 1 hour, 24 hours)
- Learn from actual prediction outcomes (feedback loop)
- Integrate multiple data sources: price history, technical indicators, news sentiment, and position state

### Key Features

- **Multi-Modal Input**: Price data, news embeddings, technical indicators, position state, time features
- **Multi-Task Learning**: Simultaneously learns trading policy, value estimation, and return predictions
- **Attention Mechanism**: Focuses on relevant news headlines
- **Auxiliary Prediction Heads**: Predicts returns at 15m, 1h, and 24h horizons
- **Continuous Learning**: Trains on actual prediction outcomes to improve over time

---

## Model Architecture

### High-Level Architecture

```
Input Features
    ├── Price Branch (1D-CNN) ──────┐
    ├── News Branch (Attention) ────┤
    ├── Position Branch (FC) ───────┤──→ Shared Latent (256) ──→ Output Heads
    └── Time Branch (FC) ───────────┘
                                        ├── Actor (Policy)
                                        ├── Critic (Value)
                                        ├── Aux 15m (Return Prediction)
                                        ├── Aux 1h (Return Prediction)
                                        └── Aux 24h (Return Prediction)
```

### Detailed Component Breakdown

#### 1. Price Branch (`PriceBranch`)
- **Type**: 1D Convolutional Neural Network
- **Input**: 
  - Price time-series window (60 points = last 60 minutes)
  - Technical indicators (10 features)
- **Architecture**:
  ```
  Input: (60 price returns + 10 indicators) = 70 features
  ├── Conv1d(1, 32, kernel=3) → ReLU
  ├── Conv1d(32, 64, kernel=3) → ReLU
  ├── MaxPool1d
  ├── Linear(64, 32) → ReLU (for indicators)
  └── Output: 128-dim latent vector
  ```
- **Purpose**: Extracts temporal patterns from price movements and technical indicators

#### 2. News Branch (`NewsBranch`)
- **Type**: Multi-Head Attention Network
- **Input**:
  - News embeddings: (20 headlines × 384 dimensions)
  - News sentiment scores: (20 headlines)
- **Architecture**:
  ```
  Input: (20, 384) embeddings + (20,) sentiment
  ├── Multi-Head Attention (4 heads)
  ├── Layer Normalization
  ├── Feed-Forward Network
  └── Output: 128-dim latent vector
  ```
- **Purpose**: Identifies which news headlines are most relevant to price movements
- **Attention**: Model learns to weight headlines by importance

#### 3. Position Branch
- **Type**: Fully Connected Network
- **Input**: 5 position features
- **Architecture**:
  ```
  Input: 5 features
  ├── Linear(5, 16) → ReLU
  ├── Linear(16, 32) → ReLU
  └── Output: 32-dim latent vector
  ```
- **Features**:
  1. Position size ratio (position_value / portfolio_value)
  2. Position value (normalized)
  3. Entry price ratio (current_price / entry_price)
  4. Unrealized P&L (normalized)
  5. Time since last trade (hours)

#### 4. Time Branch
- **Type**: Fully Connected Network
- **Input**: 4 time features
- **Architecture**:
  ```
  Input: 4 features
  ├── Linear(4, 16) → ReLU
  ├── Linear(16, 16) → ReLU
  └── Output: 16-dim latent vector
  ```
- **Features**:
  1. Hour of day (0-23, normalized to 0-1)
  2. Day of week (0-6, normalized to 0-1)
  3. Minute of day (0-1439, normalized to 0-1)
  4. Is weekend (0 or 1)

#### 5. Shared Latent Layer
- **Input**: Concatenated latents (128 + 128 + 32 + 16 = 256)
- **Architecture**:
  ```
  Input: 256-dim
  ├── Linear(256, 256) → ReLU
  ├── Linear(256, 256) → ReLU
  └── Output: 256-dim shared representation
  ```
- **Purpose**: Combines all input modalities into unified representation

#### 6. Output Heads

**Actor Head (Policy)**
- **Output**: Action logits for [SELL, HOLD, BUY]
- **Architecture**: `Linear(256, 128) → ReLU → Linear(128, 3)`
- **Purpose**: Determines which action to take

**Critic Head (Value)**
- **Output**: State value estimate (scalar)
- **Architecture**: `Linear(256, 128) → ReLU → Linear(128, 1)`
- **Purpose**: Estimates expected future returns from current state

**Auxiliary Prediction Heads**
- **15-Minute Head**: `Linear(256, 64) → ReLU → Linear(64, 1)`
- **1-Hour Head**: `Linear(256, 64) → ReLU → Linear(64, 1)`
- **24-Hour Head**: `Linear(256, 64) → ReLU → Linear(64, 1)`
- **Purpose**: Predict price returns at different horizons
- **Output Format**: Decimal return (0.01 = +1%, -0.02 = -2%)

---

## Input Features

### Complete Input Structure

The model receives 5 types of inputs:

#### 1. Price Features (70 dimensions)

**Price Time-Series** (60 values):
- Last 60 price points (5-minute intervals = 5 hours of history)
- Converted to returns: `(price[t] - price[t-1]) / price[t-1]`
- Normalized and padded/truncated to exactly 60 points

**Technical Indicators** (10 values):
1. `current_price` - Current SOL price (normalized by /100)
2. `sma_1h` - Simple Moving Average (1 hour)
3. `sma_4h` - Simple Moving Average (4 hours)
4. `sma_24h` - Simple Moving Average (24 hours)
5. `price/sma_1h` - Price ratio to 1h SMA
6. `price/sma_4h` - Price ratio to 4h SMA
7. `rsi` - Relative Strength Index (0-100, normalized to 0-1)
8. `std_dev` - Price volatility (standard deviation, normalized)
9. `momentum_15m` - 15-minute momentum
10. `percent_change` - Percent change over window (normalized)

**Example**:
```python
price_features = {
    "current_price": 150.25,
    "sma_1h": 149.80,
    "sma_4h": 148.50,
    "sma_24h": 145.00,
    "rsi": 65.5,  # 0-100 scale
    "std_dev": 2.5,
    "momentum_15m": 0.5,
    "percent_change": 3.2
}
```

#### 2. News Features

**News Embeddings** (20 headlines × 384 dimensions):
- Each headline embedded using `all-MiniLM-L6-v2` (sentence-transformers)
- 384-dimensional dense vectors
- Padded with zeros if fewer than 20 headlines
- Masked during attention (invalid headlines = 0)

**News Sentiment** (20 values):
- Sentiment scores from -1.0 (negative) to +1.0 (positive)
- 0.0 indicates no news/padding

**Example**:
```python
news_data = [
    {
        "embedding": np.array([0.123, -0.456, ...], shape=(384,)),
        "sentiment_score": 0.75,  # Positive sentiment
        "headline": "Solana breaks new ATH",
        "cluster_id": 5
    },
    # ... up to 20 headlines
]
```

#### 3. Position Features (5 dimensions)

1. **Position Size Ratio**: `(position_size * current_price) / portfolio_value`
   - Example: 0.1 = 10% of portfolio in position
2. **Position Value**: `position_size * current_price` (normalized by /1000)
3. **Entry Price Ratio**: `current_price / entry_price`
   - Example: 1.05 = 5% profit from entry
4. **Unrealized P&L**: `(current_price - entry_price) / entry_price` (normalized)
5. **Time Since Last Trade**: Minutes since last trade (normalized to hours)

**Example**:
```python
position_state = {
    "position_size": 10.0,  # SOL
    "portfolio_value": 10000.0,  # USD
    "entry_price": 145.00,
    "current_price": 150.25,
    "time_since_last_trade": 30.0,  # minutes
    "unrealized_pnl": 0.0362  # 3.62% profit
}
```

#### 4. Time Features (4 dimensions)

1. **Hour of Day**: 0-23, normalized to 0-1
2. **Day of Week**: 0-6 (Monday=0), normalized to 0-1
3. **Minute of Day**: 0-1439, normalized to 0-1
4. **Is Weekend**: 0.0 (weekday) or 1.0 (weekend)

**Example**:
```python
timestamp = datetime(2026, 1, 23, 14, 30)  # Wednesday 2:30 PM
time_features = [0.604, 0.286, 0.604, 0.0]  # [hour/24, day/7, minute/1440, weekend]
```

#### 5. Complete State Dictionary

```python
state_dict = {
    "price": np.array([...], shape=(70,)),  # 60 returns + 10 indicators
    "news_embeddings": np.array([...], shape=(20, 384)),
    "news_sentiment": np.array([...], shape=(20,)),
    "news_clusters": np.array([...], shape=(20,), dtype=int),
    "position": np.array([...], shape=(5,)),
    "time": np.array([...], shape=(4,))
}
```

---

## Outputs

### Model Output Structure

```python
output = {
    "action_logits": torch.Tensor([-0.5, 2.3, 1.1]),  # [SELL, HOLD, BUY]
    "value": torch.Tensor([0.025]),  # Expected future return
    "pred_15m": torch.Tensor([0.0012]),  # Predicted 15-min return (0.12%)
    "pred_1h": torch.Tensor([0.0035]),  # Predicted 1h return (0.35%)
    "pred_24h": torch.Tensor([0.015]),  # Predicted 24h return (1.5%)
    "attention_weights": torch.Tensor([...], shape=(20, 20))  # News attention
}
```

### Action Selection

- **Action Logits**: Raw scores for each action
- **Softmax**: Converts to probabilities
- **Selected Action**: Action with highest probability
- **Actions**:
  - `0` = SELL
  - `1` = HOLD
  - `2` = BUY

**Example**:
```python
action_logits = [-0.5, 2.3, 1.1]
action_probs = softmax([-0.5, 2.3, 1.1]) = [0.10, 0.70, 0.20]
selected_action = 1  # HOLD (highest probability)
confidence = 0.70  # 70% confidence
```

### Return Predictions

All predictions are in **decimal form**:
- `0.01` = +1% return
- `-0.02` = -2% return
- `0.15` = +15% return

**Predicted Price Calculation**:
```python
predicted_price_15m = current_price * (1 + pred_15m)
predicted_price_1h = current_price * (1 + pred_1h)
predicted_price_24h = current_price * (1 + pred_24h)
```

---

## How It Learns

### Learning Algorithm: Proximal Policy Optimization (PPO)

The model uses **PPO with Generalized Advantage Estimation (GAE)**:

1. **Collect Rollout**: Model makes decisions on historical data
2. **Compute Advantages**: GAE estimates how good each action was
3. **Update Policy**: PPO updates model to increase probability of good actions
4. **Auxiliary Losses**: Simultaneously trains prediction heads

### Loss Components

**Total Loss**:
```
L_total = L_policy + λ_value * L_value - λ_entropy * H + 
          λ_1h * L_aux_1h + λ_24h * L_aux_24h + λ_15m * L_aux_15m
```

Where:
- **L_policy**: PPO clipped policy loss (maximizes good actions)
- **L_value**: MSE between predicted and actual returns
- **H**: Entropy bonus (encourages exploration)
- **L_aux_1h/24h/15m**: MSE between predicted and actual returns at each horizon

**Default Coefficients**:
- `λ_value = 0.5`
- `λ_entropy = 0.01`
- `λ_1h = 1.0`
- `λ_24h = 1.0`
- `λ_15m = 1.0`

### Reward Calculation

**Reward** is based on:
- Price movement in direction of action
- Transaction costs (0.1% per trade)
- Risk penalties (if constraints violated)

**Example**:
```python
# If action = BUY and price goes up 2%:
reward = 0.02 - 0.001  # Return minus transaction cost = 0.019

# If action = SELL and price goes up 2%:
reward = -0.02 - 0.001  # Negative return minus cost = -0.021
```

### Training Data Sources

The model trains on **two types of data**:

1. **Historical Price Data**:
   - Pre-calculated episodes from `sol_prices.db`
   - Future prices known (for supervised learning of predictions)
   - Used for initial training and data augmentation

2. **Prediction Outcomes** (NEW - Feedback Loop):
   - Actual predictions made by the model
   - Actual returns that occurred
   - Model learns from its own mistakes
   - Enables continuous improvement

---

## Training Process

### Training Pipeline

```
1. Load Training Data
   ├── Historical episodes (from sol_prices.db)
   └── Prediction outcomes (from rl_prediction_accuracy table)

2. For Each Episode:
   ├── Encode state (price, news, position, time)
   ├── Get action from model
   ├── Calculate reward from future price
   ├── Store experience in buffer
   └── Calculate actual returns (15m, 1h, 24h)

3. Train on Buffer:
   ├── Compute advantages (GAE)
   ├── Compute losses (policy, value, auxiliary)
   ├── Backpropagate gradients
   └── Update model weights

4. Save Checkpoint
   └── Model state, optimizer state, training step
```

### Training Command

```bash
python scripts/train_rl_agent.py \
    --use-prediction-outcomes \
    --combine-with-historical \
    --epochs 10 \
    --batch-size 32 \
    --aux-15m-coef 1.0
```

### Training Metrics

After each epoch, you'll see:
```
Epoch 10 Average Losses:
  policy_loss: -0.000249
  value_loss: 0.000002
  entropy: 0.004237
  aux_1h_loss: 0.000004
  aux_24h_loss: 0.000218
  aux_15m_loss: 0.000123  # ← 15m head being trained!
  clip_fraction: 0.000246
  total_loss: -0.000068
```

---

## Prediction Generation

### How Predictions Are Generated

1. **Encode Current State**:
   - Get last 60 price points
   - Get latest 20 news headlines
   - Calculate technical indicators
   - Encode position and time features

2. **Forward Pass Through Model**:
   - Model processes all inputs
   - Extracts shared latent representation
   - Auxiliary heads output return predictions

3. **Extract Predictions**:
   - `pred_15m`: 15-minute return prediction
   - `pred_1h`: 1-hour return prediction
   - `pred_24h`: 24-hour return prediction

4. **Calculate Confidence**:
   - Based on prediction magnitude and value estimate
   - Higher magnitude + higher value = higher confidence

### Prediction Endpoints

**15-Minute Prediction**:
```bash
curl http://localhost:5030/api/sol-price/predict-15m
```

**Multi-Horizon Predictions**:
```bash
curl http://localhost:5030/api/rl-agent/predictions?limit=1
```

### Prediction Storage

All predictions are stored in `rl_prediction_accuracy` table:
- Timestamp
- Predicted returns (15m, 1h, 24h)
- Confidence scores
- Price at prediction time
- Actual returns (updated later)
- Error metrics (MAE, RMSE)

---

## Example Data Structures

### Example State Encoding

```python
# Input: Current market state
prices = [150.00, 150.25, 150.50, ..., 152.00]  # Last 60 prices
price_features = {
    "current_price": 152.00,
    "sma_1h": 151.50,
    "sma_4h": 150.00,
    "sma_24h": 148.00,
    "rsi": 65.5,
    "std_dev": 2.5,
    "momentum_15m": 0.5,
    "percent_change": 2.5
}
news_data = [
    {"embedding": [...], "sentiment_score": 0.75, "headline": "..."},
    # ... up to 20 headlines
]
position_state = {
    "position_size": 10.0,
    "portfolio_value": 10000.0,
    "entry_price": 150.00,
    "time_since_last_trade": 30.0,
    "unrealized_pnl": 0.0133
}

# Encoded State
state_dict = {
    "price": np.array([...], shape=(70,)),  # 60 returns + 10 indicators
    "news_embeddings": np.array([...], shape=(20, 384)),
    "news_sentiment": np.array([0.75, 0.50, ...], shape=(20,)),
    "position": np.array([0.152, 1520.0, 1.013, 0.0133, 0.5], shape=(5,)),
    "time": np.array([0.604, 0.286, 0.604, 0.0], shape=(4,))
}
```

### Example Model Output

```python
output = {
    "action_logits": torch.Tensor([-0.2, 1.5, 2.1]),  # [SELL, HOLD, BUY]
    "value": torch.Tensor([0.025]),  # Expected 2.5% return
    "pred_15m": torch.Tensor([0.0012]),  # +0.12% in 15 min
    "pred_1h": torch.Tensor([0.0035]),  # +0.35% in 1 hour
    "pred_24h": torch.Tensor([0.015]),  # +1.5% in 24 hours
    "attention_weights": torch.Tensor([...])  # Which headlines matter
}

# Action selection
action_probs = softmax([-0.2, 1.5, 2.1]) = [0.11, 0.33, 0.56]
selected_action = 2  # BUY (56% confidence)
```

### Example Training Episode

```python
episode = {
    "prices": [150.00, 150.25, 150.50, ...],  # 100 steps
    "timestamps": [datetime(...), ...],
    "future_prices_15m": [150.18, 150.43, ...],  # 15 min ahead
    "future_prices_1h": [150.50, 150.75, ...],  # 1 hour ahead
    "future_prices_24h": [152.00, 152.25, ...],  # 24 hours ahead
    "price_features": [{...}, {...}, ...],  # Technical indicators per step
    "news_data": [[...], [...], ...],  # News at each step
    "actual_returns_15m": [0.0012, 0.0012, ...],  # Actual returns
    "actual_returns_1h": [0.0033, 0.0033, ...],
    "actual_returns_24h": [0.0133, 0.0133, ...]
}
```

---

## Feedback Loop

### Complete Learning Cycle

```
1. State Observation
   └── Encode: prices, news, position, time

2. Action & Prediction
   ├── Model outputs: action, value, pred_15m, pred_1h, pred_24h
   └── Store prediction in database

3. Wait for Outcome
   ├── Background loop checks every 15 minutes
   ├── Updates actual returns when available (1h, 24h)
   └── Calculates prediction errors

4. Training on Outcomes
   ├── Load predictions with actual returns
   ├── Train auxiliary heads on prediction errors
   └── Model learns from its mistakes

5. Improved Predictions
   └── Next predictions are more accurate
```

### Training from Outcomes

When using `--use-prediction-outcomes`:

1. **Load Prediction Outcomes**:
   ```python
   # From rl_prediction_accuracy table
   predictions = [
       {
           "predicted_return_1h": 0.0035,
           "actual_return_1h": 0.0042,  # What actually happened
           "error_1h": 0.0007,  # Model was off by 0.07%
           ...
       },
       # ... more predictions
   ]
   ```

2. **Create Training Episodes**:
   - Reconstruct state from stored decision data
   - Use actual returns as training targets
   - Model learns: "When I predicted X, actual was Y"

3. **Train Auxiliary Heads**:
   - `aux_15m` learns from 15-minute outcomes
   - `aux_1h` learns from 1-hour outcomes
   - `aux_24h` learns from 24-hour outcomes

### Continuous Improvement

- **Week 1**: Model makes predictions, some are wrong
- **Week 2**: Model trains on Week 1 outcomes, learns patterns
- **Week 3**: Model makes better predictions (learned from mistakes)
- **Week 4**: Model trains on Week 3 outcomes, improves further
- **Result**: Model gets smarter over time

---

## Model Specifications

### Architecture Details

**Total Parameters**: ~500K-1M (depending on configuration)

**Key Hyperparameters**:
- Learning Rate: `3e-4` (Adam optimizer)
- Discount Factor (γ): `0.99`
- GAE Lambda: `0.95`
- PPO Clip Epsilon: `0.2`
- Max Gradient Norm: `0.5`
- Batch Size: `32`
- Rollout Length: `128` steps

**Input Dimensions**:
- Price features: `70` (60 returns + 10 indicators)
- News embeddings: `(20, 384)`
- News sentiment: `20`
- Position features: `5`
- Time features: `4`

**Output Dimensions**:
- Action logits: `3` (SELL, HOLD, BUY)
- Value: `1` (scalar)
- Predictions: `1` each (15m, 1h, 24h)

### Model Initialization

- **Weights**: Xavier/Glorot uniform initialization (gain=0.5)
- **Biases**: Zero initialization
- **Convolutional Layers**: Kaiming uniform initialization

---

## Usage Examples

### Making a Decision

```python
from rl_agent.integration import RLAgentIntegration

integration = RLAgentIntegration(model=model, device="cpu")
decision = integration.make_decision()

# Returns:
{
    "action": "BUY",
    "confidence": 0.75,
    "current_price": 150.25,
    "prediction_1h": 0.0035,  # +0.35%
    "prediction_24h": 0.015,  # +1.5%
    "confidence_1h": 0.65,
    "confidence_24h": 0.58
}
```

### Generating 15-Minute Prediction

```python
from rl_agent.prediction_generator import generate_15m_price_prediction

pred_15m, conf_15m, price_15m, method = generate_15m_price_prediction(
    model=model,
    state_encoder=encoder,
    price_data=prices,
    price_features=features,
    news_data=news,
    position_state=position,
    current_price=150.25,
    timestamp=datetime.now(),
    device="cpu"
)

# Returns:
# pred_15m = 0.0012 (0.12% return)
# conf_15m = 0.65 (65% confidence)
# price_15m = 150.43 (predicted price)
# method = "aux_head" (using trained head) or "scaled_1h" (fallback)
```

### Training the Model

```python
from scripts.train_rl_agent import train_on_historical_data

train_on_historical_data(
    episodes_path="training_data/episodes.pkl",
    num_epochs=10,
    batch_size=32,
    enable_auxiliary=True,
    aux_15m_coef=1.0,
    use_prediction_outcomes=True,
    combine_with_historical=True
)
```

---

## Performance Metrics

### Prediction Accuracy

The model tracks accuracy via `PredictionManager`:

- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors more
- **Win Rate**: Percentage of predictions that were profitable

**Example Stats**:
```python
{
    "count_1h": 45,  # 45 predictions with 1h outcomes
    "count_24h": 30,  # 30 predictions with 24h outcomes
    "mae_1h": 0.002,  # Average error: 0.2%
    "mae_24h": 0.015,  # Average error: 1.5%
    "rmse_1h": 0.003,
    "rmse_24h": 0.020
}
```

### Model Performance

- **Policy Loss**: Should decrease (model learning better actions)
- **Value Loss**: Should decrease (better return estimation)
- **Auxiliary Losses**: Should decrease (better predictions)
- **Entropy**: Should stabilize (balanced exploration/exploitation)

---

## Key Design Decisions

### Why Multi-Horizon Predictions?

- **15m**: Short-term volatility, quick trades
- **1h**: Medium-term trends, swing trades
- **24h**: Long-term direction, position sizing

### Why Auxiliary Heads?

- **Shared Representation**: All heads use same state encoding
- **Multi-Task Learning**: Predictions help policy learning
- **Explainability**: Can see what model expects

### Why Attention for News?

- **Selective Focus**: Model learns which headlines matter
- **Variable Input**: Handles 0-20 headlines gracefully
- **Interpretability**: Attention weights show reasoning

### Why PPO?

- **Stability**: Clipped objective prevents large policy updates
- **Sample Efficiency**: Works well with limited data
- **Continuous Actions**: Handles continuous return predictions

---

## Limitations & Future Improvements

### Current Limitations

1. **Fixed Architecture**: Model size doesn't adapt to data
2. **Single Model**: No ensemble for uncertainty estimation
3. **Static Features**: Technical indicators are pre-calculated
4. **News Lag**: News embeddings may be delayed

### Potential Improvements

1. **Ensemble Models**: Multiple models for uncertainty
2. **Adaptive Architecture**: Dynamic model sizing
3. **Online Learning**: Update model incrementally
4. **Feature Engineering**: Learn optimal technical indicators
5. **Multi-Asset**: Extend to other cryptocurrencies

---

## References

- **PPO Paper**: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- **GAE**: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
- **Multi-Task RL**: Auxiliary tasks improve main task performance

---

**Documentation Version**: 1.0  
**Last Updated**: 2026-01-23
