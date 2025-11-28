# RL Agent Training Guide

> **Status**: Ready to train on historical data! ✅

## Important: Historical vs Real-Time Training

### Historical Training (What We're Doing) ✅

**We DON'T need to wait 24 hours** because we're training on **past data** where we already know what happened 24 hours later!

- ✅ Training uses historical price data (July–November 2025)
- ✅ We already have future prices (1h and 24h later) in the database
- ✅ Rewards are calculated immediately from known future prices
- ✅ No waiting required - we can train right now

### News Data Availability ⚠️

**Important**: News data is limited compared to price data:
- **Price data**: July 2025 - November 2025 (5 months, 202K+ points)
- **News data**: November 27-28, 2025 only (2 days, 550 articles)

**What this means:**
- ✅ **Training will work fine** - the model handles missing news gracefully
- ✅ **Empty news is padded with zeros** - state encoder automatically handles this
- ✅ **Model learns from price patterns first** - news enhances it later
- ✅ **As news accumulates**, retraining will incorporate it

**Training Strategy:**
1. **Phase 1 (Now)**: Train on price data + empty news (padded zeros)
   - Model learns price patterns and technical indicators
   - News branch gets zero inputs (model learns to ignore/use zeros)
   
2. **Phase 2 (Later)**: Retrain with accumulated news data
   - As news data grows (weeks/months), retraining incorporates it
   - Model learns to use news when available

**Example:**
- Training step at timestamp: `2025-11-01 10:00:00` (price: $150)
- We already know price at `2025-11-02 10:00:00` (price: $155)
- Reward = (155 - 150) / 150 = +3.3% (if BUY action)
- **No waiting needed!**

### Real-Time Training (Future - Not Implemented)

If we wanted to train in real-time as new data arrives:
- ❌ Would need to wait 24 hours to see outcome
- ❌ Would need to store decisions and update rewards later
- ❌ Much slower learning process

**We're NOT doing this** - we're using historical data where outcomes are already known.

### Real-Time Decision Making (After Training)

Once the model is trained:
- ✅ Model makes predictions immediately (no waiting)
- ✅ Predicts 1h and 24h returns based on learned patterns
- ✅ No need to wait - just uses current state to predict future

---

## Quick Start

### Step 1: Prepare Training Data

Extract historical data and create training episodes:

```bash
python -m rl_agent.training_data_prep
```

This will:
- Load price data from `sol_prices.db` (202,077+ data points available)
- Load news data from `news_sentiment.db` (547+ articles available)
- Create training episodes with state-action-reward sequences
- Save to `training_data/episodes.pkl`

**Expected output:**
```
Preparing training data from [start] to [end]
Retrieved 202077 price points
Created 100+ training episodes
✅ Created X training episodes
```

### Step 2: Train the Model

Train the RL agent on the prepared episodes:

```bash
python train_rl_agent.py --epochs 10 --batch-size 32
```

**Options:**
- `--episodes`: Path to episodes file (default: `training_data/episodes.pkl`)
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--checkpoint-dir`: Directory for checkpoints (default: `models/rl_agent`)
- `--device`: Device to use (`cpu` or `cuda`, default: `cpu`)
- `--resume`: Path to checkpoint to resume from

**Example with GPU:**
```bash
python train_rl_agent.py --epochs 20 --device cuda --batch-size 64
```

**Example resuming from checkpoint:**
```bash
python train_rl_agent.py --resume models/rl_agent/checkpoint_epoch_5.pt --epochs 10
```

### Step 3: Load Trained Model

After training, load the model in your application:

```python
from rl_agent.model import TradingActorCritic
from rl_agent.integration import RLAgentIntegration
import torch

# Load trained model
checkpoint = torch.load("models/rl_agent/checkpoint_epoch_10.pt")
model = TradingActorCritic(...)
model.load_state_dict(checkpoint['model_state_dict'])

# Initialize integration
integration = RLAgentIntegration(model=model)
decision = integration.make_decision()
```

---

## Data Availability

### Current Historical Data

✅ **Price Data**: 
- **202,077 data points** from July 2025 to November 2025
- 5-minute intervals
- ~5 months of data
- Sufficient for training!

⚠️ **News Data**:
- **550 articles** from November 27-28, 2025 only
- Includes embeddings and sentiment
- **Limited coverage** - only 2 days of news vs 5 months of prices
- More news will accumulate over time

### Data Coverage

- **Price coverage**: Excellent (5 months, 202K+ points)
- **News coverage**: Limited (2 days, 550 articles) - will grow over time
- **Training feasibility**: ✅ Yes, can train now on price data

### Handling Missing News

**The system handles missing news gracefully:**

1. **State Encoder**: Automatically pads empty news with zeros
   - If no news at timestamp T, returns empty list `[]`
   - State encoder pads to `max_news_headlines` with zero embeddings
   - Model receives: `(20, 384)` array of zeros for news embeddings

2. **Model Training**: Works fine with zero-padded news
   - Model learns from price patterns (primary signal)
   - News attention mechanism learns to ignore zero-padded inputs
   - As news accumulates, model learns to use it

3. **Training Strategy**:
   - **Now**: Train on price data + empty news (model learns price patterns)
   - **Later**: Retrain with accumulated news (model learns to use news)

---

## Training Process

### What Happens During Training

1. **Data Loading**: Loads preprocessed episodes from disk
   - Each episode contains: states, prices, timestamps, **future_prices_1h**, **future_prices_24h**
   - Future prices are already known (from historical data)

2. **Episode Processing**: For each episode:
   - Encodes states (price, news, position, time) at time T
   - Gets actions from model
   - **Calculates rewards from future prices** (already known!):
     - Reward uses price at T+24h (already in database)
     - No waiting needed - we're training on past data
   - Stores experiences in buffer

3. **Training**: When buffer is full:
   - Computes advantages using GAE
   - Updates model using PPO
   - Clears buffer

4. **Checkpointing**: Saves model after each epoch

**Key Point**: We're training on historical data where outcomes are already known, so we can calculate 24h rewards immediately without waiting.

### Training Metrics

The training script logs:
- Policy loss
- Value loss
- Entropy (exploration)
- Auxiliary losses (1h/24h prediction accuracy)
- Total training steps

---

## Periodic Retraining (Recommended)

### Automated Retraining

Use the `retrain_rl_agent.py` script for automated periodic retraining:

**Weekly retraining:**
```bash
python retrain_rl_agent.py --mode weekly --epochs 5
```

**Monthly retraining:**
```bash
python retrain_rl_agent.py --mode monthly --epochs 10
```

**Adaptive retraining (smart - retrains when enough new data):**
```bash
python retrain_rl_agent.py --mode adaptive
```

**Schedule with cron:**
```bash
# Weekly on Sunday 2 AM
0 2 * * 0 cd /path/to/project && python retrain_rl_agent.py --mode weekly --epochs 5
```

See [RETRAINING_STRATEGY.md](RETRAINING_STRATEGY.md) for detailed retraining strategies and automation setup.

---

## Troubleshooting

### "No episodes created"

**Problem**: Insufficient historical data

**Solution**: 
- Check database has enough data: `sqlite3 sol_prices.db "SELECT COUNT(*) FROM sol_prices;"`
- Reduce `episode_length` or `min_points` in `training_data_prep.py`
- Wait for more data to accumulate

### "Out of memory"

**Problem**: Too much data in memory

**Solution**:
- Reduce `batch_size`
- Process episodes in smaller batches
- Use `--device cpu` if GPU memory is limited

### "No news data"

**Problem**: News database empty or missing

**Solution**:
- Training can proceed with empty news (will use padding)
- News will improve model but isn't required
- Run news fetcher to collect more data

---

## Next Steps After Training

1. **Evaluate Model**: Test on held-out data
2. **Paper Trading**: Run in simulation for 1-2 weeks
3. **Load in App**: Update `app.py` to load trained model
4. **Monitor Performance**: Track decision quality

---

## Files Created

- `rl_agent/training_data_prep.py` - Data preparation script
- `train_rl_agent.py` - Main training script
- `training_data/episodes.pkl` - Preprocessed training data
- `models/rl_agent/checkpoint_epoch_*.pt` - Model checkpoints

---

## Summary

✅ **We have enough historical data to train now!**
- 202K+ price points (5 months)
- Training pipeline ready
- Can start training immediately

The model will learn from historical price patterns and can be improved as more news data accumulates.

