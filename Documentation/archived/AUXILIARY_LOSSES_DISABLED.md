# Auxiliary Losses Temporarily Disabled

## Status

Auxiliary losses for 1h/24h return prediction have been **temporarily disabled** in the training script due to persistent NaN gradient issues.

## Why Disabled

Despite extensive NaN protection measures, the auxiliary heads continue to produce NaN gradients during training. The protection code is working (detecting and zeroing them), but the constant warnings indicate a deeper issue that needs investigation.

## Current Training Configuration

In `train_rl_agent.py`, auxiliary losses are disabled:

```python
trainer = PPOTrainer(
    model=model,
    environment=environment,
    lr=3e-4,
    device=device,
    checkpoint_dir=checkpoint_dir,
    enable_auxiliary_losses=False,  # Temporarily disabled
)
```

## Impact

**What Still Works:**
- ✅ Policy learning (action selection)
- ✅ Value estimation (state values)
- ✅ PPO training loop
- ✅ Model checkpoints
- ✅ Model inference

**What's Disabled:**
- ❌ Auxiliary loss for 1h return prediction
- ❌ Auxiliary loss for 24h return prediction
- ⚠️ Predictions will still be generated but won't be trained via supervised loss

## Next Steps

### Option 1: Continue Training Without Auxiliary Losses

The model will learn to make good trading decisions (BUY/SELL/HOLD) based on policy and value losses. Predictions will be generated but may not be accurate until we fix the auxiliary heads.

**Pros:**
- Training proceeds without NaN issues
- Model learns core trading policy
- Can be used for decision-making

**Cons:**
- Predictions won't be trained
- May need to retrain later with auxiliary losses

### Option 2: Investigate Root Cause

The NaN gradients suggest one of:
1. **Corrupted auxiliary head weights** - Even after reinitialization, they may be getting corrupted again
2. **Numerical instability** - The loss computation or gradient flow has numerical issues
3. **Invalid training data** - Returns may have extreme values causing issues

**Investigation steps:**
1. Check if auxiliary head weights become NaN during training
2. Add more detailed logging of predictions vs. actual returns
3. Verify training data has valid return values
4. Try reducing learning rate for auxiliary heads only

### Option 3: Reduce Auxiliary Loss Coefficients

Instead of disabling, reduce their impact:

```python
trainer = PPOTrainer(
    ...
    enable_auxiliary_losses=True,
    aux_1h_coef=0.001,  # Very small
    aux_24h_coef=0.001,
)
```

## Re-enabling Auxiliary Losses

Once the root cause is identified and fixed:

1. Set `enable_auxiliary_losses=True` in `train_rl_agent.py`
2. Optionally reduce coefficients: `aux_1h_coef=0.01, aux_24h_coef=0.01`
3. Monitor training logs for NaN gradient warnings
4. If warnings persist, investigate further

## Current Training Status

Training will proceed normally without auxiliary losses. The model will:
- Learn optimal trading actions (policy)
- Estimate state values (critic)
- Generate predictions (though untrained via supervised loss)

This is still valuable for the core RL objective of learning to trade!

