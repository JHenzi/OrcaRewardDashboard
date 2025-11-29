# Training NaN Gradient Issue - Enhanced Fixes

## Problem

During training, NaN gradients are being generated in the auxiliary prediction heads (`aux_1h` and `aux_24h`), even after initial fixes.

## Root Causes

1. **Extreme return values**: Training data may contain extreme return values (>100% or <-100%) that cause numerical instability
2. **NaN propagation**: NaN values in predictions or returns propagate through loss computation
3. **Gradient explosion**: Large gradients in auxiliary heads cause NaN during backpropagation

## Enhanced Fixes Applied

### 1. Model Output Validation (`rl_agent/model.py`)
- ✅ Clamp auxiliary predictions to ±100% return range (`[-1.0, 1.0]`)
- ✅ Validate all outputs before returning

### 2. Training Loss Computation (`rl_agent/trainer.py`)
- ✅ **Immediate validation** of model outputs after forward pass
- ✅ **Clamp predictions** to reasonable range before loss computation
- ✅ **Clamp returns** to ±100% range to prevent extreme values
- ✅ **Skip loss computation** if data is invalid (set loss to 0)
- ✅ **Enhanced gradient clipping** with NaN detection and zeroing
- ✅ **Option to disable auxiliary losses** if they continue causing issues

### 3. Training Data Validation (`scripts/train_rl_agent.py`)
- ✅ Validates returns before storing in buffer
- ✅ Replaces NaN/inf with 0.0

## Usage

### Option 1: Continue Training with Enhanced Protection

The enhanced NaN protection should prevent the issue. Continue training:

```bash
python scripts/train_rl_agent.py --epochs 10
```

The warnings about NaN gradients should decrease as the model stabilizes.

### Option 2: Disable Auxiliary Losses Temporarily

If NaN gradients persist, you can disable auxiliary losses:

```python
trainer = PPOTrainer(
    model=model,
    environment=environment,
    enable_auxiliary_losses=False,  # Disable if causing issues
    ...
)
```

This will:
- Still generate predictions (for inference)
- Skip auxiliary loss computation during training
- Prevent NaN gradient issues
- Model will learn from policy/value losses only

### Option 3: Reduce Auxiliary Loss Coefficients

Reduce the impact of auxiliary losses:

```python
trainer = PPOTrainer(
    model=model,
    environment=environment,
    aux_1h_coef=0.01,  # Reduced from 0.1
    aux_24h_coef=0.01,  # Reduced from 0.1
    ...
)
```

## Monitoring

Watch for these log messages:
- `"NaN/inf gradient detected in aux_*"` - Gradient clipping is working
- `"Model output pred_* contains NaN/inf"` - Model outputs need attention
- `"aux_*_loss was NaN/inf, setting to 0"` - Loss computation issue

If these persist, consider:
1. Checking training data for extreme return values
2. Reducing learning rate
3. Temporarily disabling auxiliary losses
4. Increasing gradient clipping threshold

## Expected Behavior

After these fixes:
- ✅ NaN gradients should be caught and zeroed out
- ✅ Training should continue without crashing
- ✅ Predictions should stabilize over time
- ⚠️ Warnings may appear initially but should decrease as model stabilizes

## Next Steps

1. **Monitor training logs** - Watch for decreasing NaN warnings
2. **Check prediction values** - They should be in reasonable range (±100%)
3. **Validate training data** - Ensure returns are not extreme
4. **Adjust if needed** - Reduce coefficients or disable if issues persist

