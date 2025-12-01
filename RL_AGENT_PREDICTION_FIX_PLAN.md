# RL Agent Prediction Fix Plan

## 🔴 Critical Issue

**Problem**: RL Agent is returning **0 values** for 1-hour and 24-hour return predictions.

**Root Cause**: The model was trained with `enable_auxiliary_losses=False`, meaning the auxiliary prediction heads (`aux_1h` and `aux_24h`) were **never trained**. They remain in their randomly initialized state, outputting zeros.

**Evidence**:
- `scripts/train_rl_agent.py` line 117: `enable_auxiliary_losses=False`
- Model architecture has auxiliary heads (lines 263-273 in `rl_agent/model.py`)
- Prediction generation code exists and is correct (`rl_agent/prediction_generator.py`)
- But heads were never updated during training

---

## 🎯 Solution Strategy

**Approach**: **Fix training infrastructure now** so that when the next scheduled retraining occurs (scheduled for 2025-12-05), it will train correctly with auxiliary losses enabled.

**We will NOT retrain immediately**, but prepare everything so the automated retraining scheduler will fix the issue automatically.

---

## 📋 Detailed Fix Plan

### Phase 1: Fix Training Script Configuration (1-2 hours)

#### 1.1 Investigate NaN Gradient Issues
- [ ] Review `rl_agent/trainer.py` auxiliary loss computation (lines 460-523)
- [ ] Check if issue is:
  - Invalid return targets (NaN/inf in actual returns)
  - Gradient explosion in auxiliary heads
  - Numerical instability in loss computation
- [ ] Add better validation and logging
- [ ] Document findings

**Files to check**:
- `rl_agent/trainer.py` - Auxiliary loss computation
- `rl_agent/training_data_prep.py` - Return target generation
- Training logs (if available)

#### 1.2 Fix Auxiliary Loss Computation in Trainer
- [ ] Add robust NaN/inf handling in `rl_agent/trainer.py`
- [ ] Add gradient clipping specifically for auxiliary heads
- [ ] Add validation that return targets are valid before computing loss
- [ ] Add logging to track auxiliary loss values during training
- [ ] Test loss computation with sample data

**Changes needed in `rl_agent/trainer.py`**:
```python
# 1. Better validation of return targets before computing loss
# 2. Gradient clipping for auxiliary heads (separate from main model)
# 3. More robust NaN/inf handling
# 4. Better logging for debugging
```

#### 1.3 Update Training Script to Enable Auxiliary Losses
- [ ] Modify `scripts/train_rl_agent.py`:
  - Change `enable_auxiliary_losses=False` to `enable_auxiliary_losses=True`
  - Set reduced coefficients: `aux_1h_coef=0.01, aux_24h_coef=0.01` (start conservative)
  - Add command-line arguments to override these settings
  - Add validation checks
  - Add better logging for auxiliary losses
- [ ] Ensure retraining script (`scripts/retrain_rl_agent.py`) will use these settings

**Changes needed in `scripts/train_rl_agent.py`**:
```python
# Line 117: Change from
enable_auxiliary_losses=False,
# To:
enable_auxiliary_losses=True,
aux_1h_coef=0.01,  # Reduced from 0.1 to prevent NaN issues
aux_24h_coef=0.01,  # Reduced from 0.1 to prevent NaN issues
```

#### 1.4 Add Command-Line Arguments for Flexibility
- [ ] Add `--enable-auxiliary-losses` flag (default: True)
- [ ] Add `--aux-1h-coef` argument (default: 0.01)
- [ ] Add `--aux-24h-coef` argument (default: 0.01)
- [ ] Update retraining script to pass these through if needed

### Phase 2: Validate Training Infrastructure (1-2 hours)

#### 2.1 Create Test Script
- [ ] Create `scripts/test_auxiliary_training.py`
- [ ] Test auxiliary loss computation with sample data
- [ ] Verify gradients flow correctly through auxiliary heads
- [ ] Test that training step completes without NaN/inf errors
- [ ] Validate predictions are non-zero after test training

**Test script should**:
```python
# 1. Load small sample of training data
# 2. Run one training step with auxiliary losses enabled
# 3. Check for NaN/inf in losses and gradients
# 4. Verify auxiliary head weights are updated
# 5. Test prediction generation after training step
```

#### 2.2 Validate Data Preparation
- [ ] Check `rl_agent/training_data_prep.py` generates valid return targets
- [ ] Verify return targets don't contain NaN/inf
- [ ] Ensure return targets are in reasonable range (±1.0 = ±100%)
- [ ] Add validation checks to data prep script

#### 2.3 Test Training on Small Dataset
- [ ] Run training on small subset (1-2 episodes)
- [ ] Monitor for NaN/inf issues
- [ ] Verify auxiliary losses decrease
- [ ] Check predictions are non-zero after training
- [ ] Document any issues found

### Phase 3: Update Retraining Integration (30 minutes)

#### 3.1 Verify Retraining Script Uses Fixed Training Script
- [ ] Check `scripts/retrain_rl_agent.py` calls `train_rl_agent.py` correctly
- [ ] Ensure retraining script doesn't override auxiliary loss settings
- [ ] Verify scheduler will use updated training script
- [ ] Test manual retraining trigger: `POST /api/rl-agent/retrain`

#### 3.2 Add Validation to Retraining
- [ ] Add check that trained model produces non-zero predictions
- [ ] Add validation before model deployment
- [ ] Add logging for prediction values in retraining logs

### Phase 4: Documentation & Monitoring (1 hour)

#### 4.1 Update Documentation
- [ ] Document auxiliary loss configuration
- [ ] Add notes about NaN gradient issues and fixes
- [ ] Update training guide with auxiliary loss settings
- [ ] Document how to verify predictions are working

#### 4.2 Add Monitoring/Alerting
- [ ] Add check in prediction generator to log warning if predictions are zero
- [ ] Add validation in model loading to check prediction heads
- [ ] Consider adding health check endpoint for predictions

---

## 🔧 Immediate Actions (Do Now)

### Step 1: Fix Training Script
```python
# In scripts/train_rl_agent.py, line 111-120
trainer = PPOTrainer(
    model=model,
    environment=environment,
    lr=3e-4,
    device=device,
    checkpoint_dir=checkpoint_dir,
    enable_auxiliary_losses=True,  # CHANGE: Enable auxiliary losses
    aux_1h_coef=0.01,  # ADD: Reduced coefficient to prevent NaN
    aux_24h_coef=0.01,  # ADD: Reduced coefficient to prevent NaN
)
```

### Step 2: Improve Auxiliary Loss Handling
```python
# In rl_agent/trainer.py, improve NaN handling in auxiliary loss computation
# Add validation before computing loss
# Add gradient clipping for auxiliary heads
# Add better logging
```

### Step 3: Test Training Infrastructure
```bash
# Create test script to verify training works
python scripts/test_auxiliary_training.py
```

### Step 4: Verify Next Retraining Will Use Fixed Script
```python
# Check that retraining scheduler will use updated train_rl_agent.py
# Verify next retrain time (should be 2025-12-05)
curl http://localhost:5030/api/rl-agent/status
```

---

## 📝 Implementation Checklist

### Priority 1: Fix Training Configuration (Do First)
- [ ] Update `scripts/train_rl_agent.py` to enable auxiliary losses
- [ ] Set conservative coefficients (0.01 for both)
- [ ] Improve NaN handling in `rl_agent/trainer.py`
- [ ] Add validation checks for return targets
- [ ] Test training step on small dataset

### Priority 2: Validate Infrastructure (Before Next Training)
- [ ] Create test script for auxiliary training
- [ ] Test auxiliary loss computation
- [ ] Verify gradients flow correctly
- [ ] Test on small dataset end-to-end
- [ ] Document any issues

### Priority 3: Integration & Monitoring
- [ ] Verify retraining script integration
- [ ] Add prediction validation to retraining
- [ ] Add monitoring/alerting for zero predictions
- [ ] Update documentation

---

## 🧪 Testing Plan

### Unit Tests
- [ ] Test auxiliary loss computation with valid data
- [ ] Test auxiliary loss computation with invalid data (NaN/inf)
- [ ] Test gradient flow through auxiliary heads
- [ ] Test prediction extraction from model output

### Integration Tests
- [ ] Test training step with auxiliary losses enabled
- [ ] Test full training cycle on small dataset
- [ ] Test that predictions are non-zero after training
- [ ] Test retraining script uses fixed configuration

### Validation Tests (After Next Training)
- [ ] Predictions are non-zero after retraining
- [ ] Predictions are in reasonable range (±1.0 = ±100%)
- [ ] Predictions vary with different market states
- [ ] No NaN/inf values in predictions
- [ ] Training completes without errors

---

## 📊 Success Criteria

**Before Next Training**:
1. ✅ Training script has `enable_auxiliary_losses=True`
2. ✅ Auxiliary loss coefficients set to 0.01 (conservative)
3. ✅ NaN/inf handling improved in trainer
4. ✅ Test training completes without errors
5. ✅ Test predictions are non-zero after training

**After Next Scheduled Training** (2025-12-05):
1. ✅ Training completes successfully
2. ✅ Predictions are **non-zero** (at least ±0.001 = ±0.1%)
3. ✅ Predictions are **reasonable** (within ±1.0 = ±100% return)
4. ✅ Predictions **vary** with different market states
5. ✅ No NaN/inf errors during training

---

## 🚨 Risk Mitigation

### If Training Still Fails with NaN
- Keep `enable_auxiliary_losses=False` as fallback
- Use fine-tuning approach instead (separate script)
- Consider separate prediction model

### If Predictions Are Still Zero After Training
- Check if auxiliary heads are actually being updated
- Verify return targets are valid
- Consider increasing auxiliary loss coefficients gradually
- May need more training epochs

### If Next Training Doesn't Happen
- Manual trigger: `POST /api/rl-agent/retrain`
- Or run: `python scripts/retrain_rl_agent.py --mode incremental`

---

## 📅 Timeline

- **Today**: Fix training script configuration, improve NaN handling
- **Day 1**: Create test script, validate training infrastructure
- **Day 2**: Test on small dataset, document any issues
- **Before 2025-12-05**: All fixes in place, ready for scheduled retraining
- **After 2025-12-05**: Validate predictions are fixed after automated retraining

---

## 📚 Related Files

**Files to Modify**:
- `scripts/train_rl_agent.py` - Enable auxiliary losses, set coefficients
- `rl_agent/trainer.py` - Improve NaN handling, add validation
- `rl_agent/training_data_prep.py` - Validate return targets (if needed)

**Files to Create**:
- `scripts/test_auxiliary_training.py` - Test script for validation

**Files to Verify**:
- `scripts/retrain_rl_agent.py` - Uses fixed training script
- `rl_agent/retraining_scheduler.py` - Will trigger with fixed script
- `rl_agent/model.py` - Auxiliary heads architecture (no changes needed)
- `rl_agent/prediction_generator.py` - Prediction extraction (no changes needed)

---

## ✅ Verification Steps

### Before Next Training
```bash
# 1. Verify training script is fixed
grep "enable_auxiliary_losses" scripts/train_rl_agent.py
# Should show: enable_auxiliary_losses=True

# 2. Test training infrastructure
python scripts/test_auxiliary_training.py

# 3. Check next retraining time
curl http://localhost:5030/api/rl-agent/status | grep next_retrain_time
```

### After Next Training
```bash
# 1. Test predictions are non-zero
curl -X POST http://localhost:5030/api/rl-agent/decision
# Check: predicted_return_1h and predicted_return_24h should not be 0

# 2. Check training logs for auxiliary losses
# Should see aux_1h_loss and aux_24h_loss decreasing

# 3. Verify model checkpoint was updated
ls -la models/rl_agent/checkpoint_*.pt
```

---

**Status**: 🔴 **CRITICAL** - Predictions are broken, but we're fixing infrastructure for next training
**Priority**: **HIGH** - Core feature not working, but fix is scheduled
**Approach**: **Fix training infrastructure now**, let automated retraining fix predictions on 2025-12-05
**Estimated Fix Time**: 1-2 days to prepare infrastructure, then wait for scheduled retraining
