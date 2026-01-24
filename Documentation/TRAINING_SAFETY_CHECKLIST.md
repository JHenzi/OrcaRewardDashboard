# Training Safety Checklist

## ✅ Ready to Train (All Fixes Applied)

### Memory Fixes
- ✅ **Checkpoint loading**: Memory-efficient (drops optimizer state immediately)
- ✅ **Low-memory mode**: `--low-memory` flag reduces batch size to 8
- ✅ **Garbage collection**: Added after loading, training steps, and epochs
- ✅ **Episode limiting**: `--max-episodes` limits how many episodes load
- ✅ **Buffer threshold**: Lower in low-memory mode (2x batch_size vs 4x)

### Data Fixes
- ✅ **Corrupted embeddings**: Fixed (11,791 corrupted embeddings cleared and regenerated)
- ✅ **Class imbalance**: Fixed thresholds (15m: ±0.2%, 1h: ±0.5%, 24h: ±1%)
- ✅ **Class weights**: Added (2x penalty for bearish/bullish to prevent always-neutral)

### Model Fixes
- ✅ **Return value unpacking**: Fixed (5 values from generate_15m_price_prediction)
- ✅ **External markets**: Safe (zeros during training, won't crash)

## ⚠️ Known Issues

### Large Episodes File
- **Size**: `episodes.pkl` is 2.9GB
- **Risk**: Loading entire file into memory could still be problematic
- **Mitigation**: Use `--max-episodes` to limit (e.g., `--max-episodes 300`)

### No Historical External Markets
- **Status**: BTC/S&P 500 features are zeros during training
- **Impact**: None (model learns to ignore zeros)
- **Action**: None needed for now

## 🚀 Safe Training Commands

### Option 1: Conservative (Recommended First)
```bash
python scripts/train_rl_agent.py \
  --low-memory \
  --max-episodes 300 \
  --epochs 3 \
  --batch-size 4
```
**Memory**: ~2-4GB
**Time**: ~5-10 minutes

### Option 2: Moderate
```bash
python scripts/train_rl_agent.py \
  --low-memory \
  --max-episodes 500 \
  --epochs 5 \
  --batch-size 8
```
**Memory**: ~4-6GB
**Time**: ~10-15 minutes

### Option 3: Full (If Mac has 16GB+ RAM)
```bash
python scripts/train_rl_agent.py \
  --low-memory \
  --max-episodes 1000 \
  --epochs 10 \
  --batch-size 8
```
**Memory**: ~6-8GB
**Time**: ~20-30 minutes

## 📊 What to Watch For

### During Training
1. **Memory usage**: Monitor Activity Monitor - should stay under 8GB
2. **Class distribution logs**: Should show balanced classes (not 90%+ neutral)
   ```
   📊 Aux 1h (±0.5% threshold) | Acc: 45% | Pred: B=30%/N=40%/Bu=30% | Actual: B=28%/N=42%/Bu=30%
   ```
3. **No crashes**: Training should complete without system freeze

### After Training
1. **Model predictions**: Should NOT be 100% neutral anymore
2. **Confidence**: Should vary (not always 1.0)
3. **Predictions**: Should show bearish/bullish predictions

## 🔧 If It Still Crashes

1. **Reduce episodes further**: `--max-episodes 100`
2. **Reduce epochs**: `--epochs 1` (just test)
3. **Check system memory**: Close other apps
4. **Check episodes.pkl**: Might need to recreate with fewer episodes

## ✅ Pre-Training Verification

Run this to verify everything is ready:
```bash
# Check episodes file exists
ls -lh training_data/episodes.pkl

# Check database has valid embeddings
python -c "
import sqlite3
conn = sqlite3.connect('news_sentiment.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM news_articles WHERE embedding IS NOT NULL')
print(f'Valid embeddings: {cursor.fetchone()[0]}')
conn.close()
"

# Quick test (should not crash)
python scripts/train_rl_agent.py --low-memory --max-episodes 10 --epochs 1 --batch-size 2
```

## Summary

**Status**: ✅ **SAFE TO TRAIN** with low-memory flags

**Recommended Command**:
```bash
python scripts/train_rl_agent.py --low-memory --max-episodes 300 --epochs 3
```

This should:
- ✅ Not crash your Mac
- ✅ Use reasonable memory (~2-4GB)
- ✅ Train with fixed class thresholds
- ✅ Complete in ~5-10 minutes
