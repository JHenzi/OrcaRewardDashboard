# RL Agent Retraining Strategy

> **Goal**: Keep the model up-to-date with new market data and patterns

---

## Data Accumulation Rate

Based on your current setup:

### Price Data
- **Frequency**: Every 5 minutes (configurable, typically 5-30 min)
- **Daily**: ~288 data points (5-min intervals)
- **Weekly**: ~2,016 data points
- **Monthly**: ~8,640 data points

### News Data
- **Frequency**: Every 6 hours (with 5-min cooldown)
- **Daily**: ~4-10 articles (varies by feed activity)
- **Weekly**: ~28-70 articles
- **Monthly**: ~120-300 articles

---

## Recommended Retraining Frequencies

### Option 1: Weekly Retraining (Recommended) ⭐

**Why Weekly:**
- ✅ ~2,000 new price points per week (good for learning)
- ✅ ~30-70 new news articles (enough for pattern updates)
- ✅ Balances freshness vs. computational cost
- ✅ Catches weekly market patterns

**When to Run:**
- Every Sunday night (low market activity)
- Or Monday morning before market opens

**Command:**
```bash
python retrain_rl_agent.py --mode weekly --epochs 5
```

### Option 2: Monthly Retraining

**Why Monthly:**
- ✅ ~8,600 new price points (very comprehensive)
- ✅ ~120-300 news articles (rich context)
- ✅ Less frequent = less computational overhead
- ✅ Good for major pattern shifts

**When to Run:**
- First Sunday of each month
- Or first Monday of month

**Command:**
```bash
python retrain_rl_agent.py --mode monthly --epochs 10
```

### Option 3: Adaptive Retraining (Smart) 🧠

**Why Adaptive:**
- ✅ Retrains only when enough new data accumulates
- ✅ Saves compute when data is sparse
- ✅ Automatically adapts to data availability

**Thresholds:**
- Minimum 2,000 new price points
- Minimum 50 new news articles (optional)
- At least 7 days since last training

**Command:**
```bash
python retrain_rl_agent.py --mode adaptive
```

**When it retrains:**
- Automatically checks conditions
- Retrains if thresholds met
- Skips if insufficient data

---

## Retraining Modes

### Full Retraining
- Uses **all historical data** (from beginning)
- More comprehensive but slower
- Best for major updates or initial training

```bash
python retrain_rl_agent.py --mode full --epochs 10
```

### Incremental Retraining
- Uses **recent data only** (last 30 days)
- Faster, focuses on recent patterns
- Best for regular updates

```bash
python retrain_rl_agent.py --mode incremental --epochs 5
```

---

## Automation Setup

### Cron Job (Linux/Mac)

**Weekly retraining (Sunday 2 AM):**
```bash
# Add to crontab: crontab -e
0 2 * * 0 cd /path/to/OrcaRedemptionTracker && /usr/bin/python3 retrain_rl_agent.py --mode weekly --epochs 5 >> retrain.log 2>&1
```

**Monthly retraining (1st of month, 2 AM):**
```bash
0 2 1 * * cd /path/to/OrcaRedemptionTracker && /usr/bin/python3 retrain_rl_agent.py --mode monthly --epochs 10 >> retrain.log 2>&1
```

**Adaptive (daily check, 3 AM):**
```bash
0 3 * * * cd /path/to/OrcaRedemptionTracker && /usr/bin/python3 retrain_rl_agent.py --mode adaptive >> retrain.log 2>&1
```

### Systemd Timer (Linux)

Create `/etc/systemd/system/rl-agent-retrain.service`:
```ini
[Unit]
Description=RL Agent Weekly Retraining
After=network.target

[Service]
Type=oneshot
User=your-user
WorkingDirectory=/path/to/OrcaRedemptionTracker
ExecStart=/usr/bin/python3 retrain_rl_agent.py --mode weekly --epochs 5
StandardOutput=append:/path/to/retrain.log
StandardError=append:/path/to/retrain.log
```

Create `/etc/systemd/system/rl-agent-retrain.timer`:
```ini
[Unit]
Description=RL Agent Weekly Retraining Timer
Requires=rl-agent-retrain.service

[Timer]
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable rl-agent-retrain.timer
sudo systemctl start rl-agent-retrain.timer
```

### Manual Scheduling

For testing or manual control:
```bash
# Check if retraining is needed
python retrain_rl_agent.py --mode adaptive

# Force retraining
python retrain_rl_agent.py --mode weekly --force

# Full retraining with more epochs
python retrain_rl_agent.py --mode full --epochs 20
```

---

## What Gets Retrained

### Data Used
- **Price history**: All or recent (depending on mode)
- **News articles**: All or recent (depending on mode)
- **Technical indicators**: Recalculated from price data
- **Rewards**: Calculated from future price movements

### Model Updates
- Policy network (action selection)
- Value network (state value estimation)
- Auxiliary heads (1h/24h return predictions)
- All parameters updated via PPO

### Checkpoints
- Saved after each epoch
- Named: `checkpoint_epoch_N.pt`
- Stored in: `models/rl_agent/`
- Latest checkpoint is automatically used

---

## Monitoring Retraining

### Check Last Training Time
```bash
python retrain_rl_agent.py --mode adaptive
# Shows: "Last training: 2025-11-28 10:00:00"
```

### View Training Logs
```bash
tail -f retrain.log
```

### Check Model Performance
- Compare metrics before/after retraining
- Monitor prediction accuracy
- Track decision quality

---

## Cost Considerations

### Computational Cost
- **Weekly (5 epochs)**: ~30-60 minutes (CPU) or ~5-10 minutes (GPU)
- **Monthly (10 epochs)**: ~1-2 hours (CPU) or ~15-20 minutes (GPU)
- **Full (20 epochs)**: ~4-8 hours (CPU) or ~1-2 hours (GPU)

### Storage Cost
- Each checkpoint: ~10-50 MB
- Keep last 3-5 checkpoints
- Total: ~50-250 MB

---

## Recommendations

### For Production Use

1. **Start with Weekly Adaptive**:
   ```bash
   # Daily check, retrains if conditions met
   0 3 * * * python retrain_rl_agent.py --mode adaptive
   ```

2. **Monitor for 1 month**:
   - Track how often it actually retrains
   - Adjust thresholds if needed
   - Check model performance improvements

3. **Optimize based on results**:
   - If retraining too often → increase thresholds
   - If retraining too rarely → decrease thresholds
   - If performance degrades → retrain more frequently

### For Development/Testing

1. **Manual retraining**:
   ```bash
   python retrain_rl_agent.py --mode incremental --epochs 3 --force
   ```

2. **Quick iterations**:
   - Use fewer epochs (3-5)
   - Use incremental mode
   - Test on new data quickly

---

## Troubleshooting

### "Retraining not needed"
- **Cause**: Not enough new data or too recent
- **Solution**: Use `--force` flag or wait for more data

### "Out of memory"
- **Cause**: Too much data in memory
- **Solution**: Use incremental mode or reduce batch size

### "Training failed"
- **Cause**: Data corruption or insufficient data
- **Solution**: Check data quality, verify databases

---

## Summary

**Recommended Setup:**
- **Frequency**: Weekly adaptive retraining
- **Mode**: Incremental (last 30 days)
- **Epochs**: 5 epochs per retraining
- **Schedule**: Sunday 2 AM (or daily check with adaptive)

**Command:**
```bash
python retrain_rl_agent.py --mode adaptive
```

This gives you the best balance of:
- ✅ Model freshness
- ✅ Computational efficiency  
- ✅ Automatic adaptation to data availability

