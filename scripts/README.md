# Scripts Directory

This directory contains standalone utility scripts that are not part of the main application but are used for maintenance, training, and database management.

## Training Scripts

- **`train_rl_agent.py`** - Train the RL agent on historical data
  ```bash
  python scripts/train_rl_agent.py --epochs 10
  ```

- **`retrain_rl_agent.py`** - Automated retraining with adaptive scheduling
  ```bash
  python scripts/retrain_rl_agent.py --mode adaptive
  ```

## Database Migration Scripts

- **`migrate_rl_agent_tables.py`** - Create RL agent database tables
  ```bash
  python scripts/migrate_rl_agent_tables.py
  ```

- **`migrate_add_rl_confidence_column.py`** - Add confidence columns to RL agent decisions table
  ```bash
  python scripts/migrate_add_rl_confidence_column.py
  ```

- **`migrate_add_redemption_values.py`** - Add USD value tracking to redemption records
  ```bash
  python scripts/migrate_add_redemption_values.py
  ```

## Utility Scripts

- **`fix_auxiliary_heads.py`** - Fix NaN weights in auxiliary prediction heads
  ```bash
  python scripts/fix_auxiliary_heads.py --checkpoint models/rl_agent/checkpoint_epoch_10.pt
  ```

- **`reprocess_news_sentiment.py`** - Reprocess news articles for sentiment analysis

- **`wipe_bandits_table.py`** - Clear contextual bandit data (deprecated feature)

- **`unified_trade_predictor.py`** - Unified trade prediction system (deprecated)

## Note

These scripts are standalone and not imported by the main application (`app.py`). They can be run independently for maintenance, training, or database operations.

