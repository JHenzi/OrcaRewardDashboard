-- Database Indexes for Performance Optimization
-- Run this script to add all recommended indexes

-- ============================================================================
-- rewards.db indexes
-- ============================================================================

-- Indexes for collect_fees table
CREATE INDEX IF NOT EXISTS idx_collect_fees_timestamp ON collect_fees(timestamp);
CREATE INDEX IF NOT EXISTS idx_collect_fees_token_mint ON collect_fees(token_mint);
CREATE INDEX IF NOT EXISTS idx_collect_fees_to_user ON collect_fees(to_user_account);
CREATE INDEX IF NOT EXISTS idx_collect_fees_composite ON collect_fees(token_mint, timestamp);
CREATE INDEX IF NOT EXISTS idx_collect_fees_signature ON collect_fees(signature);
CREATE INDEX IF NOT EXISTS idx_collect_fees_redemption_date ON collect_fees(redemption_date);

-- ============================================================================
-- sol_prices.db indexes
-- ============================================================================

-- Indexes for sol_prices table
CREATE INDEX IF NOT EXISTS idx_sol_prices_timestamp ON sol_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_sol_prices_rate ON sol_prices(rate);
CREATE INDEX IF NOT EXISTS idx_sol_prices_timestamp_rate ON sol_prices(timestamp, rate);

-- Indexes for bandit_logs table
CREATE INDEX IF NOT EXISTS idx_bandit_logs_timestamp ON bandit_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_bandit_logs_action ON bandit_logs(action);
CREATE INDEX IF NOT EXISTS idx_bandit_logs_reward ON bandit_logs(reward);
CREATE INDEX IF NOT EXISTS idx_bandit_logs_timestamp_action ON bandit_logs(timestamp, action);

-- Indexes for trades table
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_action ON trades(action);
CREATE INDEX IF NOT EXISTS idx_trades_price ON trades(price);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp_action ON trades(timestamp, action);

-- Indexes for sol_predictions table (if still in use)
CREATE INDEX IF NOT EXISTS idx_sol_predictions_timestamp ON sol_predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_sol_predictions_created_at ON sol_predictions(created_at);

-- ============================================================================
-- Analyze tables for query optimization
-- ============================================================================

-- Run ANALYZE to update statistics for query planner
ANALYZE collect_fees;
ANALYZE sol_prices;
ANALYZE bandit_logs;
ANALYZE trades;

