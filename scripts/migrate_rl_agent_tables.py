"""
Database Migration: RL Agent Tables

Creates database tables for the RL agent system.
Run this script once to set up the database schema.
"""

import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

# Get database path from environment or use default (relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
default_db_path = os.path.join(project_root, "sol_prices.db")
DB_PATH = os.getenv("DATABASE_PATH", default_db_path)


def create_rl_agent_tables(db_path: str = DB_PATH):
    """Create all RL agent tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Creating RL agent tables in {db_path}...")
    
    # RL Agent Decisions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_agent_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,  -- BUY, SELL, HOLD
            confidence REAL,  -- Confidence in the action (0.0 to 1.0)
            action_probabilities TEXT,  -- JSON: {BUY: 0.3, SELL: 0.1, HOLD: 0.6}
            state_features TEXT,  -- JSON of state at decision time
            price_features TEXT,  -- JSON of price features at decision time
            current_price REAL,  -- Price at time of decision
            predicted_return_1h REAL,
            predicted_return_24h REAL,
            predicted_return_1h_confidence REAL,
            predicted_confidence_1h REAL,  -- Alias for predicted_return_1h_confidence
            predicted_return_24h_confidence REAL,
            predicted_confidence_24h REAL,  -- Alias for predicted_return_24h_confidence
            value_estimate REAL,
            entropy REAL,
            portfolio_value REAL,
            position_size REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rl_decisions_timestamp 
        ON rl_agent_decisions(timestamp)
    """)
    
    # RL Attention Logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_attention_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER NOT NULL,
            headline_id INTEGER,  -- Reference to news_sentiment table
            headline_text TEXT,
            attention_weight REAL,
            cluster_id INTEGER,
            FOREIGN KEY (decision_id) REFERENCES rl_agent_decisions(id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_attention_decision 
        ON rl_attention_logs(decision_id)
    """)
    
    # Discovered Rules
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS discovered_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_text TEXT NOT NULL,
            rule_conditions TEXT,  -- JSON of conditions
            action TEXT NOT NULL,
            win_rate REAL,
            avg_return_1h REAL,
            avg_return_24h REAL,
            sample_size INTEGER,
            confidence_interval_lower REAL,
            confidence_interval_upper REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_validated_at DATETIME
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rules_action 
        ON discovered_rules(action)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rules_win_rate 
        ON discovered_rules(win_rate DESC)
    """)
    
    # News Clusters
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cluster_id INTEGER NOT NULL,
            cluster_label TEXT,
            representative_headlines TEXT,  -- JSON array
            embedding_centroid BLOB,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_clusters_id 
        ON news_clusters(cluster_id)
    """)
    
    # RL Training Metrics
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            training_step INTEGER NOT NULL,
            policy_loss REAL,
            value_loss REAL,
            entropy REAL,
            total_loss REAL,
            learning_rate REAL,
            clip_fraction REAL,
            explained_variance REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_training_step 
        ON rl_training_metrics(training_step)
    """)
    
    # RL Rule Firings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_rule_firings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER NOT NULL,
            decision_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            outcome_1h REAL,  -- Filled after 1h
            outcome_24h REAL,  -- Filled after 24h
            FOREIGN KEY (rule_id) REFERENCES discovered_rules(id),
            FOREIGN KEY (decision_id) REFERENCES rl_agent_decisions(id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rule_firings_rule 
        ON rl_rule_firings(rule_id)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rule_firings_timestamp 
        ON rl_rule_firings(timestamp)
    """)
    
    # RL Prediction Accuracy (for Phase 4.2)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_prediction_accuracy (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER,
            timestamp TEXT NOT NULL,
            predicted_return_1h REAL,
            predicted_return_24h REAL,
            predicted_confidence_1h REAL,
            predicted_confidence_24h REAL,
            actual_return_1h REAL,
            actual_return_24h REAL,
            price_at_prediction REAL,
            price_1h_later REAL,
            price_24h_later REAL,
            error_1h REAL,
            error_24h REAL,
            mae_1h REAL,
            mae_24h REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME,
            FOREIGN KEY (decision_id) REFERENCES rl_agent_decisions(id)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_accuracy_timestamp 
        ON rl_prediction_accuracy(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_accuracy_decision 
        ON rl_prediction_accuracy(decision_id)
    """)
    
    conn.commit()
    conn.close()
    
    print("✅ RL agent tables created successfully!")
    print("\nCreated tables:")
    print("  - rl_agent_decisions")
    print("  - rl_attention_logs")
    print("  - discovered_rules")
    print("  - news_clusters")
    print("  - rl_training_metrics")
    print("  - rl_rule_firings")
    print("  - rl_prediction_accuracy (Phase 4.2)")


if __name__ == "__main__":
    create_rl_agent_tables()

