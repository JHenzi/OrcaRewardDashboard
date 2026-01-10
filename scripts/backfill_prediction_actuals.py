#!/usr/bin/env python3
"""
Backfill missing actual returns for all predictions.

This script will:
1. Find all predictions missing 1h or 24h actual returns
2. Look up the actual prices from sol_prices.db
3. Calculate and update the actual returns

Run this to fix predictions that weren't updated by the background loop.
"""

import sqlite3
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.prediction_manager import PredictionManager

def backfill_actual_returns():
    """Backfill all missing actual returns."""
    # Get database paths - use environment or find relative to project root
    project_root = Path(__file__).parent.parent
    db_path = os.getenv("DATABASE_PATH", str(project_root / "rewards.db"))
    sol_prices_path = os.getenv("SOL_PRICES_PATH", str(project_root / "sol_prices.db"))
    
    # Fallback to main project directory if not found in worktree
    if not Path(db_path).exists():
        main_project = Path("/Users/joe/Local Development/OrcaRedemptionTracker")
        if (main_project / "rewards.db").exists():
            db_path = str(main_project / "rewards.db")
            sol_prices_path = str(main_project / "sol_prices.db")
            print(f"📁 Using databases from main project: {main_project}")
    
    print(f"🔍 Connecting to predictions database: {db_path}")
    print(f"🔍 Using price database: {sol_prices_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all predictions missing 1h returns
    cursor.execute("""
        SELECT id, timestamp, price_at_prediction
        FROM rl_prediction_accuracy
        WHERE actual_return_1h IS NULL
        AND price_at_prediction IS NOT NULL
        AND price_at_prediction > 0
        ORDER BY timestamp ASC
    """)
    predictions_1h = cursor.fetchall()
    
    # Get all predictions missing 24h returns
    cursor.execute("""
        SELECT id, timestamp, price_at_prediction
        FROM rl_prediction_accuracy
        WHERE actual_return_24h IS NULL
        AND price_at_prediction IS NOT NULL
        AND price_at_prediction > 0
        ORDER BY timestamp ASC
    """)
    predictions_24h = cursor.fetchall()
    
    print(f"📊 Found {len(predictions_1h)} predictions missing 1h returns")
    print(f"📊 Found {len(predictions_24h)} predictions missing 24h returns")
    
    conn.close()
    
    prediction_manager = PredictionManager(db_path=db_path)
    updated_1h = 0
    updated_24h = 0
    failed_1h = 0
    failed_24h = 0
    
    # Update 1h predictions
    print("\n🔄 Updating 1h predictions...")
    for pred_id, pred_timestamp, price_at_pred in predictions_1h:
        try:
            # Parse timestamp
            if isinstance(pred_timestamp, str):
                if 'Z' in pred_timestamp:
                    pred_dt = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
                elif '+' in pred_timestamp or pred_timestamp.count('-') >= 3:
                    pred_dt = datetime.fromisoformat(pred_timestamp)
                else:
                    # Try parsing as naive datetime
                    pred_dt = datetime.fromisoformat(pred_timestamp)
            else:
                pred_dt = datetime.fromisoformat(pred_timestamp)
            
            target_dt = pred_dt + timedelta(hours=1)
            
            # Get price 1 hour later (wider window for better matching)
            # Use sol_prices.db for price data (separate from predictions database)
            price_conn = sqlite3.connect(sol_prices_path)
            price_cursor = price_conn.cursor()
            
            # Use wider time window: 30 minutes before to 1 hour after
            target_start = (target_dt - timedelta(minutes=30)).isoformat()
            target_end = (target_dt + timedelta(hours=1)).isoformat()
            
            price_cursor.execute("""
                SELECT rate, timestamp
                FROM sol_prices
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY ABS(julianday(timestamp) - julianday(?))
                LIMIT 1
            """, (target_start, target_end, target_dt.isoformat()))
            
            price_row = price_cursor.fetchone()
            price_conn.close()
            
            if price_row and price_at_pred and price_at_pred > 0:
                price_1h_later = price_row[0]
                actual_return_1h = (price_1h_later - price_at_pred) / price_at_pred
                actual_return_1h = max(-1.0, min(1.0, actual_return_1h))
                
                prediction_manager.update_actual_returns(
                    prediction_id=pred_id,
                    actual_return_1h=actual_return_1h,
                    price_1h_later=price_1h_later,
                )
                updated_1h += 1
                if updated_1h % 10 == 0:
                    print(f"  ✅ Updated {updated_1h} 1h predictions...")
            else:
                failed_1h += 1
                if failed_1h <= 5:  # Only show first few failures
                    print(f"  ⚠️  Could not find price for prediction {pred_id} at {target_dt}")
        except Exception as e:
            failed_1h += 1
            if failed_1h <= 5:
                print(f"  ❌ Error updating prediction {pred_id}: {e}")
    
    # Update 24h predictions
    print("\n🔄 Updating 24h predictions...")
    for pred_id, pred_timestamp, price_at_pred in predictions_24h:
        try:
            # Parse timestamp
            if isinstance(pred_timestamp, str):
                if 'Z' in pred_timestamp:
                    pred_dt = datetime.fromisoformat(pred_timestamp.replace('Z', '+00:00'))
                elif '+' in pred_timestamp or pred_timestamp.count('-') >= 3:
                    pred_dt = datetime.fromisoformat(pred_timestamp)
                else:
                    pred_dt = datetime.fromisoformat(pred_timestamp)
            else:
                pred_dt = datetime.fromisoformat(pred_timestamp)
            
            target_dt = pred_dt + timedelta(hours=24)
            
            # Get price 24 hours later (wider window for better matching)
            # Use sol_prices.db for price data (separate from predictions database)
            price_conn = sqlite3.connect(sol_prices_path)
            price_cursor = price_conn.cursor()
            
            # Use wider time window: 1 hour before to 2 hours after
            target_start = (target_dt - timedelta(hours=1)).isoformat()
            target_end = (target_dt + timedelta(hours=2)).isoformat()
            
            price_cursor.execute("""
                SELECT rate, timestamp
                FROM sol_prices
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY ABS(julianday(timestamp) - julianday(?))
                LIMIT 1
            """, (target_start, target_end, target_dt.isoformat()))
            
            price_row = price_cursor.fetchone()
            price_conn.close()
            
            if price_row and price_at_pred and price_at_pred > 0:
                price_24h_later = price_row[0]
                actual_return_24h = (price_24h_later - price_at_pred) / price_at_pred
                actual_return_24h = max(-1.0, min(1.0, actual_return_24h))
                
                prediction_manager.update_actual_returns(
                    prediction_id=pred_id,
                    actual_return_24h=actual_return_24h,
                    price_24h_later=price_24h_later,
                )
                updated_24h += 1
                if updated_24h % 10 == 0:
                    print(f"  ✅ Updated {updated_24h} 24h predictions...")
            else:
                failed_24h += 1
                if failed_24h <= 5:
                    print(f"  ⚠️  Could not find price for prediction {pred_id} at {target_dt}")
        except Exception as e:
            failed_24h += 1
            if failed_24h <= 5:
                print(f"  ❌ Error updating prediction {pred_id}: {e}")
    
    print(f"\n✅ Backfill complete!")
    print(f"   - Updated {updated_1h} 1h predictions ({failed_1h} failed)")
    print(f"   - Updated {updated_24h} 24h predictions ({failed_24h} failed)")
    
    # Show final stats
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_1h IS NOT NULL")
    with_1h = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_24h IS NOT NULL")
    with_24h = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
    total = cursor.fetchone()[0]
    conn.close()
    
    print(f"\n📊 Final stats:")
    print(f"   - Total predictions: {total}")
    print(f"   - With 1h returns: {with_1h} ({with_1h/total*100:.1f}%)")
    print(f"   - With 24h returns: {with_24h} ({with_24h/total*100:.1f}%)")

if __name__ == "__main__":
    backfill_actual_returns()
