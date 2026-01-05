#!/usr/bin/env python3
"""
Check what prediction data exists in the database.

This diagnostic script shows:
- Total predictions
- How many have actual returns
- Sample of recent predictions
- Whether error columns are populated
"""

import sqlite3
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

db_path = os.getenv("DATABASE_PATH", "sol_prices.db")

print(f"🔍 Checking database: {db_path}\n")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if table exists
cursor.execute("""
    SELECT name FROM sqlite_master 
    WHERE type='table' AND name='rl_prediction_accuracy'
""")
table_exists = cursor.fetchone()

if not table_exists:
    print("❌ Table 'rl_prediction_accuracy' does not exist!")
    print("   Run: python scripts/migrate_rl_agent_tables.py")
    conn.close()
    exit(1)

# Get table schema
cursor.execute("PRAGMA table_info(rl_prediction_accuracy)")
columns = cursor.fetchall()
print("📋 Table columns:")
for col in columns:
    print(f"   - {col[1]} ({col[2]})")
print()

# Count total predictions
cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy")
total = cursor.fetchone()[0]
print(f"📊 Total predictions: {total}")

if total == 0:
    print("⚠️  No predictions found in database!")
    print("   Make sure the RL agent is making decisions and storing predictions.")
    conn.close()
    exit(0)

# Count predictions with actual returns
cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_1h IS NOT NULL")
with_1h = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE actual_return_24h IS NOT NULL")
with_24h = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE error_1h IS NOT NULL")
with_error_1h = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM rl_prediction_accuracy WHERE error_24h IS NOT NULL")
with_error_24h = cursor.fetchone()[0]

print(f"   - With 1h actual returns: {with_1h} ({with_1h/total*100:.1f}%)")
print(f"   - With 24h actual returns: {with_24h} ({with_24h/total*100:.1f}%)")
print(f"   - With 1h error calculated: {with_error_1h} ({with_error_1h/total*100:.1f}%)")
print(f"   - With 24h error calculated: {with_error_24h} ({with_error_24h/total*100:.1f}%)")
print()

# Check predictions needing updates
now = datetime.now()
one_hour_ago = (now - timedelta(hours=1)).isoformat()
twenty_four_hours_ago = (now - timedelta(hours=24)).isoformat()

cursor.execute("""
    SELECT COUNT(*) 
    FROM rl_prediction_accuracy
    WHERE datetime(timestamp) <= datetime(?)
    AND actual_return_1h IS NULL
    AND price_at_prediction IS NOT NULL
    AND price_at_prediction > 0
""", (one_hour_ago,))
needing_1h = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) 
    FROM rl_prediction_accuracy
    WHERE datetime(timestamp) <= datetime(?)
    AND actual_return_24h IS NULL
    AND price_at_prediction IS NOT NULL
    AND price_at_prediction > 0
""", (twenty_four_hours_ago,))
needing_24h = cursor.fetchone()[0]

print(f"⏰ Predictions needing updates:")
print(f"   - Missing 1h returns (old enough): {needing_1h}")
print(f"   - Missing 24h returns (old enough): {needing_24h}")
print()

# Get sample of recent predictions
cursor.execute("""
    SELECT 
        id, timestamp, 
        predicted_return_1h, predicted_return_24h,
        actual_return_1h, actual_return_24h,
        error_1h, error_24h,
        price_at_prediction, price_1h_later, price_24h_later
    FROM rl_prediction_accuracy
    ORDER BY timestamp DESC
    LIMIT 5
""")
recent = cursor.fetchall()

print("📝 Sample of 5 most recent predictions:")
for row in recent:
    pred_id, ts, pred_1h, pred_24h, actual_1h, actual_24h, err_1h, err_24h, price_at, price_1h, price_24h = row
    print(f"\n   Prediction ID: {pred_id}")
    print(f"   Timestamp: {ts}")
    print(f"   Price at prediction: {price_at}")
    actual_1h_str = f"{actual_1h*100:.2f}%" if actual_1h is not None else "None"
    err_1h_str = f"{err_1h*100:.2f}%" if err_1h is not None else "None"
    actual_24h_str = f"{actual_24h*100:.2f}%" if actual_24h is not None else "None"
    err_24h_str = f"{err_24h*100:.2f}%" if err_24h is not None else "None"
    
    print(f"   Predicted 1h: {pred_1h*100:.2f}% | Actual: {actual_1h_str} | Error: {err_1h_str}")
    print(f"   Predicted 24h: {pred_24h*100:.2f}% | Actual: {actual_24h_str} | Error: {err_24h_str}")
    print(f"   Price 1h later: {price_1h if price_1h else 'None'}")
    print(f"   Price 24h later: {price_24h if price_24h else 'None'}")

# Check if we have predictions with actual returns in the last 24 hours
time_threshold = (datetime.now() - timedelta(hours=24)).isoformat()
cursor.execute("""
    SELECT COUNT(*) 
    FROM rl_prediction_accuracy
    WHERE timestamp >= ?
    AND actual_return_1h IS NOT NULL
""", (time_threshold,))
recent_1h = cursor.fetchone()[0]

cursor.execute("""
    SELECT COUNT(*) 
    FROM rl_prediction_accuracy
    WHERE timestamp >= ?
    AND actual_return_24h IS NOT NULL
""", (time_threshold,))
recent_24h = cursor.fetchone()[0]

print(f"\n📈 Predictions with actual returns in last 24 hours:")
print(f"   - 1h returns: {recent_1h}")
print(f"   - 24h returns: {recent_24h}")

if recent_1h == 0 and with_1h > 0:
    print("\n⚠️  WARNING: You have predictions with 1h returns, but none in the last 24 hours!")
    print("   The accuracy stats query looks at last 24 hours only.")
    print("   Consider increasing the 'hours' parameter in the query.")

conn.close()

print("\n✅ Diagnostic complete!")
