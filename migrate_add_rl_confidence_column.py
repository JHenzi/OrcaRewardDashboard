#!/usr/bin/env python3
"""
Migration: Add missing columns to rl_agent_decisions table

Adds confidence, price_features, and current_price columns that are used
by the integration layer but missing from the original schema.
"""

import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", "sol_prices.db")


def migrate_rl_agent_decisions():
    """Add missing columns to rl_agent_decisions table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print(f"Migrating rl_agent_decisions table in {DB_PATH}...")
    
    # Check if table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='rl_agent_decisions'
    """)
    if not cursor.fetchone():
        print("❌ rl_agent_decisions table does not exist. Run migrate_rl_agent_tables.py first.")
        conn.close()
        return
    
    # Get existing columns
    cursor.execute("PRAGMA table_info(rl_agent_decisions)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    print(f"Existing columns: {existing_columns}")
    
    # Add missing columns
    columns_to_add = []
    
    if 'confidence' not in existing_columns:
        columns_to_add.append(('confidence', 'REAL'))
    
    if 'price_features' not in existing_columns:
        columns_to_add.append(('price_features', 'TEXT'))
    
    if 'current_price' not in existing_columns:
        columns_to_add.append(('current_price', 'REAL'))
    
    if 'predicted_confidence_1h' not in existing_columns:
        columns_to_add.append(('predicted_confidence_1h', 'REAL'))
    
    if 'predicted_confidence_24h' not in existing_columns:
        columns_to_add.append(('predicted_confidence_24h', 'REAL'))
    
    if columns_to_add:
        for col_name, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE rl_agent_decisions ADD COLUMN {col_name} {col_type}")
                print(f"✅ Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    print(f"⚠️  Column {col_name} already exists, skipping")
                else:
                    print(f"❌ Error adding {col_name}: {e}")
    else:
        print("✅ All required columns already exist")
    
    conn.commit()
    conn.close()
    print("✅ Migration complete")


if __name__ == "__main__":
    migrate_rl_agent_decisions()

