#!/usr/bin/env python3
"""
Database Migration Script: Add USD Value Tracking to Redemptions

This script:
1. Adds new columns to collect_fees table for USD value tracking
2. Backfills historical redemption values (approximation using current price)
3. Updates the schema for future redemptions

Run this script once to migrate existing data.
"""

import sqlite3
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# SOL and USDC mint addresses
SOL_MINT = 'So11111111111111111111111111111111111111112'
USDC_MINT = 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'

def get_sol_price_data():
    """Fetch current SOL price from LiveCoinWatch"""
    api_key = os.getenv('LIVECOINWATCH_API_KEY')
    if not api_key:
        logger.warning("LIVECOINWATCH_API_KEY not found, using default price")
        return {'rate': 150.0}  # Default fallback
    
    url = 'https://api.livecoinwatch.com/coins/single'
    headers = {
        'content-type': 'application/json',
        'x-api-key': api_key
    }
    payload = {
        "currency": "USD",
        "code": "SOL",
        "meta": True
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching SOL price: {e}")
        return {'rate': 150.0}  # Default fallback

def add_redemption_value_columns(db_path):
    """Add new columns to collect_fees table"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(collect_fees)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'usd_value_at_redemption' not in columns:
            logger.info("Adding usd_value_at_redemption column...")
            cursor.execute('''
                ALTER TABLE collect_fees 
                ADD COLUMN usd_value_at_redemption REAL
            ''')
        
        if 'sol_price_at_redemption' not in columns:
            logger.info("Adding sol_price_at_redemption column...")
            cursor.execute('''
                ALTER TABLE collect_fees 
                ADD COLUMN sol_price_at_redemption REAL
            ''')
        
        if 'usdc_price_at_redemption' not in columns:
            logger.info("Adding usdc_price_at_redemption column...")
            cursor.execute('''
                ALTER TABLE collect_fees 
                ADD COLUMN usdc_price_at_redemption REAL DEFAULT 1.0
            ''')
        
        if 'redemption_date' not in columns:
            logger.info("Adding redemption_date column...")
            cursor.execute('''
                ALTER TABLE collect_fees 
                ADD COLUMN redemption_date TEXT
            ''')
        
        conn.commit()
        logger.info("✅ Successfully added new columns to collect_fees table")
        return True
    except sqlite3.Error as e:
        logger.error(f"❌ Error adding columns: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def backfill_redemption_values(db_path, use_current_price=True):
    """
    Backfill USD values for historical redemptions
    
    Args:
        db_path: Path to rewards.db
        use_current_price: If True, use current SOL price (approximation)
                          If False, would need historical price API (not implemented)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current SOL price
    sol_price_data = get_sol_price_data()
    current_sol_price = sol_price_data.get('rate', 150.0)
    logger.info(f"Using SOL price: ${current_sol_price:.2f} for backfill")
    
    # Get all redemptions without USD values
    cursor.execute('''
        SELECT id, timestamp, token_mint, token_amount
        FROM collect_fees
        WHERE usd_value_at_redemption IS NULL
        ORDER BY timestamp
    ''')
    
    redemptions = cursor.fetchall()
    logger.info(f"Found {len(redemptions)} redemptions to backfill")
    
    if not redemptions:
        logger.info("No redemptions need backfilling")
        conn.close()
        return
    
    updated_count = 0
    for redemption_id, timestamp, mint, amount in redemptions:
        usd_value = 0.0
        sol_price_at_redemption = 0.0
        usdc_price_at_redemption = 1.0
        
        if mint == SOL_MINT:
            # SOL redemption
            sol_price_at_redemption = current_sol_price if use_current_price else current_sol_price  # TODO: Fetch historical price
            usd_value = amount * sol_price_at_redemption
        elif mint == USDC_MINT:
            # USDC redemption (1:1 with USD)
            usd_value = amount
            sol_price_at_redemption = current_sol_price  # Store for reference
            usdc_price_at_redemption = 1.0
        
        redemption_date = datetime.utcfromtimestamp(timestamp).isoformat()
        
        try:
            cursor.execute('''
                UPDATE collect_fees
                SET usd_value_at_redemption = ?,
                    sol_price_at_redemption = ?,
                    usdc_price_at_redemption = ?,
                    redemption_date = ?
                WHERE id = ?
            ''', (
                usd_value,
                sol_price_at_redemption,
                usdc_price_at_redemption,
                redemption_date,
                redemption_id
            ))
            updated_count += 1
            
            if updated_count % 100 == 0:
                logger.info(f"Updated {updated_count} redemptions...")
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating redemption {redemption_id}: {e}")
    
    conn.commit()
    conn.close()
    logger.info(f"✅ Successfully backfilled {updated_count} redemptions")
    
    if use_current_price:
        logger.warning("⚠️  Note: Used current SOL price for all historical redemptions.")
        logger.warning("    For accurate historical values, implement historical price fetching.")

def main():
    """Main migration function"""
    db_path = os.getenv('DATABASE_PATH', 'rewards.db')
    
    if not os.path.exists(db_path):
        logger.error(f"❌ Database file not found: {db_path}")
        sys.exit(1)
    
    logger.info(f"Starting migration for database: {db_path}")
    
    # Step 1: Add columns
    if not add_redemption_value_columns(db_path):
        logger.error("Failed to add columns. Exiting.")
        sys.exit(1)
    
    # Step 2: Backfill values
    logger.info("Starting backfill of historical redemption values...")
    backfill_redemption_values(db_path, use_current_price=True)
    
    logger.info("✅ Migration completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Update app.py insert_collect_fee() function to capture USD values")
    logger.info("2. Test with a new redemption to verify it works")
    logger.info("3. Consider implementing historical price fetching for accurate backfill")

if __name__ == "__main__":
    main()

