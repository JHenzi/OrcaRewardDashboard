"""
Signal Performance Tracker
Tracks the performance of trading signals (RSI, Bandit, etc.) over time.

This module logs signals when they're generated and tracks their outcomes
to calculate reliability metrics like win rate, average return, etc.
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class SignalPerformanceTracker:
    """Track and analyze the performance of trading signals"""
    
    def __init__(self, db_path="sol_prices.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create signal_performance table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,  -- 'rsi_buy', 'rsi_sell', 'rsi_hold', 'bandit_buy', 'bandit_sell', 'bandit_hold'
                signal_timestamp TEXT NOT NULL,  -- ISO format
                price_at_signal REAL NOT NULL,
                price_1h_later REAL,
                price_4h_later REAL,
                price_24h_later REAL,
                price_7d_later REAL,
                return_1h REAL,
                return_4h REAL,
                return_24h REAL,
                return_7d REAL,
                was_profitable BOOLEAN,
                signal_metadata TEXT,  -- JSON string for additional context (RSI value, bandit reward, etc.)
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signal_performance_type 
            ON signal_performance(signal_type)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signal_performance_timestamp 
            ON signal_performance(signal_timestamp)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signal_performance_type_timestamp 
            ON signal_performance(signal_type, signal_timestamp)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Signal performance database initialized")
    
    def log_signal(self, signal_type: str, price: float, metadata: Optional[Dict] = None):
        """
        Log a new signal
        
        Args:
            signal_type: Type of signal (e.g., 'rsi_buy', 'rsi_sell', 'bandit_buy')
            price: Price at the time of signal
            metadata: Optional dictionary with additional context (RSI value, bandit reward, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        import json
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute('''
            INSERT INTO signal_performance 
            (signal_type, signal_timestamp, price_at_signal, signal_metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            signal_type,
            datetime.utcnow().isoformat(),
            price,
            metadata_json
        ))
        conn.commit()
        conn.close()
        logger.debug(f"Logged signal: {signal_type} at price ${price:.2f}")
    
    def update_performance_metrics(self):
        """
        Update performance metrics for signals that are old enough to have outcomes.
        This should be run periodically (e.g., every hour) to update prices at 1h, 4h, 24h, 7d later.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get signals that need updating
        now = datetime.utcnow()
        
        # Update 1h later prices (signals older than 1 hour)
        one_hour_ago = (now - timedelta(hours=1)).isoformat()
        cursor.execute('''
            SELECT id, signal_timestamp, price_at_signal
            FROM signal_performance
            WHERE signal_timestamp <= ? AND price_1h_later IS NULL
        ''', (one_hour_ago,))
        
        signals_1h = cursor.fetchall()
        for signal_id, signal_ts, price_at_signal in signals_1h:
            price_1h = self._get_price_at_time(signal_ts, hours_later=1)
            if price_1h:
                return_1h = ((price_1h - price_at_signal) / price_at_signal) * 100
                cursor.execute('''
                    UPDATE signal_performance
                    SET price_1h_later = ?, return_1h = ?
                    WHERE id = ?
                ''', (price_1h, return_1h, signal_id))
        
        # Update 4h later prices (signals older than 4 hours)
        four_hours_ago = (now - timedelta(hours=4)).isoformat()
        cursor.execute('''
            SELECT id, signal_timestamp, price_at_signal
            FROM signal_performance
            WHERE signal_timestamp <= ? AND price_4h_later IS NULL
        ''', (four_hours_ago,))
        
        signals_4h = cursor.fetchall()
        for signal_id, signal_ts, price_at_signal in signals_4h:
            price_4h = self._get_price_at_time(signal_ts, hours_later=4)
            if price_4h:
                return_4h = ((price_4h - price_at_signal) / price_at_signal) * 100
                cursor.execute('''
                    UPDATE signal_performance
                    SET price_4h_later = ?, return_4h = ?
                    WHERE id = ?
                ''', (price_4h, return_4h, signal_id))
        
        # Update 24h later prices (signals older than 24 hours)
        one_day_ago = (now - timedelta(hours=24)).isoformat()
        cursor.execute('''
            SELECT id, signal_timestamp, price_at_signal
            FROM signal_performance
            WHERE signal_timestamp <= ? AND price_24h_later IS NULL
        ''', (one_day_ago,))
        
        signals_24h = cursor.fetchall()
        for signal_id, signal_ts, price_at_signal in signals_24h:
            price_24h = self._get_price_at_time(signal_ts, hours_later=24)
            if price_24h:
                return_24h = ((price_24h - price_at_signal) / price_at_signal) * 100
                was_profitable = return_24h > 0
                cursor.execute('''
                    UPDATE signal_performance
                    SET price_24h_later = ?, return_24h = ?, was_profitable = ?
                    WHERE id = ?
                ''', (price_24h, return_24h, was_profitable, signal_id))
        
        # Update 7d later prices (signals older than 7 days)
        seven_days_ago = (now - timedelta(days=7)).isoformat()
        cursor.execute('''
            SELECT id, signal_timestamp, price_at_signal
            FROM signal_performance
            WHERE signal_timestamp <= ? AND price_7d_later IS NULL
        ''', (seven_days_ago,))
        
        signals_7d = cursor.fetchall()
        for signal_id, signal_ts, price_at_signal in signals_7d:
            price_7d = self._get_price_at_time(signal_ts, hours_later=168)  # 7 days = 168 hours
            if price_7d:
                return_7d = ((price_7d - price_at_signal) / price_at_signal) * 100
                cursor.execute('''
                    UPDATE signal_performance
                    SET price_7d_later = ?, return_7d = ?
                    WHERE id = ?
                ''', (price_7d, return_7d, signal_id))
        
        conn.commit()
        conn.close()
        logger.info(f"Updated performance metrics: {len(signals_1h)} 1h, {len(signals_4h)} 4h, {len(signals_24h)} 24h, {len(signals_7d)} 7d")
    
    def _get_price_at_time(self, signal_timestamp: str, hours_later: int) -> Optional[float]:
        """
        Get the price at a specific time (signal_timestamp + hours_later)
        
        Args:
            signal_timestamp: ISO format timestamp of the signal
            hours_later: Hours after the signal to get price
        
        Returns:
            Price at that time, or None if not available
        """
        try:
            signal_dt = datetime.fromisoformat(signal_timestamp)
            target_dt = signal_dt + timedelta(hours=hours_later)
            target_iso = target_dt.isoformat()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find the closest price to the target time (within 30 minutes)
            cursor.execute('''
                SELECT rate, timestamp
                FROM sol_prices
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY ABS(julianday(timestamp) - julianday(?))
                LIMIT 1
            ''', (
                (target_dt - timedelta(minutes=30)).isoformat(),
                (target_dt + timedelta(minutes=30)).isoformat(),
                target_iso
            ))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return row[0]
            return None
        except Exception as e:
            logger.warning(f"Error getting price at time {signal_timestamp} + {hours_later}h: {e}")
            return None
    
    def get_performance_stats(self, signal_type: Optional[str] = None, hours_later: int = 24) -> Dict:
        """
        Get performance statistics for signals
        
        Args:
            signal_type: Optional filter by signal type (e.g., 'rsi_buy')
            hours_later: Time horizon for performance (1, 4, 24, or 168 for 7d)
        
        Returns:
            Dictionary with performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine which return column to use
        return_column_map = {
            1: 'return_1h',
            4: 'return_4h',
            24: 'return_24h',
            168: 'return_7d'
        }
        return_column = return_column_map.get(hours_later, 'return_24h')
        
        # Build query
        if signal_type:
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_signals,
                    COUNT({return_column}) as signals_with_outcome,
                    AVG({return_column}) as avg_return,
                    SUM(CASE WHEN was_profitable = 1 THEN 1 ELSE 0 END) as profitable_count,
                    MIN({return_column}) as min_return,
                    MAX({return_column}) as max_return,
                    AVG(CASE WHEN {return_column} > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
                FROM signal_performance
                WHERE signal_type = ? AND {return_column} IS NOT NULL
            ''', (signal_type,))
        else:
            cursor.execute(f'''
                SELECT 
                    signal_type,
                    COUNT(*) as total_signals,
                    COUNT({return_column}) as signals_with_outcome,
                    AVG({return_column}) as avg_return,
                    SUM(CASE WHEN was_profitable = 1 THEN 1 ELSE 0 END) as profitable_count,
                    MIN({return_column}) as min_return,
                    MAX({return_column}) as max_return,
                    AVG(CASE WHEN {return_column} > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate
                FROM signal_performance
                WHERE {return_column} IS NOT NULL
                GROUP BY signal_type
            ''')
        
        if signal_type:
            row = cursor.fetchone()
            if row and row[1] > 0:  # signals_with_outcome > 0
                return {
                    'signal_type': signal_type,
                    'total_signals': row[0],
                    'signals_with_outcome': row[1],
                    'avg_return': round(row[2], 2) if row[2] else 0,
                    'profitable_count': row[3],
                    'min_return': round(row[4], 2) if row[4] else 0,
                    'max_return': round(row[5], 2) if row[5] else 0,
                    'win_rate': round(row[6], 1) if row[6] else 0
                }
            return {
                'signal_type': signal_type,
                'total_signals': 0,
                'signals_with_outcome': 0,
                'avg_return': 0,
                'profitable_count': 0,
                'min_return': 0,
                'max_return': 0,
                'win_rate': 0
            }
        else:
            results = []
            rows = cursor.fetchall()
            for row in rows:
                if row[2] > 0:  # signals_with_outcome > 0
                    results.append({
                        'signal_type': row[0],
                        'total_signals': row[1],
                        'signals_with_outcome': row[2],
                        'avg_return': round(row[3], 2) if row[3] else 0,
                        'profitable_count': row[4],
                        'min_return': round(row[5], 2) if row[5] else 0,
                        'max_return': round(row[6], 2) if row[6] else 0,
                        'win_rate': round(row[7], 1) if row[7] else 0
                    })
            conn.close()
            return {'by_type': results}
    
    def get_recent_signals(self, limit: int = 10, signal_type: Optional[str] = None) -> List[Dict]:
        """
        Get recent signals with their performance
        
        Args:
            limit: Number of recent signals to return
            signal_type: Optional filter by signal type
        
        Returns:
            List of signal dictionaries with performance data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if signal_type:
            cursor.execute('''
                SELECT 
                    id, signal_type, signal_timestamp, price_at_signal,
                    price_24h_later, return_24h, was_profitable, signal_metadata
                FROM signal_performance
                WHERE signal_type = ?
                ORDER BY signal_timestamp DESC
                LIMIT ?
            ''', (signal_type, limit))
        else:
            cursor.execute('''
                SELECT 
                    id, signal_type, signal_timestamp, price_at_signal,
                    price_24h_later, return_24h, was_profitable, signal_metadata
                FROM signal_performance
                ORDER BY signal_timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        import json
        signals = []
        for row in rows:
            metadata = json.loads(row[7]) if row[7] else {}
            signals.append({
                'id': row[0],
                'signal_type': row[1],
                'timestamp': row[2],
                'price_at_signal': row[3],
                'price_24h_later': row[4],
                'return_24h': round(row[5], 2) if row[5] else None,
                'was_profitable': bool(row[6]) if row[6] is not None else None,
                'metadata': metadata
            })
        
        return signals

