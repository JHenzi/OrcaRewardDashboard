"""
Consensus Tracker
Tracks when all 7 technical indicators agree (all BUY or all SELL) and measures returns.

This module:
1. Detects when all indicators are in agreement
2. Logs consensus signals
3. Tracks 1hr and 24hr returns after consensus signals
4. Provides statistics on consensus signal performance
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ConsensusTracker:
    """Track consensus signals when all 7 indicators agree"""
    
    def __init__(self, db_path=None):
        # Resolve database path - consensus signals stored in sol_prices.db
        if db_path is None:
            env_db_path = os.getenv("DATABASE_PATH")
            if env_db_path:
                if env_db_path == "rewards.db" or env_db_path.endswith("rewards.db"):
                    db_path = "sol_prices.db"
                else:
                    db_path = env_db_path
            else:
                db_path = "sol_prices.db"
        
        # Convert to absolute path to avoid issues with working directory
        if not os.path.isabs(db_path):
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir
            db_file = project_root / db_path
            if db_file.exists():
                self.db_path = str(db_file)
            else:
                self.db_path = db_path
        else:
            self.db_path = db_path
        
        self.init_database()
    
    def init_database(self):
        """Create consensus_signals table if it doesn't exist"""
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS consensus_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_type TEXT NOT NULL,  -- 'BUY' or 'SELL'
                    signal_timestamp TEXT NOT NULL,  -- ISO format
                    price_at_signal REAL NOT NULL,
                    price_1h_later REAL,
                    price_24h_later REAL,
                    return_1h REAL,
                    return_24h REAL,
                    was_profitable_1h BOOLEAN,
                    was_profitable_24h BOOLEAN,
                    indicator_values TEXT,  -- JSON string with all indicator values
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_consensus_type 
                ON consensus_signals(signal_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_consensus_timestamp 
                ON consensus_signals(signal_timestamp)
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Consensus signals database initialized: {self.db_path}")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to initialize consensus signals database: {e}")
            logger.error(f"Database path: {self.db_path}")
            logger.error(f"Absolute path: {os.path.abspath(self.db_path)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            raise
    
    def check_consensus(
        self,
        rsi: Optional[float],
        price_vs_sma_1h: Optional[float],
        price_vs_sma_4h: Optional[float],
        price_vs_sma_24h: Optional[float],
        macd_line: Optional[float],
        macd_signal: Optional[float],
        bb_position: Optional[float],
        momentum_10: Optional[float],
        current_price: float,
        indicator_values: Optional[Dict] = None
    ) -> Tuple[Optional[str], Dict]:
        """
        Check if all 7 indicators are in agreement.
        
        Returns:
            Tuple of (signal_type, indicator_signals_dict)
            signal_type: 'BUY', 'SELL', or None (no consensus)
            indicator_signals_dict: Dict showing each indicator's signal
        """
        signals = {}
        buy_count = 0
        sell_count = 0
        
        # 1. RSI
        if rsi is not None:
            if rsi < 30:
                signals['rsi'] = 'BUY'
                buy_count += 1
            elif rsi > 70:
                signals['rsi'] = 'SELL'
                sell_count += 1
            else:
                signals['rsi'] = 'NEUTRAL'
        else:
            signals['rsi'] = 'N/A'
        
        # 2. Price vs SMA 1h
        if price_vs_sma_1h is not None:
            if price_vs_sma_1h > 2:
                signals['sma_1h'] = 'BUY'
                buy_count += 1
            elif price_vs_sma_1h < -2:
                signals['sma_1h'] = 'SELL'
                sell_count += 1
            else:
                signals['sma_1h'] = 'NEUTRAL'
        else:
            signals['sma_1h'] = 'N/A'
        
        # 3. Price vs SMA 4h
        if price_vs_sma_4h is not None:
            if price_vs_sma_4h > 2:
                signals['sma_4h'] = 'BUY'
                buy_count += 1
            elif price_vs_sma_4h < -2:
                signals['sma_4h'] = 'SELL'
                sell_count += 1
            else:
                signals['sma_4h'] = 'NEUTRAL'
        else:
            signals['sma_4h'] = 'N/A'
        
        # 4. Price vs SMA 24h
        if price_vs_sma_24h is not None:
            if price_vs_sma_24h > 2:
                signals['sma_24h'] = 'BUY'
                buy_count += 1
            elif price_vs_sma_24h < -2:
                signals['sma_24h'] = 'SELL'
                sell_count += 1
            else:
                signals['sma_24h'] = 'NEUTRAL'
        else:
            signals['sma_24h'] = 'N/A'
        
        # 5. MACD
        if macd_line is not None and macd_signal is not None:
            if macd_line > macd_signal:
                signals['macd'] = 'BUY'
                buy_count += 1
            elif macd_line < macd_signal:
                signals['macd'] = 'SELL'
                sell_count += 1
            else:
                signals['macd'] = 'NEUTRAL'
        else:
            signals['macd'] = 'N/A'
        
        # 6. Bollinger Bands
        if bb_position is not None:
            if bb_position < 20:  # Oversold
                signals['bollinger'] = 'BUY'
                buy_count += 1
            elif bb_position > 80:  # Overbought
                signals['bollinger'] = 'SELL'
                sell_count += 1
            else:
                signals['bollinger'] = 'NEUTRAL'
        else:
            signals['bollinger'] = 'N/A'
        
        # 7. Momentum
        if momentum_10 is not None:
            if momentum_10 > 2:
                signals['momentum'] = 'BUY'
                buy_count += 1
            elif momentum_10 < -2:
                signals['momentum'] = 'SELL'
                sell_count += 1
            else:
                signals['momentum'] = 'NEUTRAL'
        else:
            signals['momentum'] = 'N/A'
        
        # Check for consensus (all 7 must agree, excluding N/A)
        valid_signals = [s for s in signals.values() if s != 'N/A']
        if len(valid_signals) == 7:  # All 7 indicators must be valid
            if buy_count == 7:
                return 'BUY', signals
            elif sell_count == 7:
                return 'SELL', signals
        
        return None, signals
    
    def log_consensus_signal(
        self,
        signal_type: str,
        price: float,
        indicator_values: Optional[Dict] = None,
        indicator_signals: Optional[Dict] = None
    ):
        """
        Log a consensus signal when all indicators agree.
        
        Args:
            signal_type: 'BUY' or 'SELL'
            price: Price at the time of signal
            indicator_values: Dict with raw indicator values
            indicator_signals: Dict with each indicator's signal (BUY/SELL/NEUTRAL)
        """
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Combine indicator values and signals
            metadata = {
                'indicator_values': indicator_values or {},
                'indicator_signals': indicator_signals or {}
            }
            metadata_json = json.dumps(metadata)
            
            cursor.execute('''
                INSERT INTO consensus_signals 
                (signal_type, signal_timestamp, price_at_signal, indicator_values)
                VALUES (?, ?, ?, ?)
            ''', (
                signal_type,
                datetime.utcnow().isoformat(),
                price,
                metadata_json
            ))
            conn.commit()
            conn.close()
            logger.info(f"Logged consensus {signal_type} signal at price ${price:.2f}")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to log consensus signal to database: {e}")
            logger.error(f"Database path: {self.db_path}")
            logger.error(f"Absolute path: {os.path.abspath(self.db_path)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            raise
    
    def update_returns(self):
        """
        Update 1hr and 24hr returns for consensus signals that are old enough.
        This should be run periodically (e.g., every hour).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get signals that need 1hr updates (older than 1hr, but price_1h_later is NULL)
            cursor.execute('''
                SELECT id, signal_timestamp, price_at_signal, signal_type
                FROM consensus_signals
                WHERE price_1h_later IS NULL
                AND datetime(signal_timestamp) <= datetime('now', '-1 hour')
            ''')
            signals_1h = cursor.fetchall()
            
            for signal_id, signal_time, price_at_signal, signal_type in signals_1h:
                # Get price 1 hour after signal
                signal_dt = datetime.fromisoformat(signal_time)
                target_time = (signal_dt + timedelta(hours=1)).isoformat()
                
                cursor.execute('''
                    SELECT rate FROM sol_prices
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                ''', (target_time,))
                result = cursor.fetchone()
                
                if result:
                    price_1h_later = result[0]
                    return_1h = ((price_1h_later - price_at_signal) / price_at_signal) * 100
                    was_profitable_1h = (return_1h > 0) if signal_type == 'BUY' else (return_1h < 0)
                    
                    cursor.execute('''
                        UPDATE consensus_signals
                        SET price_1h_later = ?,
                            return_1h = ?,
                            was_profitable_1h = ?
                        WHERE id = ?
                    ''', (price_1h_later, return_1h, was_profitable_1h, signal_id))
            
            # Get signals that need 24hr updates (older than 24hr, but price_24h_later is NULL)
            cursor.execute('''
                SELECT id, signal_timestamp, price_at_signal, signal_type
                FROM consensus_signals
                WHERE price_24h_later IS NULL
                AND datetime(signal_timestamp) <= datetime('now', '-24 hours')
            ''')
            signals_24h = cursor.fetchall()
            
            for signal_id, signal_time, price_at_signal, signal_type in signals_24h:
                # Get price 24 hours after signal
                signal_dt = datetime.fromisoformat(signal_time)
                target_time = (signal_dt + timedelta(hours=24)).isoformat()
                
                cursor.execute('''
                    SELECT rate FROM sol_prices
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                    LIMIT 1
                ''', (target_time,))
                result = cursor.fetchone()
                
                if result:
                    price_24h_later = result[0]
                    return_24h = ((price_24h_later - price_at_signal) / price_at_signal) * 100
                    was_profitable_24h = (return_24h > 0) if signal_type == 'BUY' else (return_24h < 0)
                    
                    cursor.execute('''
                        UPDATE consensus_signals
                        SET price_24h_later = ?,
                            return_24h = ?,
                            was_profitable_24h = ?
                        WHERE id = ?
                    ''', (price_24h_later, return_24h, was_profitable_24h, signal_id))
            
            conn.commit()
            conn.close()
            
            if signals_1h or signals_24h:
                logger.info(f"Updated returns for {len(signals_1h)} 1hr signals and {len(signals_24h)} 24hr signals")
        except Exception as e:
            logger.error(f"Error updating consensus returns: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def get_consensus_stats(self, signal_type: Optional[str] = None) -> Dict:
        """
        Get statistics on consensus signal performance.
        
        Args:
            signal_type: 'BUY', 'SELL', or None for all
            
        Returns:
            Dict with statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if signal_type:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        AVG(return_1h) as avg_return_1h,
                        AVG(return_24h) as avg_return_24h,
                        SUM(CASE WHEN was_profitable_1h = 1 THEN 1 ELSE 0 END) as profitable_1h,
                        SUM(CASE WHEN was_profitable_24h = 1 THEN 1 ELSE 0 END) as profitable_24h,
                        MIN(return_1h) as min_return_1h,
                        MAX(return_1h) as max_return_1h,
                        MIN(return_24h) as min_return_24h,
                        MAX(return_24h) as max_return_24h
                    FROM consensus_signals
                    WHERE signal_type = ?
                    AND return_1h IS NOT NULL
                    AND return_24h IS NOT NULL
                ''', (signal_type,))
            else:
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        AVG(return_1h) as avg_return_1h,
                        AVG(return_24h) as avg_return_24h,
                        SUM(CASE WHEN was_profitable_1h = 1 THEN 1 ELSE 0 END) as profitable_1h,
                        SUM(CASE WHEN was_profitable_24h = 1 THEN 1 ELSE 0 END) as profitable_24h,
                        MIN(return_1h) as min_return_1h,
                        MAX(return_1h) as max_return_1h,
                        MIN(return_24h) as min_return_24h,
                        MAX(return_24h) as max_return_24h
                    FROM consensus_signals
                    WHERE return_1h IS NOT NULL
                    AND return_24h IS NOT NULL
                ''')
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] > 0:
                total, avg_1h, avg_24h, prof_1h, prof_24h, min_1h, max_1h, min_24h, max_24h = row
                return {
                    'total': total,
                    'avg_return_1h': round(avg_1h, 2) if avg_1h else None,
                    'avg_return_24h': round(avg_24h, 2) if avg_24h else None,
                    'win_rate_1h': round((prof_1h / total) * 100, 1) if total > 0 else 0,
                    'win_rate_24h': round((prof_24h / total) * 100, 1) if total > 0 else 0,
                    'min_return_1h': round(min_1h, 2) if min_1h else None,
                    'max_return_1h': round(max_1h, 2) if max_1h else None,
                    'min_return_24h': round(min_24h, 2) if min_24h else None,
                    'max_return_24h': round(max_24h, 2) if max_24h else None,
                }
            else:
                return {
                    'total': 0,
                    'avg_return_1h': None,
                    'avg_return_24h': None,
                    'win_rate_1h': 0,
                    'win_rate_24h': 0,
                    'min_return_1h': None,
                    'max_return_1h': None,
                    'min_return_24h': None,
                    'max_return_24h': None,
                }
        except Exception as e:
            logger.error(f"Error getting consensus stats: {e}")
            return {
                'total': 0,
                'error': str(e)
            }
    
    def get_recent_consensus_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent consensus signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT 
                    id, signal_type, signal_timestamp, price_at_signal,
                    price_1h_later, price_24h_later, return_1h, return_24h,
                    was_profitable_1h, was_profitable_24h, indicator_values
                FROM consensus_signals
                ORDER BY signal_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            signals = []
            for row in rows:
                signals.append({
                    'id': row[0],
                    'signal_type': row[1],
                    'timestamp': row[2],
                    'price_at_signal': row[3],
                    'price_1h_later': row[4],
                    'price_24h_later': row[5],
                    'return_1h': row[6],
                    'return_24h': row[7],
                    'was_profitable_1h': row[8],
                    'was_profitable_24h': row[9],
                    'indicator_values': json.loads(row[10]) if row[10] else {}
                })
            
            return signals
        except Exception as e:
            logger.error(f"Error getting recent consensus signals: {e}")
            return []
