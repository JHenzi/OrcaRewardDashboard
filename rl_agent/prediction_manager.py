"""
Prediction Manager

Manages storage, retrieval, and accuracy tracking for multi-horizon return predictions.
"""

import sqlite3
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DB_PATH = os.getenv("DATABASE_PATH", "sol_prices.db")


class PredictionManager:
    """
    Manages RL agent predictions and accuracy tracking.
    
    Features:
    - Store predictions with timestamps
    - Track actual returns when they become available
    - Compute prediction accuracy metrics
    - Retrieve predictions for display
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize prediction manager.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure prediction tracking table exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for prediction accuracy tracking
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
    
    def store_prediction(
        self,
        decision_id: Optional[int],
        timestamp: datetime,
        predicted_return_1h: float,
        predicted_return_24h: float,
        predicted_confidence_1h: Optional[float] = None,
        predicted_confidence_24h: Optional[float] = None,
        price_at_prediction: float = 0.0,
    ) -> int:
        """
        Store a new prediction.
        
        Args:
            decision_id: ID of the decision that generated this prediction
            timestamp: When the prediction was made
            predicted_return_1h: Predicted 1h return
            predicted_return_24h: Predicted 24h return
            predicted_confidence_1h: Confidence/uncertainty for 1h prediction
            predicted_confidence_24h: Confidence/uncertainty for 24h prediction
            price_at_prediction: Price at time of prediction
            
        Returns:
            ID of the stored prediction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO rl_prediction_accuracy (
                decision_id, timestamp, predicted_return_1h, predicted_return_24h,
                predicted_confidence_1h, predicted_confidence_24h, price_at_prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            decision_id,
            timestamp.isoformat(),
            predicted_return_1h,
            predicted_return_24h,
            predicted_confidence_1h,
            predicted_confidence_24h,
            price_at_prediction,
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.debug(f"Stored prediction {prediction_id} at {timestamp}")
        return prediction_id
    
    def update_actual_returns(
        self,
        prediction_id: int,
        actual_return_1h: Optional[float] = None,
        actual_return_24h: Optional[float] = None,
        price_1h_later: Optional[float] = None,
        price_24h_later: Optional[float] = None,
    ):
        """
        Update prediction with actual returns when they become available.
        
        Args:
            prediction_id: ID of the prediction to update
            actual_return_1h: Actual 1h return (if available)
            actual_return_24h: Actual 24h return (if available)
            price_1h_later: Price 1 hour after prediction
            price_24h_later: Price 24 hours after prediction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current prediction
        cursor.execute("""
            SELECT predicted_return_1h, predicted_return_24h, price_at_prediction
            FROM rl_prediction_accuracy
            WHERE id = ?
        """, (prediction_id,))
        
        row = cursor.fetchone()
        if not row:
            logger.warning(f"Prediction {prediction_id} not found")
            conn.close()
            return
        
        pred_1h, pred_24h, price_at_pred = row
        
        # Calculate errors
        error_1h = None
        error_24h = None
        mae_1h = None
        mae_24h = None
        
        if actual_return_1h is not None:
            error_1h = abs(actual_return_1h - pred_1h)
            mae_1h = error_1h  # For individual prediction, MAE = absolute error
        
        if actual_return_24h is not None:
            error_24h = abs(actual_return_24h - pred_24h)
            mae_24h = error_24h
        
        # Update
        cursor.execute("""
            UPDATE rl_prediction_accuracy
            SET actual_return_1h = ?,
                actual_return_24h = ?,
                price_1h_later = ?,
                price_24h_later = ?,
                error_1h = ?,
                error_24h = ?,
                mae_1h = ?,
                mae_24h = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            actual_return_1h,
            actual_return_24h,
            price_1h_later,
            price_24h_later,
            error_1h,
            error_24h,
            mae_1h,
            mae_24h,
            prediction_id,
        ))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Updated prediction {prediction_id} with actual returns")
    
    def get_latest_predictions(self, limit: int = 10) -> List[Dict]:
        """
        Get latest predictions.
        
        Args:
            limit: Number of predictions to retrieve
            
        Returns:
            List of prediction dicts
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id, decision_id, timestamp, 
                predicted_return_1h, predicted_return_24h,
                predicted_confidence_1h, predicted_confidence_24h,
                actual_return_1h, actual_return_24h,
                price_at_prediction, price_1h_later, price_24h_later,
                error_1h, error_24h, mae_1h, mae_24h,
                created_at
            FROM rl_prediction_accuracy
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in rows:
            predictions.append({
                "id": row[0],
                "decision_id": row[1],
                "timestamp": row[2],
                "predicted_return_1h": row[3],
                "predicted_return_24h": row[4],
                "predicted_confidence_1h": row[5],
                "predicted_confidence_24h": row[6],
                "actual_return_1h": row[7],
                "actual_return_24h": row[8],
                "price_at_prediction": row[9],
                "price_1h_later": row[10],
                "price_24h_later": row[11],
                "error_1h": row[12],
                "error_24h": row[13],
                "mae_1h": row[14],
                "mae_24h": row[15],
                "created_at": row[16],
            })
        
        return predictions
    
    def get_current_prediction(self) -> Optional[Dict]:
        """
        Get the most recent prediction.
        
        Returns:
            Most recent prediction dict or None
        """
        predictions = self.get_latest_predictions(limit=1)
        return predictions[0] if predictions else None
    
    def get_prediction_accuracy_stats(
        self,
        hours: int = 24,
    ) -> Dict[str, float]:
        """
        Get prediction accuracy statistics.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict with accuracy metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        time_threshold = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Get predictions with actual returns
        cursor.execute("""
            SELECT 
                predicted_return_1h, actual_return_1h, error_1h,
                predicted_return_24h, actual_return_24h, error_24h
            FROM rl_prediction_accuracy
            WHERE timestamp >= ?
            AND actual_return_1h IS NOT NULL
            AND actual_return_24h IS NOT NULL
        """, (time_threshold,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                "count": 0,
                "mae_1h": 0.0,
                "mae_24h": 0.0,
                "rmse_1h": 0.0,
                "rmse_24h": 0.0,
                "mean_error_1h": 0.0,
                "mean_error_24h": 0.0,
            }
        
        errors_1h = [row[2] for row in rows if row[2] is not None]
        errors_24h = [row[5] for row in rows if row[5] is not None]
        
        pred_1h = [row[0] for row in rows]
        actual_1h = [row[1] for row in rows]
        pred_24h = [row[3] for row in rows]
        actual_24h = [row[4] for row in rows]
        
        # Calculate metrics
        mae_1h = np.mean(errors_1h) if errors_1h else 0.0
        mae_24h = np.mean(errors_24h) if errors_24h else 0.0
        
        rmse_1h = np.sqrt(np.mean([e**2 for e in errors_1h])) if errors_1h else 0.0
        rmse_24h = np.sqrt(np.mean([e**2 for e in errors_24h])) if errors_24h else 0.0
        
        mean_error_1h = np.mean([a - p for a, p in zip(actual_1h, pred_1h)]) if actual_1h else 0.0
        mean_error_24h = np.mean([a - p for a, p in zip(actual_24h, pred_24h)]) if actual_24h else 0.0
        
        return {
            "count": len(rows),
            "mae_1h": float(mae_1h),
            "mae_24h": float(mae_24h),
            "rmse_1h": float(rmse_1h),
            "rmse_24h": float(rmse_24h),
            "mean_error_1h": float(mean_error_1h),
            "mean_error_24h": float(mean_error_24h),
        }
    
    def get_predictions_for_chart(
        self,
        hours: int = 24,
    ) -> Dict[str, List]:
        """
        Get predictions formatted for chart display.
        
        Args:
            hours: Number of hours to retrieve
            
        Returns:
            Dict with 'timestamps', 'predicted_1h', 'predicted_24h', 'actual_1h', 'actual_24h'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        time_threshold = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT 
                timestamp,
                predicted_return_1h,
                predicted_return_24h,
                actual_return_1h,
                actual_return_24h,
                price_at_prediction,
                price_1h_later,
                price_24h_later
            FROM rl_prediction_accuracy
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (time_threshold,))
        
        rows = cursor.fetchall()
        conn.close()
        
        timestamps = []
        predicted_1h = []
        predicted_24h = []
        actual_1h = []
        actual_24h = []
        prices_at_pred = []
        prices_1h = []
        prices_24h = []
        
        for row in rows:
            timestamps.append(row[0])
            predicted_1h.append(row[1])
            predicted_24h.append(row[2])
            actual_1h.append(row[3] if row[3] is not None else None)
            actual_24h.append(row[4] if row[4] is not None else None)
            prices_at_pred.append(row[5])
            prices_1h.append(row[6] if row[6] is not None else None)
            prices_24h.append(row[7] if row[7] is not None else None)
        
        return {
            "timestamps": timestamps,
            "predicted_1h": predicted_1h,
            "predicted_24h": predicted_24h,
            "actual_1h": actual_1h,
            "actual_24h": actual_24h,
            "prices_at_prediction": prices_at_pred,
            "prices_1h_later": prices_1h,
            "prices_24h_later": prices_24h,
        }

