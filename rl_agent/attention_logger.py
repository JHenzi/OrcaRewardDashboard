"""
Attention Logger

Logs attention weights from the RL agent's news branch for explainability.
Stores which headlines influenced each decision.
"""

import sqlite3
import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get database path from environment or use default
DB_PATH = os.getenv("DATABASE_PATH", "sol_prices.db")


class AttentionLogger:
    """
    Logs attention weights for explainability.
    
    Features:
    - Store attention weights for each decision
    - Link headlines to decisions
    - Retrieve top-k influential headlines
    - Aggregate attention by topic/cluster
    """
    
    def __init__(self, db_path: str = DB_PATH):
        """
        Initialize attention logger.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_table_exists()
    
    def _ensure_table_exists(self):
        """Ensure attention logs table exists (already created in migration)."""
        # Table is created in migrate_rl_agent_tables.py
        # Just verify it exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='rl_attention_logs'
        """)
        exists = cursor.fetchone() is not None
        conn.close()
        
        if not exists:
            logger.warning("rl_attention_logs table does not exist. Run migration script.")
    
    def log_attention(
        self,
        decision_id: int,
        headlines: List[Dict],
        attention_weights: np.ndarray,
        cluster_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Log attention weights for a decision.
        
        Args:
            decision_id: ID of the decision
            headlines: List of headline dicts with keys:
                - 'headline_text': str
                - 'headline_id': int (optional, reference to news_articles table)
                - 'link': str (optional)
            attention_weights: Attention weights array (num_heads, num_headlines, num_headlines)
                               or (num_headlines,) for averaged weights
            cluster_ids: Optional list of cluster IDs for each headline
            
        Returns:
            List of log entry IDs
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        log_ids = []
        
        # Handle different attention weight formats
        if len(attention_weights.shape) == 3:
            # Multi-head attention: (num_heads, num_headlines, num_headlines)
            # Average across heads and get attention to each headline
            attn_avg = attention_weights.mean(axis=0)  # (num_headlines, num_headlines)
            # Get attention received by each headline (sum over query positions)
            headline_weights = attn_avg.sum(axis=0)  # (num_headlines,)
        elif len(attention_weights.shape) == 2:
            # (num_headlines, num_headlines) - already averaged
            headline_weights = attention_weights.sum(axis=0)
        else:
            # (num_headlines,) - already processed
            headline_weights = attention_weights
        
        # Normalize weights
        if headline_weights.sum() > 0:
            headline_weights = headline_weights / headline_weights.sum()
        
        # Log each headline with its attention weight
        for i, headline in enumerate(headlines):
            if i >= len(headline_weights):
                break
            
            weight = float(headline_weights[i])
            headline_text = headline.get("headline_text", "")
            headline_id = headline.get("headline_id")
            cluster_id = cluster_ids[i] if cluster_ids and i < len(cluster_ids) else None
            
            cursor.execute("""
                INSERT INTO rl_attention_logs (
                    decision_id, headline_id, headline_text, attention_weight, cluster_id
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                decision_id,
                headline_id,
                headline_text,
                weight,
                cluster_id,
            ))
            
            log_ids.append(cursor.lastrowid)
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged attention for decision {decision_id}: {len(log_ids)} headlines")
        return log_ids
    
    def get_top_headlines_for_decision(
        self,
        decision_id: int,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Get top-k most influential headlines for a decision.
        
        Args:
            decision_id: ID of the decision
            top_k: Number of headlines to return
            
        Returns:
            List of headline dicts with attention weights, sorted by weight
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT headline_id, headline_text, attention_weight, cluster_id
            FROM rl_attention_logs
            WHERE decision_id = ?
            ORDER BY attention_weight DESC
            LIMIT ?
        """, (decision_id, top_k))
        
        rows = cursor.fetchall()
        conn.close()
        
        headlines = []
        for row in rows:
            headlines.append({
                "headline_id": row[0],
                "headline_text": row[1],
                "attention_weight": row[2],
                "cluster_id": row[3],
            })
        
        return headlines
    
    def get_recent_attention(
        self,
        limit: int = 10,
    ) -> List[Dict]:
        """
        Get recent attention logs with decision info.
        
        Args:
            limit: Number of recent decisions to retrieve
            
        Returns:
            List of dicts with decision and top headlines
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent decisions with their attention
        cursor.execute("""
            SELECT DISTINCT 
                d.id, d.timestamp, d.action,
                al.headline_text, al.attention_weight, al.cluster_id
            FROM rl_agent_decisions d
            LEFT JOIN rl_attention_logs al ON d.id = al.decision_id
            ORDER BY d.timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Group by decision
        decisions = {}
        for row in rows:
            decision_id = row[0]
            if decision_id not in decisions:
                decisions[decision_id] = {
                    "decision_id": decision_id,
                    "timestamp": row[1],
                    "action": row[2],
                    "headlines": [],
                }
            
            if row[3]:  # headline_text
                decisions[decision_id]["headlines"].append({
                    "headline_text": row[3],
                    "attention_weight": row[4],
                    "cluster_id": row[5],
                })
        
        # Sort headlines by attention weight for each decision
        for decision in decisions.values():
            decision["headlines"].sort(key=lambda x: x["attention_weight"], reverse=True)
            decision["headlines"] = decision["headlines"][:5]  # Top 5
        
        return list(decisions.values())
    
    def get_attention_by_cluster(
        self,
        hours: int = 24,
    ) -> Dict[int, Dict]:
        """
        Get attention aggregated by news cluster.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dict mapping cluster_id to aggregate stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        time_threshold = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute("""
            SELECT 
                al.cluster_id,
                COUNT(*) as count,
                AVG(al.attention_weight) as avg_weight,
                SUM(al.attention_weight) as total_weight
            FROM rl_attention_logs al
            JOIN rl_agent_decisions d ON al.decision_id = d.id
            WHERE d.timestamp >= ?
            AND al.cluster_id IS NOT NULL
            GROUP BY al.cluster_id
            ORDER BY total_weight DESC
        """, (time_threshold,))
        
        rows = cursor.fetchall()
        conn.close()
        
        cluster_stats = {}
        for row in rows:
            cluster_id = row[0]
            cluster_stats[cluster_id] = {
                "cluster_id": cluster_id,
                "count": row[1],
                "avg_attention": row[2],
                "total_attention": row[3],
            }
        
        return cluster_stats

