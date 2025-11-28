"""
Training Data Preparation for RL Agent

Extracts historical data from databases and prepares it for RL training.
Creates state-action-reward sequences from historical price and news data.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Database paths
PRICE_DB = "sol_prices.db"
NEWS_DB = "news_sentiment.db"


class TrainingDataPrep:
    """
    Prepares historical data for RL agent training.
    
    Creates training episodes from historical price and news data.
    Each episode contains:
    - State: price features, news embeddings, position state
    - Action: simulated action (will be replaced by model during training)
    - Reward: calculated from future price movements
    """
    
    def __init__(
        self,
        price_db_path: str = PRICE_DB,
        news_db_path: str = NEWS_DB,
        price_window_size: int = 60,  # Last 60 price points
        max_news_headlines: int = 20,
        embedding_dim: int = 384,
    ):
        self.price_db_path = price_db_path
        self.news_db_path = news_db_path
        self.price_window_size = price_window_size
        self.max_news_headlines = max_news_headlines
        self.embedding_dim = embedding_dim
    
    def get_price_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_points: int = 100,
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical price data.
        
        Args:
            start_time: Start timestamp (None = from beginning)
            end_time: End timestamp (None = to now)
            min_points: Minimum number of data points required
            
        Returns:
            List of (timestamp, price) tuples
        """
        conn = sqlite3.connect(self.price_db_path)
        cursor = conn.cursor()
        
        query = "SELECT timestamp, rate FROM sol_prices WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < min_points:
            logger.warning(f"Only {len(rows)} price points available (minimum {min_points} required)")
            return []
        
        # Parse timestamps
        price_data = []
        for row in rows:
            try:
                if isinstance(row[0], str):
                    ts = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                else:
                    ts = row[0]
                price = float(row[1])
                price_data.append((ts, price))
            except Exception as e:
                logger.warning(f"Error parsing price row: {e}")
                continue
        
        logger.info(f"Retrieved {len(price_data)} price points")
        return price_data
    
    def get_news_at_time(
        self,
        timestamp: datetime,
        hours_back: int = 24,
    ) -> List[Dict]:
        """
        Get news articles available at a given timestamp.
        
        Args:
            timestamp: Current timestamp
            hours_back: How many hours of news to include
            
        Returns:
            List of news dicts with embeddings and sentiment
        """
        if not Path(self.news_db_path).exists():
            logger.warning(f"News database {self.news_db_path} not found")
            return []
        
        conn = sqlite3.connect(self.news_db_path)
        cursor = conn.cursor()
        
        cutoff_time = timestamp - timedelta(hours=hours_back)
        
        query = """
            SELECT id, title, embedding, sentiment_score
            FROM news_articles
            WHERE published_date >= ? AND published_date <= ?
            ORDER BY published_date DESC
            LIMIT ?
        """
        
        cursor.execute(query, (cutoff_time.isoformat(), timestamp.isoformat(), self.max_news_headlines))
        rows = cursor.fetchall()
        conn.close()
        
        news_items = []
        for row in rows:
            article_id, title, embedding_blob, sentiment_score = row
            # cluster_id is not stored in news_articles table
            # Use -1 to indicate no cluster (state encoder handles this)
            cluster_id = -1
            
            # Decode embedding (stored as bytes/pickle)
            try:
                if embedding_blob:
                    if isinstance(embedding_blob, bytes):
                        embedding = pickle.loads(embedding_blob)
                    else:
                        embedding = embedding_blob
                else:
                    continue  # Skip if no embedding
                
                # Ensure it's the right shape
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.astype(np.float32)
                else:
                    embedding = np.array(embedding, dtype=np.float32)
                
                news_items.append({
                    "embedding": embedding,
                    "sentiment_score": float(sentiment_score) if sentiment_score else 0.0,
                    "headline": title or "",
                    "cluster_id": int(cluster_id) if cluster_id else -1,
                })
            except Exception as e:
                logger.warning(f"Error processing news embedding: {e}")
                continue
        
        return news_items
    
    def calculate_technical_indicators(
        self,
        prices: List[float],
    ) -> Dict[str, float]:
        """
        Calculate technical indicators from price history.
        
        Args:
            prices: List of prices
            
        Returns:
            Dict of technical indicators
        """
        if len(prices) < 2:
            return {}
        
        features = {}
        
        # Current price
        features["current_price"] = prices[-1]
        
        # Simple moving averages
        if len(prices) >= 12:
            features["sma_1h"] = np.mean(prices[-12:])
        if len(prices) >= 48:
            features["sma_4h"] = np.mean(prices[-48:])
        if len(prices) >= 288:
            features["sma_24h"] = np.mean(prices[-288:])
        
        # RSI (simplified - would use proper RSI calculation)
        if len(prices) >= 15:
            # Simple momentum-based RSI approximation
            gains = [max(0, prices[i] - prices[i-1]) for i in range(1, len(prices))]
            losses = [max(0, prices[i-1] - prices[i]) for i in range(1, len(prices))]
            if len(gains) >= 14:
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    features["rsi"] = 100 - (100 / (1 + rs))
                else:
                    features["rsi"] = 100.0
            else:
                features["rsi"] = 50.0
        else:
            features["rsi"] = 50.0
        
        # Standard deviation (volatility)
        if len(prices) >= 20:
            features["std_dev"] = np.std(prices[-20:])
        else:
            features["std_dev"] = np.std(prices) if len(prices) > 1 else 0.0
        
        # Momentum (15-minute)
        if len(prices) >= 3:
            features["momentum_15m"] = ((prices[-1] - prices[-3]) / prices[-3]) * 100
        else:
            features["momentum_15m"] = 0.0
        
        # Percent change
        if len(prices) >= 2:
            features["percent_change"] = ((prices[-1] - prices[0]) / prices[0]) * 100
        else:
            features["percent_change"] = 0.0
        
        return features
    
    def calculate_reward(
        self,
        action: int,  # 0=SELL, 1=HOLD, 2=BUY
        current_price: float,
        future_price_1h: Optional[float],
        future_price_24h: Optional[float],
        position_size: float = 0.0,
        entry_price: Optional[float] = None,
        transaction_cost: float = 0.001,
    ) -> Tuple[float, float]:
        """
        Calculate reward for an action based on future prices.
        
        Args:
            action: Action taken (0=SELL, 1=HOLD, 2=BUY)
            current_price: Price when action was taken
            future_price_1h: Price 1 hour later
            future_price_24h: Price 24 hours later
            position_size: Current position size
            entry_price: Entry price for position
            transaction_cost: Transaction cost rate
            
        Returns:
            Tuple of (reward_1h, reward_24h)
        """
        # Simplified reward calculation
        # In real training, this would be more sophisticated
        
        reward_1h = 0.0
        reward_24h = 0.0
        
        if future_price_1h:
            return_1h = (future_price_1h - current_price) / current_price
            if action == 2:  # BUY
                reward_1h = return_1h - transaction_cost
            elif action == 0:  # SELL
                reward_1h = -return_1h - transaction_cost  # Profit if price goes down
            else:  # HOLD
                reward_1h = 0.0
        
        if future_price_24h:
            return_24h = (future_price_24h - current_price) / current_price
            if action == 2:  # BUY
                reward_24h = return_24h - transaction_cost
            elif action == 0:  # SELL
                reward_24h = -return_24h - transaction_cost
            else:  # HOLD
                reward_24h = 0.0
        
        return reward_1h, reward_24h
    
    def create_training_episodes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        episode_length: int = 100,  # Number of steps per episode
        step_interval_minutes: int = 5,  # Minutes between steps
    ) -> List[Dict]:
        """
        Create training episodes from historical data.
        
        Args:
            start_time: Start time for data extraction
            end_time: End time for data extraction
            episode_length: Number of steps per episode
            step_interval_minutes: Minutes between training steps
            
        Returns:
            List of episode dicts, each containing:
            - states: List of state dicts
            - prices: List of prices at each step
            - timestamps: List of timestamps
            - future_prices_1h: List of prices 1h later
            - future_prices_24h: List of prices 24h later
            - news_data: List of news data at each step
        """
        # Get price history
        price_data = self.get_price_history(start_time, end_time)
        
        if len(price_data) < episode_length + 1440:  # Need extra for 24h future prices
            logger.error(f"Insufficient price data: {len(price_data)} points")
            return []
        
        episodes = []
        
        # Create sliding window episodes
        # Each episode starts at a different point in time
        num_episodes = max(1, (len(price_data) - episode_length - 1440) // (episode_length // 2))
        
        logger.info(f"Creating {num_episodes} training episodes from {len(price_data)} price points")
        
        for ep_idx in range(num_episodes):
            start_idx = ep_idx * (episode_length // 2)  # Overlap episodes by 50%
            
            if start_idx + episode_length + 1440 >= len(price_data):
                break  # Not enough data for future prices
            
            episode = {
                "states": [],
                "prices": [],
                "timestamps": [],
                "future_prices_1h": [],
                "future_prices_24h": [],
                "news_data": [],
                "price_features": [],
            }
            
            for step in range(episode_length):
                idx = start_idx + step
                timestamp, price = price_data[idx]
                
                # Get price window (last N prices)
                price_window_start = max(0, idx - self.price_window_size)
                price_window = [p for _, p in price_data[price_window_start:idx+1]]
                
                # Calculate technical indicators
                price_features = self.calculate_technical_indicators(price_window)
                
                # Get news at this timestamp
                news_data = self.get_news_at_time(timestamp, hours_back=24)
                
                # Get future prices (for reward calculation)
                future_idx_1h = min(len(price_data) - 1, idx + 12)  # ~1 hour later (12 * 5min)
                future_idx_24h = min(len(price_data) - 1, idx + 288)  # ~24 hours later (288 * 5min)
                
                future_price_1h = price_data[future_idx_1h][1] if future_idx_1h < len(price_data) else None
                future_price_24h = price_data[future_idx_24h][1] if future_idx_24h < len(price_data) else None
                
                episode["prices"].append(price)
                episode["timestamps"].append(timestamp)
                episode["future_prices_1h"].append(future_price_1h)
                episode["future_prices_24h"].append(future_price_24h)
                episode["news_data"].append(news_data)
                episode["price_features"].append(price_features)
            
            episodes.append(episode)
            
            if (ep_idx + 1) % 10 == 0:
                logger.info(f"Created {ep_idx + 1}/{num_episodes} episodes")
        
        logger.info(f"Created {len(episodes)} training episodes")
        return episodes
    
    def save_episodes(
        self,
        episodes: List[Dict],
        output_path: str = "training_data/episodes.pkl",
    ):
        """
        Save training episodes to disk.
        
        Args:
            episodes: List of episode dicts
            output_path: Path to save episodes
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(episodes, f)
        
        logger.info(f"Saved {len(episodes)} episodes to {output_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    prep = TrainingDataPrep()
    
    # Get data from last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    print(f"Preparing training data from {start_time} to {end_time}")
    
    episodes = prep.create_training_episodes(
        start_time=start_time,
        end_time=end_time,
        episode_length=100,
        step_interval_minutes=5,
    )
    
    if episodes:
        prep.save_episodes(episodes, "training_data/episodes.pkl")
        print(f"✅ Created {len(episodes)} training episodes")
    else:
        print("❌ No episodes created - check data availability")

