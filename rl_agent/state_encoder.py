"""
State Encoder for RL Agent

Encodes market state (price, technical indicators, news, position) into
feature vectors suitable for the RL agent's neural network.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StateEncoder:
    """
    Encodes trading state into feature vectors.
    
    Handles:
    - Price time-series window (last N minutes)
    - Technical indicators (SMA, EMA, RSI, ATR, volatility, returns)
    - News embeddings (latest M headlines with sentiment)
    - Position state (current position size, P&L, time-since-last-trade)
    - Time features (minute-of-day, day-of-week)
    """
    
    def __init__(
        self,
        price_window_size: int = 60,  # Last 60 minutes
        max_news_headlines: int = 20,  # Latest 20 headlines
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
    ):
        """
        Initialize state encoder.
        
        Args:
            price_window_size: Number of recent price points to include
            max_news_headlines: Maximum number of news headlines to include
            embedding_dim: Dimension of news embeddings
        """
        self.price_window_size = price_window_size
        self.max_news_headlines = max_news_headlines
        self.embedding_dim = embedding_dim
        
    def encode_price_features(
        self,
        prices: List[float],
        price_features: Dict[str, float],
    ) -> np.ndarray:
        """
        Encode price time-series and technical indicators.
        
        Args:
            prices: List of recent prices (last N minutes)
            price_features: Dict with technical indicators (SMA, RSI, etc.)
            
        Returns:
            Feature vector for price branch
        """
        # Normalize prices to returns (percentage changes)
        if len(prices) < 2:
            returns = np.zeros(self.price_window_size)
        else:
            returns = np.diff(prices) / prices[:-1]
            # Pad or truncate to window_size
            if len(returns) > self.price_window_size:
                returns = returns[-self.price_window_size:]
            elif len(returns) < self.price_window_size:
                returns = np.pad(returns, (self.price_window_size - len(returns), 0), 'constant')
        
        # Extract technical indicators
        features = []
        
        # Price statistics
        features.append(price_features.get("current_price", 0.0) / 100.0)  # Normalize
        features.append(price_features.get("sma_1h", 0.0) / 100.0 if price_features.get("sma_1h") else 0.0)
        features.append(price_features.get("sma_4h", 0.0) / 100.0 if price_features.get("sma_4h") else 0.0)
        features.append(price_features.get("sma_24h", 0.0) / 100.0 if price_features.get("sma_24h") else 0.0)
        
        # Price ratios
        current_price = price_features.get("current_price", 1.0)
        features.append(current_price / price_features.get("sma_1h", 1.0) if price_features.get("sma_1h") else 1.0)
        features.append(current_price / price_features.get("sma_4h", 1.0) if price_features.get("sma_4h") else 1.0)
        
        # RSI (0-100 scale, normalize to 0-1)
        rsi = price_features.get("rsi", 50.0)
        features.append(rsi / 100.0 if rsi else 0.5)
        
        # Volatility
        features.append(price_features.get("std_dev", 0.0) / 100.0)
        
        # Momentum
        momentum = price_features.get("momentum_15m", 0.0)
        features.append(momentum if momentum else 0.0)
        
        # Percent change
        features.append(price_features.get("percent_change", 0.0) / 100.0)
        
        # Combine returns and features
        price_features_array = np.concatenate([returns, np.array(features)])
        
        # Replace NaN and inf with 0
        price_features_array = np.nan_to_num(price_features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return price_features_array.astype(np.float32)
    
    def encode_news_features(
        self,
        news_data: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode news embeddings and sentiment.
        
        Args:
            news_data: List of dicts with keys:
                - 'embedding': np.ndarray (embedding_dim,)
                - 'sentiment_score': float (-1 to 1)
                - 'headline': str
                - 'cluster_id': int (optional)
                
        Returns:
            Tuple of (embeddings, sentiment_scores, cluster_ids)
            - embeddings: (M, embedding_dim) array
            - sentiment_scores: (M,) array
            - cluster_ids: (M,) array
        """
        # Pad or truncate to max_news_headlines
        if len(news_data) > self.max_news_headlines:
            news_data = news_data[:self.max_news_headlines]
        
        embeddings = []
        sentiment_scores = []
        cluster_ids = []
        
        for item in news_data:
            embedding = item.get("embedding")
            if embedding is not None:
                if isinstance(embedding, bytes):
                    # If stored as bytes, need to decode (would need pickle or similar)
                    # For now, assume it's already an array
                    logger.warning("Embedding stored as bytes, skipping")
                    continue
                embeddings.append(np.array(embedding, dtype=np.float32))
                sentiment_scores.append(item.get("sentiment_score", 0.0))
                cluster_ids.append(item.get("cluster_id", -1))
        
        # Pad to max_news_headlines
        while len(embeddings) < self.max_news_headlines:
            embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            sentiment_scores.append(0.0)
            cluster_ids.append(-1)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        sentiment_array = np.array(sentiment_scores, dtype=np.float32)
        cluster_array = np.array(cluster_ids, dtype=np.int32)
        
        # Replace NaN and inf with 0
        embeddings_array = np.nan_to_num(embeddings_array, nan=0.0, posinf=0.0, neginf=0.0)
        sentiment_array = np.nan_to_num(sentiment_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return embeddings_array, sentiment_array, cluster_array
    
    def encode_position_features(
        self,
        position_size: float,
        portfolio_value: float,
        entry_price: Optional[float],
        current_price: float,
        time_since_last_trade: float,  # minutes
        unrealized_pnl: float = 0.0,
    ) -> np.ndarray:
        """
        Encode position and portfolio state.
        
        Args:
            position_size: Current position size (SOL amount)
            portfolio_value: Total portfolio value
            entry_price: Entry price for current position
            current_price: Current SOL price
            time_since_last_trade: Minutes since last trade
            unrealized_pnl: Unrealized profit/loss
            
        Returns:
            Feature vector for position branch
        """
        features = []
        
        # Position size (normalized by portfolio value)
        position_ratio = position_size * current_price / portfolio_value if portfolio_value > 0 else 0.0
        features.append(position_ratio)
        
        # Position value
        position_value = position_size * current_price
        features.append(position_value / 1000.0)  # Normalize
        
        # Entry price ratio
        if entry_price and entry_price > 0:
            price_ratio = current_price / entry_price
            features.append(price_ratio)
        else:
            features.append(1.0)  # No position
        
        # Unrealized P&L (normalized)
        features.append(unrealized_pnl / portfolio_value if portfolio_value > 0 else 0.0)
        
        # Time since last trade (normalized to hours)
        features.append(time_since_last_trade / 60.0)
        
        features_array = np.array(features, dtype=np.float32)
        
        # Replace NaN and inf with 0
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features_array
    
    def encode_time_features(self, timestamp: Optional[datetime]) -> np.ndarray:
        """
        Encode temporal features.
        
        Args:
            timestamp: Current timestamp (or None for default)
            
        Returns:
            Feature vector for time features
        """
        features = []
        
        # Use current time if timestamp is None
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now()
        
        # Hour of day (0-23, normalize to 0-1)
        features.append(timestamp.hour / 24.0)
        
        # Day of week (0-6, normalize to 0-1)
        features.append(timestamp.weekday() / 7.0)
        
        # Minute of day (0-1439, normalize to 0-1)
        features.append((timestamp.hour * 60 + timestamp.minute) / 1440.0)
        
        # Is weekend
        features.append(1.0 if timestamp.weekday() >= 5 else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def encode_full_state(
        self,
        prices: List[float],
        price_features: Dict[str, float],
        news_data: List[Dict],
        position_size: float,
        portfolio_value: float,
        entry_price: Optional[float],
        current_price: float,
        time_since_last_trade: float,
        timestamp: datetime,
        unrealized_pnl: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Encode complete trading state.
        
        Returns:
            Dict with keys:
                - 'price': price features array
                - 'news_embeddings': news embeddings array (M, embedding_dim)
                - 'news_sentiment': sentiment scores array (M,)
                - 'news_clusters': cluster IDs array (M,)
                - 'position': position features array
                - 'time': time features array
        """
        price_features_array = self.encode_price_features(prices, price_features)
        news_embeddings, news_sentiment, news_clusters = self.encode_news_features(news_data)
        position_features = self.encode_position_features(
            position_size, portfolio_value, entry_price,
            current_price, time_since_last_trade, unrealized_pnl
        )
        time_features = self.encode_time_features(timestamp)
        
        return {
            "price": price_features_array,
            "news_embeddings": news_embeddings,
            "news_sentiment": news_sentiment,
            "news_clusters": news_clusters,
            "position": position_features,
            "time": time_features,
        }

