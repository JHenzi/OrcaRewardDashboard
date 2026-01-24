"""
State Encoder for RL Agent

Encodes market state (price, technical indicators, news, position) into
feature vectors suitable for the RL agent's neural network.

POST-AUDIT CHANGES:
- FIX #2: Multi-resolution price input (5min, 1hr, daily candles)
- FIX #3: Removed raw price features, use only stationary returns/ratios
- FIX #4: Added news age feature for time decay
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
    - Multi-resolution price time-series (FIX #2):
      - Short-term: 60 points @ 5-min intervals (5 hours)
      - Medium-term: 60 points @ 1-hour intervals (2.5 days)
      - Long-term: 30 points @ 1-day intervals (1 month)
    - Stationary technical indicators only (FIX #3): SMA ratios, RSI, volatility
    - News embeddings with time decay (FIX #4): embedding + sentiment + age
    - Position state (current position size, P&L, time-since-last-trade)
    - Time features (minute-of-day, day-of-week)
    """
    
    def __init__(
        self,
        price_window_size: int = 60,  # Short-term: Last 60 5-min intervals
        medium_window_size: int = 60,  # Medium-term: Last 60 1-hour intervals
        long_window_size: int = 30,  # Long-term: Last 30 daily candles
        max_news_headlines: int = 20,  # Latest 20 headlines
        embedding_dim: int = 384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
    ):
        """
        Initialize state encoder.
        
        Args:
            price_window_size: Number of recent 5-min price points (short-term)
            medium_window_size: Number of recent 1-hour price points (medium-term)
            long_window_size: Number of recent daily price points (long-term)
            max_news_headlines: Maximum number of news headlines to include
            embedding_dim: Dimension of news embeddings
        """
        self.price_window_size = price_window_size
        self.medium_window_size = medium_window_size
        self.long_window_size = long_window_size
        self.max_news_headlines = max_news_headlines
        self.embedding_dim = embedding_dim
        
    def encode_price_features(
        self,
        prices: List[float],
        price_features: Dict[str, float],
    ) -> np.ndarray:
        """
        Encode price time-series and technical indicators.
        
        FIX #3: Only STATIONARY features - no raw prices!
        - Log returns instead of prices (stationary)
        - Price/SMA ratios (stationary - always around 1.0)
        - RSI, volatility, momentum (all bounded/stationary)
        
        Args:
            prices: List of recent prices (last N minutes)
            price_features: Dict with technical indicators (SMA, RSI, etc.)
            
        Returns:
            Feature vector for price branch (60 returns + 9 indicators = 69 features)
        """
        # Convert prices to LOG RETURNS (stationary - FIX #3)
        if len(prices) < 2:
            returns = np.zeros(self.price_window_size)
        else:
            # Log returns: ln(p_t / p_{t-1}) - more stationary than simple returns
            prices_array = np.array(prices, dtype=np.float64)
            prices_array = np.maximum(prices_array, 1e-8)  # Prevent log(0)
            log_returns = np.diff(np.log(prices_array))
            
            # Pad or truncate to window_size
            if len(log_returns) > self.price_window_size:
                returns = log_returns[-self.price_window_size:]
            elif len(log_returns) < self.price_window_size:
                returns = np.pad(log_returns, (self.price_window_size - len(log_returns), 0), 'constant')
            else:
                returns = log_returns
        
        # Extract STATIONARY technical indicators ONLY (FIX #3)
        # REMOVED: raw current_price, sma_1h, sma_4h, sma_24h (non-stationary)
        features = []
        
        current_price = price_features.get("current_price", 1.0)
        if current_price is None or current_price <= 0:
            current_price = 1.0
        
        # Price RATIOS (stationary - centered around 1.0)
        sma_1h = price_features.get("sma_1h") or current_price
        sma_4h = price_features.get("sma_4h") or current_price
        sma_24h = price_features.get("sma_24h") or current_price
        
        features.append(current_price / sma_1h if sma_1h > 0 else 1.0)  # Price/SMA_1h
        features.append(current_price / sma_4h if sma_4h > 0 else 1.0)  # Price/SMA_4h
        features.append(current_price / sma_24h if sma_24h > 0 else 1.0)  # Price/SMA_24h (NEW)
        
        # SMA ratios (stationary - detect trend)
        features.append(sma_1h / sma_4h if sma_4h > 0 else 1.0)  # Short vs Medium trend
        features.append(sma_4h / sma_24h if sma_24h > 0 else 1.0)  # Medium vs Long trend
        
        # RSI (0-100 scale, normalize to 0-1) - already bounded/stationary
        rsi = price_features.get("rsi", 50.0)
        features.append(rsi / 100.0 if rsi else 0.5)
        
        # Volatility (relative to price, so somewhat stationary)
        std_dev = price_features.get("std_dev", 0.0)
        # Normalize as percentage of current price
        features.append((std_dev / current_price) if std_dev and current_price > 0 else 0.0)
        
        # Momentum (already a return/ratio - stationary)
        momentum = price_features.get("momentum_15m", 0.0)
        features.append(momentum if momentum else 0.0)
        
        # Percent change (already a return - stationary)
        features.append(price_features.get("percent_change", 0.0) / 100.0)
        
        # Total: 60 returns + 9 indicators = 69 features
        # (reduced from 10 indicators - removed 4 raw prices, added 3 ratios)
        
        # Combine returns and features
        price_features_array = np.concatenate([returns, np.array(features)])
        
        # Replace NaN and inf with 0
        price_features_array = np.nan_to_num(price_features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        return price_features_array.astype(np.float32)
    
    def encode_price_medium(
        self,
        hourly_prices: List[float],
    ) -> np.ndarray:
        """
        Encode medium-term price data (hourly candles).
        
        FIX #2: Multi-resolution input for 24h predictions.
        
        Args:
            hourly_prices: List of hourly prices (last 60 hours = 2.5 days)
            
        Returns:
            Log returns array for medium-term branch
        """
        if len(hourly_prices) < 2:
            return np.zeros(self.medium_window_size, dtype=np.float32)
        
        # Log returns
        prices_array = np.array(hourly_prices, dtype=np.float64)
        prices_array = np.maximum(prices_array, 1e-8)
        log_returns = np.diff(np.log(prices_array))
        
        # Pad or truncate
        if len(log_returns) > self.medium_window_size:
            returns = log_returns[-self.medium_window_size:]
        elif len(log_returns) < self.medium_window_size:
            returns = np.pad(log_returns, (self.medium_window_size - len(log_returns), 0), 'constant')
        else:
            returns = log_returns
        
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns.astype(np.float32)
    
    def encode_price_long(
        self,
        daily_prices: List[float],
    ) -> np.ndarray:
        """
        Encode long-term price data (daily candles).
        
        FIX #2: Multi-resolution input for macro context.
        
        Args:
            daily_prices: List of daily prices (last 30 days = 1 month)
            
        Returns:
            Log returns array for long-term branch
        """
        if len(daily_prices) < 2:
            return np.zeros(self.long_window_size, dtype=np.float32)
        
        # Log returns
        prices_array = np.array(daily_prices, dtype=np.float64)
        prices_array = np.maximum(prices_array, 1e-8)
        log_returns = np.diff(np.log(prices_array))
        
        # Pad or truncate
        if len(log_returns) > self.long_window_size:
            returns = log_returns[-self.long_window_size:]
        elif len(log_returns) < self.long_window_size:
            returns = np.pad(log_returns, (self.long_window_size - len(log_returns), 0), 'constant')
        else:
            returns = log_returns
        
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns.astype(np.float32)
    
    def encode_news_features(
        self,
        news_data: List[Dict],
        current_timestamp: Optional[datetime] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode news embeddings, sentiment, and age (FIX #4: Time Decay).
        
        Args:
            news_data: List of dicts with keys:
                - 'embedding': np.ndarray (embedding_dim,)
                - 'sentiment_score': float (-1 to 1)
                - 'headline': str
                - 'cluster_id': int (optional)
                - 'published_at': datetime (optional, for age calculation)
            current_timestamp: Current time for calculating news age
                
        Returns:
            Tuple of (embeddings, sentiment_scores, cluster_ids, news_ages)
            - embeddings: (M, embedding_dim) array
            - sentiment_scores: (M,) array
            - cluster_ids: (M,) array
            - news_ages: (M,) array - age in minutes, normalized (0=fresh, 1=1 day old)
        """
        if current_timestamp is None:
            current_timestamp = datetime.now()
        
        # Pad or truncate to max_news_headlines
        if len(news_data) > self.max_news_headlines:
            news_data = news_data[:self.max_news_headlines]
        
        embeddings = []
        sentiment_scores = []
        cluster_ids = []
        news_ages = []  # FIX #4: Time decay
        
        for item in news_data:
            embedding = item.get("embedding")
            if embedding is not None:
                # Handle bytes embeddings (shouldn't happen if training_data_prep is correct, but be safe)
                if isinstance(embedding, bytes):
                    try:
                        import pickle
                        embedding = pickle.loads(embedding)
                    except (pickle.UnpicklingError, EOFError, ValueError):
                        try:
                            embedding = np.frombuffer(embedding, dtype=np.float32)
                        except (ValueError, TypeError):
                            logger.warning("Could not decode bytes embedding, skipping")
                            continue
                
                # Convert to numpy array and validate
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                else:
                    embedding = embedding.astype(np.float32)
                
                # Validate shape
                if embedding.shape != (self.embedding_dim,):
                    logger.warning(f"Invalid embedding shape: {embedding.shape}, expected ({self.embedding_dim},), skipping")
                    continue
                
                # Validate values (should be finite)
                if not np.isfinite(embedding).all():
                    logger.warning("Non-finite values in embedding, skipping")
                    continue
                
                embeddings.append(embedding)
                sentiment_scores.append(item.get("sentiment_score", 0.0))
                
                # Handle cluster_id - ensure it's an int, default to -1 if None
                cluster_id = item.get("cluster_id", -1)
                if cluster_id is None:
                    cluster_id = -1
                else:
                    try:
                        cluster_id = int(cluster_id)
                    except (ValueError, TypeError):
                        cluster_id = -1
                cluster_ids.append(cluster_id)
                
                # FIX #4: Calculate news age
                published_at = item.get("published_at")
                if published_at is not None:
                    try:
                        if isinstance(published_at, str):
                            # Parse ISO format string
                            published_at = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                        age_minutes = (current_timestamp - published_at).total_seconds() / 60.0
                        # Normalize: 0 = fresh, 1 = 1 day old (1440 minutes)
                        age_normalized = min(age_minutes / 1440.0, 1.0)  # Cap at 1 day
                        age_normalized = max(age_normalized, 0.0)  # No negative ages
                    except (ValueError, TypeError, AttributeError):
                        age_normalized = 0.5  # Default to medium age if parsing fails
                else:
                    age_normalized = 0.5  # Default if no timestamp
                news_ages.append(age_normalized)
        
        # Pad to max_news_headlines
        while len(embeddings) < self.max_news_headlines:
            embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
            sentiment_scores.append(0.0)
            cluster_ids.append(-1)
            news_ages.append(1.0)  # Padding = old (ignored by attention mask anyway)
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        sentiment_array = np.array(sentiment_scores, dtype=np.float32)
        cluster_array = np.array([int(cid) if cid is not None else -1 for cid in cluster_ids], dtype=np.int32)
        age_array = np.array(news_ages, dtype=np.float32)
        
        # Replace NaN and inf with 0
        embeddings_array = np.nan_to_num(embeddings_array, nan=0.0, posinf=0.0, neginf=0.0)
        sentiment_array = np.nan_to_num(sentiment_array, nan=0.0, posinf=0.0, neginf=0.0)
        age_array = np.nan_to_num(age_array, nan=0.5, posinf=1.0, neginf=0.0)
        
        return embeddings_array, sentiment_array, cluster_array, age_array
    
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
    
    def encode_external_market_returns(
        self,
        prices: List[float],
        window_size: int = 30,
    ) -> np.ndarray:
        """
        Encode external market (BTC or S&P 500) log returns.
        
        Args:
            prices: List of recent prices (5-min intervals)
            window_size: Number of returns to include (default: 30 = 2.5 hours)
            
        Returns:
            Log returns array (window_size,)
        """
        if len(prices) < 2:
            return np.zeros(window_size, dtype=np.float32)
        
        # Log returns
        prices_array = np.array(prices, dtype=np.float64)
        prices_array = np.maximum(prices_array, 1e-8)
        log_returns = np.diff(np.log(prices_array))
        
        # Pad or truncate to window_size
        if len(log_returns) > window_size:
            returns = log_returns[-window_size:]
        elif len(log_returns) < window_size:
            returns = np.pad(log_returns, (window_size - len(log_returns), 0), 'constant')
        else:
            returns = log_returns
        
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns.astype(np.float32)
    
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
        # Multi-resolution price data (FIX #2)
        hourly_prices: Optional[List[float]] = None,
        daily_prices: Optional[List[float]] = None,
        # External market data (NEW)
        btc_prices: Optional[List[float]] = None,
        sp500_prices: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Encode complete trading state with multi-resolution, time decay, and external markets.
        
        Args:
            prices: Short-term prices (5-min intervals)
            price_features: Technical indicators
            news_data: News with embeddings, sentiment, and timestamps
            position_size, portfolio_value, etc.: Position state
            timestamp: Current time (for news age calculation)
            hourly_prices: Medium-term prices (1-hour intervals) - FIX #2
            daily_prices: Long-term prices (daily candles) - FIX #2
            btc_prices: BTC prices (5-min intervals) - NEW
            sp500_prices: S&P 500 prices (5-min intervals) - NEW
        
        Returns:
            Dict with keys:
                - 'price': short-term price features (60 returns + 9 indicators)
                - 'price_medium': medium-term log returns (60 hourly) - FIX #2
                - 'price_long': long-term log returns (30 daily) - FIX #2
                - 'news_embeddings': news embeddings array (M, embedding_dim)
                - 'news_sentiment': sentiment scores array (M,)
                - 'news_clusters': cluster IDs array (M,)
                - 'news_age': news age array (M,) - FIX #4
                - 'position': position features array
                - 'time': time features array
                - 'btc_returns': BTC log returns (30 returns) - NEW
                - 'sp500_returns': S&P 500 log returns (30 returns) - NEW
        """
        # Short-term price features (stationary - FIX #3)
        price_features_array = self.encode_price_features(prices, price_features)
        
        # Multi-resolution price features (FIX #2)
        price_medium_array = self.encode_price_medium(hourly_prices or [])
        price_long_array = self.encode_price_long(daily_prices or [])
        
        # News with time decay (FIX #4)
        news_embeddings, news_sentiment, news_clusters, news_age = self.encode_news_features(
            news_data, current_timestamp=timestamp
        )
        
        # Position features
        position_features = self.encode_position_features(
            position_size, portfolio_value, entry_price,
            current_price, time_since_last_trade, unrealized_pnl
        )
        
        # Time features
        time_features = self.encode_time_features(timestamp)
        
        # External market returns (NEW)
        btc_returns = self.encode_external_market_returns(btc_prices or [], window_size=30)
        sp500_returns = self.encode_external_market_returns(sp500_prices or [], window_size=30)
        
        return {
            "price": price_features_array,
            "price_medium": price_medium_array,  # FIX #2
            "price_long": price_long_array,  # FIX #2
            "news_embeddings": news_embeddings,
            "news_sentiment": news_sentiment,
            "news_clusters": news_clusters,
            "news_age": news_age,  # FIX #4
            "position": position_features,
            "time": time_features,
            "btc_returns": btc_returns,  # NEW
            "sp500_returns": sp500_returns,  # NEW
        }

