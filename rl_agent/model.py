"""
Trading Actor-Critic Model

PyTorch implementation of actor-critic architecture with:
- Multi-resolution price branches (5min, 1hr, daily candles)
- News branch with attention + time decay
- Disentangled latent: Market Latent (for predictions) vs Account Latent (for trading)
- Actor head (policy) for action selection
- Critic head (value) for state value estimation
- Auxiliary heads for 15m/1h/24h return CLASSIFICATION (Bearish/Neutral/Bullish)

Architecture (Post-Audit):
- Market Latent = Price + News + Time (NO position data - prevents causal leakage)
- Account Latent = Position features
- Aux Heads (15m/1h/24h) → Market Latent ONLY (prevents "my entry price affects SOL price" fallacy)
- Actor/Critic Heads → Market Latent + Account Latent (needs position for trading decisions)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for news embeddings."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, embed_dim) - news embeddings
            mask: (batch_size, seq_len) - mask for padding (1 for valid, 0 for padding)
            
        Returns:
            output: (batch_size, seq_len, embed_dim) - attended embeddings
            attention_weights: (batch_size, num_heads, seq_len, seq_len) - attention weights
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Clamp scores to prevent extreme values that could cause NaN in softmax
        scores = torch.clamp(scores, min=-50.0, max=50.0)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # Use large negative value instead of -inf to prevent NaN in softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Validate scores before softmax
        scores = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Validate attention weights
        attention_weights = torch.where(torch.isfinite(attention_weights), attention_weights, torch.zeros_like(attention_weights))
        
        # Renormalize to ensure valid probability distribution
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights


class NewsBranch(nn.Module):
    """News branch with attention mechanism and time decay.
    
    Time decay allows the model to learn that "old bad news" matters less than "fresh bad news".
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        num_heads: int = 4,
        max_headlines: int = 20,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_headlines = max_headlines
        
        # Project embeddings to hidden dimension (embedding + sentiment + age = embedding_dim + 2)
        # But we process age separately for time decay
        self.embed_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Time decay layer: learns how to weight news by age
        # Input: age_in_minutes (normalized), Output: decay weight
        self.time_decay = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # Output 0-1: 0 = ignore old news, 1 = keep fresh news
        )
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Pooling: weighted sum using attention
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        sentiment: torch.Tensor, 
        news_age: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (batch_size, max_headlines, embedding_dim)
            sentiment: (batch_size, max_headlines) - sentiment scores
            news_age: (batch_size, max_headlines) - age in minutes (normalized by /1440 for day)
            mask: (batch_size, max_headlines) - 1 for valid headlines, 0 for padding
            
        Returns:
            pooled: (batch_size, hidden_dim) - pooled news representation
            attention_weights: (batch_size, num_heads, max_headlines, max_headlines)
        """
        # Project embeddings
        x = self.embed_proj(embeddings)  # (batch_size, max_headlines, hidden_dim)
        
        # Add sentiment as a feature
        sentiment_expanded = sentiment.unsqueeze(-1)  # (batch_size, max_headlines, 1)
        x = x + sentiment_expanded * 10.0  # Scale sentiment influence
        
        # Apply time decay if provided (FIX #4: News Time Decay)
        if news_age is not None:
            # Normalize age: 0 = fresh, 1 = 1 day old
            age_normalized = news_age.unsqueeze(-1)  # (batch_size, max_headlines, 1)
            decay_weights = self.time_decay(age_normalized)  # (batch_size, max_headlines, 1)
            # Fresh news (age=0) should have high weight, old news (age=1) should have low weight
            # Invert: fresh = 1.0, old = decay learned
            x = x * decay_weights  # Weight embeddings by freshness
        
        # Apply attention
        attended, attention_weights = self.attention(x, mask)
        
        # Pool: weighted sum using attention weights
        # Use mean of attention weights across heads
        attn_mean = attention_weights.mean(dim=1)  # (batch_size, max_headlines, max_headlines)
        attn_pool = attn_mean.mean(dim=1, keepdim=True)  # (batch_size, 1, max_headlines)
        
        pooled = torch.bmm(attn_pool, attended).squeeze(1)  # (batch_size, hidden_dim)
        pooled = self.pool_proj(pooled)
        
        return pooled, attention_weights


class PriceBranchSingleScale(nn.Module):
    """Single-scale price branch: processes one time resolution."""
    
    def __init__(self, window_size: int, output_dim: int = 32):
        super().__init__()
        self.window_size = window_size
        
        # Use 1D CNN for time-series
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, price_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_series: (batch_size, window_size) - log returns
            
        Returns:
            output: (batch_size, output_dim)
        """
        x = price_series.unsqueeze(1)  # (batch_size, 1, window_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch_size, 64)
        return F.relu(self.fc(x))  # (batch_size, output_dim)


class PriceBranch(nn.Module):
    """Multi-resolution price branch: processes multiple time scales.
    
    FIX #2: Temporal Horizon Mismatch
    - Short-term: 60 points @ 5-min intervals (5 hours) - for 15m/1h predictions
    - Medium-term: 60 points @ 1-hour intervals (2.5 days) - for 24h predictions
    - Long-term: 30 points @ 1-day intervals (1 month) - for macro context
    """
    
    def __init__(
        self,
        price_window_size: int = 60,  # Short-term window (backward compat)
        num_indicators: int = 9,  # Reduced: removed raw price (FIX #3)
        hidden_dim: int = 128,
        # Multi-resolution windows
        medium_window_size: int = 60,  # 1-hour candles
        long_window_size: int = 30,  # Daily candles
    ):
        super().__init__()
        self.price_window_size = price_window_size
        self.medium_window_size = medium_window_size
        self.long_window_size = long_window_size
        self.num_indicators = num_indicators
        self.hidden_dim = hidden_dim
        
        # Multi-scale CNN branches
        self.short_branch = PriceBranchSingleScale(price_window_size, output_dim=32)
        self.medium_branch = PriceBranchSingleScale(medium_window_size, output_dim=32)
        self.long_branch = PriceBranchSingleScale(long_window_size, output_dim=32)
        
        # Process indicators separately (stationary features only - FIX #3)
        self.indicator_fc = nn.Linear(num_indicators, 32)
        
        # Combine all scales + indicators
        # 32 (short) + 32 (medium) + 32 (long) + 32 (indicators) = 128
        self.combine = nn.Linear(128, hidden_dim)
        
    def forward(
        self, 
        price_features: torch.Tensor,
        price_medium: Optional[torch.Tensor] = None,
        price_long: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            price_features: (batch_size, price_window_size + num_indicators) - short-term + indicators
            price_medium: (batch_size, medium_window_size) - hourly returns (optional, for backward compat)
            price_long: (batch_size, long_window_size) - daily returns (optional, for backward compat)
            
        Returns:
            output: (batch_size, hidden_dim)
        """
        batch_size = price_features.shape[0]
        
        # Split short-term into time-series and indicators
        price_short = price_features[:, :self.price_window_size]  # (batch_size, window_size)
        indicators = price_features[:, self.price_window_size:]  # (batch_size, num_indicators)
        
        # Process short-term
        short_latent = self.short_branch(price_short)  # (batch_size, 32)
        
        # Process medium-term (backward compatible: use zeros if not provided)
        if price_medium is not None:
            medium_latent = self.medium_branch(price_medium)  # (batch_size, 32)
        else:
            medium_latent = torch.zeros(batch_size, 32, device=price_features.device)
        
        # Process long-term (backward compatible: use zeros if not provided)
        if price_long is not None:
            long_latent = self.long_branch(price_long)  # (batch_size, 32)
        else:
            long_latent = torch.zeros(batch_size, 32, device=price_features.device)
        
        # Process indicators
        ind_features = F.relu(self.indicator_fc(indicators))  # (batch_size, 32)
        
        # Combine all scales
        combined = torch.cat([short_latent, medium_latent, long_latent, ind_features], dim=1)
        output = F.relu(self.combine(combined))  # (batch_size, hidden_dim)
        
        return output


class TradingActorCritic(nn.Module):
    """
    Actor-Critic model for trading with DISENTANGLED architecture.
    
    FIX #1: Causal Leakage Prevention
    - Market Latent = Price + News + Time (NO position data)
    - Account Latent = Position features
    - Aux Heads (predictions) → Market Latent ONLY
    - Actor/Critic → Market Latent + Account Latent
    
    FIX #5: Classification instead of Regression
    - Aux heads output 3 classes: [Bearish (<-1%), Neutral, Bullish (>1%)]
    - CrossEntropy loss instead of MSE (avoids "predict mean" trap)
    """
    
    # Class constants for prediction classes
    CLASS_BEARISH = 0  # < -1% return
    CLASS_NEUTRAL = 1  # -1% to +1% return  
    CLASS_BULLISH = 2  # > +1% return
    NUM_PRED_CLASSES = 3
    
    def __init__(
        self,
        price_window_size: int = 60,
        num_indicators: int = 9,  # Reduced from 10: removed raw price (FIX #3)
        embedding_dim: int = 384,
        max_news_headlines: int = 20,
        num_actions: int = 3,  # BUY, SELL, HOLD
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        # Multi-resolution settings (FIX #2)
        medium_window_size: int = 60,  # 1-hour candles
        long_window_size: int = 30,  # Daily candles
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.max_news_headlines = max_news_headlines
        self.hidden_dim = hidden_dim
        
        # === MARKET BRANCHES (used for price predictions) ===
        # Price branch with multi-resolution (FIX #2)
        self.price_branch = PriceBranch(
            price_window_size, num_indicators, hidden_dim,
            medium_window_size, long_window_size
        )
        
        # News branch with time decay (FIX #4)
        self.news_branch = NewsBranch(embedding_dim, hidden_dim, num_attention_heads, max_news_headlines)
        
        # Time branch (market feature - time of day affects volatility)
        self.time_branch = nn.Sequential(
            nn.Linear(4, 16),  # 4 time features
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        
        # === ACCOUNT BRANCH (used ONLY for trading decisions) ===
        # Position branch - ISOLATED from price predictions (FIX #1)
        self.position_branch = nn.Sequential(
            nn.Linear(5, 16),  # 5 position features
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        
        # === MARKET LATENT (price + news + time, NO position) ===
        # This is what the Aux heads see - prevents causal leakage
        market_latent_dim = hidden_dim + hidden_dim + 16  # price + news + time = 272
        self.market_latent = nn.Sequential(
            nn.Linear(market_latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # === FULL LATENT (market + position, for Actor/Critic) ===
        # Actor and Critic need position info to make trading decisions
        full_latent_dim = 256 + 32  # market_latent + position = 288
        self.full_latent = nn.Sequential(
            nn.Linear(full_latent_dim, 256),
            nn.ReLU(),
        )
        
        # === HEADS ===
        
        # Actor head (policy) - needs position info
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        
        # Critic head (value) - needs position info  
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # === AUXILIARY PREDICTION HEADS (FIX #5: Classification) ===
        # These use MARKET LATENT ONLY (FIX #1: no position leakage)
        # Output: 3 classes [Bearish, Neutral, Bullish]
        
        self.aux_15m = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_PRED_CLASSES),  # 3-class output
        )
        
        self.aux_1h = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_PRED_CLASSES),  # 3-class output
        )
        
        self.aux_24h = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_PRED_CLASSES),  # 3-class output
        )
        
        # Initialize weights properly to prevent extreme values
        self._initialize_weights()
    
    @staticmethod
    def return_to_class(return_pct: float) -> int:
        """Convert a return percentage to a class label.
        
        Args:
            return_pct: Return as decimal (0.01 = 1%)
            
        Returns:
            Class label: 0=Bearish, 1=Neutral, 2=Bullish
        """
        if return_pct < -0.01:  # < -1%
            return TradingActorCritic.CLASS_BEARISH
        elif return_pct > 0.01:  # > +1%
            return TradingActorCritic.CLASS_BULLISH
        else:  # -1% to +1%
            return TradingActorCritic.CLASS_NEUTRAL
    
    @staticmethod
    def class_to_return_estimate(class_probs: torch.Tensor) -> torch.Tensor:
        """Convert class probabilities to expected return estimate.
        
        Uses expected value: E[r] = P(bear)*(-2%) + P(neutral)*(0%) + P(bull)*(+2%)
        
        Args:
            class_probs: (batch_size, 3) - softmax probabilities
            
        Returns:
            (batch_size, 1) - expected return estimate
        """
        # Representative returns for each class
        class_returns = torch.tensor([-0.02, 0.0, 0.02], device=class_probs.device)
        expected_return = torch.matmul(class_probs, class_returns).unsqueeze(-1)
        return expected_return
    
    def _initialize_weights(self):
        """Initialize model weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv1d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
    def forward(
        self,
        price_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        news_sentiment: torch.Tensor,
        position_features: torch.Tensor,
        time_features: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None,
        news_age: Optional[torch.Tensor] = None,  # FIX #4: News time decay
        price_medium: Optional[torch.Tensor] = None,  # FIX #2: Multi-resolution
        price_long: Optional[torch.Tensor] = None,  # FIX #2: Multi-resolution
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with DISENTANGLED architecture.
        
        Args:
            price_features: (batch_size, price_window_size + num_indicators) - short-term
            news_embeddings: (batch_size, max_headlines, embedding_dim)
            news_sentiment: (batch_size, max_headlines)
            position_features: (batch_size, 5)
            time_features: (batch_size, 4)
            news_mask: (batch_size, max_headlines) - 1 for valid, 0 for padding
            news_age: (batch_size, max_headlines) - age in minutes, normalized (FIX #4)
            price_medium: (batch_size, medium_window_size) - hourly returns (FIX #2)
            price_long: (batch_size, long_window_size) - daily returns (FIX #2)
            
        Returns:
            Dict with:
                - 'action_logits': (batch_size, num_actions)
                - 'value': (batch_size, 1)
                - 'pred_15m_logits': (batch_size, 3) - class logits
                - 'pred_1h_logits': (batch_size, 3) - class logits
                - 'pred_24h_logits': (batch_size, 3) - class logits
                - 'pred_15m': (batch_size, 1) - expected return (for backward compat)
                - 'pred_1h': (batch_size, 1) - expected return (for backward compat)
                - 'pred_24h': (batch_size, 1) - expected return (for backward compat)
                - 'attention_weights': attention weights from news branch
        """
        # Validate inputs - replace NaN/inf
        price_features = torch.where(torch.isfinite(price_features), price_features, torch.zeros_like(price_features))
        news_embeddings = torch.where(torch.isfinite(news_embeddings), news_embeddings, torch.zeros_like(news_embeddings))
        news_sentiment = torch.where(torch.isfinite(news_sentiment), news_sentiment, torch.zeros_like(news_sentiment))
        position_features = torch.where(torch.isfinite(position_features), position_features, torch.zeros_like(position_features))
        time_features = torch.where(torch.isfinite(time_features), time_features, torch.zeros_like(time_features))
        
        if price_medium is not None:
            price_medium = torch.where(torch.isfinite(price_medium), price_medium, torch.zeros_like(price_medium))
        if price_long is not None:
            price_long = torch.where(torch.isfinite(price_long), price_long, torch.zeros_like(price_long))
        if news_age is not None:
            news_age = torch.where(torch.isfinite(news_age), news_age, torch.zeros_like(news_age))
        
        # === PROCESS BRANCHES ===
        
        # Price branch (multi-resolution) - FIX #2
        price_latent = self.price_branch(price_features, price_medium, price_long)
        
        # News branch (with time decay) - FIX #4
        news_latent, attention_weights = self.news_branch(news_embeddings, news_sentiment, news_age, news_mask)
        
        # Time branch (market feature)
        time_latent = self.time_branch(time_features)
        
        # Position branch (account feature - ISOLATED)
        position_latent = self.position_branch(position_features)
        
        # Validate branch outputs
        price_latent = torch.where(torch.isfinite(price_latent), price_latent, torch.zeros_like(price_latent))
        news_latent = torch.where(torch.isfinite(news_latent), news_latent, torch.zeros_like(news_latent))
        time_latent = torch.where(torch.isfinite(time_latent), time_latent, torch.zeros_like(time_latent))
        position_latent = torch.where(torch.isfinite(position_latent), position_latent, torch.zeros_like(position_latent))
        
        # === MARKET LATENT (FIX #1: NO position data) ===
        # Price + News + Time only - prevents causal leakage into predictions
        market_input = torch.cat([price_latent, news_latent, time_latent], dim=1)
        market_input = torch.clamp(market_input, min=-10.0, max=10.0)
        market_latent = self.market_latent(market_input)
        market_latent = torch.where(torch.isfinite(market_latent), market_latent, torch.zeros_like(market_latent))
        market_latent = torch.clamp(market_latent, min=-10.0, max=10.0)
        
        # === FULL LATENT (for Actor/Critic) ===
        # Market + Position - trading decisions need position awareness
        full_input = torch.cat([market_latent, position_latent], dim=1)
        full_latent = self.full_latent(full_input)
        full_latent = torch.where(torch.isfinite(full_latent), full_latent, torch.zeros_like(full_latent))
        
        # === HEADS ===
        
        # Actor/Critic use FULL latent (market + position)
        action_logits = self.actor(full_latent)
        value = self.critic(full_latent)
        
        # Auxiliary predictions use MARKET latent ONLY (FIX #1: no position leakage)
        # Output is class LOGITS (FIX #5: Classification)
        pred_15m_logits = self.aux_15m(market_latent)
        pred_1h_logits = self.aux_1h(market_latent)
        pred_24h_logits = self.aux_24h(market_latent)
        
        # Validate outputs
        action_logits = torch.where(torch.isfinite(action_logits), action_logits, torch.zeros_like(action_logits))
        value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        pred_15m_logits = torch.where(torch.isfinite(pred_15m_logits), pred_15m_logits, torch.zeros_like(pred_15m_logits))
        pred_1h_logits = torch.where(torch.isfinite(pred_1h_logits), pred_1h_logits, torch.zeros_like(pred_1h_logits))
        pred_24h_logits = torch.where(torch.isfinite(pred_24h_logits), pred_24h_logits, torch.zeros_like(pred_24h_logits))
        
        # Clamp logits to prevent extreme softmax issues
        pred_15m_logits = torch.clamp(pred_15m_logits, min=-20.0, max=20.0)
        pred_1h_logits = torch.clamp(pred_1h_logits, min=-20.0, max=20.0)
        pred_24h_logits = torch.clamp(pred_24h_logits, min=-20.0, max=20.0)
        
        # Convert class logits to expected return (backward compatibility)
        pred_15m_probs = F.softmax(pred_15m_logits, dim=-1)
        pred_1h_probs = F.softmax(pred_1h_logits, dim=-1)
        pred_24h_probs = F.softmax(pred_24h_logits, dim=-1)
        
        pred_15m = self.class_to_return_estimate(pred_15m_probs)
        pred_1h = self.class_to_return_estimate(pred_1h_probs)
        pred_24h = self.class_to_return_estimate(pred_24h_probs)
        
        return {
            "action_logits": action_logits,
            "value": value,
            # Class logits for CrossEntropy loss (FIX #5)
            "pred_15m_logits": pred_15m_logits,
            "pred_1h_logits": pred_1h_logits,
            "pred_24h_logits": pred_24h_logits,
            # Class probabilities for interpretability
            "pred_15m_probs": pred_15m_probs,
            "pred_1h_probs": pred_1h_probs,
            "pred_24h_probs": pred_24h_probs,
            # Expected return estimates (backward compatibility)
            "pred_15m": pred_15m,
            "pred_1h": pred_1h,
            "pred_24h": pred_24h,
            # Market latent for analysis
            "market_latent": market_latent,
            "attention_weights": attention_weights,
        }
    
    def get_action(
        self,
        price_features: torch.Tensor,
        news_embeddings: torch.Tensor,
        news_sentiment: torch.Tensor,
        position_features: torch.Tensor,
        time_features: torch.Tensor,
        news_mask: Optional[torch.Tensor] = None,
        news_age: Optional[torch.Tensor] = None,
        price_medium: Optional[torch.Tensor] = None,
        price_long: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, torch.Tensor]]:
        """
        Sample an action from the policy.
        
        Args:
            Same as forward()
            deterministic: If True, return action with highest probability
            
        Returns:
            Tuple of (action, output_dict)
        """
        with torch.no_grad():
            output = self.forward(
                price_features, news_embeddings, news_sentiment,
                position_features, time_features, news_mask,
                news_age, price_medium, price_long
            )
            
            action_logits = output["action_logits"]
            
            # Clamp logits to prevent inf/nan
            action_logits = torch.clamp(action_logits, min=-50, max=50)
            
            # Check for NaN or inf
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                # Fallback to uniform distribution
                action_logits = torch.zeros_like(action_logits)
            
            if deterministic:
                action = torch.argmax(action_logits, dim=1).item()
            else:
                # Sample from softmax
                action_probs = F.softmax(action_logits, dim=-1)
                
                # Ensure probabilities are valid
                action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)  # Renormalize
                
                action = torch.multinomial(action_probs, 1).item()
            
            return action, output

