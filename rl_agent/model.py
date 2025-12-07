"""
Trading Actor-Critic Model

PyTorch implementation of actor-critic architecture with:
- Price branch (1D-CNN or LSTM)
- News branch with multi-head attention
- Position/time branch
- Actor head (policy) for action selection
- Critic head (value) for state value estimation
- Auxiliary heads for 1h and 24h return prediction
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
    """News branch with attention mechanism."""
    
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
        
        # Project embeddings to hidden dimension
        self.embed_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        
        # Pooling: weighted sum using attention
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, embeddings: torch.Tensor, sentiment: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (batch_size, max_headlines, embedding_dim)
            sentiment: (batch_size, max_headlines) - sentiment scores
            mask: (batch_size, max_headlines) - 1 for valid headlines, 0 for padding
            
        Returns:
            pooled: (batch_size, hidden_dim) - pooled news representation
            attention_weights: (batch_size, num_heads, max_headlines, max_headlines)
        """
        # Project embeddings
        x = self.embed_proj(embeddings)  # (batch_size, max_headlines, hidden_dim)
        
        # Add sentiment as a feature (concatenate or add)
        sentiment_expanded = sentiment.unsqueeze(-1)  # (batch_size, max_headlines, 1)
        x = x + sentiment_expanded * 10.0  # Scale sentiment influence
        
        # Apply attention
        attended, attention_weights = self.attention(x, mask)
        
        # Pool: weighted sum using attention weights
        # Use mean of attention weights across heads
        attn_mean = attention_weights.mean(dim=1)  # (batch_size, max_headlines, max_headlines)
        attn_pool = attn_mean.mean(dim=1, keepdim=True)  # (batch_size, 1, max_headlines)
        
        pooled = torch.bmm(attn_pool, attended).squeeze(1)  # (batch_size, hidden_dim)
        pooled = self.pool_proj(pooled)
        
        return pooled, attention_weights


class PriceBranch(nn.Module):
    """Price branch: processes price time-series and technical indicators."""
    
    def __init__(
        self,
        price_window_size: int = 60,
        num_indicators: int = 10,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.price_window_size = price_window_size
        self.num_indicators = num_indicators
        self.hidden_dim = hidden_dim
        
        input_dim = price_window_size + num_indicators
        
        # Use 1D CNN for time-series
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Process indicators separately
        self.indicator_fc = nn.Linear(num_indicators, 32)
        
        # Combine CNN output and indicators
        self.combine = nn.Linear(64 + 32, hidden_dim)
        
    def forward(self, price_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            price_features: (batch_size, price_window_size + num_indicators)
            
        Returns:
            output: (batch_size, hidden_dim)
        """
        batch_size = price_features.shape[0]
        
        # Split into time-series and indicators
        price_series = price_features[:, :self.price_window_size]  # (batch_size, window_size)
        indicators = price_features[:, self.price_window_size:]  # (batch_size, num_indicators)
        
        # Process time-series with CNN
        price_series = price_series.unsqueeze(1)  # (batch_size, 1, window_size)
        x = F.relu(self.conv1(price_series))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (batch_size, 64)
        
        # Process indicators
        ind_features = F.relu(self.indicator_fc(indicators))  # (batch_size, 32)
        
        # Combine
        combined = torch.cat([x, ind_features], dim=1)  # (batch_size, 96)
        output = F.relu(self.combine(combined))  # (batch_size, hidden_dim)
        
        return output


class TradingActorCritic(nn.Module):
    """
    Actor-Critic model for trading.
    
    Architecture:
    - Price branch: 1D-CNN → latent_p (128)
    - News branch: Attention → latent_n (128)
    - Position/time branch: FC → latent_s (32)
    - Shared: Concat → (256) → Actor/Critic/Aux heads
    """
    
    def __init__(
        self,
        price_window_size: int = 60,
        num_indicators: int = 10,
        embedding_dim: int = 384,
        max_news_headlines: int = 20,
        num_actions: int = 3,  # BUY, SELL, HOLD
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.max_news_headlines = max_news_headlines
        
        # Branches
        self.price_branch = PriceBranch(price_window_size, num_indicators, hidden_dim)
        self.news_branch = NewsBranch(embedding_dim, hidden_dim, num_attention_heads, max_news_headlines)
        
        # Position/time branch (simple FC)
        self.position_branch = nn.Sequential(
            nn.Linear(5, 16),  # 5 position features
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
        )
        
        # Time branch
        self.time_branch = nn.Sequential(
            nn.Linear(4, 16),  # 4 time features
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
        )
        
        # Shared latent
        shared_input_dim = hidden_dim + hidden_dim + 32 + 16  # price + news + position + time
        self.shared = nn.Sequential(
            nn.Linear(shared_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        # Auxiliary heads (1h and 24h return prediction)
        self.aux_1h = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        self.aux_24h = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        
        # Initialize weights properly to prevent extreme values
        self._initialize_weights()
    
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            price_features: (batch_size, price_window_size + num_indicators)
            news_embeddings: (batch_size, max_headlines, embedding_dim)
            news_sentiment: (batch_size, max_headlines)
            position_features: (batch_size, 5)
            time_features: (batch_size, 4)
            news_mask: (batch_size, max_headlines) - 1 for valid, 0 for padding
            
        Returns:
            Dict with:
                - 'action_logits': (batch_size, num_actions)
                - 'value': (batch_size, 1)
                - 'pred_1h': (batch_size, 1)
                - 'pred_24h': (batch_size, 1)
                - 'attention_weights': (batch_size, num_heads, max_headlines, max_headlines)
        """
        # Validate inputs - replace NaN/inf
        price_features = torch.where(torch.isfinite(price_features), price_features, torch.zeros_like(price_features))
        news_embeddings = torch.where(torch.isfinite(news_embeddings), news_embeddings, torch.zeros_like(news_embeddings))
        news_sentiment = torch.where(torch.isfinite(news_sentiment), news_sentiment, torch.zeros_like(news_sentiment))
        position_features = torch.where(torch.isfinite(position_features), position_features, torch.zeros_like(position_features))
        time_features = torch.where(torch.isfinite(time_features), time_features, torch.zeros_like(time_features))
        
        # Process each branch
        price_latent = self.price_branch(price_features)
        news_latent, attention_weights = self.news_branch(news_embeddings, news_sentiment, news_mask)
        position_latent = self.position_branch(position_features)
        time_latent = self.time_branch(time_features)
        
        # Validate branch outputs
        price_latent = torch.where(torch.isfinite(price_latent), price_latent, torch.zeros_like(price_latent))
        news_latent = torch.where(torch.isfinite(news_latent), news_latent, torch.zeros_like(news_latent))
        position_latent = torch.where(torch.isfinite(position_latent), position_latent, torch.zeros_like(position_latent))
        time_latent = torch.where(torch.isfinite(time_latent), time_latent, torch.zeros_like(time_latent))
        
        # Concatenate
        shared_input = torch.cat([price_latent, news_latent, position_latent, time_latent], dim=1)
        shared_input = torch.where(torch.isfinite(shared_input), shared_input, torch.zeros_like(shared_input))
        
        # Clamp shared_input to prevent extreme values that could cause NaN
        shared_input = torch.clamp(shared_input, min=-10.0, max=10.0)
        
        shared_latent = self.shared(shared_input)
        shared_latent = torch.where(torch.isfinite(shared_latent), shared_latent, torch.zeros_like(shared_latent))
        
        # Clamp shared_latent to prevent extreme values
        shared_latent = torch.clamp(shared_latent, min=-10.0, max=10.0)
        
        # Heads
        action_logits = self.actor(shared_latent)
        value = self.critic(shared_latent)
        pred_1h = self.aux_1h(shared_latent)
        pred_24h = self.aux_24h(shared_latent)
        
        # Validate and clamp outputs to prevent extreme values
        action_logits = torch.where(torch.isfinite(action_logits), action_logits, torch.zeros_like(action_logits))
        value = torch.where(torch.isfinite(value), value, torch.zeros_like(value))
        
        # For auxiliary predictions, clamp to reasonable range and validate
        # Predictions should be in range [-1, 1] (i.e., ±100% return)
        pred_1h = torch.where(torch.isfinite(pred_1h), pred_1h, torch.zeros_like(pred_1h))
        pred_1h = torch.clamp(pred_1h, min=-1.0, max=1.0)  # Clamp to ±100% return
        
        pred_24h = torch.where(torch.isfinite(pred_24h), pred_24h, torch.zeros_like(pred_24h))
        pred_24h = torch.clamp(pred_24h, min=-1.0, max=1.0)  # Clamp to ±100% return
        
        return {
            "action_logits": action_logits,
            "value": value,
            "pred_1h": pred_1h,
            "pred_24h": pred_24h,
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
                position_features, time_features, news_mask
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

