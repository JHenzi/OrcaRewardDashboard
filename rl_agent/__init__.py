"""
RL Agent Module

This module implements a reinforcement learning agent for trading SOL based on:
- Price data and technical indicators
- News embeddings and sentiment
- Position state and portfolio management

The agent uses PPO (Proximal Policy Optimization) with:
- Actor-critic architecture
- Multi-head attention over news embeddings
- Auxiliary heads for multi-horizon return prediction (1h, 24h)
- Explainability features (SHAP, decision trees, attention visualization)
"""

__version__ = "0.1.0"

from .model import TradingActorCritic
from .environment import TradingEnvironment
from .state_encoder import StateEncoder
from .trainer import PPOTrainer

__all__ = [
    "TradingActorCritic",
    "TradingEnvironment",
    "StateEncoder",
    "PPOTrainer",
]

