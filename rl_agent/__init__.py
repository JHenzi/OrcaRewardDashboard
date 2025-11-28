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
from .prediction_manager import PredictionManager
from .prediction_generator import generate_prediction, store_prediction_from_decision
from .attention_logger import AttentionLogger
from .risk_manager import RiskManager
from .explainability import RuleExtractor, SHAPExplainer
from .integration import RLAgentIntegration

__all__ = [
    "TradingActorCritic",
    "TradingEnvironment",
    "StateEncoder",
    "PPOTrainer",
    "PredictionManager",
    "generate_prediction",
    "store_prediction_from_decision",
    "AttentionLogger",
    "RiskManager",
    "RuleExtractor",
    "SHAPExplainer",
    "RLAgentIntegration",
]

