"""
Example Usage of RL Agent

This script demonstrates how to use the RL agent components.
This is a basic example - full training would require integration
with the price fetcher and news sentiment systems.
"""

import torch
import numpy as np
from datetime import datetime

from .model import TradingActorCritic
from .environment import TradingEnvironment, Action
from .state_encoder import StateEncoder


def example_basic_usage():
    """Basic example of using the RL agent components."""
    
    print("=" * 60)
    print("RL Agent Example Usage")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    
    # State encoder
    state_encoder = StateEncoder(
        price_window_size=60,
        max_news_headlines=20,
        embedding_dim=384,
    )
    
    # Environment
    env = TradingEnvironment(
        initial_capital=10000.0,
        transaction_cost_rate=0.001,
        max_position_size=0.1,
    )
    
    # Model
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    )
    
    print("   ✅ Components initialized")
    
    # 2. Create sample data
    print("\n2. Creating sample data...")
    
    # Sample price data (60 minutes of prices)
    prices = [100.0 + np.random.randn() * 2 for _ in range(60)]
    
    # Sample price features
    price_features = {
        "current_price": prices[-1],
        "sma_1h": np.mean(prices[-12:]),
        "sma_4h": np.mean(prices[-48:]),
        "sma_24h": np.mean(prices),
        "rsi": 50.0,
        "std_dev": np.std(prices),
        "momentum_15m": 0.01,
        "percent_change": 0.5,
    }
    
    # Sample news data (dummy embeddings)
    news_data = [
        {
            "embedding": np.random.randn(384).astype(np.float32),
            "sentiment_score": 0.5,
            "headline": "Sample headline",
            "cluster_id": 0,
        }
        for _ in range(5)  # 5 headlines
    ]
    
    print("   ✅ Sample data created")
    
    # 3. Encode state
    print("\n3. Encoding state...")
    
    obs = env.reset()
    state_dict = state_encoder.encode_full_state(
        prices=prices,
        price_features=price_features,
        news_data=news_data,
        position_size=obs["position_size"],
        portfolio_value=obs["portfolio_value"],
        entry_price=obs.get("entry_price"),
        current_price=prices[-1],
        time_since_last_trade=obs["time_since_last_trade"],
        timestamp=datetime.now(),
        unrealized_pnl=obs.get("unrealized_pnl", 0.0),
    )
    
    print(f"   ✅ State encoded:")
    print(f"      - Price features shape: {state_dict['price'].shape}")
    print(f"      - News embeddings shape: {state_dict['news_embeddings'].shape}")
    print(f"      - Position features shape: {state_dict['position'].shape}")
    print(f"      - Time features shape: {state_dict['time'].shape}")
    
    # 4. Get action from model
    print("\n4. Getting action from model...")
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0)
    
    # Create mask for news
    news_mask = (news_sent_tensor != 0.0).float()
    
    # Get action
    action, output = model.get_action(
        price_tensor, news_emb_tensor, news_sent_tensor,
        position_tensor, time_tensor, news_mask
    )
    
    action_names = ["SELL", "HOLD", "BUY"]
    print(f"   ✅ Action: {action_names[action]}")
    print(f"      - Value estimate: {output['value'].item():.4f}")
    print(f"      - Predicted 1h return: {output['pred_1h'].item():.4f}")
    print(f"      - Predicted 24h return: {output['pred_24h'].item():.4f}")
    
    # 5. Step environment
    print("\n5. Stepping environment...")
    
    next_price = prices[-1] + np.random.randn() * 0.5
    obs, reward, done, info = env.step(action, prices[-1], next_price)
    
    print(f"   ✅ Environment stepped:")
    print(f"      - Reward: {reward:.6f}")
    print(f"      - Portfolio value: ${obs['portfolio_value']:.2f}")
    print(f"      - Position size: {obs['position_size']:.4f} SOL")
    print(f"      - Trade executed: {info.get('trade_executed', False)}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    example_basic_usage()

