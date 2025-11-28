"""
Test script for PPO training loop.

This script tests that the training loop can run without errors.
It uses dummy data to verify the implementation.
"""

import torch
import numpy as np
from datetime import datetime

from .model import TradingActorCritic
from .environment import TradingEnvironment
from .state_encoder import StateEncoder
from .trainer import PPOTrainer


def test_training_loop():
    """Test the complete training loop."""
    print("=" * 60)
    print("Testing PPO Training Loop")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing components...")
    state_encoder = StateEncoder(
        price_window_size=60,
        max_news_headlines=20,
        embedding_dim=384,
    )
    
    env = TradingEnvironment(
        initial_capital=10000.0,
        transaction_cost_rate=0.001,
        max_position_size=0.1,
    )
    
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    )
    
    trainer = PPOTrainer(
        model=model,
        environment=env,
        lr=3e-4,
        device="cpu",
    )
    
    print("   ✅ Components initialized")
    
    # Create dummy data
    print("\n2. Creating dummy data...")
    prices = [100.0 + np.random.randn() * 2 for _ in range(100)]
    
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
    
    news_data = [
        {
            "embedding": np.random.randn(384).astype(np.float32),
            "sentiment_score": np.random.uniform(-1, 1),
            "headline": f"News {i}",
            "cluster_id": i % 3,
        }
        for i in range(5)
    ]
    
    print("   ✅ Dummy data created")
    
    # Test rollout collection
    print("\n3. Testing rollout collection...")
    try:
        rollout_data = trainer.collect_rollout(
            state_encoder=state_encoder,
            price_data=prices,
            price_features=price_features,
            news_data=news_data,
            num_steps=32,  # Small number for testing
        )
        print(f"   ✅ Rollout collected: {len(rollout_data['advantages'])} steps")
        print(f"      - Advantages shape: {rollout_data['advantages'].shape}")
        print(f"      - Returns shape: {rollout_data['returns'].shape}")
    except Exception as e:
        print(f"   ❌ Rollout collection failed: {e}")
        raise
    
    # Test training step
    print("\n4. Testing training step...")
    try:
        metrics = trainer.train_step(
            rollout_data=rollout_data,
            num_epochs=2,  # Small number for testing
            batch_size=8,
        )
        print("   ✅ Training step completed")
        print(f"      - Policy loss: {metrics['policy_loss']:.6f}")
        print(f"      - Value loss: {metrics['value_loss']:.6f}")
        print(f"      - Entropy: {metrics['entropy']:.6f}")
        print(f"      - Total loss: {metrics['total_loss']:.6f}")
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        raise
    
    # Test complete training cycle
    print("\n5. Testing complete training cycle...")
    try:
        metrics = trainer.train_on_rollout(
            state_encoder=state_encoder,
            price_data=prices,
            price_features=price_features,
            news_data=news_data,
            num_steps=32,
            num_epochs=2,
            batch_size=8,
        )
        print("   ✅ Complete training cycle successful")
        print(f"      - Training step: {trainer.training_step}")
    except Exception as e:
        print(f"   ❌ Training cycle failed: {e}")
        raise
    
    # Test checkpointing
    print("\n6. Testing checkpointing...")
    try:
        trainer.save_checkpoint("test_checkpoint.pt")
        print("   ✅ Checkpoint saved")
        
        # Create new trainer and load checkpoint
        trainer2 = PPOTrainer(
            model=TradingActorCritic(
                price_window_size=60,
                num_indicators=10,
                embedding_dim=384,
                max_news_headlines=20,
                num_actions=3,
            ),
            environment=env,
            device="cpu",
        )
        trainer2.load_checkpoint("test_checkpoint.pt")
        print("   ✅ Checkpoint loaded")
        print(f"      - Training step: {trainer2.training_step}")
    except Exception as e:
        print(f"   ❌ Checkpointing failed: {e}")
        raise
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_training_loop()

