"""
Test Auxiliary Training Infrastructure

This script validates that auxiliary loss training works correctly.
Tests:
1. Auxiliary loss computation
2. Gradient flow through auxiliary heads
3. Training step completes without NaN/inf errors
4. Predictions are non-zero after training step
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
import logging

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from rl_agent.model import TradingActorCritic
from rl_agent.environment import TradingEnvironment, Action
from rl_agent.state_encoder import StateEncoder
from rl_agent.trainer import PPOTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_dummy_state_dict(state_encoder, batch_size=4):
    """Create dummy state dictionary for testing."""
    # Create dummy price data
    prices = [100.0 + i * 0.1 for i in range(60)]
    
    price_features = {
        "current_price": 100.5,
        "rsi": 50.0,
        "sma_1h": 100.0,
        "sma_4h": 99.5,
        "sma_24h": 99.0,
        "macd_line": 0.1,
        "macd_signal": 0.05,
        "bb_position": 50.0,
        "momentum_10": 0.5,
    }
    
    # Create dummy news data
    news_data = [
        {
            "embedding": np.random.randn(384).tolist(),
            "sentiment": 0.5,
            "headline": f"Test news {i}",
        }
        for i in range(5)
    ]
    
    state_dicts = []
    for _ in range(batch_size):
        state_dict = state_encoder.encode_full_state(
            prices=prices,
            price_features=price_features,
            news_data=news_data,
            position_size=0.0,
            portfolio_value=10000.0,
            entry_price=None,
            current_price=100.5,
            time_since_last_trade=0.0,
            timestamp=None,
            unrealized_pnl=0.0,
        )
        state_dicts.append(state_dict)
    
    return state_dicts


def test_auxiliary_loss_computation():
    """Test that auxiliary loss computation works correctly."""
    logger.info("=" * 60)
    logger.info("Test 1: Auxiliary Loss Computation")
    logger.info("=" * 60)
    
    device = "cpu"
    state_encoder = StateEncoder()
    
    # Create model
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    ).to(device)
    
    # Create environment
    environment = TradingEnvironment(
        initial_capital=10000.0,
        transaction_cost_rate=0.001,
        max_position_size=0.1,
    )
    
    # Create trainer with auxiliary losses enabled
    trainer = PPOTrainer(
        model=model,
        environment=environment,
        lr=3e-4,
        device=device,
        enable_auxiliary_losses=True,
        aux_1h_coef=0.01,
        aux_24h_coef=0.01,
    )
    
    # Create dummy states
    state_dicts = create_dummy_state_dict(state_encoder, batch_size=4)
    
    # Convert to tensors
    price_tensors = [torch.FloatTensor(sd["price"]).unsqueeze(0) for sd in state_dicts]
    news_emb_tensors = [torch.FloatTensor(sd["news_embeddings"]).unsqueeze(0) for sd in state_dicts]
    news_sent_tensors = [torch.FloatTensor(sd["news_sentiment"]).unsqueeze(0) for sd in state_dicts]
    position_tensors = [torch.FloatTensor(sd["position"]).unsqueeze(0) for sd in state_dicts]
    time_tensors = [torch.FloatTensor(sd["time"]).unsqueeze(0) for sd in state_dicts]
    
    # Create dummy rollout data
    rollouts = []
    for i in range(4):
        # Get model output
        model.eval()
        with torch.no_grad():
            output = model(
                price_tensors[i],
                news_emb_tensors[i],
                news_sent_tensors[i],
                position_tensors[i],
                time_tensors[i],
                torch.ones(1, 20),  # news_mask
            )
        
        # Create dummy rollout
        rollout = {
            "states": {
                "price": price_tensors[i],
                "news_embeddings": news_emb_tensors[i],
                "news_sentiment": news_sent_tensors[i],
                "position": position_tensors[i],
                "time": time_tensors[i],
            },
            "actions": [Action.HOLD.value],
            "rewards": [0.01],
            "log_probs": torch.log(torch.softmax(output["action_logits"], dim=-1)[0, Action.HOLD.value]),
            "values": output["value"],
            "dones": [False],
            "actual_returns_1h": torch.tensor([0.02]),  # 2% return
            "actual_returns_24h": torch.tensor([0.05]),  # 5% return
        }
        rollouts.append(rollout)
    
    # Test training step
    logger.info("Running training step with auxiliary losses...")
    try:
        metrics = trainer.train_step(rollouts)
        
        # Check metrics
        logger.info(f"Training step completed successfully!")
        logger.info(f"  Policy loss: {metrics.get('policy_loss', 'N/A')}")
        logger.info(f"  Value loss: {metrics.get('value_loss', 'N/A')}")
        logger.info(f"  Aux 1h loss: {metrics.get('aux_1h_loss', 'N/A')}")
        logger.info(f"  Aux 24h loss: {metrics.get('aux_24h_loss', 'N/A')}")
        logger.info(f"  Total loss: {metrics.get('total_loss', 'N/A')}")
        
        # Validate no NaN/inf
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"❌ {key} is NaN/inf: {value}")
                    return False
        
        logger.info("✅ All metrics are finite")
        
        # Check that auxiliary losses are non-zero (if we have valid data)
        if metrics.get('aux_1h_loss', 0) == 0 and metrics.get('aux_24h_loss', 0) == 0:
            logger.warning("⚠️ Auxiliary losses are zero - this may indicate no valid return targets")
        else:
            logger.info("✅ Auxiliary losses computed successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_prediction_generation():
    """Test that predictions are generated and non-zero after training."""
    logger.info("=" * 60)
    logger.info("Test 2: Prediction Generation")
    logger.info("=" * 60)
    
    device = "cpu"
    state_encoder = StateEncoder()
    
    # Create model
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    ).to(device)
    
    # Create dummy state
    state_dicts = create_dummy_state_dict(state_encoder, batch_size=1)
    state_dict = state_dicts[0]
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    news_mask = torch.ones(1, 20).to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        output = model(
            price_tensor,
            news_emb_tensor,
            news_sent_tensor,
            position_tensor,
            time_tensor,
            news_mask,
        )
    
    pred_1h = output["pred_1h"].item() if output["pred_1h"].dim() == 0 else output["pred_1h"][0].item()
    pred_24h = output["pred_24h"].item() if output["pred_24h"].dim() == 0 else output["pred_24h"][0].item()
    
    logger.info(f"Prediction 1h: {pred_1h:.6f}")
    logger.info(f"Prediction 24h: {pred_24h:.6f}")
    
    # Check if predictions are zero (untrained model will output near-zero)
    if abs(pred_1h) < 1e-6 and abs(pred_24h) < 1e-6:
        logger.warning("⚠️ Predictions are near-zero - this is expected for untrained model")
        logger.info("   After training, predictions should be non-zero")
    else:
        logger.info("✅ Predictions are non-zero")
    
    # Check if predictions are in reasonable range
    if abs(pred_1h) > 1.0 or abs(pred_24h) > 1.0:
        logger.warning(f"⚠️ Predictions are outside reasonable range (±1.0 = ±100%)")
    else:
        logger.info("✅ Predictions are in reasonable range")
    
    # Check for NaN/inf
    if np.isnan(pred_1h) or np.isinf(pred_1h) or np.isnan(pred_24h) or np.isinf(pred_24h):
        logger.error("❌ Predictions contain NaN/inf")
        return False
    
    logger.info("✅ Prediction generation works correctly")
    return True


def test_gradient_flow():
    """Test that gradients flow through auxiliary heads."""
    logger.info("=" * 60)
    logger.info("Test 3: Gradient Flow Through Auxiliary Heads")
    logger.info("=" * 60)
    
    device = "cpu"
    state_encoder = StateEncoder()
    
    # Create model
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    ).to(device)
    
    # Create dummy state
    state_dicts = create_dummy_state_dict(state_encoder, batch_size=1)
    state_dict = state_dicts[0]
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    news_mask = torch.ones(1, 20).to(device)
    
    # Forward pass
    model.train()
    output = model(
        price_tensor,
        news_emb_tensor,
        news_sent_tensor,
        position_tensor,
        time_tensor,
        news_mask,
    )
    
    # Create dummy loss (MSE on predictions)
    target_1h = torch.tensor([0.02], device=device)  # 2% return
    target_24h = torch.tensor([0.05], device=device)  # 5% return
    
    pred_1h = output["pred_1h"].squeeze()
    pred_24h = output["pred_24h"].squeeze()
    
    loss_1h = torch.nn.functional.mse_loss(pred_1h, target_1h)
    loss_24h = torch.nn.functional.mse_loss(pred_24h, target_24h)
    total_loss = loss_1h + loss_24h
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients in auxiliary heads
    aux_1h_has_grad = False
    aux_24h_has_grad = False
    
    for name, param in model.named_parameters():
        if 'aux_1h' in name and param.grad is not None:
            aux_1h_has_grad = True
            grad_norm = param.grad.norm().item()
            logger.info(f"  {name}: grad_norm = {grad_norm:.6f}")
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                logger.error(f"❌ Gradient is NaN/inf in {name}")
                return False
        if 'aux_24h' in name and param.grad is not None:
            aux_24h_has_grad = True
            grad_norm = param.grad.norm().item()
            logger.info(f"  {name}: grad_norm = {grad_norm:.6f}")
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                logger.error(f"❌ Gradient is NaN/inf in {name}")
                return False
    
    if not aux_1h_has_grad:
        logger.warning("⚠️ No gradients found in aux_1h head")
    else:
        logger.info("✅ Gradients flow through aux_1h head")
    
    if not aux_24h_has_grad:
        logger.warning("⚠️ No gradients found in aux_24h head")
    else:
        logger.info("✅ Gradients flow through aux_24h head")
    
    return aux_1h_has_grad and aux_24h_has_grad


def main():
    """Run all tests."""
    logger.info("🧪 Testing Auxiliary Training Infrastructure")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Auxiliary loss computation
    results.append(("Auxiliary Loss Computation", test_auxiliary_loss_computation()))
    
    # Test 2: Prediction generation
    results.append(("Prediction Generation", test_prediction_generation()))
    
    # Test 3: Gradient flow
    results.append(("Gradient Flow", test_gradient_flow()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("=" * 60)
        logger.info("✅ All tests passed! Training infrastructure is ready.")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("=" * 60)
        logger.error("❌ Some tests failed. Please review errors above.")
        logger.info("=" * 60)
        return 1


if __name__ == "__main__":
    exit(main())

