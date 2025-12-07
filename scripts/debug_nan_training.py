"""
Debug script to identify where NaN originates during training.

This will run a single training step and log detailed information
about where NaN appears.
"""

import torch
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from rl_agent.model import TradingActorCritic
from rl_agent.trainer import PPOTrainer
from rl_agent.environment import TradingEnvironment
from rl_agent.state_encoder import StateEncoder
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading training episodes...")
    episodes_path = os.path.join(project_root, "training_data", "episodes.pkl")
    
    with open(episodes_path, 'rb') as f:
        episodes = pickle.load(f)
    
    logger.info(f"Loaded {len(episodes)} episodes")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    )
    
    # Check if model has NaN weights
    logger.info("Checking model weights for NaN...")
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        logger.error(f"Model has NaN in {len(nan_params)} parameters!")
        for name in nan_params[:10]:
            logger.error(f"  - {name}")
    else:
        logger.info("✅ Model weights are clean (no NaN)")
    
    # Initialize other components
    state_encoder = StateEncoder(
        price_window_size=60,
        max_news_headlines=20,
        embedding_dim=384,
    )
    
    environment = TradingEnvironment(
        initial_capital=10000.0,
        transaction_cost_rate=0.001,
        max_position_size=0.1,
    )
    
    trainer = PPOTrainer(
        model=model,
        environment=environment,
        lr=3e-4,
        device="cpu",
        enable_auxiliary_losses=False,  # Disable to isolate issue
        max_grad_norm=0.5,
    )
    
    # Get first episode
    if not episodes:
        logger.error("No episodes available!")
        return
    
    episode = episodes[0]
    logger.info(f"Using episode with {len(episode['prices'])} steps")
    
    # Collect a small rollout
    logger.info("Collecting rollout...")
    price_data = episode['prices']
    price_features_list = episode['price_features']
    news_data_list = episode['news_data']
    
    # Collect just 10 steps for testing
    for step in range(min(10, len(price_data))):
        current_price = price_data[step]
        price_features = price_features_list[step]
        news_data = news_data_list[step]
        
        # Get price window
        price_window_start = max(0, step - 60)
        price_window = price_data[price_window_start:step+1]
        
        # Encode state
        state_dict = state_encoder.encode_full_state(
            prices=price_window,
            price_features=price_features,
            news_data=news_data,
            position_size=0.0,
            portfolio_value=10000.0,
            entry_price=None,
            current_price=current_price,
            time_since_last_trade=0.0,
            timestamp=episode['timestamps'][step],
            unrealized_pnl=0.0,
        )
        
        # Check state for NaN
        state_has_nan = False
        for key, value in state_dict.items():
            if isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    logger.error(f"State {key} has NaN/inf at step {step}!")
                    state_has_nan = True
        
        if state_has_nan:
            logger.error(f"Skipping step {step} due to NaN in state")
            continue
        
        # Convert to tensors
        price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0)
        news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0)
        news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0)
        position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0)
        time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0)
        news_mask = (news_sent_tensor != 0.0).float()
        
        # Test forward pass
        logger.info(f"Testing forward pass for step {step}...")
        model.eval()
        with torch.no_grad():
            try:
                output = model(
                    price_tensor,
                    news_emb_tensor,
                    news_sent_tensor,
                    position_tensor,
                    time_tensor,
                    news_mask,
                )
                
                # Check outputs
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor):
                        if not torch.isfinite(tensor).all():
                            logger.error(f"  Model output {key} has NaN/inf!")
                        else:
                            logger.info(f"  {key}: OK (min={tensor.min().item():.6f}, max={tensor.max().item():.6f})")
            except Exception as e:
                logger.error(f"  Forward pass failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return
    
    logger.info("✅ Forward pass test complete")
    
    # Now try training step
    logger.info("\n" + "="*60)
    logger.info("Testing training step...")
    logger.info("="*60)
    
    # Collect full rollout
    rollout = trainer.collect_rollout(
        state_encoder=state_encoder,
        price_data=price_data[:100],  # Just first 100 prices
        price_features=price_features_list[0],
        news_data=news_data_list[0],
        num_steps=10,  # Small rollout
    )
    
    logger.info(f"Collected rollout with {len(rollout['advantages'])} steps")
    
    # Check rollout data
    advantages = np.array(rollout['advantages'])
    returns = np.array(rollout['returns'])
    
    logger.info(f"Advantages: min={advantages.min():.6f}, max={advantages.max():.6f}, "
              f"mean={advantages.mean():.6f}, has_nan={np.isnan(advantages).any()}")
    logger.info(f"Returns: min={returns.min():.6f}, max={returns.max():.6f}, "
              f"mean={returns.mean():.6f}, has_nan={np.isnan(returns).any()}")
    
    if np.isnan(advantages).any() or np.isnan(returns).any():
        logger.error("❌ Rollout data has NaN - this is the problem!")
        return
    
    # Try training step
    try:
        metrics = trainer.train_step(rollout, num_epochs=1, batch_size=4)
        logger.info(f"✅ Training step completed: {metrics}")
    except Exception as e:
        logger.error(f"❌ Training step failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()

