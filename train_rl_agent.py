"""
Train RL Agent on Historical Data

This script trains the RL agent using historical price and news data.
It loads preprocessed training episodes and trains the model using PPO.
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
from pathlib import Path
import pickle
import argparse

from rl_agent.model import TradingActorCritic
from rl_agent.environment import TradingEnvironment
from rl_agent.state_encoder import StateEncoder
from rl_agent.trainer import PPOTrainer
from rl_agent.training_data_prep import TrainingDataPrep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_on_historical_data(
    episodes_path: str = "training_data/episodes.pkl",
    num_epochs: int = 10,
    batch_size: int = 32,
    checkpoint_dir: str = "models/rl_agent",
    device: str = "cpu",
    resume_from: Optional[str] = None,
):
    """
    Train RL agent on historical data.
    
    Args:
        episodes_path: Path to preprocessed training episodes
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_dir: Directory to save model checkpoints
        device: Device to use ('cpu' or 'cuda')
        resume_from: Path to checkpoint to resume from
    """
    logger.info("=" * 60)
    logger.info("RL Agent Training on Historical Data")
    logger.info("=" * 60)
    
    # Load training episodes
    logger.info(f"Loading training episodes from {episodes_path}")
    if not Path(episodes_path).exists():
        logger.error(f"Episodes file not found: {episodes_path}")
        logger.info("Run training_data_prep.py first to create episodes")
        return
    
    with open(episodes_path, 'rb') as f:
        episodes = pickle.load(f)
    
    logger.info(f"Loaded {len(episodes)} training episodes")
    
    # Initialize components
    logger.info("Initializing model and environment...")
    
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
    
    model = TradingActorCritic(
        price_window_size=60,
        num_indicators=10,
        embedding_dim=384,
        max_news_headlines=20,
        num_actions=3,
    )
    
    # Load checkpoint if resuming
    if resume_from and Path(resume_from).exists():
        logger.info(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded")
    
    trainer = PPOTrainer(
        model=model,
        environment=environment,
        lr=3e-4,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    
    logger.info("✅ Components initialized")
    
    # Training loop
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    
    total_steps = 0
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        epoch_losses = []
        
        # Train on each episode
        for ep_idx, episode in enumerate(episodes):
            if ep_idx % 10 == 0:
                logger.info(f"Processing episode {ep_idx + 1}/{len(episodes)}")
            
            # Reset environment
            obs = environment.reset()
            position_size = obs["position_size"]
            portfolio_value = obs["portfolio_value"]
            entry_price = obs.get("entry_price")
            time_since_last_trade = obs["time_since_last_trade"]
            
            # Process episode steps
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            episode_dones = []
            episode_pred_1h = []
            episode_pred_24h = []
            episode_returns_1h = []
            episode_returns_24h = []
            
            for step in range(len(episode["prices"])):
                timestamp = episode["timestamps"][step]
                current_price = episode["prices"][step]
                price_features = episode["price_features"][step]
                news_data = episode["news_data"][step]
                
                # Get price window for state encoding
                price_window_start = max(0, step - 60)
                price_window = episode["prices"][price_window_start:step+1]
                
                # Encode state
                state_dict = state_encoder.encode_full_state(
                    prices=price_window,
                    price_features=price_features,
                    news_data=news_data,
                    position_size=position_size,
                    portfolio_value=portfolio_value,
                    entry_price=entry_price,
                    current_price=current_price,
                    time_since_last_trade=time_since_last_trade,
                    timestamp=timestamp,
                    unrealized_pnl=0.0,  # Simplified
                )
                
                # Convert to tensors and validate
                price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
                news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
                news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
                position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
                time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
                news_mask = (news_sent_tensor != 0.0).float()
                
                # Validate inputs (replace NaN/inf with 0)
                price_tensor = torch.where(torch.isfinite(price_tensor), price_tensor, torch.zeros_like(price_tensor))
                news_emb_tensor = torch.where(torch.isfinite(news_emb_tensor), news_emb_tensor, torch.zeros_like(news_emb_tensor))
                news_sent_tensor = torch.where(torch.isfinite(news_sent_tensor), news_sent_tensor, torch.zeros_like(news_sent_tensor))
                position_tensor = torch.where(torch.isfinite(position_tensor), position_tensor, torch.zeros_like(position_tensor))
                time_tensor = torch.where(torch.isfinite(time_tensor), time_tensor, torch.zeros_like(time_tensor))
                
                # Get action from model (use deterministic=False for training)
                try:
                    action, output = model.get_action(
                        price_tensor, news_emb_tensor, news_sent_tensor,
                        position_tensor, time_tensor, news_mask,
                        deterministic=False
                    )
                except Exception as e:
                    logger.warning(f"Error getting action at step {step}: {e}, using random action")
                    action = np.random.randint(0, 3)  # Random action as fallback
                    # Create dummy output
                    output = {
                        "value": torch.tensor(0.0),
                        "pred_1h": torch.tensor(0.0),
                        "pred_24h": torch.tensor(0.0),
                        "action_logits": torch.zeros(1, 3),
                    }
                
                # Calculate reward from future prices
                future_price_1h = episode["future_prices_1h"][step]
                future_price_24h = episode["future_prices_24h"][step]
                
                if future_price_1h and future_price_24h:
                    # Calculate returns
                    return_1h = (future_price_1h - current_price) / current_price if current_price > 0 else 0.0
                    return_24h = (future_price_24h - current_price) / current_price if current_price > 0 else 0.0
                    
                    # Calculate reward based on action
                    if action == 2:  # BUY
                        reward = return_24h - 0.001  # Transaction cost
                    elif action == 0:  # SELL
                        reward = -return_24h - 0.001
                    else:  # HOLD
                        reward = 0.0
                else:
                    reward = 0.0
                    return_1h = 0.0
                    return_24h = 0.0
                
                # Store experience
                action_probs = torch.softmax(output["action_logits"], dim=-1)
                log_prob = torch.log(action_probs[0, action] + 1e-8)
                
                episode_states.append(state_dict)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(output["value"].item())
                episode_log_probs.append(log_prob.item())
                episode_dones.append(False)  # Not terminal within episode
                episode_pred_1h.append(output["pred_1h"].item())
                episode_pred_24h.append(output["pred_24h"].item())
                episode_returns_1h.append(return_1h)
                episode_returns_24h.append(return_24h)
                
                # Update position state (simplified)
                if action == 2:  # BUY
                    position_size = 0.1  # Simplified
                    entry_price = current_price
                elif action == 0:  # SELL
                    position_size = 0.0
                    entry_price = None
                
                time_since_last_trade += 5.0  # 5 minutes per step
                total_steps += 1
            
            # Add episode to buffer
            trainer.buffer["states"].extend(episode_states)
            trainer.buffer["actions"].extend(episode_actions)
            trainer.buffer["rewards"].extend(episode_rewards)
            trainer.buffer["values"].extend(episode_values)
            trainer.buffer["log_probs"].extend(episode_log_probs)
            trainer.buffer["dones"].extend(episode_dones)
            trainer.buffer["pred_1h"].extend(episode_pred_1h)
            trainer.buffer["pred_24h"].extend(episode_pred_24h)
            trainer.buffer["returns_1h"].extend(episode_returns_1h)
            trainer.buffer["returns_24h"].extend(episode_returns_24h)
            
            # Train when buffer is large enough
            if len(trainer.buffer["states"]) >= batch_size * 4:
                logger.info(f"Training on {len(trainer.buffer['states'])} experiences...")
                
                # Compute GAE for the buffer
                advantages, returns = trainer._compute_gae()
                
                # Create rollout data
                rollout_data = {
                    "advantages": advantages,
                    "returns": returns,
                    "buffer": trainer.buffer.copy(),
                }
                
                # Train step
                loss_dict = trainer.train_step(rollout_data, num_epochs=4, batch_size=batch_size)
                epoch_losses.append(loss_dict)
                
                # Clear buffer
                trainer.buffer = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "values": [],
                    "log_probs": [],
                    "dones": [],
                    "pred_1h": [],
                    "pred_24h": [],
                    "returns_1h": [],
                    "returns_24h": [],
                }
        
        # Epoch summary
        if epoch_losses:
            avg_loss = {
                k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0].keys()
            }
            logger.info(f"\nEpoch {epoch + 1} Average Losses:")
            for key, value in avg_loss.items():
                logger.info(f"  {key}: {value:.6f}")
        
        # Save checkpoint
        checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pt"
        trainer.save_checkpoint(str(checkpoint_path), epoch=epoch + 1, total_steps=total_steps)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info(f"Total steps: {total_steps}")
    logger.info(f"Final checkpoint: {checkpoint_dir}/checkpoint_epoch_{num_epochs}.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL Agent on Historical Data")
    parser.add_argument(
        "--episodes",
        type=str,
        default="training_data/episodes.pkl",
        help="Path to training episodes file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/rl_agent",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    train_on_historical_data(
        episodes_path=args.episodes,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        resume_from=args.resume,
    )

