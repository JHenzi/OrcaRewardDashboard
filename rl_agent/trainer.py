"""
PPO Trainer

Proximal Policy Optimization trainer for the trading agent.
Implements PPO with GAE (Generalized Advantage Estimation) and
auxiliary losses for multi-horizon return prediction.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from collections import deque
import json
from pathlib import Path
import os
from datetime import datetime

from .model import TradingActorCritic
from .environment import TradingEnvironment, Action

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    PPO trainer for trading agent.
    
    Features:
    - PPO with clipped objective
    - GAE for advantage estimation
    - Auxiliary losses for 1h/24h return prediction
    - Experience replay buffer
    - Model checkpointing
    """
    
    def __init__(
        self,
        model: TradingActorCritic,
        environment: TradingEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,  # Discount factor
        gae_lambda: float = 0.95,  # GAE lambda
        clip_epsilon: float = 0.2,  # PPO clip epsilon
        value_coef: float = 0.5,  # Value loss coefficient
        entropy_coef: float = 0.01,  # Entropy bonus coefficient
        aux_1h_coef: float = 0.1,  # Auxiliary 1h loss coefficient (increased from 0.01 for better training)
        aux_24h_coef: float = 0.1,  # Auxiliary 24h loss coefficient (increased from 0.01 for better training)
        enable_auxiliary_losses: bool = True,  # Can disable if causing issues
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        checkpoint_dir: str = "models/rl_agent",
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model: TradingActorCritic model
            environment: TradingEnvironment
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            aux_1h_coef: Auxiliary 1h loss coefficient
            aux_24h_coef: Auxiliary 24h loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use ('cpu' or 'cuda')
            checkpoint_dir: Directory for model checkpoints
        """
        self.model = model.to(device)
        self.env = environment
        self.device = device
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.aux_1h_coef = aux_1h_coef
        self.aux_24h_coef = aux_24h_coef
        self.enable_auxiliary_losses = enable_auxiliary_losses
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Experience buffer
        self.buffer = {
            "states": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
            "pred_1h": [],  # Predicted 1h returns
            "pred_24h": [],  # Predicted 24h returns
            "returns_1h": [],  # Actual 1h returns (filled later)
            "returns_24h": [],  # Actual 24h returns (filled later)
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_step = 0
        
        # Store rollout data for later updates (when 1h/24h returns become available)
        self.pending_rollouts = []
        
        # Track NaN gradient occurrences for adaptive coefficient reduction
        self.aux_nan_count = 0
        self.aux_nan_threshold = 50  # Reduce coefficients after 50 NaN occurrences
        self.original_aux_1h_coef = aux_1h_coef
        self.original_aux_24h_coef = aux_24h_coef
        
    def collect_rollout(
        self,
        state_encoder,
        price_data: List[float],
        price_features: Dict[str, float],
        news_data: List[Dict],
        num_steps: int = 128,
    ) -> Dict:
        """
        Collect a rollout of experiences.
        
        Args:
            state_encoder: StateEncoder instance
            price_data: List of recent prices
            price_features: Dict of technical indicators
            news_data: List of news dicts with embeddings
            num_steps: Number of steps to collect
            
        Returns:
            Dict with rollout data
        """
        self.model.eval()
        self.buffer = {key: [] for key in self.buffer.keys()}
        
        obs = self.env.reset()
        current_price = price_data[-1] if price_data else 100.0
        
        for step in range(num_steps):
            # Encode state
            state_dict = state_encoder.encode_full_state(
                prices=price_data[-60:] if len(price_data) >= 60 else price_data,
                price_features=price_features,
                news_data=news_data,
                position_size=obs["position_size"],
                portfolio_value=obs["portfolio_value"],
                entry_price=obs.get("entry_price"),
                current_price=current_price,
                time_since_last_trade=obs["time_since_last_trade"],
                timestamp=obs.get("timestamp", datetime.now()),
                unrealized_pnl=obs.get("unrealized_pnl", 0.0),
            )
            
            # Convert to tensors
            price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(self.device)
            news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(self.device)
            news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(self.device)
            position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(self.device)
            time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(self.device)
            
            # Create mask for news (1 for valid headlines)
            news_mask = (news_sent_tensor != 0.0).float()
            
            # Get action
            action, output = self.model.get_action(
                price_tensor, news_emb_tensor, news_sent_tensor,
                position_tensor, time_tensor, news_mask
            )
            
            # Get action probability
            action_probs = torch.softmax(output["action_logits"], dim=-1)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            
            # Store in buffer
            self.buffer["states"].append(state_dict)
            self.buffer["actions"].append(action)
            self.buffer["values"].append(output["value"].item())
            self.buffer["log_probs"].append(log_prob.item())
            self.buffer["pred_1h"].append(output["pred_1h"].item())
            self.buffer["pred_24h"].append(output["pred_24h"].item())
            
            # Step environment
            next_price = price_data[step] if step < len(price_data) else current_price
            obs, reward, done, info = self.env.step(action, current_price, next_price)
            
            self.buffer["rewards"].append(reward)
            self.buffer["dones"].append(done)
            
            if done:
                obs = self.env.reset()
            
            current_price = next_price
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae()
        
        return {
            "advantages": advantages,
            "returns": returns,
            "buffer": self.buffer,
        }
    
    def _compute_gae(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Returns:
            Tuple of (advantages, returns)
        """
        rewards = np.array(self.buffer["rewards"])
        values = np.array(self.buffer["values"])
        dones = np.array(self.buffer["dones"])
        
        # Validate inputs - replace NaN/inf with 0
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for empty buffer
        if len(rewards) == 0:
            logger.warning("Empty buffer in GAE computation")
            return np.array([]), np.array([])
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Add bootstrap value for last step
        next_value = 0.0  # Terminal state value is 0
        if len(values) > 0 and not dones[-1]:
            # Use last value as bootstrap if not done
            next_value = values[-1]
            next_value = np.nan_to_num(next_value, nan=0.0, posinf=0.0, neginf=0.0)
        
        last_advantage = 0
        last_return = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                # Terminal state: no bootstrap
                delta = rewards[t] - values[t]
                last_advantage = 0
                last_return = values[t]
            else:
                # Non-terminal: bootstrap with next value
                next_val = values[t + 1] if t + 1 < len(values) else next_value
                next_val = np.nan_to_num(next_val, nan=0.0, posinf=0.0, neginf=0.0)
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
                last_return = rewards[t] + self.gamma * last_return
            
            # Ensure no NaN/inf
            last_advantage = np.nan_to_num(last_advantage, nan=0.0, posinf=0.0, neginf=0.0)
            last_return = np.nan_to_num(last_return, nan=0.0, posinf=0.0, neginf=0.0)
            
            advantages[t] = last_advantage
            returns[t] = last_return
        
        # Normalize advantages (only if std > 0)
        if len(advantages) > 0 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = np.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)
        
        return advantages, returns
    
    def _batch_states(self, states: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Batch a list of state dictionaries into tensors.
        
        Args:
            states: List of state dicts from StateEncoder
            
        Returns:
            Dict of batched tensors
        """
        batch_size = len(states)
        
        # Stack each component
        price_features = torch.stack([
            torch.FloatTensor(s["price"]) for s in states
        ]).to(self.device)
        
        news_embeddings = torch.stack([
            torch.FloatTensor(s["news_embeddings"]) for s in states
        ]).to(self.device)
        
        news_sentiment = torch.stack([
            torch.FloatTensor(s["news_sentiment"]) for s in states
        ]).to(self.device)
        
        position_features = torch.stack([
            torch.FloatTensor(s["position"]) for s in states
        ]).to(self.device)
        
        time_features = torch.stack([
            torch.FloatTensor(s["time"]) for s in states
        ]).to(self.device)
        
        # Create news mask (1 for valid headlines, 0 for padding)
        news_mask = (news_sentiment != 0.0).float()
        
        # CRITICAL: Validate and fix NaN/inf in all batched tensors
        price_features = torch.where(torch.isfinite(price_features), price_features, torch.zeros_like(price_features))
        news_embeddings = torch.where(torch.isfinite(news_embeddings), news_embeddings, torch.zeros_like(news_embeddings))
        news_sentiment = torch.where(torch.isfinite(news_sentiment), news_sentiment, torch.zeros_like(news_sentiment))
        position_features = torch.where(torch.isfinite(position_features), position_features, torch.zeros_like(position_features))
        time_features = torch.where(torch.isfinite(time_features), time_features, torch.zeros_like(time_features))
        news_mask = torch.where(torch.isfinite(news_mask), news_mask, torch.zeros_like(news_mask))
        
        return {
            "price_features": price_features,
            "news_embeddings": news_embeddings,
            "news_sentiment": news_sentiment,
            "position_features": position_features,
            "time_features": time_features,
            "news_mask": news_mask,
        }
    
    def train_step(
        self,
        rollout_data: Dict,
        num_epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Perform one training step (multiple epochs over rollout data).
        
        Args:
            rollout_data: Data from collect_rollout()
            num_epochs: Number of epochs to train on rollout
            batch_size: Batch size for training
            
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        # CRITICAL: Check if model weights already have NaN before training
        model_has_nan = False
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param.data).all():
                logger.error(f"CRITICAL: Model parameter {name} already has NaN weights before training!")
                model_has_nan = True
                # Reinitialize this parameter
                if 'weight' in name:
                    if 'shared' in name or 'actor' in name or 'critic' in name:
                        nn.init.xavier_uniform_(param.data, gain=1.0)
                    elif 'aux' in name:
                        nn.init.xavier_uniform_(param.data, gain=0.5)
                    else:
                        nn.init.xavier_uniform_(param.data, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)
                logger.warning(f"Reinitialized {name} to fix NaN weights")
        
        if model_has_nan:
            logger.error("Model had NaN weights - reinitialized corrupted parameters. Training may be unstable.")
        
        # Validate and convert to tensors
        advantages_array = np.array(rollout_data["advantages"])
        returns_array = np.array(rollout_data["returns"])
        log_probs_array = np.array(self.buffer["log_probs"])
        
        # Replace NaN/inf with 0
        advantages_array = np.nan_to_num(advantages_array, nan=0.0, posinf=0.0, neginf=0.0)
        returns_array = np.nan_to_num(returns_array, nan=0.0, posinf=0.0, neginf=0.0)
        log_probs_array = np.nan_to_num(log_probs_array, nan=-10.0, posinf=-10.0, neginf=-10.0)  # Use -10 for log probs
        
        advantages = torch.FloatTensor(advantages_array).to(self.device)
        returns = torch.FloatTensor(returns_array).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs_array).to(self.device)
        actions = torch.LongTensor(self.buffer["actions"]).to(self.device)
        
        # Check for empty or invalid data
        if len(advantages) == 0 or len(returns) == 0:
            logger.warning("Empty advantages or returns, skipping training step")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "aux_1h_loss": 0.0,
                "aux_24h_loss": 0.0,
                "clip_fraction": 0.0,
                "total_loss": 0.0,
            }
        
        # Get actual returns for auxiliary losses (if available)
        actual_returns_1h = None
        actual_returns_24h = None
        if len(self.buffer["returns_1h"]) > 0 and len(self.buffer["returns_1h"]) == len(actions):
            returns_1h_array = np.array(self.buffer["returns_1h"])
            returns_1h_array = np.nan_to_num(returns_1h_array, nan=0.0, posinf=0.0, neginf=0.0)
            actual_returns_1h = torch.FloatTensor(returns_1h_array).to(self.device)
        if len(self.buffer["returns_24h"]) > 0 and len(self.buffer["returns_24h"]) == len(actions):
            returns_24h_array = np.array(self.buffer["returns_24h"])
            returns_24h_array = np.nan_to_num(returns_24h_array, nan=0.0, posinf=0.0, neginf=0.0)
            actual_returns_24h = torch.FloatTensor(returns_24h_array).to(self.device)
        
        states = self.buffer["states"]
        num_samples = len(states)
        
        # Create indices for shuffling
        indices = np.arange(num_samples)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_aux_1h_loss = 0.0
        total_aux_24h_loss = 0.0
        total_clip_fraction = 0.0
        
        for epoch in range(num_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Train in batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = [states[i] for i in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_actions = actions[batch_indices]
                
                # CRITICAL: Validate state data before batching
                # Check if any state contains NaN/inf in its numpy arrays
                state_has_nan = False
                for state in batch_states:
                    for key, value in state.items():
                        if isinstance(value, np.ndarray):
                            if not np.isfinite(value).all():
                                logger.error(f"CRITICAL: State {key} contains NaN/inf in raw state data! Skipping batch.")
                                state_has_nan = True
                                break
                    if state_has_nan:
                        break
                
                if state_has_nan:
                    continue
                
                # Batch states
                batched = self._batch_states(batch_states)
                
                # CRITICAL: Validate ALL inputs before forward pass
                input_has_nan = False
                for key, tensor in batched.items():
                    if isinstance(tensor, torch.Tensor) and not torch.isfinite(tensor).all():
                        logger.error(f"CRITICAL: Input {key} contains NaN/inf before forward pass! Skipping batch.")
                        input_has_nan = True
                        break
                
                if input_has_nan:
                    continue
                
                # Validate batch data (advantages, returns, etc.)
                if not torch.isfinite(batch_advantages).all():
                    logger.error(f"CRITICAL: batch_advantages contains NaN/inf! Skipping batch.")
                    continue
                if not torch.isfinite(batch_returns).all():
                    logger.error(f"CRITICAL: batch_returns contains NaN/inf! Skipping batch.")
                    continue
                if not torch.isfinite(batch_old_log_probs).all():
                    logger.error(f"CRITICAL: batch_old_log_probs contains NaN/inf! Skipping batch.")
                    continue
                
                # Forward pass with error handling
                try:
                    output = self.model(
                        batched["price_features"],
                        batched["news_embeddings"],
                        batched["news_sentiment"],
                        batched["position_features"],
                        batched["time_features"],
                        batched["news_mask"],
                    )
                except Exception as e:
                    logger.error(f"CRITICAL: Model forward pass failed: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                # CRITICAL: Validate ALL model outputs immediately
                # If ANY output has NaN, the backward pass will produce NaN in ALL layers
                output_has_nan = False
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor):
                        if not torch.isfinite(tensor).all():
                            logger.error(f"CRITICAL: Model output {key} contains NaN/inf!")
                            logger.error(f"  {key} stats: min={tensor.min().item() if tensor.numel() > 0 else 'N/A'}, "
                                       f"max={tensor.max().item() if tensor.numel() > 0 else 'N/A'}, "
                                       f"finite_count={torch.isfinite(tensor).sum().item()}/{tensor.numel()}")
                            output_has_nan = True
                
                if output_has_nan:
                    logger.error("CRITICAL: Model forward pass produced NaN - skipping batch to prevent NaN gradients in ALL layers")
                    # Check if model weights have NaN
                    model_has_nan = False
                    for name, param in self.model.named_parameters():
                        if not torch.isfinite(param.data).all():
                            logger.error(f"  Model parameter {name} has NaN weights!")
                            model_has_nan = True
                    if model_has_nan:
                        logger.error("  Model weights have NaN - this will cause NaN in all forward passes!")
                    continue
                if not torch.isfinite(output["pred_1h"]).all():
                    logger.warning("Model output pred_1h contains NaN/inf, skipping batch")
                    # Zero out NaN predictions to prevent gradient issues
                    output["pred_1h"] = torch.where(
                        torch.isfinite(output["pred_1h"]),
                        output["pred_1h"],
                        torch.zeros_like(output["pred_1h"])
                    )
                if not torch.isfinite(output["pred_24h"]).all():
                    logger.warning("Model output pred_24h contains NaN/inf, skipping batch")
                    # Zero out NaN predictions to prevent gradient issues
                    output["pred_24h"] = torch.where(
                        torch.isfinite(output["pred_24h"]),
                        output["pred_24h"],
                        torch.zeros_like(output["pred_24h"])
                    )
                
                # Get action probabilities
                action_logits = output["action_logits"]
                action_probs = torch.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs + 1e-8)
                
                # Validate log_probs before using
                if not torch.isfinite(log_probs).all():
                    logger.error("CRITICAL: log_probs contains NaN/inf after softmax, skipping batch")
                    continue
                
                # Get log prob for taken actions
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Validate action_log_probs
                if not torch.isfinite(action_log_probs).all():
                    logger.error("CRITICAL: action_log_probs contains NaN/inf, skipping batch")
                    continue
                
                # Compute policy loss (PPO clipped objective)
                # CRITICAL: Clamp the difference to prevent exp() overflow
                log_prob_diff = action_log_probs - batch_old_log_probs
                log_prob_diff = torch.clamp(log_prob_diff, min=-10.0, max=10.0)  # Prevent exp() overflow
                ratio = torch.exp(log_prob_diff)
                
                # Validate ratio
                if not torch.isfinite(ratio).all():
                    logger.error("CRITICAL: ratio contains NaN/inf after exp(), skipping batch")
                    continue
                
                # CRITICAL: Clamp advantages to prevent extreme values that could cause NaN
                # Extreme advantages multiplied by ratio can cause overflow
                batch_advantages_clamped = torch.clamp(batch_advantages, min=-100.0, max=100.0)
                
                surr1 = ratio * batch_advantages_clamped
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages_clamped
                
                # Validate surrogates
                if not torch.isfinite(surr1).all() or not torch.isfinite(surr2).all():
                    logger.error("CRITICAL: surr1 or surr2 contains NaN/inf, skipping batch")
                    logger.error(f"  ratio stats: min={ratio.min().item():.6f}, max={ratio.max().item():.6f}, "
                               f"mean={ratio.mean().item():.6f}")
                    logger.error(f"  advantages stats: min={batch_advantages.min().item():.6f}, "
                               f"max={batch_advantages.max().item():.6f}, mean={batch_advantages.mean().item():.6f}")
                    continue
                
                # Clamp surrogates to prevent extreme values
                surr1 = torch.clamp(surr1, min=-1000.0, max=1000.0)
                surr2 = torch.clamp(surr2, min=-1000.0, max=1000.0)
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Final clamp on policy_loss
                if torch.isfinite(policy_loss):
                    policy_loss = torch.clamp(policy_loss, min=-1000.0, max=1000.0)
                
                # Final validation of policy_loss
                if not torch.isfinite(policy_loss):
                    logger.error("CRITICAL: policy_loss is NaN/inf, skipping batch")
                    continue
                
                # Compute clip fraction (for monitoring)
                clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
                
                # Compute value loss
                values = output["value"].squeeze(1)
                
                # Validate values before loss computation
                if not torch.isfinite(values).all():
                    logger.error("CRITICAL: values contains NaN/inf, skipping batch")
                    continue
                
                # Clamp values and returns to prevent extreme MSE
                values_clamped = torch.clamp(values, min=-1000.0, max=1000.0)
                returns_clamped = torch.clamp(batch_returns, min=-1000.0, max=1000.0)
                
                value_loss = nn.functional.mse_loss(values_clamped, returns_clamped)
                
                # Validate value_loss
                if not torch.isfinite(value_loss):
                    logger.error("CRITICAL: value_loss is NaN/inf, skipping batch")
                    continue
                
                # Compute entropy bonus
                entropy = -(action_probs * log_probs).sum(dim=1).mean()
                
                # Validate entropy
                if not torch.isfinite(entropy):
                    logger.error("CRITICAL: entropy is NaN/inf, skipping batch")
                    continue
                
                # Compute auxiliary losses (can be disabled if causing issues)
                aux_1h_loss = 0.0
                aux_24h_loss = 0.0
                
                if self.enable_auxiliary_losses:
                    pred_1h = output["pred_1h"].squeeze(1)
                    pred_24h = output["pred_24h"].squeeze(1)
                    
                    # Validate predictions before computing loss
                    pred_1h = torch.where(torch.isfinite(pred_1h), pred_1h, torch.zeros_like(pred_1h))
                    pred_24h = torch.where(torch.isfinite(pred_24h), pred_24h, torch.zeros_like(pred_24h))
                    
                    if actual_returns_1h is not None:
                        batch_returns_1h = actual_returns_1h[batch_indices]
                        # Validate returns - check for extreme values that could cause issues
                        batch_returns_1h = torch.where(
                            torch.isfinite(batch_returns_1h), 
                            batch_returns_1h, 
                            torch.zeros_like(batch_returns_1h)
                        )
                        # Clip extreme returns to prevent numerical issues
                        batch_returns_1h = torch.clamp(batch_returns_1h, min=-1.0, max=1.0)  # Max ±100% return
                        
                        # Only compute loss if we have valid data
                        if torch.isfinite(pred_1h).all() and torch.isfinite(batch_returns_1h).all():
                            # Clip predictions to reasonable range
                            pred_1h_clipped = torch.clamp(pred_1h, min=-1.0, max=1.0)
                            aux_1h_loss = nn.functional.mse_loss(pred_1h_clipped, batch_returns_1h)
                            # Validate loss
                            if not torch.isfinite(aux_1h_loss):
                                aux_1h_loss = torch.tensor(0.0, device=self.device)
                                logger.warning("aux_1h_loss was NaN/inf, setting to 0")
                            else:
                                # Log auxiliary loss periodically for monitoring
                                if hasattr(self, '_aux_1h_log_counter'):
                                    self._aux_1h_log_counter += 1
                                else:
                                    self._aux_1h_log_counter = 0
                                if self._aux_1h_log_counter % 100 == 0:
                                    pred_std = pred_1h_clipped.std().item()
                                    logger.debug(f"aux_1h_loss: {aux_1h_loss.item():.6f}, pred_mean: {pred_1h_clipped.mean().item():.4f}, pred_std: {pred_std:.4f}, target_mean: {batch_returns_1h.mean().item():.4f}")
                                    if pred_std < 0.001:
                                        logger.warning(f"⚠️ aux_1h predictions have very low variance (std={pred_std:.6f}) - predictions may be constant!")
                        else:
                            # Skip auxiliary loss if data is invalid
                            aux_1h_loss = torch.tensor(0.0, device=self.device)
                            logger.debug("Skipping aux_1h_loss due to invalid data")
                    
                    if actual_returns_24h is not None:
                        batch_returns_24h = actual_returns_24h[batch_indices]
                        # Validate returns - check for extreme values that could cause issues
                        batch_returns_24h = torch.where(
                            torch.isfinite(batch_returns_24h), 
                            batch_returns_24h, 
                            torch.zeros_like(batch_returns_24h)
                        )
                        # Clip extreme returns to prevent numerical issues
                        batch_returns_24h = torch.clamp(batch_returns_24h, min=-1.0, max=1.0)  # Max ±100% return
                        
                        # Only compute loss if we have valid data
                        if torch.isfinite(pred_24h).all() and torch.isfinite(batch_returns_24h).all():
                            # Clip predictions to reasonable range
                            pred_24h_clipped = torch.clamp(pred_24h, min=-1.0, max=1.0)
                            aux_24h_loss = nn.functional.mse_loss(pred_24h_clipped, batch_returns_24h)
                            # Validate loss
                            if not torch.isfinite(aux_24h_loss):
                                aux_24h_loss = torch.tensor(0.0, device=self.device)
                                logger.warning("aux_24h_loss was NaN/inf, setting to 0")
                            else:
                                # Log auxiliary loss periodically for monitoring
                                if hasattr(self, '_aux_24h_log_counter'):
                                    self._aux_24h_log_counter += 1
                                else:
                                    self._aux_24h_log_counter = 0
                                if self._aux_24h_log_counter % 100 == 0:
                                    pred_std = pred_24h_clipped.std().item()
                                    logger.debug(f"aux_24h_loss: {aux_24h_loss.item():.6f}, pred_mean: {pred_24h_clipped.mean().item():.4f}, pred_std: {pred_std:.4f}, target_mean: {batch_returns_24h.mean().item():.4f}")
                                    if pred_std < 0.001:
                                        logger.warning(f"⚠️ aux_24h predictions have very low variance (std={pred_std:.6f}) - predictions may be constant!")
                        else:
                            # Skip auxiliary loss if data is invalid
                            aux_24h_loss = torch.tensor(0.0, device=self.device)
                            logger.debug("Skipping aux_24h_loss due to invalid data")
                else:
                    # Auxiliary losses disabled
                    aux_1h_loss = torch.tensor(0.0, device=self.device)
                    aux_24h_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.aux_1h_coef * aux_1h_loss
                    + self.aux_24h_coef * aux_24h_loss
                )
                
                # Validate total loss before backward
                if not torch.isfinite(total_loss):
                    logger.error(f"CRITICAL: Total loss is NaN/inf: {total_loss}, skipping backward pass")
                    logger.error(f"  policy_loss: {policy_loss.item() if torch.isfinite(policy_loss) else 'NaN'}")
                    logger.error(f"  value_loss: {value_loss.item() if torch.isfinite(value_loss) else 'NaN'}")
                    logger.error(f"  entropy: {entropy.item() if torch.isfinite(entropy) else 'NaN'}")
                    logger.error(f"  aux_1h_loss: {aux_1h_loss.item() if isinstance(aux_1h_loss, torch.Tensor) and torch.isfinite(aux_1h_loss) else aux_1h_loss}")
                    logger.error(f"  aux_24h_loss: {aux_24h_loss.item() if isinstance(aux_24h_loss, torch.Tensor) and torch.isfinite(aux_24h_loss) else aux_24h_loss}")
                    continue
                
                # Log loss components for debugging (first batch only)
                if not hasattr(self, '_logged_loss_components'):
                    logger.info(f"Loss components: policy={policy_loss.item():.6f}, value={value_loss.item():.6f}, "
                              f"entropy={entropy.item():.6f}, aux_1h={aux_1h_loss.item() if isinstance(aux_1h_loss, torch.Tensor) else aux_1h_loss:.6f}, "
                              f"aux_24h={aux_24h_loss.item() if isinstance(aux_24h_loss, torch.Tensor) else aux_24h_loss:.6f}, "
                              f"total={total_loss.item():.6f}")
                    # Also log input statistics
                    logger.info(f"Input stats: advantages mean={batch_advantages.mean().item():.6f} std={batch_advantages.std().item():.6f}, "
                              f"returns mean={batch_returns.mean().item():.6f} std={batch_returns.std().item():.6f}, "
                              f"old_log_probs mean={batch_old_log_probs.mean().item():.6f}")
                    self._logged_loss_components = True
                
                # CRITICAL: Before backward, check if loss or any component is NaN
                # If so, the backward will produce NaN gradients in ALL layers
                loss_components = {
                    'policy_loss': policy_loss.item() if torch.isfinite(policy_loss) else float('nan'),
                    'value_loss': value_loss.item() if torch.isfinite(value_loss) else float('nan'),
                    'entropy': entropy.item() if torch.isfinite(entropy) else float('nan'),
                    'total_loss': total_loss.item() if torch.isfinite(total_loss) else float('nan'),
                }
                
                if any(np.isnan(v) or np.isinf(v) for v in loss_components.values()):
                    logger.error(f"CRITICAL: Loss has NaN/inf BEFORE backward: {loss_components}")
                    logger.error(f"  This will cause NaN gradients in ALL layers!")
                    logger.error(f"  Model outputs - action_logits: finite={torch.isfinite(output['action_logits']).all()}, "
                               f"value: finite={torch.isfinite(output['value']).all()}")
                    logger.error(f"  Batch stats - advantages: min={batch_advantages.min().item():.6f}, "
                               f"max={batch_advantages.max().item():.6f}, "
                               f"returns: min={batch_returns.min().item():.6f}, max={batch_returns.max().item():.6f}")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                
                # Check model weights before backward
                weights_have_nan_before = any(not torch.isfinite(p.data).all() for p in self.model.parameters())
                if weights_have_nan_before:
                    logger.error("CRITICAL: Model weights have NaN BEFORE backward pass!")
                    # Reinitialize all NaN weights
                    for name, param in self.model.named_parameters():
                        if not torch.isfinite(param.data).all():
                            logger.error(f"Reinitializing {name} due to NaN weights")
                            if 'weight' in name:
                                if param.dim() == 2:
                                    nn.init.xavier_uniform_(param.data, gain=0.5)
                                elif param.dim() == 3:
                                    nn.init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
                            elif 'bias' in name:
                                nn.init.constant_(param.data, 0.0)
                    continue  # Skip this batch after reinitializing
                
                try:
                    total_loss.backward()
                except RuntimeError as e:
                    logger.error(f"CRITICAL: RuntimeError during backward: {e}")
                    logger.error(f"  Loss value: {total_loss.item()}")
                    continue
                
                # Check if backward pass produced NaN in loss (shouldn't happen, but check)
                if not torch.isfinite(total_loss):
                    logger.error("CRITICAL: total_loss became NaN during backward pass!")
                    continue
                
                # CRITICAL: Check and fix auxiliary head gradients BEFORE clipping
                # This prevents NaN from propagating and corrupting weights
                aux_grad_issues = False
                for name, param in self.model.named_parameters():
                    if 'aux' in name and param.grad is not None:
                        if not torch.isfinite(param.grad).all():
                            aux_grad_issues = True
                            self.aux_nan_count += 1
                            logger.warning(f"NaN/inf gradient detected in {name} (count: {self.aux_nan_count}), zeroing out")
                            param.grad = torch.where(
                                torch.isfinite(param.grad),
                                param.grad,
                                torch.zeros_like(param.grad)
                            )
                        # Additional protection: clip auxiliary gradients more aggressively
                        # Use smaller max norm for auxiliary heads to prevent instability
                        if param.grad is not None and torch.isfinite(param.grad).all():
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 1.0:  # More aggressive clipping for aux heads
                                param.grad = param.grad / grad_norm * 1.0
                                logger.debug(f"Clipped {name} gradient from {grad_norm:.4f} to 1.0")
                
                # Adaptive coefficient reduction if NaN persists
                if self.aux_nan_count >= self.aux_nan_threshold:
                    reduction_factor = 0.5
                    self.aux_1h_coef = max(0.001, self.aux_1h_coef * reduction_factor)
                    self.aux_24h_coef = max(0.001, self.aux_24h_coef * reduction_factor)
                    logger.warning(
                        f"Reduced auxiliary loss coefficients due to persistent NaN gradients: "
                        f"aux_1h_coef={self.aux_1h_coef:.6f}, aux_24h_coef={self.aux_24h_coef:.6f}"
                    )
                    self.aux_nan_count = 0  # Reset counter after reduction
                
                # If auxiliary gradients had issues, skip updating those parameters this step
                if aux_grad_issues:
                    logger.warning("Auxiliary gradients had NaN/inf - skipping optimizer step for aux heads")
                    # Temporarily disable gradients for auxiliary heads
                    for name, param in self.model.named_parameters():
                        if 'aux' in name:
                            param.grad = None
                
                # Gradient clipping - with special handling for auxiliary heads
                # Clip all gradients (aux heads already handled above)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.grad is not None], 
                    self.max_grad_norm
                )
                
                # Final safety check: verify no NaN gradients remain
                has_any_nan_grad = False
                for name, param in self.model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        logger.error(f"CRITICAL: {name} still has NaN gradient after all fixes! Zeroing completely.")
                        param.grad = torch.zeros_like(param.grad)
                        has_any_nan_grad = True
                
                # CRITICAL: If ANY gradient has NaN, skip optimizer step entirely
                # This prevents corrupting the model weights
                if has_any_nan_grad:
                    logger.error("CRITICAL: NaN gradients detected in ALL layers - skipping optimizer step to prevent model corruption")
                    logger.error("This indicates a fundamental numerical instability. Check:")
                    logger.error("  1. Model weights may have NaN (check before training)")
                    logger.error("  2. Input data may have extreme values")
                    logger.error("  3. Loss computation may be producing NaN")
                    # Don't update weights at all - skip this batch completely
                    continue
                
                # Optimize
                self.optimizer.step()
                
                # CRITICAL: Validate weights after optimizer step to catch NaN corruption
                # If auxiliary head weights become NaN, reinitialize them immediately
                for name, param in self.model.named_parameters():
                    if 'aux' in name:
                        if not torch.isfinite(param.data).all():
                            logger.error(f"CRITICAL: {name} weights became NaN after optimizer step! Reinitializing...")
                            # Reinitialize this specific parameter
                            if 'weight' in name:
                                if 'aux_1h.0' in name or 'aux_24h.0' in name:
                                    # First layer: Linear(256, 64)
                                    nn.init.xavier_uniform_(param.data, gain=0.5)
                                elif 'aux_1h.2' in name or 'aux_24h.2' in name:
                                    # Output layer: Linear(64, 1)
                                    nn.init.xavier_uniform_(param.data, gain=0.5)
                            elif 'bias' in name:
                                nn.init.constant_(param.data, 0.0)
                            logger.info(f"Reinitialized {name} to prevent NaN propagation")
                
                # Accumulate metrics (check for NaN before adding)
                policy_loss_val = policy_loss.item()
                value_loss_val = value_loss.item()
                entropy_val = entropy.item()
                aux_1h_loss_val = aux_1h_loss.item() if isinstance(aux_1h_loss, torch.Tensor) else aux_1h_loss
                aux_24h_loss_val = aux_24h_loss.item() if isinstance(aux_24h_loss, torch.Tensor) else aux_24h_loss
                clip_fraction_val = clip_fraction.item()
                
                # Replace NaN with 0
                if np.isnan(policy_loss_val) or np.isinf(policy_loss_val):
                    logger.warning(f"NaN/inf policy_loss detected, replacing with 0")
                    policy_loss_val = 0.0
                if np.isnan(value_loss_val) or np.isinf(value_loss_val):
                    logger.warning(f"NaN/inf value_loss detected, replacing with 0")
                    value_loss_val = 0.0
                if np.isnan(entropy_val) or np.isinf(entropy_val):
                    logger.warning(f"NaN/inf entropy detected, replacing with 0")
                    entropy_val = 0.0
                if np.isnan(aux_1h_loss_val) or np.isinf(aux_1h_loss_val):
                    aux_1h_loss_val = 0.0
                if np.isnan(aux_24h_loss_val) or np.isinf(aux_24h_loss_val):
                    aux_24h_loss_val = 0.0
                if np.isnan(clip_fraction_val) or np.isinf(clip_fraction_val):
                    clip_fraction_val = 0.0
                
                total_policy_loss += policy_loss_val
                total_value_loss += value_loss_val
                total_entropy += entropy_val
                total_aux_1h_loss += aux_1h_loss_val
                total_aux_24h_loss += aux_24h_loss_val
                total_clip_fraction += clip_fraction_val
        
        # Average metrics over all batches and epochs
        num_batches = (num_samples + batch_size - 1) // batch_size
        num_total_batches = num_epochs * num_batches
        
        metrics = {
            "policy_loss": total_policy_loss / num_total_batches,
            "value_loss": total_value_loss / num_total_batches,
            "entropy": total_entropy / num_total_batches,
            "aux_1h_loss": total_aux_1h_loss / num_total_batches,
            "aux_24h_loss": total_aux_24h_loss / num_total_batches,
            "clip_fraction": total_clip_fraction / num_total_batches,
            "total_loss": (
                total_policy_loss / num_total_batches
                + self.value_coef * (total_value_loss / num_total_batches)
                - self.entropy_coef * (total_entropy / num_total_batches)
                + self.aux_1h_coef * (total_aux_1h_loss / num_total_batches)
                + self.aux_24h_coef * (total_aux_24h_loss / num_total_batches)
            ),
        }
        
        self.training_step += 1
        
        return metrics
    
    def save_checkpoint(self, filename: Optional[str] = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.training_step}.pt"
        
        # Handle both absolute paths and relative paths
        if os.path.isabs(filename):
            checkpoint_path = Path(filename)
        else:
            checkpoint_path = self.checkpoint_dir / filename
        
        # Ensure parent directory exists
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step,
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def update_auxiliary_targets(
        self,
        decision_indices: List[int],
        returns_1h: Optional[List[float]] = None,
        returns_24h: Optional[List[float]] = None,
    ):
        """
        Update auxiliary return targets when 1h/24h outcomes become available.
        
        Args:
            decision_indices: List of indices in buffer corresponding to decisions
            returns_1h: Actual 1h returns (if available)
            returns_24h: Actual 24h returns (if available)
        """
        if returns_1h is not None:
            for idx, ret in zip(decision_indices, returns_1h):
                if idx < len(self.buffer["returns_1h"]):
                    self.buffer["returns_1h"][idx] = ret
                else:
                    # Pad if needed
                    while len(self.buffer["returns_1h"]) <= idx:
                        self.buffer["returns_1h"].append(0.0)
                    self.buffer["returns_1h"][idx] = ret
        
        if returns_24h is not None:
            for idx, ret in zip(decision_indices, returns_24h):
                if idx < len(self.buffer["returns_24h"]):
                    self.buffer["returns_24h"][idx] = ret
                else:
                    # Pad if needed
                    while len(self.buffer["returns_24h"]) <= idx:
                        self.buffer["returns_24h"].append(0.0)
                    self.buffer["returns_24h"][idx] = ret
    
    def train_on_rollout(
        self,
        state_encoder,
        price_data: List[float],
        price_features: Dict[str, float],
        news_data: List[Dict],
        num_steps: int = 128,
        num_epochs: int = 4,
        batch_size: int = 32,
    ) -> Dict[str, float]:
        """
        Complete training cycle: collect rollout and train.
        
        Args:
            state_encoder: StateEncoder instance
            price_data: List of recent prices
            price_features: Dict of technical indicators
            news_data: List of news dicts with embeddings
            num_steps: Number of steps to collect in rollout
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dict with training metrics
        """
        # Collect rollout
        rollout_data = self.collect_rollout(
            state_encoder, price_data, price_features, news_data, num_steps
        )
        
        # Train on rollout
        metrics = self.train_step(rollout_data, num_epochs, batch_size)
        
        return metrics

