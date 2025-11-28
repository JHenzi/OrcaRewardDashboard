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
        aux_1h_coef: float = 0.1,  # Auxiliary 1h loss coefficient
        aux_24h_coef: float = 0.1,  # Auxiliary 24h loss coefficient
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
        
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Add bootstrap value for last step
        next_value = 0.0  # Terminal state value is 0
        if len(values) > 0 and not dones[-1]:
            # Use last value as bootstrap if not done
            next_value = values[-1]
        
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
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_advantage = delta + self.gamma * self.gae_lambda * last_advantage
                last_return = rewards[t] + self.gamma * last_return
            
            advantages[t] = last_advantage
            returns[t] = last_return
        
        # Normalize advantages (only if std > 0)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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
        
        advantages = torch.FloatTensor(rollout_data["advantages"]).to(self.device)
        returns = torch.FloatTensor(rollout_data["returns"]).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer["log_probs"]).to(self.device)
        actions = torch.LongTensor(self.buffer["actions"]).to(self.device)
        
        # Get actual returns for auxiliary losses (if available)
        actual_returns_1h = None
        actual_returns_24h = None
        if len(self.buffer["returns_1h"]) > 0 and len(self.buffer["returns_1h"]) == len(actions):
            actual_returns_1h = torch.FloatTensor(self.buffer["returns_1h"]).to(self.device)
        if len(self.buffer["returns_24h"]) > 0 and len(self.buffer["returns_24h"]) == len(actions):
            actual_returns_24h = torch.FloatTensor(self.buffer["returns_24h"]).to(self.device)
        
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
                
                # Batch states
                batched = self._batch_states(batch_states)
                
                # Forward pass
                output = self.model(
                    batched["price_features"],
                    batched["news_embeddings"],
                    batched["news_sentiment"],
                    batched["position_features"],
                    batched["time_features"],
                    batched["news_mask"],
                )
                
                # Get action probabilities
                action_logits = output["action_logits"]
                action_probs = torch.softmax(action_logits, dim=-1)
                log_probs = torch.log(action_probs + 1e-8)
                
                # Get log prob for taken actions
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute clip fraction (for monitoring)
                clip_fraction = ((ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)).float().mean()
                
                # Compute value loss
                values = output["value"].squeeze(1)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Compute entropy bonus
                entropy = -(action_probs * log_probs).sum(dim=1).mean()
                
                # Compute auxiliary losses
                aux_1h_loss = 0.0
                aux_24h_loss = 0.0
                
                pred_1h = output["pred_1h"].squeeze(1)
                pred_24h = output["pred_24h"].squeeze(1)
                
                if actual_returns_1h is not None:
                    batch_returns_1h = actual_returns_1h[batch_indices]
                    aux_1h_loss = nn.functional.mse_loss(pred_1h, batch_returns_1h)
                
                if actual_returns_24h is not None:
                    batch_returns_24h = actual_returns_24h[batch_indices]
                    aux_24h_loss = nn.functional.mse_loss(pred_24h, batch_returns_24h)
                
                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                    + self.aux_1h_coef * aux_1h_loss
                    + self.aux_24h_coef * aux_24h_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimize
                self.optimizer.step()
                
                # Accumulate metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_aux_1h_loss += aux_1h_loss.item() if isinstance(aux_1h_loss, torch.Tensor) else aux_1h_loss
                total_aux_24h_loss += aux_24h_loss.item() if isinstance(aux_24h_loss, torch.Tensor) else aux_24h_loss
                total_clip_fraction += clip_fraction.item()
        
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
        
        checkpoint_path = self.checkpoint_dir / filename
        
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

