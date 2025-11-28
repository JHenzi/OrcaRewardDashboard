"""
Trading Environment

Gym-style environment for RL agent training.
Simulates trading with realistic transaction costs and risk constraints.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from datetime import datetime, timedelta
from enum import IntEnum

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions."""
    SELL = 0
    HOLD = 1
    BUY = 2


class TradingEnvironment:
    """
    Trading environment for RL agent.
    
    Simulates:
    - Position management (buy/sell/hold)
    - Transaction costs
    - Portfolio value tracking
    - Risk constraints (position size, trade frequency, daily loss cap)
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,  # USD
        transaction_cost_rate: float = 0.001,  # 0.1% per trade
        max_position_size: float = 0.1,  # 10% of capital initially
        max_trades_per_hour: int = 5,
        daily_loss_cap: float = 0.05,  # 5% daily loss limit
        risk_penalty_coef: float = 0.01,
    ):
        """
        Initialize trading environment.
        
        Args:
            initial_capital: Starting capital in USD
            transaction_cost_rate: Transaction cost as fraction (0.001 = 0.1%)
            max_position_size: Maximum position as fraction of capital
            max_trades_per_hour: Maximum number of trades per hour
            daily_loss_cap: Maximum daily loss as fraction (0.05 = 5%)
            risk_penalty_coef: Coefficient for risk penalty in reward
        """
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.max_position_size = max_position_size
        self.max_trades_per_hour = max_trades_per_hour
        self.daily_loss_cap = daily_loss_cap
        self.risk_penalty_coef = risk_penalty_coef
        
        # State
        self.reset()
        
    def reset(self) -> Dict:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation dict
        """
        self.portfolio_value = self.initial_capital
        self.position_size = 0.0  # SOL amount
        self.entry_price = None
        self.cash = self.initial_capital
        self.last_trade_time = None
        self.trades_today = 0
        self.daily_start_value = self.initial_capital
        self.trade_history = []
        self.step_count = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        """Get current observation (position state)."""
        return {
            "position_size": self.position_size,
            "portfolio_value": self.portfolio_value,
            "entry_price": self.entry_price,
            "cash": self.cash,
            "time_since_last_trade": self._get_time_since_last_trade(),
            "unrealized_pnl": self._get_unrealized_pnl(),
        }
    
    def _get_time_since_last_trade(self) -> float:
        """Get minutes since last trade."""
        if self.last_trade_time is None:
            return 0.0
        delta = datetime.now() - self.last_trade_time
        return delta.total_seconds() / 60.0
    
    def _get_unrealized_pnl(self) -> float:
        """Calculate unrealized profit/loss."""
        if self.entry_price is None or self.position_size == 0:
            return 0.0
        # This will be updated with current_price in step()
        return 0.0  # Placeholder
    
    def _calculate_transaction_cost(self, trade_value: float) -> float:
        """Calculate transaction cost for a trade."""
        return trade_value * self.transaction_cost_rate
    
    def _check_risk_limits(self, action: int, current_price: float) -> Tuple[bool, str]:
        """
        Check if action violates risk limits.
        
        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        # Check daily loss cap
        daily_return = (self.portfolio_value - self.daily_start_value) / self.daily_start_value
        if daily_return < -self.daily_loss_cap:
            return False, "daily_loss_cap_exceeded"
        
        # Check trade frequency
        time_since_last = self._get_time_since_last_trade()
        if time_since_last < 60.0 / self.max_trades_per_hour and self.last_trade_time is not None:
            return False, "trade_frequency_limit"
        
        # Check position size for BUY
        if action == Action.BUY:
            max_position_value = self.portfolio_value * self.max_position_size
            if self.position_size * current_price >= max_position_value:
                return False, "max_position_size"
        
        return True, ""
    
    def step(
        self,
        action: int,
        current_price: float,
        next_price: Optional[float] = None,
    ) -> Tuple[Dict, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=SELL, 1=HOLD, 2=BUY)
            current_price: Current SOL price
            next_price: Next price (for reward calculation, if available)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.step_count += 1
        reward = 0.0
        info = {
            "action_taken": action,
            "action_allowed": True,
            "reason": "",
        }
        
        # Check risk limits
        is_allowed, reason = self._check_risk_limits(action, current_price)
        if not is_allowed:
            # Force HOLD if action not allowed
            action = Action.HOLD
            info["action_allowed"] = False
            info["reason"] = reason
        
        # Execute action
        if action == Action.BUY and self.position_size == 0:
            # Buy SOL
            max_position_value = self.portfolio_value * self.max_position_size
            trade_value = min(max_position_value, self.cash * 0.95)  # Use 95% of cash
            
            if trade_value > 0:
                transaction_cost = self._calculate_transaction_cost(trade_value)
                total_cost = trade_value + transaction_cost
                
                if total_cost <= self.cash:
                    sol_amount = (trade_value - transaction_cost) / current_price
                    self.position_size = sol_amount
                    self.entry_price = current_price
                    self.cash -= total_cost
                    self.last_trade_time = datetime.now()
                    self.trades_today += 1
                    
                    info["trade_executed"] = True
                    info["trade_value"] = trade_value
                    info["transaction_cost"] = transaction_cost
                    info["sol_amount"] = sol_amount
                    
        elif action == Action.SELL and self.position_size > 0:
            # Sell SOL
            trade_value = self.position_size * current_price
            transaction_cost = self._calculate_transaction_cost(trade_value)
            proceeds = trade_value - transaction_cost
            
            self.cash += proceeds
            self.position_size = 0.0
            self.entry_price = None
            self.last_trade_time = datetime.now()
            self.trades_today += 1
            
            info["trade_executed"] = True
            info["trade_value"] = trade_value
            info["transaction_cost"] = transaction_cost
            info["proceeds"] = proceeds
            
        else:
            # HOLD
            info["trade_executed"] = False
        
        # Update portfolio value
        position_value = self.position_size * current_price
        self.portfolio_value = self.cash + position_value
        
        # Calculate reward
        if self.step_count > 1:
            # Log portfolio value change
            prev_value = getattr(self, '_prev_portfolio_value', self.initial_capital)
            
            # Reward = log return minus transaction cost and risk penalty
            if prev_value > 0:
                log_return = np.log(self.portfolio_value / prev_value)
            else:
                log_return = 0.0
            
            # Transaction cost penalty (already deducted from portfolio)
            # Risk penalty (position size squared)
            position_ratio = position_value / self.portfolio_value if self.portfolio_value > 0 else 0.0
            risk_penalty = self.risk_penalty_coef * (position_ratio ** 2)
            
            reward = log_return - risk_penalty
            
            # Store for next step
            self._prev_portfolio_value = self.portfolio_value
        
        # Update daily tracking
        if self.step_count % (24 * 12) == 0:  # Reset daily tracking every 24 hours (assuming 5-min steps)
            self.daily_start_value = self.portfolio_value
            self.trades_today = 0
        
        # Done if portfolio value drops too low
        done = self.portfolio_value < self.initial_capital * 0.5  # Stop if lose 50%
        
        observation = self._get_observation()
        observation["current_price"] = current_price
        observation["unrealized_pnl"] = (current_price - self.entry_price) * self.position_size if self.entry_price else 0.0
        
        info["portfolio_value"] = self.portfolio_value
        info["position_size"] = self.position_size
        info["cash"] = self.cash
        
        return observation, reward, done, info
    
    def get_state_for_encoder(self) -> Dict:
        """Get state dict for StateEncoder."""
        return {
            "position_size": self.position_size,
            "portfolio_value": self.portfolio_value,
            "entry_price": self.entry_price,
            "current_price": getattr(self, '_current_price', 0.0),
            "time_since_last_trade": self._get_time_since_last_trade(),
            "unrealized_pnl": self._get_unrealized_pnl(),
        }

