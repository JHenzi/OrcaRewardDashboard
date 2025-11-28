"""
Risk Manager

Monitors and enforces risk constraints for the RL agent.
Tracks position size, trade frequency, daily P&L, and uncertainty metrics.
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Manages risk constraints and monitoring for the RL agent.
    
    Features:
    - Position size limits
    - Trade frequency limits
    - Daily loss caps
    - Uncertainty monitoring
    - Risk metrics tracking
    """
    
    def __init__(
        self,
        max_position_size: float = 0.1,  # 10% of capital
        max_trades_per_hour: int = 5,
        daily_loss_cap: float = 0.05,  # 5% daily loss
        uncertainty_threshold: float = 0.8,  # High uncertainty threshold
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position as fraction of capital
            max_trades_per_hour: Maximum number of trades per hour
            daily_loss_cap: Maximum daily loss as fraction
            uncertainty_threshold: Uncertainty level above which to force HOLD
        """
        self.max_position_size = max_position_size
        self.max_trades_per_hour = max_trades_per_hour
        self.daily_loss_cap = daily_loss_cap
        self.uncertainty_threshold = uncertainty_threshold
        
        # Track trade history
        self.trade_times = deque(maxlen=100)  # Last 100 trades
        self.daily_start_value = None
        self.daily_start_time = None
        self.current_portfolio_value = None
        
    def reset_daily_tracking(self, portfolio_value: float):
        """Reset daily tracking (call at start of each day)."""
        self.daily_start_value = portfolio_value
        self.daily_start_time = datetime.now()
        self.current_portfolio_value = portfolio_value
    
    def check_position_limit(
        self,
        current_position_size: float,
        portfolio_value: float,
        proposed_trade_value: float,
    ) -> Tuple[bool, str]:
        """
        Check if position size is within limits.
        
        Args:
            current_position_size: Current position size (SOL amount)
            portfolio_value: Total portfolio value
            proposed_trade_value: Value of proposed trade
            
        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        current_position_value = current_position_size * (portfolio_value / (1 - self.max_position_size)) if portfolio_value > 0 else 0
        max_position_value = portfolio_value * self.max_position_size
        
        if current_position_value + proposed_trade_value > max_position_value:
            return False, f"position_limit_exceeded: {current_position_value + proposed_trade_value:.2f} > {max_position_value:.2f}"
        
        return True, ""
    
    def check_trade_frequency(self) -> Tuple[bool, str]:
        """
        Check if trade frequency is within limits.
        
        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        now = datetime.now()
        
        # Remove trades older than 1 hour
        while self.trade_times and (now - self.trade_times[0]).total_seconds() > 3600:
            self.trade_times.popleft()
        
        if len(self.trade_times) >= self.max_trades_per_hour:
            return False, f"trade_frequency_limit: {len(self.trade_times)} trades in last hour"
        
        return True, ""
    
    def check_daily_loss_cap(self, current_portfolio_value: float) -> Tuple[bool, str, float]:
        """
        Check if daily loss cap is exceeded.
        
        Args:
            current_portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (is_allowed, reason_if_not, daily_return)
        """
        if self.daily_start_value is None:
            self.reset_daily_tracking(current_portfolio_value)
            return True, "", 0.0
        
        daily_return = (current_portfolio_value - self.daily_start_value) / self.daily_start_value if self.daily_start_value > 0 else 0.0
        
        if daily_return < -self.daily_loss_cap:
            return False, f"daily_loss_cap_exceeded: {daily_return*100:.2f}%", daily_return
        
        self.current_portfolio_value = current_portfolio_value
        return True, "", daily_return
    
    def check_uncertainty(self, uncertainty: float) -> Tuple[bool, str]:
        """
        Check if uncertainty is too high (force HOLD if so).
        
        Args:
            uncertainty: Uncertainty metric (0-1, higher = more uncertain)
            
        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        if uncertainty > self.uncertainty_threshold:
            return False, f"uncertainty_too_high: {uncertainty:.2f} > {self.uncertainty_threshold}"
        
        return True, ""
    
    def record_trade(self):
        """Record that a trade was executed."""
        self.trade_times.append(datetime.now())
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dict with risk metrics
        """
        now = datetime.now()
        
        # Count recent trades
        recent_trades = sum(
            1 for trade_time in self.trade_times
            if (now - trade_time).total_seconds() <= 3600
        )
        
        # Calculate daily return
        daily_return = 0.0
        if self.daily_start_value and self.current_portfolio_value:
            daily_return = (self.current_portfolio_value - self.daily_start_value) / self.daily_start_value
        
        # Calculate time until next trade allowed
        time_until_next_trade = 0.0
        if self.trade_times:
            last_trade = self.trade_times[-1]
            time_since_last = (now - last_trade).total_seconds()
            min_interval = 3600.0 / self.max_trades_per_hour  # Minimum seconds between trades
            time_until_next_trade = max(0.0, min_interval - time_since_last)
        
        return {
            "max_position_size": self.max_position_size,
            "max_trades_per_hour": self.max_trades_per_hour,
            "daily_loss_cap": self.daily_loss_cap,
            "recent_trades_count": recent_trades,
            "trades_remaining_this_hour": max(0, self.max_trades_per_hour - recent_trades),
            "time_until_next_trade_allowed": time_until_next_trade,
            "daily_return": daily_return,
            "daily_return_percent": daily_return * 100,
            "daily_loss_cap_remaining": max(0.0, -self.daily_loss_cap - daily_return),
            "daily_start_value": self.daily_start_value,
            "current_portfolio_value": self.current_portfolio_value,
        }
    
    def check_all_constraints(
        self,
        action: str,
        current_position_size: float,
        portfolio_value: float,
        proposed_trade_value: float,
        uncertainty: Optional[float] = None,
    ) -> Tuple[bool, str, Dict]:
        """
        Check all risk constraints at once.
        
        Args:
            action: Action to check (BUY, SELL, HOLD)
            current_position_size: Current position size
            portfolio_value: Portfolio value
            proposed_trade_value: Value of proposed trade
            uncertainty: Optional uncertainty metric
            
        Returns:
            Tuple of (is_allowed, reason_if_not, risk_metrics)
        """
        risk_metrics = self.get_risk_metrics()
        
        # Check daily loss cap
        allowed, reason, daily_return = self.check_daily_loss_cap(portfolio_value)
        if not allowed:
            return False, reason, risk_metrics
        
        # Check trade frequency (only for BUY/SELL, not HOLD)
        if action in ["BUY", "SELL"]:
            allowed, reason = self.check_trade_frequency()
            if not allowed:
                return False, reason, risk_metrics
        
        # Check position limit (only for BUY)
        if action == "BUY":
            allowed, reason = self.check_position_limit(
                current_position_size, portfolio_value, proposed_trade_value
            )
            if not allowed:
                return False, reason, risk_metrics
        
        # Check uncertainty (only if provided)
        if uncertainty is not None:
            allowed, reason = self.check_uncertainty(uncertainty)
            if not allowed:
                return False, reason, risk_metrics
        
        return True, "", risk_metrics

