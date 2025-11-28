"""
Unified Trade Predictor

This module combines all trading signals (RSI, Technical Indicators, News Sentiment, Bandit)
into a unified trade prediction system that uses the bandit_state.json portfolio simulation.

The predictor:
1. Aggregates all signals (RSI, MACD, Bollinger Bands, Momentum, News, Bandit)
2. Weights signals based on their historical performance
3. Makes a unified buy/sell/hold recommendation
4. Simulates trades using bandit_state.json portfolio tracking
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple
from pathlib import Path

# Import signal performance tracker to weight signals
try:
    from signal_performance_tracker import SignalPerformanceTracker
    SIGNAL_TRACKER_AVAILABLE = True
except ImportError:
    SIGNAL_TRACKER_AVAILABLE = False
    logging.warning("signal_performance_tracker not available")

# Import news sentiment analyzer
try:
    from news_sentiment import NewsSentimentAnalyzer
    NEWS_ANALYZER_AVAILABLE = True
except ImportError:
    NEWS_ANALYZER_AVAILABLE = False
    logging.warning("news_sentiment not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BANDIT_STATE_FILE = "bandit_state.json"


class UnifiedTradePredictor:
    """
    Unified trade predictor that combines all signals and uses portfolio simulation.
    """
    
    def __init__(self):
        """Initialize the unified trade predictor."""
        self.signal_tracker = SignalPerformanceTracker() if SIGNAL_TRACKER_AVAILABLE else None
        self.news_analyzer = NewsSentimentAnalyzer() if NEWS_ANALYZER_AVAILABLE else None
        self.bandit_state = self.load_bandit_state()
        
    def load_bandit_state(self) -> Dict:
        """Load bandit state from JSON file."""
        try:
            if Path(BANDIT_STATE_FILE).exists():
                with open(BANDIT_STATE_FILE, "r") as f:
                    state = json.load(f)
                return state
            else:
                logger.warning(f"{BANDIT_STATE_FILE} not found. Using default state.")
                return self._default_state()
        except Exception as e:
            logger.error(f"Error loading bandit state: {e}")
            return self._default_state()
    
    def _default_state(self) -> Dict:
        """Return default bandit state."""
        return {
            "last_action": "hold",
            "entry_price": 0.0,
            "position_open": False,
            "fee": 0.001,
            "portfolio": {
                "sol_balance": 0.0,
                "usd_balance": 1000.0,
                "total_cost_basis": 0.0,
                "realized_pnl": 0.0,
                "entry_price": 0.0
            }
        }
    
    def save_bandit_state(self):
        """Save bandit state to JSON file."""
        try:
            with open(BANDIT_STATE_FILE, "w") as f:
                json.dump(self.bandit_state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving bandit state: {e}")
    
    def get_signal_weights(self) -> Dict[str, float]:
        """
        Get weights for each signal type based on historical performance.
        
        Returns:
            Dictionary mapping signal types to weights (0-1)
        """
        if not self.signal_tracker:
            # Default equal weights if tracker not available
            return {
                "rsi_buy": 0.2,
                "rsi_sell": 0.2,
                "rsi_hold": 0.1,
                "bandit_buy": 0.2,
                "bandit_sell": 0.2,
                "bandit_hold": 0.1,
            }
        
        weights = {}
        signal_types = ["rsi_buy", "rsi_sell", "rsi_hold", "bandit_buy", "bandit_sell", "bandit_hold"]
        
        for signal_type in signal_types:
            stats = self.signal_tracker.get_performance_stats(signal_type, hours_later=24)
            
            # Weight based on win rate and average return
            if stats["total_signals"] > 0:
                win_rate = stats["win_rate"] / 100.0  # Convert to 0-1
                avg_return = abs(stats["avg_return"]) / 10.0  # Normalize (assume max 10% return)
                weight = (win_rate * 0.7 + min(avg_return, 1.0) * 0.3)  # 70% win rate, 30% return
            else:
                weight = 0.1  # Default low weight for untested signals
            
            weights[signal_type] = weight
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def aggregate_signals(
        self,
        rsi_value: Optional[float],
        technical_signals: Dict,
        news_features: Optional[Dict],
        bandit_prediction: Optional[Dict],
        current_price: float
    ) -> Dict:
        """
        Aggregate all signals into a unified prediction.
        
        Args:
            rsi_value: Current RSI value
            technical_signals: Dictionary of technical indicator signals
            news_features: News sentiment features
            bandit_prediction: Bandit model prediction
            current_price: Current SOL price
            
        Returns:
            Dictionary with unified prediction and confidence
        """
        # Initialize signal scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        
        # Get signal weights
        weights = self.get_signal_weights()
        
        # RSI signals
        if rsi_value is not None:
            if rsi_value < 30:
                buy_score += weights.get("rsi_buy", 0.2) * (30 - rsi_value) / 30  # Stronger as RSI gets lower
            elif rsi_value > 70:
                sell_score += weights.get("rsi_sell", 0.2) * (rsi_value - 70) / 30  # Stronger as RSI gets higher
            else:
                hold_score += weights.get("rsi_hold", 0.1)
        
        # Technical indicators
        if technical_signals:
            # MACD
            if technical_signals.get("macd_signal") == "bullish":
                buy_score += 0.15
            elif technical_signals.get("macd_signal") == "bearish":
                sell_score += 0.15
            
            # Bollinger Bands
            if technical_signals.get("bb_signal") == "oversold":
                buy_score += 0.1
            elif technical_signals.get("bb_signal") == "overbought":
                sell_score += 0.1
            
            # Momentum
            momentum = technical_signals.get("momentum_signal", 0)
            if momentum > 0.02:  # Strong positive momentum
                buy_score += 0.1
            elif momentum < -0.02:  # Strong negative momentum
                sell_score += 0.1
            
            # Price vs SMA
            if technical_signals.get("price_vs_sma_1h", 1.0) > 1.02:  # Price 2% above SMA
                buy_score += 0.05
            elif technical_signals.get("price_vs_sma_1h", 1.0) < 0.98:  # Price 2% below SMA
                sell_score += 0.05
        
        # News sentiment
        if news_features:
            news_sentiment = news_features.get("news_sentiment_score", 0.0)
            news_count = news_features.get("news_count", 0)
            
            if news_count > 0:
                # Weight news sentiment (less weight than technical signals)
                news_weight = 0.15
                if news_sentiment > 0.3:
                    buy_score += news_weight * min(news_sentiment, 1.0)
                elif news_sentiment < -0.3:
                    sell_score += news_weight * min(abs(news_sentiment), 1.0)
        
        # Bandit prediction
        if bandit_prediction:
            action = bandit_prediction.get("action", "hold")
            confidence = abs(bandit_prediction.get("reward", 0.0)) / 10.0  # Normalize
            
            if action == "buy":
                buy_score += weights.get("bandit_buy", 0.2) * confidence
            elif action == "sell":
                sell_score += weights.get("bandit_sell", 0.2) * confidence
            else:
                hold_score += weights.get("bandit_hold", 0.1) * confidence
        
        # Normalize scores
        total_score = buy_score + sell_score + hold_score
        if total_score > 0:
            buy_score /= total_score
            sell_score /= total_score
            hold_score /= total_score
        
        # Determine action
        if buy_score > sell_score and buy_score > hold_score and buy_score > 0.4:
            action = "buy"
            confidence = buy_score
        elif sell_score > buy_score and sell_score > hold_score and sell_score > 0.4:
            action = "sell"
            confidence = sell_score
        else:
            action = "hold"
            confidence = hold_score
        
        return {
            "action": action,
            "confidence": confidence,
            "buy_score": buy_score,
            "sell_score": sell_score,
            "hold_score": hold_score,
            "signals_used": {
                "rsi": rsi_value is not None,
                "technical": technical_signals is not None,
                "news": news_features is not None,
                "bandit": bandit_prediction is not None,
            }
        }
    
    def simulate_trade(
        self,
        action: str,
        current_price: float,
        fee: float = 0.001
    ) -> Dict:
        """
        Simulate a trade using the bandit_state.json portfolio.
        
        Args:
            action: "buy", "sell", or "hold"
            current_price: Current SOL price
            fee: Trading fee (default 0.1%)
            
        Returns:
            Dictionary with trade result and updated portfolio state
        """
        portfolio = self.bandit_state["portfolio"]
        sol_balance = portfolio["sol_balance"]
        usd_balance = portfolio["usd_balance"]
        
        result = {
            "action": action,
            "price": current_price,
            "fee": fee,
            "success": False,
            "message": "",
            "portfolio_before": portfolio.copy(),
        }
        
        if action == "buy":
            # Calculate how much SOL we can buy
            cost_per_sol = current_price * (1 + fee)
            max_sol = usd_balance / cost_per_sol
            
            if max_sol > 0.001:  # Minimum trade size
                # Buy with 50% of available cash (conservative)
                trade_amount_usd = usd_balance * 0.5
                sol_to_buy = trade_amount_usd / cost_per_sol
                total_cost = sol_to_buy * cost_per_sol
                
                if total_cost <= usd_balance:
                    portfolio["sol_balance"] += sol_to_buy
                    portfolio["usd_balance"] -= total_cost
                    portfolio["total_cost_basis"] += total_cost
                    portfolio["entry_price"] = current_price
                    
                    self.bandit_state["last_action"] = "buy"
                    self.bandit_state["entry_price"] = current_price
                    self.bandit_state["position_open"] = True
                    
                    result["success"] = True
                    result["message"] = f"Bought {sol_to_buy:.4f} SOL at ${current_price:.2f}"
                    result["sol_bought"] = sol_to_buy
                    result["cost"] = total_cost
            else:
                result["message"] = "Insufficient funds to buy"
        
        elif action == "sell":
            if sol_balance > 0.001:  # Minimum trade size
                # Sell 50% of position (conservative)
                sol_to_sell = sol_balance * 0.5
                revenue = sol_to_sell * current_price * (1 - fee)
                
                # Calculate P&L
                avg_entry = portfolio["total_cost_basis"] / portfolio["sol_balance"] if portfolio["sol_balance"] > 0 else 0
                cost_basis_sold = sol_to_sell * avg_entry
                pnl = revenue - cost_basis_sold
                
                portfolio["sol_balance"] -= sol_to_sell
                portfolio["usd_balance"] += revenue
                portfolio["total_cost_basis"] -= cost_basis_sold
                portfolio["realized_pnl"] += pnl
                
                if portfolio["sol_balance"] < 0.001:
                    self.bandit_state["position_open"] = False
                    portfolio["entry_price"] = 0.0
                
                self.bandit_state["last_action"] = "sell"
                
                result["success"] = True
                result["message"] = f"Sold {sol_to_sell:.4f} SOL at ${current_price:.2f}, P&L: ${pnl:.2f}"
                result["sol_sold"] = sol_to_sell
                result["revenue"] = revenue
                result["pnl"] = pnl
        
        else:  # hold
            result["success"] = True
            result["message"] = "Holding position"
        
        # Calculate current equity
        current_equity = portfolio["usd_balance"] + (portfolio["sol_balance"] * current_price)
        unrealized_pnl = 0.0
        if portfolio["sol_balance"] > 0:
            avg_entry = portfolio["total_cost_basis"] / portfolio["sol_balance"]
            unrealized_pnl = (current_price - avg_entry) * portfolio["sol_balance"]
        
        result["portfolio_after"] = portfolio.copy()
        result["current_equity"] = current_equity
        result["unrealized_pnl"] = unrealized_pnl
        result["total_pnl"] = portfolio["realized_pnl"] + unrealized_pnl
        
        # Save updated state
        self.save_bandit_state()
        
        return result
    
    def predict_and_simulate(
        self,
        rsi_value: Optional[float],
        technical_signals: Dict,
        news_features: Optional[Dict],
        bandit_prediction: Optional[Dict],
        current_price: float
    ) -> Dict:
        """
        Make unified prediction and simulate trade.
        
        Returns:
            Dictionary with prediction, trade simulation, and portfolio state
        """
        # Get unified prediction
        prediction = self.aggregate_signals(
            rsi_value, technical_signals, news_features, bandit_prediction, current_price
        )
        
        # Simulate trade
        trade_result = self.simulate_trade(
            prediction["action"],
            current_price,
            fee=self.bandit_state.get("fee", 0.001)
        )
        
        return {
            "prediction": prediction,
            "trade": trade_result,
            "portfolio": self.bandit_state["portfolio"].copy(),
            "bandit_state": self.bandit_state.copy(),
        }


def main():
    """Test the unified trade predictor."""
    print("Initializing Unified Trade Predictor...")
    predictor = UnifiedTradePredictor()
    
    # Example signals
    rsi_value = 25.0  # Oversold
    technical_signals = {
        "macd_signal": "bullish",
        "bb_signal": "oversold",
        "momentum_signal": 0.03,
        "price_vs_sma_1h": 0.95,
    }
    
    news_features = None
    if NEWS_ANALYZER_AVAILABLE:
        news_features = predictor.news_analyzer.get_recent_news_features(hours=24)
    
    bandit_prediction = {
        "action": "buy",
        "reward": 0.5,
    }
    
    current_price = 150.0
    
    print("\nMaking unified prediction...")
    result = predictor.predict_and_simulate(
        rsi_value, technical_signals, news_features, bandit_prediction, current_price
    )
    
    print(f"\nPrediction: {result['prediction']}")
    print(f"Trade: {result['trade']}")
    print(f"Portfolio: {result['portfolio']}")
    print("\n✅ Unified trade prediction complete!")


if __name__ == "__main__":
    main()

