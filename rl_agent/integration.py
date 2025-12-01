"""
RL Agent Integration Module

Connects the RL agent to existing systems:
- sol_price_fetcher.py for price data
- news_sentiment.py for news embeddings and sentiment
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import os
from dotenv import load_dotenv

from .state_encoder import StateEncoder
from .model import TradingActorCritic
from .environment import TradingEnvironment
from .prediction_generator import generate_prediction, store_prediction_from_decision
from .attention_logger import AttentionLogger
from .risk_manager import RiskManager

load_dotenv()

logger = logging.getLogger(__name__)

# Get database paths from environment
# Price data is in sol_prices.db (separate from rewards.db)
PRICE_DB = "sol_prices.db"  # Price data is always in sol_prices.db
DECISIONS_DB = os.getenv("DATABASE_PATH", "rewards.db")  # Decisions stored in rewards.db
NEWS_DB = "news_sentiment.db"


class RLAgentIntegration:
    """
    Integration layer connecting RL agent to existing price and news systems.
    """
    
    def __init__(
        self,
        model: Optional[TradingActorCritic] = None,
        state_encoder: Optional[StateEncoder] = None,
        device: str = "cpu",
    ):
        """
        Initialize RL agent integration.
        
        Args:
            model: Trained TradingActorCritic model (if None, will need to load)
            state_encoder: StateEncoder instance
            device: Device to run model on
        """
        self.model = model
        self.state_encoder = state_encoder or StateEncoder()
        self.device = device
        
        # Initialize supporting components
        # AttentionLogger uses rewards.db (same as decisions)
        self.attention_logger = AttentionLogger(db_path=DECISIONS_DB)
        self.risk_manager = RiskManager()
        
        # Track current state
        self.current_position = 0.0
        self.portfolio_value = 10000.0  # Starting capital
        self.entry_price = None
        self.last_trade_time = None
    
    def get_price_data(self, hours: int = 24) -> Tuple[List[float], Dict]:
        """
        Get price data from sol_prices database.
        
        Args:
            hours: Number of hours of price history to retrieve
            
        Returns:
            Tuple of (price_list, price_features_dict)
        """
        # Price data is always in sol_prices.db (not rewards.db)
        conn = sqlite3.connect(PRICE_DB)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Get price history
        cursor.execute("""
            SELECT timestamp, rate FROM sol_prices
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            logger.warning("No price data available")
            return [], {}
        
        prices = [row[1] for row in rows]
        current_price = prices[-1] if prices else 0.0
        
        # Calculate technical indicators
        price_features = self._calculate_technical_indicators(prices)
        price_features["current_price"] = current_price
        
        return prices, price_features
    
    def _calculate_technical_indicators(self, prices: List[float]) -> Dict:
        """
        Calculate technical indicators from price history.
        
        Args:
            prices: List of prices
            
        Returns:
            Dict of technical indicators
        """
        if len(prices) < 2:
            return {
                "sma_1h": 0.0,
                "sma_4h": 0.0,
                "sma_24h": 0.0,
                "rsi": 50.0,
                "std_dev": 0.0,
                "momentum_15m": 0.0,
                "percent_change": 0.0,
            }
        
        current_price = prices[-1]
        
        # Simple Moving Averages (assuming 1 price per minute)
        sma_1h = np.mean(prices[-60:]) if len(prices) >= 60 else np.mean(prices)
        sma_4h = np.mean(prices[-240:]) if len(prices) >= 240 else np.mean(prices)
        sma_24h = np.mean(prices[-1440:]) if len(prices) >= 1440 else np.mean(prices)
        
        # RSI calculation (simplified)
        if len(prices) >= 14:
            deltas = np.diff(prices[-14:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0
        else:
            rsi = 50.0
        
        # Standard deviation (volatility)
        std_dev = np.std(prices[-60:]) if len(prices) >= 60 else np.std(prices)
        
        # Momentum (15-minute change)
        if len(prices) >= 15:
            momentum_15m = (current_price - prices[-15]) / prices[-15]
        else:
            momentum_15m = 0.0
        
        # Percent change (24h)
        if len(prices) >= 1440:
            percent_change = (current_price - prices[-1440]) / prices[-1440] * 100
        elif len(prices) > 0:
            percent_change = (current_price - prices[0]) / prices[0] * 100
        else:
            percent_change = 0.0
        
        return {
            "sma_1h": float(sma_1h),
            "sma_4h": float(sma_4h),
            "sma_24h": float(sma_24h),
            "rsi": float(rsi),
            "std_dev": float(std_dev),
            "momentum_15m": float(momentum_15m),
            "percent_change": float(percent_change),
        }
    
    def get_news_data(self, hours: int = 24, max_headlines: int = 20) -> List[Dict]:
        """
        Get news data from news_sentiment database.
        
        Args:
            hours: Number of hours of news to retrieve
            max_headlines: Maximum number of headlines
            
        Returns:
            List of news dicts with embeddings and sentiment
        """
        try:
            from news_sentiment import NewsSentimentAnalyzer
            
            analyzer = NewsSentimentAnalyzer()
            news_items = analyzer.get_recent_news_for_rl_agent(
                hours=hours,
                max_headlines=max_headlines,
                crypto_only=False,
            )
            
            return news_items
        except ImportError:
            logger.warning("News sentiment analyzer not available")
            return []
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return []
    
    def make_decision(
        self,
        force_action: Optional[str] = None,
    ) -> Dict:
        """
        Make a trading decision using the RL agent.
        
        Args:
            force_action: If provided, force this action (for testing)
            
        Returns:
            Dict with decision details
        """
        if not self.model:
            logger.error("Model not loaded. Cannot make decision.")
            return {
                "action": "HOLD",
                "reason": "Model not available",
                "confidence": 0.0,
            }
        
        # Get current market state
        prices, price_features = self.get_price_data(hours=24)
        news_data = self.get_news_data(hours=24, max_headlines=20)
        
        if not prices:
            logger.warning("No price data available for decision")
            return {
                "action": "HOLD",
                "reason": "No price data",
                "confidence": 0.0,
            }
        
        current_price = price_features["current_price"]
        
        # Calculate position state
        time_since_last_trade = 0.0
        if self.last_trade_time:
            time_since_last_trade = (datetime.now() - self.last_trade_time).total_seconds() / 3600.0
        
        unrealized_pnl = 0.0
        if self.entry_price and self.current_position != 0:
            if self.current_position > 0:  # Long position
                unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:  # Short position
                unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        position_state = {
            "position_size": self.current_position,
            "portfolio_value": self.portfolio_value,
            "entry_price": self.entry_price,
            "time_since_last_trade": time_since_last_trade,
            "unrealized_pnl": unrealized_pnl,
        }
        
        # Encode state
        state_dict = self.state_encoder.encode_full_state(
            prices=prices[-60:] if len(prices) >= 60 else prices,
            price_features=price_features,
            news_data=news_data,
            position_size=position_state["position_size"],
            portfolio_value=position_state["portfolio_value"],
            entry_price=position_state["entry_price"],
            current_price=current_price,
            time_since_last_trade=position_state["time_since_last_trade"],
            timestamp=datetime.now(),
            unrealized_pnl=position_state["unrealized_pnl"],
        )
        
        # Get action from model
        if force_action:
            action = force_action.upper()
        else:
            import torch
            price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(self.device)
            news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(self.device)
            news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(self.device)
            position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(self.device)
            time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(self.device)
            news_mask = (news_sent_tensor != 0.0).float()
            
            self.model.eval()
            with torch.no_grad():
                output = self.model(
                    price_tensor, news_emb_tensor, news_sent_tensor,
                    position_tensor, time_tensor, news_mask
                )
                # Model returns action_logits, need to convert to probabilities
                import torch.nn.functional as F
                action_logits = output["action_logits"]
                action_probs = F.softmax(action_logits, dim=-1)[0].cpu().numpy()
                action_idx = np.argmax(action_probs)
                actions = ["BUY", "SELL", "HOLD"]
                action = actions[action_idx]
                confidence = float(action_probs[action_idx])
        
        # Check risk constraints
        proposed_trade_value = self.portfolio_value * 0.1 if action == "BUY" else 0.0
        allowed, reason, risk_metrics = self.risk_manager.check_all_constraints(
            action=action,
            current_position_size=abs(self.current_position),
            portfolio_value=self.portfolio_value,
            proposed_trade_value=proposed_trade_value,
        )
        
        if not allowed:
            logger.info(f"Action {action} blocked by risk manager: {reason}")
            action = "HOLD"
            confidence = 0.5
        
        # Generate predictions
        pred_1h, pred_24h, conf_1h, conf_24h = generate_prediction(
            self.model,
            self.state_encoder,
            prices,
            price_features,
            news_data,
            position_state,
            current_price,
            datetime.now(),
            self.device,
        )
        
        # Store decision in database
        decision_id = None
        try:
            decision_id = self._store_decision(
                action=action,
                confidence=confidence if not force_action else 0.8,
                state_features=state_dict,
                price_features=price_features,
                current_price=current_price,
                pred_1h=pred_1h,
                pred_24h=pred_24h,
                conf_1h=conf_1h,
                conf_24h=conf_24h,
            )
            
            # Store prediction in PredictionManager for tracking
            if decision_id and decision_id > 0:
                try:
                    from .prediction_manager import PredictionManager
                    prediction_manager = PredictionManager()
                    prediction_manager.store_prediction(
                        decision_id=decision_id,
                        timestamp=datetime.now(),
                        predicted_return_1h=pred_1h,
                        predicted_return_24h=pred_24h,
                        predicted_confidence_1h=conf_1h,
                        predicted_confidence_24h=conf_24h,
                        price_at_prediction=current_price,
                    )
                except Exception as e:
                    logger.warning(f"Failed to store prediction in PredictionManager: {e}")
            
            # Validate decision_id - ensure it's an integer
            if decision_id is None:
                logger.warning("Failed to store decision (decision_id is None)")
                decision_id = 0  # Use 0 as fallback
            else:
                # Ensure decision_id is an integer
                try:
                    decision_id = int(decision_id)
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid decision_id type: {type(decision_id)}, value: {decision_id}, error: {e}")
                    decision_id = 0
        except Exception as e:
            logger.error(f"Error storing decision: {e}")
            import traceback
            logger.error(traceback.format_exc())
            decision_id = 0  # Use 0 as fallback
        
        # Log attention weights (if available and decision_id is valid)
        # Only log if decision_id is a valid integer > 0
        if news_data and not force_action:
            # Ensure decision_id is valid before logging
            try:
                # decision_id should already be an int at this point, but double-check
                if decision_id is None:
                    logger.debug("Skipping attention logging: decision_id is None")
                elif not isinstance(decision_id, int):
                    logger.warning(f"Skipping attention logging: decision_id is not int ({type(decision_id)})")
                elif decision_id > 0:
                    # Get attention weights from model output if available
                    attention_weights = None
                    if "attention_weights" in output:
                        attention_weights = output["attention_weights"].cpu().numpy()
                    else:
                        # Fallback: use equal weights as placeholder
                        attention_weights = np.ones((len(news_data), len(news_data))) / len(news_data)
                    
                    # Prepare headlines with proper handling of None values
                    headlines = []
                    for item in news_data:
                        headline_id = item.get("article_id")
                        # Convert to int if not None, otherwise use None
                        if headline_id is not None:
                            try:
                                headline_id = int(headline_id)
                            except (ValueError, TypeError):
                                headline_id = None
                        
                        headlines.append({
                            "headline_text": item.get("headline", ""),
                            "headline_id": headline_id,
                        })
                    
                    # Don't pass cluster_ids (they might contain None values)
                    # cluster_ids will be None by default in log_attention
                    self.attention_logger.log_attention(decision_id, headlines, attention_weights, cluster_ids=None)
                else:
                    logger.debug(f"Skipping attention logging: invalid decision_id ({decision_id})")
            except Exception as e:
                logger.warning(f"Failed to log attention: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        return {
            "decision_id": decision_id,
            "action": action,
            "confidence": confidence if not force_action else 0.8,
            "current_price": current_price,
            "prediction_1h": pred_1h,
            "prediction_24h": pred_24h,
            "confidence_1h": conf_1h,
            "confidence_24h": conf_24h,
            "risk_metrics": risk_metrics,
        }
    
    def _store_decision(
        self,
        action: str,
        confidence: float,
        state_features: Dict,
        price_features: Dict,
        current_price: float,
        pred_1h: float,
        pred_24h: float,
        conf_1h: float,
        conf_24h: float,
    ) -> Optional[int]:
        """Store decision in database."""
        # Decisions are stored in rewards.db (same as app.py)
        conn = sqlite3.connect(DECISIONS_DB)
        cursor = conn.cursor()
        
        import json
        
        # Helper function to convert numpy arrays to lists for JSON serialization
        def convert_numpy_to_list(obj):
            """Recursively convert numpy arrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            state_features_serializable = convert_numpy_to_list(state_features)
            price_features_serializable = convert_numpy_to_list(price_features)
            
            cursor.execute("""
                INSERT INTO rl_agent_decisions (
                    timestamp, action, confidence, state_features, price_features,
                    current_price, predicted_return_1h, predicted_return_24h,
                    predicted_confidence_1h, predicted_confidence_24h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                action,
                confidence,
                json.dumps(state_features_serializable),
                json.dumps(price_features_serializable),
                current_price,
                pred_1h,
                pred_24h,
                conf_1h,
                conf_24h,
            ))
            
            decision_id = cursor.lastrowid
            conn.commit()
            
            # Validate decision_id - should never be None after successful INSERT
            if decision_id is None:
                logger.error("INSERT succeeded but lastrowid is None - trying alternative method")
                # Try to get the last inserted ID manually
                cursor.execute("SELECT last_insert_rowid()")
                row = cursor.fetchone()
                if row and row[0] is not None:
                    decision_id = int(row[0])
                else:
                    logger.error("Could not retrieve decision_id after INSERT")
                    conn.close()
                    return None
            
            # Ensure decision_id is an integer
            if not isinstance(decision_id, int):
                try:
                    decision_id = int(decision_id)
                except (ValueError, TypeError) as e:
                    logger.error(f"Cannot convert decision_id to int: {decision_id}, type: {type(decision_id)}, error: {e}")
                    conn.close()
                    return None
            
            conn.close()
            return decision_id
            
        except sqlite3.Error as e:
            conn.rollback()
            conn.close()
            logger.error(f"Database error storing decision: {e}")
            raise
        except Exception as e:
            conn.rollback()
            conn.close()
            logger.error(f"Unexpected error storing decision: {e}")
            raise

