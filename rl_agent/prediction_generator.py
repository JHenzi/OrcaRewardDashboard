"""
Prediction Generator

Helper functions to generate predictions from the RL agent model.

POST-AUDIT CHANGES:
- Classification output: predictions now include class probabilities (Bearish/Neutral/Bullish)
- Multi-resolution support: can accept hourly/daily price data
- News age support: includes news age for time decay
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

from .model import TradingActorCritic
from .state_encoder import StateEncoder
from .prediction_manager import PredictionManager

logger = logging.getLogger(__name__)

# Time-horizon specific thresholds (must match trainer.py!)
THRESHOLD_15M = 0.002   # ±0.2% for 15-minute
THRESHOLD_1H = 0.005    # ±0.5% for 1-hour
THRESHOLD_24H = 0.01    # ±1.0% for 24-hour

# Class labels for interpretability (default for backward compat)
CLASS_LABELS = ["Bearish (<-1%)", "Neutral (-1% to +1%)", "Bullish (>+1%)"]
CLASS_LABELS_15M = ["Bearish (<-0.2%)", "Neutral (-0.2% to +0.2%)", "Bullish (>+0.2%)"]
CLASS_LABELS_1H = ["Bearish (<-0.5%)", "Neutral (-0.5% to +0.5%)", "Bullish (>+0.5%)"]
CLASS_LABELS_24H = ["Bearish (<-1%)", "Neutral (-1% to +1%)", "Bullish (>+1%)"]


def generate_prediction(
    model: TradingActorCritic,
    state_encoder: StateEncoder,
    price_data: list,
    price_features: Dict,
    news_data: list,
    position_state: Dict,
    current_price: float,
    timestamp: datetime,
    device: str = "cpu",
    # Multi-resolution data (POST-AUDIT FIX #2)
    hourly_prices: Optional[List[float]] = None,
    daily_prices: Optional[List[float]] = None,
) -> Tuple[float, float, Optional[float], Optional[float], Optional[Dict]]:
    """
    Generate 1h and 24h return predictions from the RL agent.
    
    POST-AUDIT: Now uses classification (Bearish/Neutral/Bullish) internally
    but returns expected return values for backward compatibility.
    
    Args:
        model: Trained TradingActorCritic model
        state_encoder: StateEncoder instance
        price_data: List of recent prices (5-min intervals)
        price_features: Dict of technical indicators
        news_data: List of news dicts with embeddings
        position_state: Dict with position info (size, portfolio_value, etc.)
        current_price: Current SOL price
        timestamp: Current timestamp
        device: Device to run model on
        hourly_prices: Optional list of hourly prices (for multi-resolution)
        daily_prices: Optional list of daily prices (for multi-resolution)
        
    Returns:
        Tuple of (pred_1h, pred_24h, confidence_1h, confidence_24h, class_info)
        - pred_1h/pred_24h: Expected returns in decimal form (0.01 = 1%)
        - confidence_1h/confidence_24h: Confidence scores (0.0 to 1.0)
        - class_info: Optional dict with class probabilities for interpretability
    """
    model.eval()
    
    # Encode state with multi-resolution support (POST-AUDIT)
    state_dict = state_encoder.encode_full_state(
        prices=price_data[-60:] if len(price_data) >= 60 else price_data,
        price_features=price_features,
        news_data=news_data,
        position_size=position_state.get("position_size", 0.0),
        portfolio_value=position_state.get("portfolio_value", 10000.0),
        entry_price=position_state.get("entry_price"),
        current_price=current_price,
        time_since_last_trade=position_state.get("time_since_last_trade", 0.0),
        timestamp=timestamp,
        unrealized_pnl=position_state.get("unrealized_pnl", 0.0),
        hourly_prices=hourly_prices,  # Multi-resolution (FIX #2)
        daily_prices=daily_prices,  # Multi-resolution (FIX #2)
    )
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    
    # Multi-resolution tensors (POST-AUDIT FIX #2)
    price_medium_tensor = torch.FloatTensor(state_dict.get("price_medium", np.zeros(60))).unsqueeze(0).to(device)
    price_long_tensor = torch.FloatTensor(state_dict.get("price_long", np.zeros(30))).unsqueeze(0).to(device)
    
    # News age tensor (POST-AUDIT FIX #4)
    news_age_tensor = torch.FloatTensor(state_dict.get("news_age", np.zeros(20))).unsqueeze(0).to(device)
    
    # Create mask for news
    news_mask = (news_sent_tensor != 0.0).float()
    
    # Get predictions
    with torch.no_grad():
        output = model(
            price_tensor, news_emb_tensor, news_sent_tensor,
            position_tensor, time_tensor, news_mask,
            news_age=news_age_tensor,  # FIX #4
            price_medium=price_medium_tensor,  # FIX #2
            price_long=price_long_tensor,  # FIX #2
        )
        
        # POST-AUDIT: Handle classification output (FIX #5)
        class_info = None
        
        if "pred_1h_probs" in output:
            # New classification model - use class probabilities
            pred_1h_probs = output["pred_1h_probs"][0].cpu().numpy()
            pred_24h_probs = output["pred_24h_probs"][0].cpu().numpy()
            
            # Expected return = weighted sum of class returns
            # Use threshold-appropriate expected returns (2x threshold for bear/bull)
            # 1h: threshold=0.5%, so bear=-1%, neutral=0%, bull=+1%
            pred_1h = float(pred_1h_probs[0] * (-2 * THRESHOLD_1H) + pred_1h_probs[1] * 0.0 + pred_1h_probs[2] * (2 * THRESHOLD_1H))
            # 24h: threshold=1%, so bear=-2%, neutral=0%, bull=+2%
            pred_24h = float(pred_24h_probs[0] * (-2 * THRESHOLD_24H) + pred_24h_probs[1] * 0.0 + pred_24h_probs[2] * (2 * THRESHOLD_24H))
            
            # Confidence = max class probability (how certain is the model?)
            confidence_1h = float(pred_1h_probs.max())
            confidence_24h = float(pred_24h_probs.max())
            
            # Build class info for interpretability
            class_info = {
                "1h": {
                    "class_probs": {
                        "bearish": float(pred_1h_probs[0]),
                        "neutral": float(pred_1h_probs[1]),
                        "bullish": float(pred_1h_probs[2]),
                    },
                    "predicted_class": CLASS_LABELS_1H[int(pred_1h_probs.argmax())],
                },
                "24h": {
                    "class_probs": {
                        "bearish": float(pred_24h_probs[0]),
                        "neutral": float(pred_24h_probs[1]),
                        "bullish": float(pred_24h_probs[2]),
                    },
                    "predicted_class": CLASS_LABELS_24H[int(pred_24h_probs.argmax())],
                },
            }
            
            logger.info(f"Classification predictions - 1h: {class_info['1h']['predicted_class']} "
                       f"({confidence_1h:.1%}), 24h: {class_info['24h']['predicted_class']} ({confidence_24h:.1%})")
        else:
            # Old regression model (backward compatibility)
            pred_1h_tensor = output["pred_1h"]
            pred_24h_tensor = output["pred_24h"]
            
            if pred_1h_tensor.dim() > 1:
                pred_1h = pred_1h_tensor[0].item()
            else:
                pred_1h = pred_1h_tensor.item()
                
            if pred_24h_tensor.dim() > 1:
                pred_24h = pred_24h_tensor[0].item()
            else:
                pred_24h = pred_24h_tensor.item()
            
            # Old confidence calculation
            value_estimate = output["value"].item() if output["value"].dim() == 0 else output["value"][0].item()
            confidence_1h = min(1.0, max(0.1, (abs(value_estimate) * 0.5 + abs(pred_1h) * 10.0)))
            confidence_24h = min(1.0, max(0.1, (abs(value_estimate) * 0.4 + abs(pred_24h) * 8.0)))
            
            if abs(pred_1h) < 1e-6:
                confidence_1h = 0.1
            if abs(pred_24h) < 1e-6:
                confidence_24h = 0.1
            
            logger.info(f"Regression predictions - 1h: {pred_1h:.6f}, 24h: {pred_24h:.6f}")
    
    return pred_1h, pred_24h, confidence_1h, confidence_24h, class_info


def generate_15m_price_prediction(
    model: TradingActorCritic,
    state_encoder: StateEncoder,
    price_data: list,
    price_features: Dict,
    news_data: list,
    position_state: Dict,
    current_price: float,
    timestamp: datetime,
    device: str = "cpu",
    # Multi-resolution data (POST-AUDIT FIX #2)
    hourly_prices: Optional[List[float]] = None,
    daily_prices: Optional[List[float]] = None,
) -> Tuple[float, float, float, str, Optional[Dict]]:
    """
    Generate 15-minute price prediction from the RL agent.
    
    POST-AUDIT: Now uses classification (Bearish/Neutral/Bullish) internally.
    
    Args:
        model: Trained TradingActorCritic model
        state_encoder: StateEncoder instance
        price_data: List of recent prices (5-min intervals)
        price_features: Dict of technical indicators
        news_data: List of news dicts with embeddings
        position_state: Dict with position info (size, portfolio_value, etc.)
        current_price: Current SOL price
        timestamp: Current timestamp
        device: Device to run model on
        hourly_prices: Optional list of hourly prices (for multi-resolution)
        daily_prices: Optional list of daily prices (for multi-resolution)
        
    Returns:
        Tuple of (pred_15m, confidence_15m, predicted_price_15m, method, class_info)
        - pred_15m: Predicted 15-minute return (decimal form, 0.01 = 1%)
        - confidence_15m: Confidence score (0.0 to 1.0)
        - predicted_price_15m: Predicted price in 15 minutes
        - method: "aux_head" if using dedicated head, "scaled_1h" if using fallback
        - class_info: Optional dict with class probabilities
    """
    model.eval()
    
    # Encode state with multi-resolution support (POST-AUDIT)
    state_dict = state_encoder.encode_full_state(
        prices=price_data[-60:] if len(price_data) >= 60 else price_data,
        price_features=price_features,
        news_data=news_data,
        position_size=position_state.get("position_size", 0.0),
        portfolio_value=position_state.get("portfolio_value", 10000.0),
        entry_price=position_state.get("entry_price"),
        current_price=current_price,
        time_since_last_trade=position_state.get("time_since_last_trade", 0.0),
        timestamp=timestamp,
        unrealized_pnl=position_state.get("unrealized_pnl", 0.0),
        hourly_prices=hourly_prices,  # Multi-resolution (FIX #2)
        daily_prices=daily_prices,  # Multi-resolution (FIX #2)
    )
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    
    # Multi-resolution tensors (POST-AUDIT FIX #2)
    price_medium_tensor = torch.FloatTensor(state_dict.get("price_medium", np.zeros(60))).unsqueeze(0).to(device)
    price_long_tensor = torch.FloatTensor(state_dict.get("price_long", np.zeros(30))).unsqueeze(0).to(device)
    
    # News age tensor (POST-AUDIT FIX #4)
    news_age_tensor = torch.FloatTensor(state_dict.get("news_age", np.zeros(20))).unsqueeze(0).to(device)
    
    # Create mask for news
    news_mask = (news_sent_tensor != 0.0).float()
    
    # Get predictions
    with torch.no_grad():
        output = model(
            price_tensor, news_emb_tensor, news_sent_tensor,
            position_tensor, time_tensor, news_mask,
            news_age=news_age_tensor,  # FIX #4
            price_medium=price_medium_tensor,  # FIX #2
            price_long=price_long_tensor,  # FIX #2
        )
        
        class_info = None
        
        # POST-AUDIT: Handle classification output (FIX #5)
        if "pred_15m_probs" in output and output["pred_15m_probs"] is not None:
            # New classification model
            pred_15m_probs = output["pred_15m_probs"][0].cpu().numpy()
            
            # Expected return = weighted sum with 15m-appropriate threshold
            # 15m: threshold=0.2%, so bear=-0.4%, neutral=0%, bull=+0.4%
            pred_15m = float(
                pred_15m_probs[0] * (-2 * THRESHOLD_15M) + 
                pred_15m_probs[1] * 0.0 + 
                pred_15m_probs[2] * (2 * THRESHOLD_15M)
            )
            
            # Confidence = max class probability
            confidence_15m = float(pred_15m_probs.max())
            method = "aux_head_class"
            
            class_info = {
                "class_probs": {
                    "bearish": float(pred_15m_probs[0]),
                    "neutral": float(pred_15m_probs[1]),
                    "bullish": float(pred_15m_probs[2]),
                },
                "predicted_class": CLASS_LABELS_15M[int(pred_15m_probs.argmax())],
            }
            
            logger.debug(f"15m classification: {class_info['predicted_class']} ({confidence_15m:.1%})")
            
        elif "pred_15m" in output and output["pred_15m"] is not None:
            # Old regression model with 15m head
            pred_15m_tensor = output["pred_15m"]
            
            if pred_15m_tensor.dim() > 1:
                pred_15m_raw = pred_15m_tensor[0].item()
            else:
                pred_15m_raw = pred_15m_tensor.item()
            
            # Get 1h for fallback
            pred_1h_tensor = output["pred_1h"]
            pred_1h = pred_1h_tensor[0].item() if pred_1h_tensor.dim() > 1 else pred_1h_tensor.item()
            
            # Check if producing meaningful predictions
            if abs(pred_15m_raw) < 1e-5:
                pred_15m = pred_1h * 0.25
                method = "scaled_1h"
                logger.debug(f"15m head untrained, using scaled 1h: {pred_15m:.6f}")
            else:
                pred_15m = pred_15m_raw
                method = "aux_head"
                logger.debug(f"Using trained 15m head: {pred_15m:.6f}")
            
            # Old confidence calculation
            value_estimate = output["value"].item() if output["value"].dim() == 0 else output["value"][0].item()
            confidence_15m = min(1.0, max(0.1, (abs(value_estimate) * 0.6 + abs(pred_15m) * 12.0)))
            
            if abs(pred_15m) < 1e-6:
                confidence_15m = 0.1
            if method == "scaled_1h":
                confidence_15m = confidence_15m * 0.9
        else:
            # No 15m head - use scaled 1h
            pred_1h_tensor = output["pred_1h"]
            pred_1h = pred_1h_tensor[0].item() if pred_1h_tensor.dim() > 1 else pred_1h_tensor.item()
            
            pred_15m = pred_1h * 0.25
            method = "scaled_1h"
            
            value_estimate = output["value"].item() if output["value"].dim() == 0 else output["value"][0].item()
            confidence_15m = min(1.0, max(0.1, (abs(value_estimate) * 0.6 + abs(pred_15m) * 12.0))) * 0.9
            
            logger.debug(f"No 15m head, using scaled 1h: {pred_15m:.6f}")
        
        # Calculate predicted price
        predicted_price_15m = current_price * (1.0 + pred_15m)
    
    return pred_15m, confidence_15m, predicted_price_15m, method, class_info


def store_prediction_from_decision(
    prediction_manager: PredictionManager,
    decision_id: Optional[int],
    timestamp: datetime,
    pred_1h: float,
    pred_24h: float,
    confidence_1h: Optional[float],
    confidence_24h: Optional[float],
    price_at_prediction: float,
) -> int:
    """
    Store a prediction linked to a decision.
    
    Args:
        prediction_manager: PredictionManager instance
        decision_id: ID of the decision
        timestamp: When prediction was made
        pred_1h: Predicted 1h return
        pred_24h: Predicted 24h return
        confidence_1h: Confidence for 1h prediction
        confidence_24h: Confidence for 24h prediction
        price_at_prediction: Price at time of prediction
        
    Returns:
        ID of stored prediction
    """
    return prediction_manager.store_prediction(
        decision_id=decision_id,
        timestamp=timestamp,
        predicted_return_1h=pred_1h,
        predicted_return_24h=pred_24h,
        predicted_confidence_1h=confidence_1h,
        predicted_confidence_24h=confidence_24h,
        price_at_prediction=price_at_prediction,
    )

