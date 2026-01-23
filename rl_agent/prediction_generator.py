"""
Prediction Generator

Helper functions to generate predictions from the RL agent model.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from .model import TradingActorCritic
from .state_encoder import StateEncoder
from .prediction_manager import PredictionManager

logger = logging.getLogger(__name__)


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
) -> Tuple[float, float, Optional[float], Optional[float]]:
    """
    Generate 1h and 24h return predictions from the RL agent.
    
    Args:
        model: Trained TradingActorCritic model
        state_encoder: StateEncoder instance
        price_data: List of recent prices
        price_features: Dict of technical indicators
        news_data: List of news dicts with embeddings
        position_state: Dict with position info (size, portfolio_value, etc.)
        current_price: Current SOL price
        timestamp: Current timestamp
        device: Device to run model on
        
    Returns:
        Tuple of (pred_1h, pred_24h, confidence_1h, confidence_24h)
        Returns are in decimal form (0.01 = 1%)
    """
    model.eval()
    
    # Encode state
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
    )
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    
    # Create mask for news
    news_mask = (news_sent_tensor != 0.0).float()
    
    # Get predictions
    with torch.no_grad():
        output = model(
            price_tensor, news_emb_tensor, news_sent_tensor,
            position_tensor, time_tensor, news_mask
        )
        
        # Extract predictions - handle both batched and unbatched outputs
        pred_1h_tensor = output["pred_1h"]
        pred_24h_tensor = output["pred_24h"]
        
        # If batched, take first element
        if pred_1h_tensor.dim() > 1:
            pred_1h = pred_1h_tensor[0].item()
        else:
            pred_1h = pred_1h_tensor.item()
            
        if pred_24h_tensor.dim() > 1:
            pred_24h = pred_24h_tensor[0].item()
        else:
            pred_24h = pred_24h_tensor.item()
        
        # Log raw predictions for debugging
        logger.info(f"Raw predictions - 1h: {pred_1h:.6f}, 24h: {pred_24h:.6f}")
        
        # Check if predictions are zero or near-zero (might indicate untrained model)
        if abs(pred_1h) < 1e-6 and abs(pred_24h) < 1e-6:
            logger.warning(
                "⚠️ Predictions are near-zero (1h: {:.6f}, 24h: {:.6f}). "
                "This may indicate: "
                "1) Model hasn't been trained yet, "
                "2) Auxiliary heads need more training, or "
                "3) Model needs retraining with auxiliary loss focus. "
                "Run: python scripts/fix_auxiliary_heads.py --checkpoint <checkpoint_path> "
                "or retrain with: python scripts/train_rl_agent.py --enable-auxiliary".format(pred_1h, pred_24h)
            )
        
        # For confidence, we could use:
        # 1. Model uncertainty (if we had ensemble or dropout)
        # 2. Value estimate variance
        # 3. Entropy of action distribution
        # For now, use a simple heuristic based on value estimate
        value_estimate = output["value"].item() if output["value"].dim() == 0 else output["value"][0].item()
        
        # Confidence based on prediction magnitude and value estimate
        # If predictions are near-zero, confidence should be low
        pred_magnitude_1h = abs(pred_1h)
        pred_magnitude_24h = abs(pred_24h)
        
        # Combine value estimate and prediction magnitude for confidence
        confidence_1h = min(1.0, max(0.1, (abs(value_estimate) * 0.5 + pred_magnitude_1h * 10.0)))
        confidence_24h = min(1.0, max(0.1, (abs(value_estimate) * 0.4 + pred_magnitude_24h * 8.0)))
        
        # If predictions are essentially zero, set low confidence
        if abs(pred_1h) < 1e-6:
            confidence_1h = 0.1  # Very low confidence for zero predictions
        if abs(pred_24h) < 1e-6:
            confidence_24h = 0.1  # Very low confidence for zero predictions
    
    return pred_1h, pred_24h, confidence_1h, confidence_24h


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
) -> Tuple[float, float, float, str]:
    """
    Generate 15-minute price prediction from the RL agent.
    
    Args:
        model: Trained TradingActorCritic model
        state_encoder: StateEncoder instance
        price_data: List of recent prices
        price_features: Dict of technical indicators
        news_data: List of news dicts with embeddings
        position_state: Dict with position info (size, portfolio_value, etc.)
        current_price: Current SOL price
        timestamp: Current timestamp
        device: Device to run model on
        
    Returns:
        Tuple of (pred_15m, confidence_15m, predicted_price_15m, method)
        - pred_15m: Predicted 15-minute return (decimal form, 0.01 = 1%)
        - confidence_15m: Confidence score (0.0 to 1.0)
        - predicted_price_15m: Predicted price in 15 minutes
        - method: "aux_head" if using dedicated head, "scaled_1h" if using fallback
    """
    model.eval()
    
    # Encode state (reuse existing logic)
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
    )
    
    # Convert to tensors
    price_tensor = torch.FloatTensor(state_dict["price"]).unsqueeze(0).to(device)
    news_emb_tensor = torch.FloatTensor(state_dict["news_embeddings"]).unsqueeze(0).to(device)
    news_sent_tensor = torch.FloatTensor(state_dict["news_sentiment"]).unsqueeze(0).to(device)
    position_tensor = torch.FloatTensor(state_dict["position"]).unsqueeze(0).to(device)
    time_tensor = torch.FloatTensor(state_dict["time"]).unsqueeze(0).to(device)
    
    # Create mask for news
    news_mask = (news_sent_tensor != 0.0).float()
    
    # Get predictions
    with torch.no_grad():
        output = model(
            price_tensor, news_emb_tensor, news_sent_tensor,
            position_tensor, time_tensor, news_mask
        )
        
        # Get 1h prediction first (needed for fallback)
        pred_1h_tensor = output["pred_1h"]
        if pred_1h_tensor.dim() > 1:
            pred_1h = pred_1h_tensor[0].item()
        else:
            pred_1h = pred_1h_tensor.item()
        
        # Check if model has 15m head and if it's producing meaningful predictions
        if "pred_15m" in output and output["pred_15m"] is not None:
            # Use dedicated 15m head
            pred_15m_tensor = output["pred_15m"]
            
            # Extract prediction
            if pred_15m_tensor.dim() > 1:
                pred_15m_raw = pred_15m_tensor[0].item()
            else:
                pred_15m_raw = pred_15m_tensor.item()
            
            # Check if 15m head is producing meaningful predictions (not just zeros)
            # If near-zero, it's likely untrained - fall back to scaled 1h
            if abs(pred_15m_raw) < 1e-5:
                # Head exists but is untrained (outputting zeros) - use fallback
                pred_15m = pred_1h * 0.25
                method = "scaled_1h"
                logger.debug(f"15m head exists but untrained (output: {pred_15m_raw:.6f}), using scaled 1h: {pred_15m:.6f}")
            else:
                # Head is trained and producing meaningful predictions
                pred_15m = pred_15m_raw
                method = "aux_head"
                logger.debug(f"Using trained 15m head: {pred_15m:.6f}")
        else:
            # Fallback: scale 1h prediction proportionally
            # 15 minutes = 0.25 of 1 hour
            # Scale proportionally: 15min = 1h * (15/60) = 1h * 0.25
            pred_15m = pred_1h * 0.25
            method = "scaled_1h"
            logger.debug(f"Using scaled 1h prediction: 1h={pred_1h:.6f}, 15m={pred_15m:.6f}")
        
        # Calculate confidence (similar to 1h confidence but adjusted for shorter horizon)
        value_estimate = output["value"].item() if output["value"].dim() == 0 else output["value"][0].item()
        pred_magnitude_15m = abs(pred_15m)
        
        # Confidence for 15m is typically higher than 1h (shorter horizon = more predictable)
        # But also consider that 15m predictions are more volatile
        confidence_15m = min(1.0, max(0.1, (abs(value_estimate) * 0.6 + pred_magnitude_15m * 12.0)))
        
        # If prediction is near-zero, set lower confidence
        if abs(pred_15m) < 1e-6:
            confidence_15m = 0.1
        
        # If using scaled method, reduce confidence slightly (less direct prediction)
        if method == "scaled_1h":
            confidence_15m = confidence_15m * 0.9  # 10% reduction for scaled predictions
        
        # Calculate predicted price
        predicted_price_15m = current_price * (1.0 + pred_15m)
    
    return pred_15m, confidence_15m, predicted_price_15m, method


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

