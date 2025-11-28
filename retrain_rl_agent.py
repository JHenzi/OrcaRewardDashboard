"""
Automated RL Agent Retraining Script

This script handles periodic retraining of the RL agent with new data.
Can be run manually or scheduled (cron, systemd, etc.).

Retraining Strategy:
- Full retraining: Retrain on all historical data (monthly)
- Incremental retraining: Fine-tune on new data only (weekly)
- Adaptive: Retrain based on data volume or performance metrics
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import sqlite3
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database paths
PRICE_DB = "sol_prices.db"
NEWS_DB = "news_sentiment.db"


def get_data_stats() -> dict:
    """Get statistics about available data."""
    stats = {}
    
    # Price data stats
    try:
        conn = sqlite3.connect(PRICE_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM sol_prices")
        row = cursor.fetchone()
        stats['price_count'] = row[0] if row[0] else 0
        stats['price_first'] = row[1]
        stats['price_last'] = row[2]
        conn.close()
    except Exception as e:
        logger.error(f"Error getting price stats: {e}")
        stats['price_count'] = 0
    
    # News data stats
    try:
        if Path(NEWS_DB).exists():
            conn = sqlite3.connect(NEWS_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), MIN(published_date), MAX(published_date) FROM news_articles")
            row = cursor.fetchone()
            stats['news_count'] = row[0] if row[0] else 0
            stats['news_first'] = row[1]
            stats['news_last'] = row[2]
            conn.close()
        else:
            stats['news_count'] = 0
    except Exception as e:
        logger.error(f"Error getting news stats: {e}")
        stats['news_count'] = 0
    
    return stats


def get_new_data_since(last_training_time: datetime) -> dict:
    """Get count of new data points since last training."""
    new_data = {'price_count': 0, 'news_count': 0}
    
    try:
        conn = sqlite3.connect(PRICE_DB)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sol_prices WHERE timestamp > ?",
            (last_training_time.isoformat(),)
        )
        new_data['price_count'] = cursor.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.error(f"Error getting new price data: {e}")
    
    try:
        if Path(NEWS_DB).exists():
            conn = sqlite3.connect(NEWS_DB)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM news_articles WHERE published_date > ?",
                (last_training_time.isoformat(),)
            )
            new_data['news_count'] = cursor.fetchone()[0]
            conn.close()
    except Exception as e:
        logger.error(f"Error getting new news data: {e}")
    
    return new_data


def should_retrain(
    last_training_time: Optional[datetime],
    mode: str = "adaptive",
    min_new_price_points: int = 2000,  # ~1 week of 5-min data
    min_new_news: int = 50,
    days_since_last: int = 7,
) -> Tuple[bool, str]:
    """
    Determine if retraining is needed.
    
    Args:
        last_training_time: When model was last trained
        mode: "adaptive", "weekly", "monthly", "always"
        min_new_price_points: Minimum new price points for adaptive mode
        min_new_news: Minimum new news articles for adaptive mode
        days_since_last: Days since last training for time-based modes
        
    Returns:
        Tuple of (should_retrain: bool, reason: str)
    """
    if mode == "always":
        return True, "Always retrain mode"
    
    if last_training_time is None:
        return True, "No previous training found"
    
    days_ago = (datetime.now() - last_training_time).days
    
    if mode == "weekly":
        if days_ago >= 7:
            return True, f"Last training was {days_ago} days ago (weekly threshold)"
        return False, f"Last training was {days_ago} days ago (too recent)"
    
    if mode == "monthly":
        if days_ago >= 30:
            return True, f"Last training was {days_ago} days ago (monthly threshold)"
        return False, f"Last training was {days_ago} days ago (too recent)"
    
    if mode == "adaptive":
        # Check time threshold
        if days_ago < days_since_last:
            return False, f"Last training was {days_ago} days ago (too recent)"
        
        # Check data volume
        new_data = get_new_data_since(last_training_time)
        
        if new_data['price_count'] < min_new_price_points:
            return False, f"Only {new_data['price_count']} new price points (need {min_new_price_points})"
        
        if new_data['news_count'] < min_new_news:
            logger.warning(f"Only {new_data['news_count']} new news articles (preferred: {min_new_news})")
            # News is optional, so we'll proceed anyway
        
        return True, f"Sufficient new data: {new_data['price_count']} prices, {new_data['news_count']} news"
    
    return False, f"Unknown mode: {mode}"


def get_last_training_time(checkpoint_dir: str = "models/rl_agent") -> Optional[datetime]:
    """Get timestamp of last training from checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None
    
    # Find most recent checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    most_recent = checkpoints[0]
    
    # Return modification time
    return datetime.fromtimestamp(most_recent.stat().st_mtime)


def retrain_model(
    mode: str = "incremental",
    epochs: int = 5,
    resume_from: Optional[str] = None,
    checkpoint_dir: str = "models/rl_agent",
    device: str = "cpu",
) -> bool:
    """
    Retrain the RL agent model.
    
    Args:
        mode: "full" (all data) or "incremental" (new data only)
        epochs: Number of training epochs
        resume_from: Checkpoint to resume from
        checkpoint_dir: Checkpoint directory
        device: Device to use
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Starting {mode} retraining...")
    
    # Find latest checkpoint if not specified
    if resume_from is None:
        checkpoint_dir_path = Path(checkpoint_dir)
        if checkpoint_dir_path.exists():
            checkpoints = list(checkpoint_dir_path.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
                resume_from = str(checkpoints[-1])
                logger.info(f"Resuming from: {resume_from}")
    
    # Prepare training data
    logger.info("Preparing training data...")
    prep_cmd = [sys.executable, "-m", "rl_agent.training_data_prep"]
    
    if mode == "incremental":
        # Only use recent data (last 30 days)
        # Could be enhanced to use only new data since last training
        logger.info("Using incremental mode: last 30 days of data")
    else:
        logger.info("Using full mode: all historical data")
    
    try:
        result = subprocess.run(prep_cmd, capture_output=True, text=True, check=True)
        logger.info("Data preparation completed")
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preparation failed: {e}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        return False
    
    # Train model
    logger.info("Training model...")
    train_cmd = [
        sys.executable,
        "train_rl_agent.py",
        "--epochs", str(epochs),
        "--device", device,
        "--checkpoint-dir", checkpoint_dir,
    ]
    
    if resume_from:
        train_cmd.extend(["--resume", resume_from])
    
    try:
        result = subprocess.run(train_cmd, check=True)
        logger.info("Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Retrain RL Agent with New Data")
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptive",
        choices=["adaptive", "weekly", "monthly", "always", "full", "incremental"],
        help="Retraining mode"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if conditions not met"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/rl_agent",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--min-price-points",
        type=int,
        default=2000,
        help="Minimum new price points for adaptive mode"
    )
    parser.add_argument(
        "--min-news",
        type=int,
        default=50,
        help="Minimum new news articles for adaptive mode"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("RL Agent Retraining Check")
    logger.info("=" * 60)
    
    # Get data statistics
    stats = get_data_stats()
    logger.info(f"Available data: {stats['price_count']} price points, {stats['news_count']} news articles")
    
    # Check if retraining is needed
    if args.mode in ["full", "incremental"]:
        # Direct training modes (for manual use)
        should_train = True
        reason = f"{args.mode} training mode"
    else:
        # Check conditions
        last_training = get_last_training_time(args.checkpoint_dir)
        if last_training:
            logger.info(f"Last training: {last_training}")
        else:
            logger.info("No previous training found")
        
        should_train, reason = should_retrain(
            last_training,
            mode=args.mode,
            min_new_price_points=args.min_price_points,
            min_new_news=args.min_news,
        )
    
    if args.force:
        should_train = True
        reason = "Force flag set"
    
    if not should_train:
        logger.info(f"❌ Retraining not needed: {reason}")
        return 0
    
    logger.info(f"✅ Retraining needed: {reason}")
    
    # Determine training mode
    if args.mode in ["full", "incremental"]:
        training_mode = args.mode
    elif args.mode == "adaptive":
        # Use incremental for adaptive (faster)
        training_mode = "incremental"
    else:
        training_mode = "incremental"
    
    # Retrain
    success = retrain_model(
        mode=training_mode,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )
    
    if success:
        logger.info("✅ Retraining completed successfully")
        return 0
    else:
        logger.error("❌ Retraining failed")
        return 1


if __name__ == "__main__":
    exit(main())

