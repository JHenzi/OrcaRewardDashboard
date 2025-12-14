"""
Retraining Scheduler for RL Agent

Handles automated, asynchronous retraining of the RL agent.
Runs in a background thread and integrates with ModelManager for safe deployment.
"""

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple
import json

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Schedules and executes automated RL agent retraining.
    
    Features:
    - Weekly retraining schedule
    - Asynchronous execution (non-blocking)
    - Model validation before deployment
    - Graceful error handling
    - Keeps old model if new one fails
    """
    
    def __init__(
        self,
        model_manager,
        retrain_script: str = "scripts/retrain_rl_agent.py",
        training_script: str = "scripts/train_rl_agent.py",
        data_prep_script: str = "rl_agent/training_data_prep.py",
        interval_days: int = 7,
        enabled: bool = True,
    ):
        """
        Initialize retraining scheduler.
        
        Args:
            model_manager: ModelManager instance
            retrain_script: Path to retraining script
            training_script: Path to training script
            data_prep_script: Path to data preparation script
            interval_days: Days between retraining (default: 7 for weekly)
            enabled: Whether scheduler is enabled
        """
        self.model_manager = model_manager
        
        # Resolve paths relative to project root (where app.py is located)
        # Get project root by going up from this file's directory
        project_root = Path(__file__).parent.parent.resolve()
        self.retrain_script = (project_root / retrain_script).resolve()
        self.training_script = (project_root / training_script).resolve()
        self.data_prep_script = (project_root / data_prep_script).resolve()
        
        # Validate that retrain script exists
        if not self.retrain_script.exists():
            logger.error(f"❌ Retrain script not found at: {self.retrain_script}")
            logger.error(f"   Project root: {project_root}")
            logger.error(f"   Expected path: {self.retrain_script}")
            raise FileNotFoundError(f"Retrain script not found: {self.retrain_script}")
        
        self.interval_days = interval_days
        self.enabled = enabled
        
        # State
        self.last_retrain_time: Optional[datetime] = None
        self.next_retrain_time: Optional[datetime] = None
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        
        # Load last retrain time from metadata
        self._load_schedule_state()
    
    def _load_schedule_state(self):
        """Load schedule state from metadata."""
        try:
            metadata_file = self.model_manager.metadata_file
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    last_retrain = metadata.get("last_retrain_time")
                    next_retrain = metadata.get("next_retrain_time")
                    
                    if last_retrain:
                        self.last_retrain_time = datetime.fromisoformat(last_retrain)
                        self._calculate_next_retrain()
                    elif next_retrain:
                        # If we have a scheduled time but no last retrain, use it
                        self.next_retrain_time = datetime.fromisoformat(next_retrain)
            
            # If no state loaded, initialize to future date (don't train immediately)
            if self.next_retrain_time is None:
                self._calculate_next_retrain()
                self._save_schedule_state()  # Persist the initial schedule
                logger.info(f"📅 Initialized retraining schedule: next retrain at {self.next_retrain_time}")
        except Exception as e:
            logger.warning(f"Error loading schedule state: {e}")
            # On error, still initialize to prevent immediate training
            if self.next_retrain_time is None:
                self._calculate_next_retrain()
    
    def _save_schedule_state(self):
        """Save schedule state to metadata."""
        try:
            metadata = self.model_manager.metadata
            metadata["last_retrain_time"] = self.last_retrain_time.isoformat() if self.last_retrain_time else None
            metadata["next_retrain_time"] = self.next_retrain_time.isoformat() if self.next_retrain_time else None
            self.model_manager._save_metadata()
        except Exception as e:
            logger.warning(f"Error saving schedule state: {e}")
    
    def _calculate_next_retrain(self):
        """Calculate next retrain time based on last retrain."""
        if self.last_retrain_time:
            self.next_retrain_time = self.last_retrain_time + timedelta(days=self.interval_days)
        else:
            # If never trained, schedule for next interval
            self.next_retrain_time = datetime.now() + timedelta(days=self.interval_days)
    
    def _should_retrain(self) -> bool:
        """Check if it's time to retrain."""
        if not self.enabled:
            return False
        
        if self.is_training:
            return False  # Already training
        
        # Never trigger training if next_retrain_time is None
        # This should not happen if state is properly initialized
        if self.next_retrain_time is None:
            logger.warning("next_retrain_time is None - initializing schedule to prevent immediate training")
            self._calculate_next_retrain()
            self._save_schedule_state()
            return False  # Don't train immediately
        
        return datetime.now() >= self.next_retrain_time
    
    def _run_retraining(self) -> Tuple[bool, Optional[str], Optional[Path]]:
        """
        Run the retraining process.
        
        Returns:
            Tuple of (success, error_message, model_path)
        """
        try:
            logger.info("🔄 Starting automated retraining...")
            
            checkpoint_dir = self.model_manager.model_dir
            
            # CRITICAL: Record existing checkpoints BEFORE training starts
            # This prevents deploying old checkpoints if training doesn't create new ones
            existing_checkpoints = {}
            if checkpoint_dir.exists():
                for checkpoint in checkpoint_dir.glob("checkpoint_*.pt"):
                    existing_checkpoints[checkpoint] = checkpoint.stat().st_mtime
            training_start_time = time.time()
            logger.info(f"📋 Found {len(existing_checkpoints)} existing checkpoints before training")
            
            # Run retraining (retrain script handles data prep and training)
            logger.info("Running retraining script (handles data prep and training)...")
            
            # Run training using train_rl_agent.py (retrain_rl_agent.py is a wrapper)
            train_result = subprocess.run(
                [
                    sys.executable,
                    str(self.retrain_script),
                    "--mode", "incremental",  # Use incremental for weekly
                    "--epochs", "5",  # Fewer epochs for weekly retraining
                    "--checkpoint-dir", str(checkpoint_dir),
                ],
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            
            # Log training output for debugging
            if train_result.stdout:
                logger.info(f"Training stdout: {train_result.stdout[-500:]}")  # Last 500 chars
            if train_result.stderr:
                logger.warning(f"Training stderr: {train_result.stderr[-500:]}")
            
            if train_result.returncode != 0:
                error_msg = f"Training failed (exit code {train_result.returncode}): {train_result.stderr}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg, None
            
            # Step 3: Find NEW checkpoints created during training
            # Only consider checkpoints that are NEW or were modified AFTER training started
            all_checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
            new_checkpoints = []
            
            for checkpoint in all_checkpoints:
                checkpoint_mtime = checkpoint.stat().st_mtime
                is_new = checkpoint not in existing_checkpoints
                was_modified = checkpoint in existing_checkpoints and checkpoint_mtime > existing_checkpoints[checkpoint]
                created_during_training = checkpoint_mtime >= training_start_time
                
                if is_new or was_modified or created_during_training:
                    new_checkpoints.append((checkpoint, checkpoint_mtime))
                    logger.info(f"✅ Found new/modified checkpoint: {checkpoint.name} (mtime: {checkpoint_mtime:.0f})")
            
            if not new_checkpoints:
                error_msg = (
                    f"❌ Training completed but NO NEW checkpoints were created! "
                    f"Found {len(all_checkpoints)} total checkpoints, but none were created/modified during training. "
                    f"This indicates training did not actually run or failed silently."
                )
                logger.error(error_msg)
                logger.error(f"Existing checkpoints before training: {list(existing_checkpoints.keys())}")
                logger.error(f"All checkpoints after training: {[c.name for c in all_checkpoints]}")
                return False, error_msg, None
            
            # Get the most recently created/modified NEW checkpoint
            checkpoint_path, checkpoint_mtime = max(new_checkpoints, key=lambda x: x[1])
            logger.info(f"✅ Using newly created checkpoint: {checkpoint_path.name} (modified at {checkpoint_mtime:.0f})")
            
            # Validate that this checkpoint is actually from training (not just an old file)
            if checkpoint_path in existing_checkpoints:
                old_mtime = existing_checkpoints[checkpoint_path]
                if checkpoint_mtime <= old_mtime:
                    error_msg = (
                        f"❌ Checkpoint {checkpoint_path.name} was not actually modified during training! "
                        f"Old mtime: {old_mtime:.0f}, Current mtime: {checkpoint_mtime:.0f}"
                    )
                    logger.error(error_msg)
                    return False, error_msg, None
            
            logger.info(f"✅ Model trained successfully: {checkpoint_path}")
            return True, None, checkpoint_path
            
        except subprocess.TimeoutExpired:
            error_msg = "Training process timed out"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Retraining error: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg, None
    
    def _retrain_async(self):
        """Run retraining in background thread."""
        try:
            self.is_training = True
            logger.info("🔄 Retraining started (async)")
            
            # Run retraining
            success, error_msg, model_path = self._run_retraining()
            
            if success and model_path:
                # Deploy new model (ModelManager handles validation)
                logger.info("Deploying new model...")
                from rl_agent.model import TradingActorCritic
                
                model_kwargs = {
                    "price_window_size": 60,
                    "num_indicators": 10,
                    "embedding_dim": 384,
                    "max_news_headlines": 20,
                    "num_actions": 3,
                }
                
                deploy_success, version, deploy_error = self.model_manager.deploy_model(
                    model_path=model_path,
                    model_class=TradingActorCritic,
                    model_kwargs=model_kwargs,
                )
                
                if deploy_success:
                    logger.info(f"✅ Retraining complete! New model {version} deployed")
                    self.last_retrain_time = datetime.now()
                    self._calculate_next_retrain()
                    self._save_schedule_state()
                    
                    # Cleanup old models
                    self.model_manager.cleanup_old_models()
                else:
                    logger.error(f"❌ Model deployment failed: {deploy_error}")
                    logger.info("⚠️ Keeping current model active")
            else:
                logger.error(f"❌ Retraining failed: {error_msg}")
                logger.info("⚠️ Keeping current model active")
                
        except Exception as e:
            logger.error(f"❌ Retraining thread error: {e}")
        finally:
            self.is_training = False
            logger.info("🔄 Retraining thread finished")
    
    def trigger_retrain(self, run_async: bool = True, force: bool = False):
        """
        Manually trigger retraining.
        
        Args:
            run_async: Whether to run asynchronously (default: True)
            force: If True, trigger even if already training (default: False)
        """
        if self.is_training and not force:
            logger.warning("Retraining already in progress - skipping")
            return
        
        if run_async:
            # Check if a training thread is already running
            if self.training_thread and self.training_thread.is_alive():
                logger.warning("Training thread already running - skipping duplicate trigger")
                return
            
            self.training_thread = threading.Thread(target=self._retrain_async, daemon=True)
            self.training_thread.start()
            logger.info("🔄 Retraining triggered (async)")
        else:
            self._retrain_async()
    
    def check_and_retrain(self):
        """Check if retraining is needed and trigger if so."""
        if self._should_retrain():
            logger.info(f"⏰ Time to retrain! (scheduled for {self.next_retrain_time})")
            self.trigger_retrain(run_async=True)
    
    def start_scheduler(self, check_interval_seconds: int = 3600):
        """
        Start the scheduler loop in a background thread.
        
        Args:
            check_interval_seconds: How often to check if retraining is needed (default: 1 hour)
        """
        def scheduler_loop():
            logger.info(f"📅 Retraining scheduler started (checking every {check_interval_seconds}s)")
            while True:
                try:
                    self.check_and_retrain()
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                
                time.sleep(check_interval_seconds)
        
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        logger.info("✅ Retraining scheduler thread started")
    
    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "enabled": self.enabled,
            "is_training": self.is_training,
            "last_retrain_time": self.last_retrain_time.isoformat() if self.last_retrain_time else None,
            "next_retrain_time": self.next_retrain_time.isoformat() if self.next_retrain_time else None,
            "interval_days": self.interval_days,
        }

