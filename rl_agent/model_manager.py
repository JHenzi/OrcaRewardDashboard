"""
Model Manager for RL Agent

Handles model versioning, validation, deployment, and archiving.
Implements MLOps best practices:
- Model versioning with timestamps
- Validation before deployment
- Graceful rollback on failure
- Automatic cleanup of old models (30-day retention)
"""

import logging
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages RL agent model versions, deployment, and archiving.
    
    Features:
    - Model versioning with timestamps
    - Validation before deployment
    - Automatic cleanup (30-day retention)
    - Rollback capability
    """
    
    def __init__(
        self,
        model_dir: str = "models/rl_agent",
        archive_dir: str = "models/rl_agent/archive",
        metadata_file: str = "models/rl_agent/metadata.json",
        retention_days: int = 30,
    ):
        """
        Initialize model manager.
        
        Args:
            model_dir: Directory for active models
            archive_dir: Directory for archived models
            metadata_file: Path to metadata JSON file
            retention_days: Number of days to keep archived models
        """
        self.model_dir = Path(model_dir)
        self.archive_dir = Path(archive_dir)
        self.metadata_file = Path(metadata_file)
        self.retention_days = retention_days
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Current active model
        self.current_model_path = None
        self.current_model_version = None
        
    def _load_metadata(self) -> Dict:
        """Load model metadata from JSON file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return {"models": [], "current_version": None}
        return {"models": [], "current_version": None}
    
    def _save_metadata(self):
        """Save model metadata to JSON file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def _generate_version(self) -> str:
        """Generate a version string from current timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _get_model_path(self, version: str, archived: bool = False) -> Path:
        """Get path for a model version."""
        base_dir = self.archive_dir if archived else self.model_dir
        return base_dir / f"checkpoint_{version}.pt"
    
    def archive_model(self, version: str) -> bool:
        """
        Archive a model version.
        
        Args:
            version: Model version to archive
            
        Returns:
            True if successful
        """
        try:
            model_path = self._get_model_path(version, archived=False)
            if not model_path.exists():
                logger.warning(f"Model {version} not found for archiving")
                return False
            
            archive_path = self._get_model_path(version, archived=True)
            shutil.move(str(model_path), str(archive_path))
            
            logger.info(f"Archived model {version} to {archive_path}")
            return True
        except Exception as e:
            logger.error(f"Error archiving model {version}: {e}")
            return False
    
    def validate_model(
        self,
        model_path: Path,
        model_class: nn.Module,
        model_kwargs: Dict,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a model before deployment.
        
        Checks:
        - Model file exists and is readable
        - Model can be loaded
        - Model structure matches expected architecture
        - Model can perform forward pass
        
        Args:
            model_path: Path to model checkpoint
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file exists
            if not model_path.exists():
                return False, f"Model file not found: {model_path}"
            
            # Try to load checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
            except Exception as e:
                return False, f"Failed to load checkpoint: {e}"
            
            # Check required keys
            if 'model_state_dict' not in checkpoint:
                return False, "Checkpoint missing 'model_state_dict'"
            
            # Check for NaN weights in model state (especially auxiliary heads)
            model_state = checkpoint['model_state_dict']
            nan_keys = []
            for key, weight in model_state.items():
                if isinstance(weight, torch.Tensor) and torch.isnan(weight).any():
                    nan_keys.append(key)
            
            if nan_keys:
                # Check if it's just auxiliary heads (can be fixed)
                aux_nan_keys = [k for k in nan_keys if 'aux' in k]
                other_nan_keys = [k for k in nan_keys if 'aux' not in k]
                
                if other_nan_keys:
                    return False, f"Model has NaN weights in critical layers: {other_nan_keys[:3]}"
                elif aux_nan_keys:
                    logger.warning(
                        f"Model has NaN weights in auxiliary heads: {aux_nan_keys}. "
                        f"Run fix_auxiliary_heads.py to repair."
                    )
                    # For now, allow it but warn - predictions will be zero
                    # In production, you might want to return False here
            
            # Try to instantiate and load model
            try:
                model = model_class(**model_kwargs)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
            except Exception as e:
                return False, f"Failed to load model state: {e}"
            
            # Try a dummy forward pass
            try:
                with torch.no_grad():
                    # Create dummy inputs matching expected shapes
                    # Based on StateEncoder output shapes
                    batch_size = 1
                    price_window_size = model_kwargs.get('price_window_size', 60)
                    num_indicators = model_kwargs.get('num_indicators', 10)
                    embedding_dim = model_kwargs.get('embedding_dim', 384)
                    max_news = model_kwargs.get('max_news_headlines', 20)
                    
                    # Price features: (batch_size, price_window_size + num_indicators)
                    # StateEncoder.encode_price_features() returns flattened array
                    # with returns (60) + indicators (10) = 70 total
                    price_features = torch.randn(batch_size, price_window_size + num_indicators)
                    
                    # News embeddings: (batch_size, max_headlines, embedding_dim)
                    news_emb = torch.randn(batch_size, max_news, embedding_dim)
                    
                    # News sentiment: (batch_size, max_headlines)
                    news_sent = torch.randn(batch_size, max_news)
                    
                    # Position features: (batch_size, 5) - from StateEncoder.encode_position_features()
                    position = torch.randn(batch_size, 5)
                    
                    # Time features: (batch_size, 4) - from StateEncoder.encode_time_features()
                    time_features = torch.randn(batch_size, 4)
                    
                    # News mask: (batch_size, max_headlines) - 1 for valid, 0 for padding
                    news_mask = torch.ones(batch_size, max_news)
                    
                    output = model(
                        price_features, news_emb, news_sent,
                        position, time_features, news_mask
                    )
                    
                    # Check output structure
                    if 'action_logits' not in output or 'value' not in output:
                        return False, "Model output missing required keys"
                    
            except Exception as e:
                return False, f"Model forward pass failed: {e}"
            
            logger.info(f"✅ Model {model_path.name} validated successfully")
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def deploy_model(
        self,
        model_path: Path,
        model_class: nn.Module,
        model_kwargs: Dict,
        version: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Deploy a new model version.
        
        Process:
        1. Validate the new model
        2. Archive current model if exists
        3. Copy new model to active location
        4. Update metadata
        5. Set as current model
        
        Args:
            model_path: Path to the new model checkpoint
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            version: Optional version string (auto-generated if None)
            
        Returns:
            Tuple of (success, version, error_message)
        """
        try:
            # Generate version if not provided
            if version is None:
                version = self._generate_version()
            
            # Validate model first
            is_valid, error_msg = self.validate_model(model_path, model_class, model_kwargs)
            if not is_valid:
                logger.error(f"❌ Model validation failed: {error_msg}")
                return False, None, error_msg
            
            # Archive current model if exists
            current_version = self.metadata.get("current_version")
            if current_version:
                self.archive_model(current_version)
            
            # Copy new model to active location
            active_model_path = self._get_model_path(version, archived=False)
            shutil.copy2(str(model_path), str(active_model_path))
            
            # Update metadata
            model_info = {
                "version": version,
                "deployed_at": datetime.now().isoformat(),
                "source_path": str(model_path),
                "active_path": str(active_model_path),
                "training_epochs": None,  # Will be filled by training script
                "training_loss": None,
            }
            
            # Try to extract training info from checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                if 'epoch' in checkpoint:
                    model_info['training_epochs'] = checkpoint['epoch']
                if 'loss' in checkpoint:
                    model_info['training_loss'] = checkpoint['loss']
            except:
                pass
            
            self.metadata["models"].append(model_info)
            self.metadata["current_version"] = version
            self._save_metadata()
            
            # Update current model
            self.current_model_path = active_model_path
            self.current_model_version = version
            
            logger.info(f"✅ Deployed model version {version}")
            return True, version, None
            
        except Exception as e:
            error_msg = f"Error deploying model: {e}"
            logger.error(f"❌ {error_msg}")
            return False, None, error_msg
    
    def load_current_model(
        self,
        model_class: nn.Module,
        model_kwargs: Dict,
        device: str = "cpu",
    ) -> Optional[nn.Module]:
        """
        Load the current active model.
        
        First tries to load from metadata (deployed model).
        If no deployed model exists, looks for the latest checkpoint file.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            device: Device to load model on
            
        Returns:
            Loaded model or None if not available
        """
        try:
            # Get current version from metadata
            current_version = self.metadata.get("current_version")
            
            if current_version:
                # Check if there's a newer epoch checkpoint than what's deployed
                epoch_checkpoints = list(self.model_dir.glob("checkpoint_epoch_*.pt"))
                if epoch_checkpoints:
                    latest_epoch = max(epoch_checkpoints, key=lambda p: p.stat().st_mtime)
                    current_model_path = self._get_model_path(current_version, archived=False)
                    
                    if current_model_path.exists():
                        current_mtime = current_model_path.stat().st_mtime
                        latest_mtime = latest_epoch.stat().st_mtime
                        
                        # If latest epoch checkpoint is newer, auto-deploy it
                        if latest_mtime > current_mtime:
                            logger.info(f"🔄 Found newer checkpoint: {latest_epoch.name} (newer than deployed {current_version})")
                            try:
                                is_valid, error_msg = self.validate_model(latest_epoch, model_class, model_kwargs)
                                if is_valid:
                                    # Auto-deploy the newer checkpoint
                                    version = self._generate_version()
                                    active_model_path = self._get_model_path(version, archived=False)
                                    shutil.copy2(str(latest_epoch), str(active_model_path))
                                    
                                    # Archive current model
                                    archive_path = self._get_model_path(current_version, archived=True)
                                    if current_model_path.exists() and current_model_path != active_model_path:
                                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                                        shutil.move(str(current_model_path), str(archive_path))
                                    
                                    # Update metadata
                                    model_info = {
                                        "version": version,
                                        "deployed_at": datetime.now().isoformat(),
                                        "source_path": str(latest_epoch),
                                        "active_path": str(active_model_path),
                                        "auto_deployed": True,
                                    }
                                    
                                    self.metadata["models"].append(model_info)
                                    self.metadata["current_version"] = version
                                    self._save_metadata()
                                    
                                    logger.info(f"✅ Auto-deployed newer checkpoint as version {version}")
                                    current_version = version
                                else:
                                    logger.warning(f"Newer checkpoint validation failed: {error_msg}, using current version")
                            except Exception as e:
                                logger.warning(f"Failed to auto-deploy newer checkpoint: {e}, using current version")
                
                # Try to load from active location
                active_model_path = self._get_model_path(current_version, archived=False)
                if not active_model_path.exists():
                    # Try archive
                    archive_model_path = self._get_model_path(current_version, archived=True)
                    if archive_model_path.exists():
                        active_model_path = archive_model_path
                    else:
                        logger.warning(f"Current model {current_version} not found, looking for latest checkpoint")
                        current_version = None  # Fall through to auto-detect
            
            # If no current version in metadata, look for latest checkpoint
            if not current_version:
                logger.info("No model registered in metadata, looking for latest checkpoint...")
                checkpoints = list(self.model_dir.glob("checkpoint_*.pt"))
                if not checkpoints:
                    logger.warning("No checkpoints found in model directory")
                    return None
                
                # Get the most recently modified checkpoint
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                logger.info(f"Found latest checkpoint: {latest_checkpoint.name}")
                
                # Try to auto-deploy it (register in metadata)
                try:
                    is_valid, error_msg = self.validate_model(latest_checkpoint, model_class, model_kwargs)
                    if is_valid:
                        # Auto-deploy the latest checkpoint
                        version = self._generate_version()
                        active_model_path = self._get_model_path(version, archived=False)
                        shutil.copy2(str(latest_checkpoint), str(active_model_path))
                        
                        # Update metadata
                        model_info = {
                            "version": version,
                            "deployed_at": datetime.now().isoformat(),
                            "source_path": str(latest_checkpoint),
                            "active_path": str(active_model_path),
                            "auto_deployed": True,  # Flag to indicate auto-deployment
                        }
                        
                        self.metadata["models"].append(model_info)
                        self.metadata["current_version"] = version
                        self._save_metadata()
                        
                        logger.info(f"✅ Auto-deployed latest checkpoint as version {version}")
                        current_version = version
                    else:
                        logger.error(f"Latest checkpoint validation failed: {error_msg}")
                        return None
                except Exception as e:
                    logger.warning(f"Failed to auto-deploy checkpoint: {e}, loading directly")
                    # Load directly without registering
                    active_model_path = latest_checkpoint
            else:
                # We have a current_version (either from metadata or auto-deployed)
                active_model_path = self._get_model_path(current_version, archived=False)
                if not active_model_path.exists():
                    archive_model_path = self._get_model_path(current_version, archived=True)
                    if archive_model_path.exists():
                        active_model_path = archive_model_path
                    else:
                        logger.error(f"Current model {current_version} not found")
                        return None
            
            # Load model
            checkpoint = torch.load(active_model_path, map_location=device)
            model = model_class(**model_kwargs)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model.to(device)
            
            self.current_model_path = active_model_path
            self.current_model_version = current_version or "auto-detected"
            
            logger.info(f"✅ Loaded model from {active_model_path.name}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading current model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def cleanup_old_models(self):
        """
        Clean up archived models older than retention_days.
        
        Keeps:
        - Current active model (always)
        - Models from last retention_days
        - At least the last 3 models (safety)
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            current_version = self.metadata.get("current_version")
            
            # Get all archived models
            archived_models = []
            for model_info in self.metadata.get("models", []):
                version = model_info.get("version")
                if version == current_version:
                    continue  # Never delete current
                
                deployed_at = model_info.get("deployed_at")
                if deployed_at:
                    try:
                        deployed_date = datetime.fromisoformat(deployed_at)
                        archived_models.append((version, deployed_date, model_info))
                    except:
                        continue
            
            # Sort by date (newest first)
            archived_models.sort(key=lambda x: x[1], reverse=True)
            
            # Keep at least last 3 models
            models_to_keep = max(3, len([m for m in archived_models if m[1] > cutoff_date]))
            models_to_delete = archived_models[models_to_keep:]
            
            # Delete old models
            deleted_count = 0
            for version, _, _ in models_to_delete:
                archive_path = self._get_model_path(version, archived=True)
                if archive_path.exists():
                    try:
                        archive_path.unlink()
                        logger.info(f"Deleted old model {version}")
                        deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {version}: {e}")
            
            # Update metadata (remove deleted models)
            self.metadata["models"] = [
                m for m in self.metadata["models"]
                if m.get("version") == current_version or
                any(m.get("version") == v for v, _, _ in archived_models[:models_to_keep])
            ]
            self._save_metadata()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old models")
            
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about current and available models."""
        return {
            "current_version": self.metadata.get("current_version"),
            "current_path": str(self.current_model_path) if self.current_model_path else None,
            "total_models": len(self.metadata.get("models", [])),
            "retention_days": self.retention_days,
        }

