"""
Fix Corrupted Auxiliary Heads

This script fixes NaN weights in the auxiliary prediction heads by reinitializing them.
The auxiliary heads can get corrupted during training if NaN values propagate.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_auxiliary_heads(checkpoint_path: str, output_path: str = None):
    """
    Fix NaN weights in auxiliary heads by reinitializing them.
    
    Args:
        checkpoint_path: Path to corrupted checkpoint
        output_path: Path to save fixed checkpoint (default: adds '_fixed' suffix)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_state = checkpoint['model_state_dict']
    
    # Check for NaN in auxiliary heads
    aux_1h_keys = [k for k in model_state.keys() if 'aux_1h' in k]
    aux_24h_keys = [k for k in model_state.keys() if 'aux_24h' in k]
    
    has_nan = False
    for key in aux_1h_keys + aux_24h_keys:
        weight = model_state[key]
        if torch.isnan(weight).any():
            has_nan = True
            logger.warning(f"{key} has NaN values")
    
    if not has_nan:
        logger.info("No NaN values found in auxiliary heads. Model is OK.")
        return
    
    logger.info("Reinitializing auxiliary heads...")
    
    # Reinitialize aux_1h
    # aux_1h.0: Linear(256, 64)
    if 'aux_1h.0.weight' in model_state:
        nn.init.xavier_uniform_(model_state['aux_1h.0.weight'], gain=0.5)
        logger.info("Reinitialized aux_1h.0.weight")
    if 'aux_1h.0.bias' in model_state:
        nn.init.constant_(model_state['aux_1h.0.bias'], 0.0)
        logger.info("Reinitialized aux_1h.0.bias")
    
    # aux_1h.2: Linear(64, 1)
    if 'aux_1h.2.weight' in model_state:
        nn.init.xavier_uniform_(model_state['aux_1h.2.weight'], gain=0.5)
        logger.info("Reinitialized aux_1h.2.weight")
    if 'aux_1h.2.bias' in model_state:
        nn.init.constant_(model_state['aux_1h.2.bias'], 0.0)
        logger.info("Reinitialized aux_1h.2.bias")
    
    # Reinitialize aux_24h
    # aux_24h.0: Linear(256, 64)
    if 'aux_24h.0.weight' in model_state:
        nn.init.xavier_uniform_(model_state['aux_24h.0.weight'], gain=0.5)
        logger.info("Reinitialized aux_24h.0.weight")
    if 'aux_24h.0.bias' in model_state:
        nn.init.constant_(model_state['aux_24h.0.bias'], 0.0)
        logger.info("Reinitialized aux_24h.0.bias")
    
    # aux_24h.2: Linear(64, 1)
    if 'aux_24h.2.weight' in model_state:
        nn.init.xavier_uniform_(model_state['aux_24h.2.weight'], gain=0.5)
        logger.info("Reinitialized aux_24h.2.weight")
    if 'aux_24h.2.bias' in model_state:
        nn.init.constant_(model_state['aux_24h.2.bias'], 0.0)
        logger.info("Reinitialized aux_24h.2.bias")
    
    # Verify no NaN remains
    for key in aux_1h_keys + aux_24h_keys:
        weight = model_state[key]
        if torch.isnan(weight).any():
            logger.error(f"ERROR: {key} still has NaN after reinitialization!")
        else:
            logger.info(f"✓ {key} is now valid (mean={weight.mean().item():.6f})")
    
    # Save fixed checkpoint
    if output_path is None:
        checkpoint_file = Path(checkpoint_path)
        output_path = str(checkpoint_file.parent / f"{checkpoint_file.stem}_fixed{checkpoint_file.suffix}")
    
    logger.info(f"Saving fixed checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    logger.info("✅ Fixed checkpoint saved!")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix NaN weights in auxiliary heads")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to corrupted checkpoint (default: models/rl_agent/checkpoint_epoch_10.pt in project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for fixed checkpoint (default: adds '_fixed' suffix)"
    )
    
    args = parser.parse_args()
    
    # Set default checkpoint path if not provided
    if args.checkpoint is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.checkpoint = os.path.join(project_root, "models", "rl_agent", "checkpoint_epoch_10.pt")
    
    fix_auxiliary_heads(args.checkpoint, args.output)

