"""
Fix NaN weights in entire model checkpoint.

This script detects and reinitializes ALL parameters with NaN values,
not just auxiliary heads. Use this when the entire model is corrupted.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_model_nan(checkpoint_path: str, output_path: str = None):
    """
    Fix NaN weights in entire model by reinitializing corrupted parameters.
    
    Args:
        checkpoint_path: Path to corrupted checkpoint
        output_path: Path to save fixed checkpoint (default: adds '_fixed' suffix)
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model_state = checkpoint['model_state_dict']
    
    # Find all parameters with NaN
    nan_params = []
    for key, param in model_state.items():
        if torch.isnan(param).any():
            nan_params.append(key)
    
    if not nan_params:
        logger.info("No NaN values found in model. Model is OK.")
        return checkpoint_path
    
    logger.warning(f"Found {len(nan_params)} parameters with NaN values:")
    for key in nan_params[:10]:  # Show first 10
        logger.warning(f"  - {key}")
    if len(nan_params) > 10:
        logger.warning(f"  ... and {len(nan_params) - 10} more")
    
    logger.info("Reinitializing corrupted parameters...")
    
    # Reinitialize each corrupted parameter based on its layer type
    fixed_count = 0
    for key in nan_params:
        param = model_state[key]
        
        # Determine initialization based on parameter name and shape
        if 'weight' in key:
            if param.dim() == 2:  # Linear layer
                # Use Xavier uniform with appropriate gain
                if 'aux' in key:
                    gain = 0.5  # Smaller gain for auxiliary heads
                else:
                    gain = 1.0  # Standard gain for main layers
                nn.init.xavier_uniform_(param, gain=gain)
            elif param.dim() == 4:  # Conv2d layer
                nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
            elif param.dim() == 1:  # 1D weight (unusual)
                nn.init.uniform_(param, -0.1, 0.1)
            else:
                # Fallback: uniform initialization
                nn.init.uniform_(param, -0.1, 0.1)
            fixed_count += 1
            logger.info(f"Reinitialized {key} (weight, dim={param.dim()})")
        elif 'bias' in key:
            nn.init.constant_(param, 0.0)
            fixed_count += 1
            logger.info(f"Reinitialized {key} (bias)")
        else:
            # Unknown parameter type - use uniform
            nn.init.uniform_(param, -0.1, 0.1)
            fixed_count += 1
            logger.warning(f"Reinitialized {key} (unknown type, using uniform)")
    
    # Verify no NaN remains
    remaining_nan = []
    for key, param in model_state.items():
        if torch.isnan(param).any():
            remaining_nan.append(key)
    
    if remaining_nan:
        logger.error(f"ERROR: {len(remaining_nan)} parameters still have NaN after reinitialization!")
        for key in remaining_nan:
            logger.error(f"  - {key}")
        return None
    else:
        logger.info(f"✅ Successfully fixed {fixed_count} parameters. No NaN remaining.")
    
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
    
    parser = argparse.ArgumentParser(description="Fix NaN weights in entire model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to corrupted checkpoint (default: models/rl_agent/checkpoint_20251128_180539.pt)"
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
        args.checkpoint = os.path.join(project_root, "models", "rl_agent", "checkpoint_20251128_180539.pt")
    
    result = fix_model_nan(args.checkpoint, args.output)
    
    if result:
        print(f"\n✅ Model fixed! Replace original with: {result}")
        print(f"   cp {result} {args.checkpoint}")
    else:
        print("\n❌ Failed to fix model")
        sys.exit(1)

