#!/usr/bin/env python3
"""
Unified training script for all inpainting models on OAI dataset.
Platform-agnostic and reproducible.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def train_aot_gan(config_path):
    """Train AOT-GAN model."""
    print("🚀 Training AOT-GAN...")

    # Change to AOT-GAN directory
    aot_gan_dir = project_root / "models" / "aot-gan" / "src"
    os.chdir(aot_gan_dir)

    # Import and run training
    import torch.multiprocessing as mp
    from train import main_worker

    # Parse config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create args object
    class Args:
        def __init__(self, config):
            for key, value in config.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        setattr(self, subkey, subvalue)
                else:
                    setattr(self, key, value)

    args = Args(config)

    # Set up distributed training if needed
    if args.hardware.get("distributed", False):
        mp.spawn(main_worker, nprocs=1, args=(1, args))
    else:
        main_worker(0, 1, args)


def train_ict(config_path):
    """Train ICT model."""
    print("🚀 Training ICT...")

    # Change to ICT directory
    ict_dir = project_root / "models" / "ict" / "Guided_Upsample"
    os.chdir(ict_dir)

    # Import and run training
    from main import main

    main(mode=1)  # Training mode


def train_repaint(config_path):
    """Train RePaint model (note: RePaint is inference-only)."""
    print("⚠️  RePaint is inference-only. Use test.py for inference.")


def main():
    parser = argparse.ArgumentParser(
        description="Train inpainting models on OAI dataset"
    )
    parser.add_argument(
        "--model",
        choices=["aot-gan", "ict", "repaint"],
        required=True,
        help="Model to train",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    # Create results directory
    results_dir = project_root / "results" / "logs" / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    # Train selected model
    if args.model == "aot-gan":
        train_aot_gan(config_path)
    elif args.model == "ict":
        train_ict(config_path)
    elif args.model == "repaint":
        train_repaint(config_path)

    print(f"✅ Training completed for {args.model}")


if __name__ == "__main__":
    main()
