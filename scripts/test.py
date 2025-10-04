#!/usr/bin/env python3
"""
Unified testing script for all inpainting models on OAI dataset.
Platform-agnostic and reproducible.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_aot_gan(config_path):
    """Test AOT-GAN model."""
    print("🧪 Testing AOT-GAN...")

    # Change to AOT-GAN directory
    aot_gan_dir = project_root / "AOT-GAN-for-Inpainting" / "src"
    os.chdir(aot_gan_dir)

    # Import and run testing
    from test import main_worker

    # Parse config
    with open(config_path, "r") as f:
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

    # Run testing
    main_worker(args, use_gpu=True)


def test_ict(config_path):
    """Test ICT model."""
    print("🧪 Testing ICT...")

    # Change to ICT directory
    ict_dir = project_root / "ICT" / "Guided_Upsample"
    os.chdir(ict_dir)

    # Import and run testing
    from main import main

    main(mode=2)  # Testing mode


def test_repaint(config_path):
    """Test RePaint model."""
    print("🧪 Testing RePaint...")

    # Change to RePaint directory
    repaint_dir = project_root / "RePaint"
    os.chdir(repaint_dir)

    # Import and run testing
    from test import main
    from conf_mgt.conf_base import Default_Conf

    # Parse config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create config object
    conf = Default_Conf()
    for key, value in config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                setattr(conf, subkey, subvalue)
        else:
            setattr(conf, key, value)

    # Run testing
    main(conf)


def main():
    parser = argparse.ArgumentParser(
        description="Test inpainting models on OAI dataset"
    )
    parser.add_argument(
        "--model",
        choices=["aot-gan", "ict", "repaint"],
        required=True,
        help="Model to test",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        help="Dataset subset to test on (test, subset_4)",
    )

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)

    # Create output directory
    output_dir = project_root / "output" / args.model.upper() / "OAI"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test selected model
    if args.model == "aot-gan":
        test_aot_gan(config_path)
    elif args.model == "ict":
        test_ict(config_path)
    elif args.model == "repaint":
        test_repaint(config_path)

    print(f"✅ Testing completed for {args.model}")


if __name__ == "__main__":
    main()
