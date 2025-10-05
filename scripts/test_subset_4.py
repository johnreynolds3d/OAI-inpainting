#!/usr/bin/env python3
"""
Test all model variants using subset_4 data.
This script runs AOT-GAN, ICT, and RePaint on the subset_4 evaluation set.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, description, timeout=300):
    """Run a command with timeout and error handling."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=project_root, capture_output=True, text=True, timeout=timeout
        )
        end_time = time.time()

        if result.returncode == 0:
            print(
                f"✅ {description} completed successfully ({end_time - start_time:.1f}s)"
            )
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr[-500:])  # Last 500 chars
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

    return True


def test_aot_gan():
    """Test AOT-GAN on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING AOT-GAN ON SUBSET_4")
    print("=" * 60)

    # Check if AOT-GAN pretrained models exist
    aot_gan_models = project_root / "data" / "pretrained" / "aot-gan" / "OAI"
    if not aot_gan_models.exists() or not any(aot_gan_models.iterdir()):
        print("⚠️  AOT-GAN OAI pretrained models not found, using Places2 models")
        model_path = project_root / "data" / "pretrained" / "aot-gan" / "places2"
    else:
        model_path = aot_gan_models

    # Create subset_4 config for AOT-GAN
    config_content = f"""# AOT-GAN Configuration for OAI Subset_4
model: "aotgan"
pre_train: "{model_path}/G0000000.pt"
dir_image: "{project_root}/data/oai/test/img/subset_4"
dir_mask: "{project_root}/data/oai/test/mask/subset_4"
outputs: "{project_root}/output/AOT-GAN/OAI/subset_4"
"""

    config_path = project_root / "configs" / "aot-gan" / "subset_4_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        f.write(config_content)

    # Run AOT-GAN test
    cmd = [
        "python",
        "scripts/test.py",
        "--model",
        "aot-gan",
        "--config",
        str(config_path),
        "--subset",
        "subset_4",
    ]

    return run_command(cmd, "AOT-GAN testing on subset_4")


def test_ict():
    """Test ICT on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ICT ON SUBSET_4")
    print("=" * 60)

    # Check if ICT pretrained models exist
    ict_models = project_root / "data" / "pretrained" / "ict" / "Upsample" / "OAI"
    if not ict_models.exists() or not any(ict_models.iterdir()):
        print("⚠️  ICT OAI pretrained models not found, using Places2_Nature models")
        model_path = (
            project_root / "data" / "pretrained" / "ict" / "Upsample" / "Places2_Nature"
        )
    else:
        model_path = ict_models

    # Create subset_4 config for ICT
    config_content = f"""# ICT Configuration for OAI Subset_4
MODE: 2
MODEL: 2
MASK: 3
EDGE: 1
NMS: 1
SEED: 10
GPU: [0]
DEBUG: 0
VERBOSE: 0

Generator: 0
No_Bar: True

# Dataset paths for subset_4
TEST_FLIST: "{project_root}/data/oai/test/img/subset_4"
TEST_EDGE_FLIST: "{project_root}/data/oai/test/edge/subset_4"
TEST_MASK_FLIST: "{project_root}/data/oai/test/mask/subset_4"

# Model paths
PATH: "{model_path}"
RESULTS: "{project_root}/output/ICT/OAI/subset_4"

# Inference parameters
BATCH_SIZE: 1
INPUT_SIZE: 256
"""

    config_path = project_root / "configs" / "ict" / "subset_4_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        f.write(config_content)

    # Run ICT test
    cmd = [
        "python",
        "scripts/test.py",
        "--model",
        "ict",
        "--config",
        str(config_path),
        "--subset",
        "subset_4",
    ]

    return run_command(cmd, "ICT testing on subset_4")


def test_repaint():
    """Test RePaint on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING REPAINT ON SUBSET_4")
    print("=" * 60)

    # Create subset_4 config for RePaint
    config_content = f"""# RePaint Configuration for OAI Subset_4
model:
  name: "repaint"
  model_path: "{project_root}/data/pretrained/repaint/256x256_diffusion.pt"
  classifier_path: "{project_root}/data/pretrained/repaint/256x256_classifier.pt"

# Data paths for subset_4
data:
  gt_path: "{project_root}/data/oai/test/img/subset_4"
  mask_path: "{project_root}/data/oai/test/mask/subset_4"
  output_path: "{project_root}/output/RePaint/OAI/subset_4"

# Inference parameters
inference:
  sample_num: 1
  jump_length: 10
  jump_n_sample: 10
  n_steps: 1000
  guidance_scale: 1.0

# Hardware configuration
hardware:
  device: "cuda"
  batch_size: 1

# Visualization
visualization:
  visualize_all: true
  save_intermediate: false

# Paths
paths:
  results_dir: "{project_root}/results/repaint/subset_4"
  logs_dir: "{project_root}/results/logs/repaint/subset_4"
"""

    config_path = project_root / "configs" / "repaint" / "subset_4_config.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        f.write(config_content)

    # Run RePaint test
    cmd = [
        "python",
        "scripts/test.py",
        "--model",
        "repaint",
        "--config",
        str(config_path),
        "--subset",
        "subset_4",
    ]

    return run_command(cmd, "RePaint testing on subset_4")


def evaluate_results():
    """Evaluate and compare results from all models."""
    print("\n" + "=" * 60)
    print("📊 EVALUATING RESULTS")
    print("=" * 60)

    # Run evaluation script
    cmd = [
        "python",
        "scripts/evaluate.py",
        "--models",
        "aot-gan",
        "ict",
        "repaint",
        "--subset",
        "subset_4",
        "--output",
        "./results/evaluation/subset_4",
    ]

    return run_command(cmd, "Evaluation of all models on subset_4")


def main():
    parser = argparse.ArgumentParser(
        description="Test all model variants using subset_4 data"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["aot-gan", "ict", "repaint", "all"],
        default=["all"],
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip the final evaluation step"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout for each model test in seconds (default: 300)",
    )

    args = parser.parse_args()

    # Determine which models to test
    if "all" in args.models:
        models_to_test = ["aot-gan", "ict", "repaint"]
    else:
        models_to_test = args.models

    print("🎯 Testing Models on Subset_4")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Subset: 4 images (2 osteoporotic, 2 normal)")
    print(f"Timeout: {args.timeout}s per model")

    # Check if subset_4 exists
    subset_4_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"
    if not subset_4_dir.exists():
        print(f"❌ Subset_4 directory not found: {subset_4_dir}")
        print("Please run data/oai/split.py first to generate subset_4")
        sys.exit(1)

    # Test each model
    results = {}
    for model in models_to_test:
        if model == "aot-gan":
            results["aot-gan"] = test_aot_gan()
        elif model == "ict":
            results["ict"] = test_ict()
        elif model == "repaint":
            results["repaint"] = test_repaint()

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TESTING SUMMARY")
    print("=" * 60)

    for model, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{model.upper():<10}: {status}")

    # Run evaluation if all tests passed and not skipped
    if all(results.values()) and not args.skip_evaluation:
        print("\n🎯 All tests passed! Running evaluation...")
        eval_success = evaluate_results()
        if eval_success:
            print("\n🎉 Complete! Check results in:")
            print("   - ./output/*/OAI/subset_4/")
            print("   - ./results/evaluation/subset_4/")
        else:
            print("\n⚠️  Evaluation failed, but individual tests completed")
    elif not all(results.values()):
        print("\n⚠️  Some tests failed. Check the output above for details.")
    else:
        print("\n✅ Individual tests completed (evaluation skipped)")

    print(f"\n📁 Output directories:")
    for model in models_to_test:
        output_dir = project_root / "output" / model.upper() / "OAI" / "subset_4"
        print(f"   {model}: {output_dir}")


if __name__ == "__main__":
    main()
