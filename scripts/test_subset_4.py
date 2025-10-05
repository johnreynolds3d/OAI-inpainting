#!/usr/bin/env python3
"""
Simple script to test all model variants using subset_4 data.
This script runs each model directly with proper configuration.
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


def run_command(cmd, description, cwd=None, timeout=300):
    """Run a command with timeout and error handling."""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=cwd or project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
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

    # Create output directory
    output_dir = project_root / "output" / "AOT-GAN" / "OAI" / "subset_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run AOT-GAN test directly
    cmd = [
        "python",
        "test.py",
        "--model",
        "aotgan",
        "--pre_train",
        str(model_path / "G0000000.pt"),
        "--dir_image",
        str(project_root / "data" / "oai" / "test" / "img" / "subset_4"),
        "--dir_mask",
        str(project_root / "data" / "oai" / "test" / "mask" / "subset_4"),
        "--outputs",
        str(output_dir),
    ]

    return run_command(
        cmd,
        "AOT-GAN testing on subset_4",
        cwd=project_root / "AOT-GAN-for-Inpainting" / "src",
    )


def test_ict():
    """Test ICT on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ICT ON SUBSET_4")
    print("=" * 60)

    # Check if ICT pretrained models exist (using backup structure)
    ict_models = project_root / "ICT" / "ckpts_ICT" / "Upsample" / "OAI"
    if not ict_models.exists() or not any(ict_models.iterdir()):
        print("⚠️  ICT OAI pretrained models not found, using Places2_Nature models")
        model_path = project_root / "ICT" / "ckpts_ICT" / "Upsample" / "Places2_Nature"
    else:
        model_path = ict_models

    # Create output directory
    output_dir = project_root / "output" / "ICT" / "OAI" / "subset_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config file for ICT
    config_content = f"""MODE: 2
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

TEST_FLIST: {project_root}/data/oai/test/img/subset_4
TEST_EDGE_FLIST: {project_root}/data/oai/test/edge/subset_4
TEST_MASK_FLIST: {project_root}/data/oai/test/mask/subset_4

PATH: {model_path}
RESULTS: {output_dir}

BATCH_SIZE: 1
INPUT_SIZE: 256
"""

    # Create checkpoint directory and config file
    checkpoint_dir = project_root / "ICT" / "Guided_Upsample" / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    config_path = checkpoint_dir / "config.yml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Run ICT test
    cmd = ["python", "main.py", "--path", "checkpoints"]

    return run_command(
        cmd, "ICT testing on subset_4", cwd=project_root / "ICT" / "Guided_Upsample"
    )


def test_repaint():
    """Test RePaint on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING REPAINT ON SUBSET_4")
    print("=" * 60)

    # Create output directory
    output_dir = project_root / "output" / "RePaint" / "OAI" / "subset_4"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config file for RePaint (using working parameters from backup)
    config_content = f"""attention_resolutions: 32,16,8
class_cond: false
diffusion_steps: 1000
learn_sigma: true
noise_schedule: linear
num_channels: 256
num_head_channels: 64
num_heads: 4
num_res_blocks: 2
resblock_updown: true
use_fp16: false
use_scale_shift_norm: true
classifier_scale: 4.0
lr_kernel_n_std: 2
num_samples: 4
show_progress: true
timestep_respacing: '25'
use_kl: false
predict_xstart: false
rescale_timesteps: false
rescale_learned_sigmas: false
classifier_use_fp16: false
classifier_width: 128
classifier_depth: 2
classifier_attention_resolutions: 32,16,8
classifier_use_scale_shift_norm: true
classifier_resblock_updown: true
classifier_pool: attention
num_heads_upsample: -1
channel_mult: ''
dropout: 0.0
use_checkpoint: false
use_new_attention_order: false
clip_denoised: true
use_ddim: false
latex_name: RePaint
method_name: Repaint
image_size: 256
model_path: {project_root}/data/pretrained/repaint/places256_300000.pt
name: test_subset_4
inpa_inj_sched_prev: true
n_jobs: 1
print_estimated_vars: false
inpa_inj_sched_prev_cumnoise: false
schedule_jump_params:
  t_T: 25
  n_sample: 1
  jump_length: 3
  jump_n_sample: 3
data:
  eval:
    subset_4_test:
      mask_loader: true
      gt_path: {project_root}/data/oai/test/img/subset_4
      mask_path: {project_root}/data/oai/test/mask/subset_4
      image_size: 256
      class_cond: false
      deterministic: true
      random_crop: false
      random_flip: false
      return_dict: true
      drop_last: false
      batch_size: 1
      return_dataloader: true
      offset: 0
      max_len: 4
      paths:
        srs: {output_dir}/inpainted
        lrs: {output_dir}/gt_masked
        gts: {output_dir}/gt
        gt_keep_masks: {output_dir}/gt_keep_mask
"""

    config_path = project_root / "RePaint" / "subset_4_config.yml"
    with open(config_path, "w") as f:
        f.write(config_content)

    # Run RePaint test
    cmd = ["python", "test.py", "--conf_path", "subset_4_config.yml"]

    return run_command(cmd, "RePaint testing on subset_4", cwd=project_root / "RePaint")


def check_subset_4():
    """Check if subset_4 exists and has the expected structure."""
    print("🔍 Checking subset_4 structure...")

    subset_4_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"
    if not subset_4_dir.exists():
        print(f"❌ Subset_4 directory not found: {subset_4_dir}")
        return False

    # Check for required files
    required_dirs = [
        project_root / "data" / "oai" / "test" / "img" / "subset_4",
        project_root / "data" / "oai" / "test" / "mask" / "subset_4",
        project_root / "data" / "oai" / "test" / "edge" / "subset_4",
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"❌ Required directory not found: {dir_path}")
            return False

        files = list(dir_path.glob("*.png"))
        if not files:
            print(f"❌ No PNG files found in: {dir_path}")
            return False

        print(f"✅ Found {len(files)} files in {dir_path.name}")

    # Check subset_4_info.csv
    info_file = project_root / "data" / "oai" / "test" / "subset_4_info.csv"
    if info_file.exists():
        print(f"✅ Found subset_4_info.csv with metadata")
    else:
        print(f"⚠️  subset_4_info.csv not found")

    return True


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
        "--timeout",
        type=int,
        default=600,
        help="Timeout for each model test in seconds (default: 600)",
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

    # Check subset_4 structure
    if not check_subset_4():
        print("❌ Subset_4 structure check failed")
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

    # Print output directories
    print(f"\n📁 Output directories:")
    for model in models_to_test:
        output_dir = project_root / "output" / model.upper() / "OAI" / "subset_4"
        print(f"   {model}: {output_dir}")

    if all(results.values()):
        print("\n🎉 All tests completed successfully!")
        print("Check the output directories for results.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()
