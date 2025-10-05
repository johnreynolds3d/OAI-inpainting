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


def test_aot_gan_variant(model_name, model_path, output_dir, timeout=600):
    """Test a specific AOT-GAN variant on subset_4."""
    print(f"\n🧪 Testing AOT-GAN {model_name}...")

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
        f"AOT-GAN {model_name} testing on subset_4",
        cwd=project_root / "AOT-GAN-for-Inpainting" / "src",
        timeout=timeout,
    )


def test_aot_gan(timeout=600):
    """Test all AOT-GAN variants on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING AOT-GAN VARIANTS ON SUBSET_4")
    print("=" * 60)

    results = []

    # Test CelebA-HQ variant
    celebahq_path = project_root / "data" / "pretrained" / "aot-gan" / "celebahq"
    if celebahq_path.exists() and any(celebahq_path.iterdir()):
        output_dir = project_root / "output" / "AOT-GAN" / "CelebA-HQ" / "subset_4"
        result = test_aot_gan_variant("CelebA-HQ", celebahq_path, output_dir, timeout)
        results.append(("AOT-GAN CelebA-HQ", result))
    else:
        print("⚠️  AOT-GAN CelebA-HQ models not found, skipping")
        results.append(("AOT-GAN CelebA-HQ", False))

    # Test Places2 variant
    places2_path = project_root / "data" / "pretrained" / "aot-gan" / "places2"
    if places2_path.exists() and any(places2_path.iterdir()):
        output_dir = project_root / "output" / "AOT-GAN" / "Places2" / "subset_4"
        result = test_aot_gan_variant("Places2", places2_path, output_dir, timeout)
        results.append(("AOT-GAN Places2", result))
    else:
        print("⚠️  AOT-GAN Places2 models not found, skipping")
        results.append(("AOT-GAN Places2", False))

    # Test OAI variant (if available)
    oai_path = project_root / "data" / "pretrained" / "aot-gan" / "OAI"
    if oai_path.exists() and any(oai_path.iterdir()):
        output_dir = project_root / "output" / "AOT-GAN" / "OAI" / "subset_4"
        result = test_aot_gan_variant("OAI", oai_path, output_dir, timeout)
        results.append(("AOT-GAN OAI", result))
    else:
        print("⚠️  AOT-GAN OAI models not found, skipping")
        results.append(("AOT-GAN OAI", False))

    return results


def test_ict_variant(model_name, model_path, output_dir, timeout=600):
    """Test a specific ICT variant on subset_4."""
    print(f"\n🧪 Testing ICT {model_name}...")

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
condition_num: 1
prior_size: 32
test_batch_size: 1
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
        cmd,
        f"ICT {model_name} testing on subset_4",
        cwd=project_root / "ICT" / "Guided_Upsample",
        timeout=timeout,
    )


def test_ict(timeout=600):
    """Test all ICT variants on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING ICT VARIANTS ON SUBSET_4")
    print("=" * 60)

    results = []

    # Test FFHQ variant
    ffhq_path = project_root / "data" / "pretrained" / "ict" / "Upsample" / "FFHQ"
    if ffhq_path.exists() and any(ffhq_path.iterdir()):
        output_dir = project_root / "output" / "ICT" / "FFHQ" / "subset_4"
        result = test_ict_variant("FFHQ", ffhq_path, output_dir, timeout)
        results.append(("ICT FFHQ", result))
    else:
        print("⚠️  ICT FFHQ models not found, skipping")
        results.append(("ICT FFHQ", False))

    # Test ImageNet variant
    imagenet_path = (
        project_root / "data" / "pretrained" / "ict" / "Upsample" / "ImageNet"
    )
    if imagenet_path.exists() and any(imagenet_path.iterdir()):
        output_dir = project_root / "output" / "ICT" / "ImageNet" / "subset_4"
        result = test_ict_variant("ImageNet", imagenet_path, output_dir, timeout)
        results.append(("ICT ImageNet", result))
    else:
        print("⚠️  ICT ImageNet models not found, skipping")
        results.append(("ICT ImageNet", False))

    # Test Places2_Nature variant
    places2_nature_path = (
        project_root / "data" / "pretrained" / "ict" / "Upsample" / "Places2_Nature"
    )
    if places2_nature_path.exists() and any(places2_nature_path.iterdir()):
        output_dir = project_root / "output" / "ICT" / "Places2_Nature" / "subset_4"
        result = test_ict_variant(
            "Places2_Nature", places2_nature_path, output_dir, timeout
        )
        results.append(("ICT Places2_Nature", result))
    else:
        print("⚠️  ICT Places2_Nature models not found, skipping")
        results.append(("ICT Places2_Nature", False))

    # Test OAI variant (if available)
    oai_path = project_root / "ICT" / "ckpts_ICT" / "Upsample" / "OAI"
    if oai_path.exists() and any(oai_path.iterdir()):
        output_dir = project_root / "output" / "ICT" / "OAI" / "subset_4"
        result = test_ict_variant("OAI", oai_path, output_dir, timeout)
        results.append(("ICT OAI", result))
    else:
        print("⚠️  ICT OAI models not found, skipping")
        results.append(("ICT OAI", False))

    return results


def test_repaint_variant(model_name, model_path, output_dir, timeout=600):
    """Test a specific RePaint variant on subset_4."""
    print(f"\n🧪 Testing RePaint {model_name}...")

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
model_path: {model_path}
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

    config_path = (
        project_root
        / "RePaint"
        / f"subset_4_{model_name.lower().replace('-', '_')}_config.yml"
    )
    with open(config_path, "w") as f:
        f.write(config_content)

    # Run RePaint test
    cmd = ["python", "test.py", "--conf_path", config_path.name]

    return run_command(
        cmd,
        f"RePaint {model_name} testing on subset_4",
        cwd=project_root / "RePaint",
        timeout=timeout,
    )


def test_repaint(timeout=600):
    """Test all RePaint variants on subset_4."""
    print("\n" + "=" * 60)
    print("🧪 TESTING REPAINT VARIANTS ON SUBSET_4")
    print("=" * 60)

    results = []

    # Test CelebA-HQ variant
    celebahq_model = (
        project_root / "data" / "pretrained" / "repaint" / "celeba256_250000.pt"
    )
    if celebahq_model.exists():
        output_dir = project_root / "output" / "RePaint" / "CelebA-HQ" / "subset_4"
        result = test_repaint_variant("CelebA-HQ", celebahq_model, output_dir, timeout)
        results.append(("RePaint CelebA-HQ", result))
    else:
        print("⚠️  RePaint CelebA-HQ model not found, skipping")
        results.append(("RePaint CelebA-HQ", False))

    # Test ImageNet variant
    imagenet_model = (
        project_root / "data" / "pretrained" / "repaint" / "256x256_diffusion.pt"
    )
    if imagenet_model.exists():
        output_dir = project_root / "output" / "RePaint" / "ImageNet" / "subset_4"
        result = test_repaint_variant("ImageNet", imagenet_model, output_dir, timeout)
        results.append(("RePaint ImageNet", result))
    else:
        print("⚠️  RePaint ImageNet model not found, skipping")
        results.append(("RePaint ImageNet", False))

    # Test Places2 variant
    places2_model = (
        project_root / "data" / "pretrained" / "repaint" / "places256_300000.pt"
    )
    if places2_model.exists():
        output_dir = project_root / "output" / "RePaint" / "Places2" / "subset_4"
        result = test_repaint_variant("Places2", places2_model, output_dir, timeout)
        results.append(("RePaint Places2", result))
    else:
        print("⚠️  RePaint Places2 model not found, skipping")
        results.append(("RePaint Places2", False))

    return results


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
    all_results = []
    for model in models_to_test:
        if model == "aot-gan":
            model_results = test_aot_gan(timeout=args.timeout)
            all_results.extend(model_results)
        elif model == "ict":
            model_results = test_ict(timeout=args.timeout)
            all_results.extend(model_results)
        elif model == "repaint":
            model_results = test_repaint(timeout=args.timeout)
            all_results.extend(model_results)

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TESTING SUMMARY")
    print("=" * 60)

    all_passed = True
    for model_name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{model_name:<20}: {status}")
        if not result:
            all_passed = False

    # Print output directories
    print(f"\n📁 Output directories:")
    for model_name, result in all_results:
        if result:  # Only show directories for successful tests
            if "AOT-GAN" in model_name:
                variant = model_name.split()[-1]
                print(
                    f"   {model_name}: /var/home/john/Documents/Code/OAI-inpainting/output/AOT-GAN/{variant}/subset_4"
                )
            elif "ICT" in model_name:
                variant = model_name.split()[-1]
                print(
                    f"   {model_name}: /var/home/john/Documents/Code/OAI-inpainting/output/ICT/{variant}/subset_4"
                )
            elif "RePaint" in model_name:
                variant = model_name.split()[-1]
                print(
                    f"   {model_name}: /var/home/john/Documents/Code/OAI-inpainting/output/RePaint/{variant}/subset_4"
                )

    if all_passed:
        print("\n🎉 All tests completed successfully!")
        print("Check the output directories for results.")
    else:
        print("\n❌ Some tests failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
