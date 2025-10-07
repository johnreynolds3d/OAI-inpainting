#!/usr/bin/env python3
"""
Comprehensive Test Script for All Model Variants on OAI Data
Tests all available pretrained models on subset_4 data for quick validation.
Perfect for Google Colab execution with clear progress indicators.

Model Variants Tested:
- AOT-GAN: CelebA-HQ, Places2, OAI (if available)
- ICT: FFHQ, ImageNet, Places2_Nature, OAI (if available)
- RePaint: CelebA-HQ, ImageNet, Places2

Total: 9 model variants tested on 4 OAI X-ray images
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ModelTester:
    """Comprehensive model testing with progress tracking."""

    def __init__(self, timeout_per_model: int = 600, verbose: bool = True):
        self.timeout_per_model = timeout_per_model
        self.verbose = verbose
        self.start_time = datetime.now()
        self.results = {}

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji_map = {
            "INFO": "i",  # noqa: RUF001
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "PHASE": "🔄",
            "START": "🚀",
        }
        emoji = emoji_map.get(level, "📝")
        print(f"[{timestamp}] {emoji} {message}")

    def run_command(
        self, cmd: List[str], description: str, cwd=None, timeout=None
    ) -> Tuple[bool, str, float]:
        """Run command with timeout and error handling."""
        if timeout is None:
            timeout = self.timeout_per_model

        self.log(f"Running: {description}", "PHASE")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}", "INFO")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=False,
                cwd=cwd or project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log(f"Completed in {elapsed:.1f}s", "SUCCESS")
                return True, result.stdout, elapsed
            else:
                self.log(f"Failed after {elapsed:.1f}s", "ERROR")
                if self.verbose:
                    self.log(f"Error: {result.stderr[-500:]}", "ERROR")
                return False, result.stderr, elapsed

        except subprocess.TimeoutExpired:
            self.log(f"Timeout after {timeout}s", "ERROR")
            return False, f"Timeout after {timeout}s", timeout
        except Exception as e:
            self.log(f"Exception: {e}", "ERROR")
            return False, str(e), 0.0

    def verify_subset_4(self) -> bool:
        """Verify subset_4 data structure."""
        self.log("Verifying subset_4 data...", "INFO")

        required_dirs = [
            project_root / "data" / "oai" / "test" / "img" / "subset_4",
            project_root / "data" / "oai" / "test" / "mask" / "subset_4",
            project_root / "data" / "oai" / "test" / "edge" / "subset_4",
            project_root / "data" / "oai" / "test" / "mask_inv" / "subset_4",
        ]

        for dir_path in required_dirs:
            if not dir_path.exists():
                self.log(f"Missing directory: {dir_path}", "ERROR")
                return False
            files = list(dir_path.glob("*.png"))
            if not files:
                self.log(f"No PNG files in: {dir_path}", "ERROR")
                return False
            self.log(f"Found {len(files)} files in {dir_path.name}/", "SUCCESS")

        return True

    def test_aot_gan_variant(
        self, variant_name: str, model_path: Path, output_dir: Path
    ) -> Tuple[bool, float]:
        """Test a specific AOT-GAN variant."""
        self.log(f"Testing AOT-GAN {variant_name}...", "PHASE")

        # Check if model exists
        model_file = model_path / "G0000000.pt"
        if not model_file.exists():
            self.log(f"Model file not found: {model_file}", "WARNING")
            return False, 0.0

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python",
            "test.py",
            "--model",
            "aotgan",
            "--pre_train",
            str(model_file),
            "--dir_image",
            str(project_root / "data" / "oai" / "test" / "img" / "subset_4"),
            "--dir_mask",
            str(project_root / "data" / "oai" / "test" / "mask" / "subset_4"),
            "--outputs",
            str(output_dir),
        ]

        success, _output, elapsed = self.run_command(
            cmd,
            f"AOT-GAN {variant_name}",
            cwd=project_root / "models" / "aot-gan" / "src",
        )

        return success, elapsed

    def test_aot_gan_all(self) -> List[Dict]:
        """Test all AOT-GAN variants."""
        self.log("=" * 60, "INFO")
        self.log("TESTING AOT-GAN VARIANTS", "START")
        self.log("=" * 60, "INFO")

        variants = [
            ("CelebA-HQ", "celebahq"),
            ("Places2", "places2"),
            ("OAI", "OAI"),
        ]

        results = []
        for variant_name, variant_dir in variants:
            model_path = project_root / "data" / "pretrained" / "aot-gan" / variant_dir
            output_dir = (
                project_root / "results" / "AOT-GAN" / variant_name / "subset_4"
            )

            if model_path.exists():
                success, elapsed = self.test_aot_gan_variant(
                    variant_name, model_path, output_dir
                )
                results.append(
                    {
                        "model": f"AOT-GAN {variant_name}",
                        "success": success,
                        "elapsed": elapsed,
                        "output_dir": str(output_dir) if success else None,
                    }
                )
            else:
                self.log(f"AOT-GAN {variant_name} not found, skipping", "WARNING")
                results.append(
                    {
                        "model": f"AOT-GAN {variant_name}",
                        "success": False,
                        "elapsed": 0.0,
                        "output_dir": None,
                        "reason": "Model not available",
                    }
                )

        return results

    def test_ict_variant(
        self, variant_name: str, model_path: Path, output_dir: Path
    ) -> Tuple[bool, float]:
        """Test a specific ICT variant."""
        self.log(f"Testing ICT {variant_name}...", "PHASE")

        if not model_path.exists():
            self.log(f"Model path not found: {model_path}", "WARNING")
            return False, 0.0

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary config for this test
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

        # Create config in models/ict directory
        config_path = (
            project_root
            / "models"
            / "ict"
            / f"test_subset_4_{variant_name.lower()}.yml"
        )
        with config_path.open("w") as f:
            f.write(config_content)

        # Note: ICT testing requires specific setup
        # This is a simplified version - actual implementation may need adjustment
        cmd = ["python", "run.py", "--config", str(config_path)]

        success, _output, elapsed = self.run_command(
            cmd,
            f"ICT {variant_name}",
            cwd=project_root / "models" / "ict",
        )

        # Clean up config
        if config_path.exists():
            config_path.unlink()

        return success, elapsed

    def test_ict_all(self) -> List[Dict]:
        """Test all ICT variants."""
        self.log("=" * 60, "INFO")
        self.log("TESTING ICT VARIANTS", "START")
        self.log("=" * 60, "INFO")

        variants = [
            ("FFHQ", "FFHQ"),
            ("ImageNet", "ImageNet"),
            ("Places2_Nature", "Places2_Nature"),
            ("OAI", "OAI"),
        ]

        results = []
        for variant_name, variant_dir in variants:
            model_path = (
                project_root / "data" / "pretrained" / "ict" / "Upsample" / variant_dir
            )
            # Also check in models/ict/ckpts_ICT for OAI variant
            if not model_path.exists() and variant_name == "OAI":
                model_path = (
                    project_root / "models" / "ict" / "ckpts_ICT" / "Upsample" / "OAI"
                )

            output_dir = project_root / "results" / "ICT" / variant_name / "subset_4"

            if model_path.exists():
                success, elapsed = self.test_ict_variant(
                    variant_name, model_path, output_dir
                )
                results.append(
                    {
                        "model": f"ICT {variant_name}",
                        "success": success,
                        "elapsed": elapsed,
                        "output_dir": str(output_dir) if success else None,
                    }
                )
            else:
                self.log(f"ICT {variant_name} not found, skipping", "WARNING")
                results.append(
                    {
                        "model": f"ICT {variant_name}",
                        "success": False,
                        "elapsed": 0.0,
                        "output_dir": None,
                        "reason": "Model not available",
                    }
                )

        return results

    def test_repaint_variant(
        self, variant_name: str, model_path: Path, output_dir: Path
    ) -> Tuple[bool, float]:
        """Test a specific RePaint variant."""
        self.log(f"Testing RePaint {variant_name}...", "PHASE")

        if not model_path.exists():
            self.log(f"Model file not found: {model_path}", "WARNING")
            return False, 0.0

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create config for RePaint
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
name: test_subset_4_{variant_name.lower()}
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
            / "models"
            / "repaint"
            / f"test_subset_4_{variant_name.lower().replace('-', '_')}.yml"
        )
        with config_path.open("w") as f:
            f.write(config_content)

        cmd = ["python", "test.py", "--conf_path", str(config_path.name)]

        success, _output, elapsed = self.run_command(
            cmd,
            f"RePaint {variant_name}",
            cwd=project_root / "models" / "repaint",
        )

        # Clean up config
        if config_path.exists():
            config_path.unlink()

        return success, elapsed

    def test_repaint_all(self) -> List[Dict]:
        """Test all RePaint variants."""
        self.log("=" * 60, "INFO")
        self.log("TESTING REPAINT VARIANTS", "START")
        self.log("=" * 60, "INFO")

        variants = [
            ("CelebA-HQ", "celeba256_250000.pt"),
            ("ImageNet", "256x256_diffusion.pt"),
            ("Places2", "places256_300000.pt"),
        ]

        results = []
        for variant_name, model_file in variants:
            model_path = project_root / "data" / "pretrained" / "repaint" / model_file
            output_dir = (
                project_root / "results" / "RePaint" / variant_name / "subset_4"
            )

            if model_path.exists():
                success, elapsed = self.test_repaint_variant(
                    variant_name, model_path, output_dir
                )
                results.append(
                    {
                        "model": f"RePaint {variant_name}",
                        "success": success,
                        "elapsed": elapsed,
                        "output_dir": str(output_dir) if success else None,
                    }
                )
            else:
                self.log(f"RePaint {variant_name} not found, skipping", "WARNING")
                results.append(
                    {
                        "model": f"RePaint {variant_name}",
                        "success": False,
                        "elapsed": 0.0,
                        "output_dir": None,
                        "reason": "Model not available",
                    }
                )

        return results

    def run_comprehensive_test(self, models: Optional[List[str]] = None) -> Dict:
        """Run comprehensive test of all model variants."""
        self.log("=" * 60, "START")
        self.log("COMPREHENSIVE MODEL TESTING ON OAI SUBSET_4", "START")
        self.log("=" * 60, "START")
        self.log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

        # Verify data structure
        if not self.verify_subset_4():
            self.log("Subset_4 verification failed", "ERROR")
            return {"success": False, "error": "Data verification failed"}

        # Determine which models to test
        if models is None or "all" in models:
            test_aot_gan = True
            test_ict = True
            test_repaint = True
        else:
            test_aot_gan = "aot-gan" in models
            test_ict = "ict" in models
            test_repaint = "repaint" in models

        # Run tests
        all_results = []

        if test_aot_gan:
            aot_results = self.test_aot_gan_all()
            all_results.extend(aot_results)

        if test_ict:
            ict_results = self.test_ict_all()
            all_results.extend(ict_results)

        if test_repaint:
            repaint_results = self.test_repaint_all()
            all_results.extend(repaint_results)

        # Generate summary
        total_elapsed = datetime.now() - self.start_time
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"] and "reason" not in r]
        skipped = [r for r in all_results if "reason" in r]

        # Print detailed summary
        self.log("=" * 60, "INFO")
        self.log("TEST SUMMARY", "START")
        self.log("=" * 60, "INFO")

        self.log(f"Total tests: {len(all_results)}", "INFO")
        self.log(f"Successful: {len(successful)}", "SUCCESS")
        self.log(f"Failed: {len(failed)}", "ERROR" if failed else "INFO")
        self.log(f"Skipped: {len(skipped)}", "WARNING" if skipped else "INFO")
        self.log(f"Total time: {total_elapsed}", "INFO")

        # Print individual results
        print("\n📊 DETAILED RESULTS:")
        print("-" * 60)
        for result in all_results:
            model_name = result["model"]
            if result["success"]:
                status = "✅ PASSED"
                time_str = f"({result['elapsed']:.1f}s)"
            elif "reason" in result:
                status = "⏭️ SKIPPED"
                time_str = f"({result['reason']})"
            else:
                status = "❌ FAILED"
                time_str = f"({result['elapsed']:.1f}s)"

            print(f"{model_name:<25}: {status:>10} {time_str}")

        # Print output directories
        if successful:
            print("\n📁 OUTPUT DIRECTORIES:")
            print("-" * 60)
            for result in successful:
                if result["output_dir"]:
                    print(f"  {result['model']:<25}: {result['output_dir']}")

        # Save results to JSON
        results_file = project_root / "results" / "comprehensive_test_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with results_file.open("w") as f:
            json.dump(
                {
                    "timestamp": self.start_time.isoformat(),
                    "duration": str(total_elapsed),
                    "results": all_results,
                    "summary": {
                        "total": len(all_results),
                        "successful": len(successful),
                        "failed": len(failed),
                        "skipped": len(skipped),
                    },
                },
                f,
                indent=2,
            )
        self.log(f"Results saved to: {results_file}", "SUCCESS")

        return {
            "success": len(failed) == 0,
            "results": all_results,
            "summary": {
                "total": len(all_results),
                "successful": len(successful),
                "failed": len(failed),
                "skipped": len(skipped),
            },
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test of all model variants on OAI subset_4 data"
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
        help="Timeout per model in seconds (default: 600)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )

    args = parser.parse_args()

    tester = ModelTester(timeout_per_model=args.timeout, verbose=args.verbose)
    result = tester.run_comprehensive_test(models=args.models)

    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
