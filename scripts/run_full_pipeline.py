#!/usr/bin/env python3
"""
Complete OAI Inpainting Pipeline Runner
Runs through all 5 phases with progress monitoring and timeout handling.
Perfect for Google Colab execution.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PipelineRunner:
    """Complete pipeline runner with progress monitoring and timeout handling."""

    def __init__(self, timeout_hours: int = 8, verbose: bool = True):
        self.timeout_hours = timeout_hours
        self.verbose = verbose
        self.start_time = datetime.now()
        self.results = {}

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp and level."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def check_timeout(self) -> bool:
        """Check if we've exceeded the timeout."""
        elapsed = datetime.now() - self.start_time
        if elapsed > timedelta(hours=self.timeout_hours):
            self.log(f"TIMEOUT: Pipeline exceeded {self.timeout_hours} hours", "ERROR")
            return True
        return False

    def run_command(
        self,
        cmd: List[str],
        description: str,
        cwd: Optional[Path] = None,
        timeout: int = 3600,
        phase: str = "",
    ) -> Tuple[bool, str]:
        """Run a command with timeout and error handling."""
        self.log(f"Starting: {description}", "PHASE")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}", "DEBUG")
            if cwd:
                self.log(f"Working directory: {cwd}", "DEBUG")

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
            self.log(
                f"Completed in {elapsed:.1f}s (exit code: {result.returncode})", "PHASE"
            )

            if result.returncode == 0:
                self.log(f"SUCCESS: {description}", "SUCCESS")
                return True, result.stdout
            else:
                self.log(f"FAILED: {description}", "ERROR")
                self.log(f"Error output: {result.stderr}", "ERROR")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: {description} exceeded {timeout}s", "ERROR")
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            self.log(f"ERROR: {description} failed with exception: {e}", "ERROR")
            return False, str(e)

    def phase_1_verification(self) -> bool:
        """Phase 1: Quick verification (5 minutes)."""
        self.log("=" * 60, "PHASE")
        self.log("PHASE 1: QUICK VERIFICATION", "PHASE")
        self.log("=" * 60, "PHASE")

        if self.check_timeout():
            return False

        # Test AOT-GAN on subset
        success, output = self.run_command(
            [
                "python",
                "scripts/test_subset_4.py",
                "--models",
                "aot-gan",
                "--timeout",
                "300",
            ],
            "Testing AOT-GAN on subset_4 data",
            timeout=600,  # 10 minutes max
        )

        if not success:
            self.log("Phase 1 failed - AOT-GAN test failed", "ERROR")
            return False

        # List available models
        success, output = self.run_command(
            ["python", "scripts/test_subset_4.py", "--list-models"],
            "Listing available models",
            timeout=60,
        )

        self.log("Phase 1 completed successfully", "SUCCESS")
        return True

    def phase_2_aot_gan_training(self) -> bool:
        """Phase 2: AOT-GAN training (2-4 hours)."""
        self.log("=" * 60, "PHASE")
        self.log("PHASE 2: AOT-GAN TRAINING", "PHASE")
        self.log("=" * 60, "PHASE")

        if self.check_timeout():
            return False

        # Check if AOT-GAN already trained
        aot_gan_results = project_root / "results" / "AOT-GAN" / "OAI" / "subset_4"
        if aot_gan_results.exists() and any(aot_gan_results.iterdir()):
            self.log("AOT-GAN results already exist, skipping training", "INFO")
            return True

        # Start AOT-GAN training
        success, output = self.run_command(
            [
                "python",
                "scripts/train.py",
                "--model",
                "aot-gan",
                "--config",
                "configs/oai_config.yml",
            ],
            "Training AOT-GAN model",
            timeout=14400,  # 4 hours max
        )

        if not success:
            self.log("Phase 2 failed - AOT-GAN training failed", "ERROR")
            return False

        self.log("Phase 2 completed successfully", "SUCCESS")
        return True

    def phase_3_ict_training(self) -> bool:
        """Phase 3: ICT training (1-3 hours)."""
        self.log("=" * 60, "PHASE")
        self.log("PHASE 3: ICT TRAINING", "PHASE")
        self.log("=" * 60, "PHASE")

        if self.check_timeout():
            return False

        # Check if ICT already trained
        ict_results = project_root / "results" / "ICT" / "OAI" / "subset_4"
        if ict_results.exists() and any(ict_results.iterdir()):
            self.log("ICT results already exist, skipping training", "INFO")
            return True

        # Start ICT training
        success, output = self.run_command(
            [
                "python",
                "scripts/train.py",
                "--model",
                "ict",
                "--config",
                "configs/oai_config.yml",
            ],
            "Training ICT model",
            timeout=10800,  # 3 hours max
        )

        if not success:
            self.log("Phase 3 failed - ICT training failed", "ERROR")
            return False

        self.log("Phase 3 completed successfully", "SUCCESS")
        return True

    def phase_4_repaint_inference(self) -> bool:
        """Phase 4: RePaint inference (30 minutes)."""
        self.log("=" * 60, "PHASE")
        self.log("PHASE 4: REPAINT INFERENCE", "PHASE")
        self.log("=" * 60, "PHASE")

        if self.check_timeout():
            return False

        # Check if RePaint already run
        repaint_results = project_root / "results" / "RePaint" / "OAI" / "subset_4"
        if repaint_results.exists() and any(repaint_results.iterdir()):
            self.log("RePaint results already exist, skipping inference", "INFO")
            return True

        # Run RePaint inference
        success, output = self.run_command(
            [
                "python",
                "scripts/test.py",
                "--model",
                "repaint",
                "--config",
                "configs/oai_config.yml",
            ],
            "Running RePaint inference",
            timeout=1800,  # 30 minutes max
        )

        if not success:
            self.log("Phase 4 failed - RePaint inference failed", "ERROR")
            return False

        self.log("Phase 4 completed successfully", "SUCCESS")
        return True

    def phase_5_evaluation(self) -> bool:
        """Phase 5: Comprehensive evaluation (15 minutes)."""
        self.log("=" * 60, "PHASE")
        self.log("PHASE 5: COMPREHENSIVE EVALUATION", "PHASE")
        self.log("=" * 60, "PHASE")

        if self.check_timeout():
            return False

        # Run comprehensive evaluation
        success, output = self.run_command(
            [
                "python",
                "scripts/evaluate.py",
                "--models",
                "all",
                "--output",
                "results/evaluation/",
            ],
            "Running comprehensive evaluation",
            timeout=900,  # 15 minutes max
        )

        if not success:
            self.log("Phase 5 failed - Evaluation failed", "ERROR")
            return False

        # Display results summary
        self.log("Evaluation completed, displaying results:", "SUCCESS")
        try:
            with (
                project_root / "results" / "evaluation" / "comparison_report.txt"
            ).open() as f:
                print(f.read())
        except FileNotFoundError:
            self.log("Results file not found, but evaluation completed", "WARNING")

        self.log("Phase 5 completed successfully", "SUCCESS")
        return True

    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        self.log("🚀 Starting OAI Inpainting Pipeline", "START")
        self.log(f"Timeout: {self.timeout_hours} hours", "INFO")
        self.log(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")

        phases = [
            ("Phase 1: Verification", self.phase_1_verification),
            ("Phase 2: AOT-GAN Training", self.phase_2_aot_gan_training),
            ("Phase 3: ICT Training", self.phase_3_ict_training),
            ("Phase 4: RePaint Inference", self.phase_4_repaint_inference),
            ("Phase 5: Evaluation", self.phase_5_evaluation),
        ]

        completed_phases = 0
        total_phases = len(phases)

        for phase_name, phase_func in phases:
            self.log(f"Starting {phase_name}...", "INFO")

            if self.check_timeout():
                self.log(f"Pipeline timed out during {phase_name}", "ERROR")
                break

            try:
                if phase_func():
                    completed_phases += 1
                    self.log(f"✅ {phase_name} completed successfully", "SUCCESS")
                else:
                    self.log(f"❌ {phase_name} failed", "ERROR")
                    break
            except Exception as e:
                self.log(f"❌ {phase_name} failed with exception: {e}", "ERROR")
                break

        # Final summary
        elapsed = datetime.now() - self.start_time
        self.log("=" * 60, "SUMMARY")
        self.log("PIPELINE SUMMARY", "SUMMARY")
        self.log("=" * 60, "SUMMARY")
        self.log(f"Completed phases: {completed_phases}/{total_phases}", "SUMMARY")
        self.log(f"Total time: {elapsed}", "SUMMARY")

        if completed_phases == total_phases:
            self.log("🎉 PIPELINE COMPLETED SUCCESSFULLY!", "SUCCESS")
            self.log("All models trained and evaluated successfully", "SUCCESS")
            return True
        else:
            self.log(
                f"⚠️ PIPELINE INCOMPLETE: {completed_phases}/{total_phases} phases completed",
                "WARNING",
            )
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run complete OAI inpainting pipeline")
    parser.add_argument(
        "--timeout", type=int, default=8, help="Maximum runtime in hours (default: 8)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["1", "2", "3", "4", "5", "all"],
        default=["all"],
        help="Which phases to run (default: all)",
    )

    args = parser.parse_args()

    runner = PipelineRunner(timeout_hours=args.timeout, verbose=args.verbose)

    if "all" in args.phases:
        success = runner.run_pipeline()
    else:
        # Run specific phases
        success = True
        phase_map = {
            "1": runner.phase_1_verification,
            "2": runner.phase_2_aot_gan_training,
            "3": runner.phase_3_ict_training,
            "4": runner.phase_4_repaint_inference,
            "5": runner.phase_5_evaluation,
        }

        for phase in args.phases:
            if phase in phase_map:
                if not phase_map[phase]():
                    success = False
                    break

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
