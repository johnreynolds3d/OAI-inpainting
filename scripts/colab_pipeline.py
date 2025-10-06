#!/usr/bin/env python3
"""
Colab-Optimized OAI Inpainting Pipeline
Simplified version for Google Colab with progress bars and clear output.
"""

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display

# Add project root to path
# In Colab, we're already in the OAI-inpainting directory
project_root = Path.cwd()
sys.path.insert(0, str(project_root))


class ColabPipeline:
    """Colab-optimized pipeline runner with progress bars."""

    def __init__(self, timeout_hours=8):
        self.timeout_hours = timeout_hours
        self.start_time = datetime.now()
        self.progress_bar = None
        self.status_text = None

    def setup_progress_ui(self):
        """Setup progress UI for Colab."""
        try:
            self.progress_bar = widgets.IntProgress(
                value=0,
                min=0,
                max=5,
                description="Progress:",
                bar_style="info",
                orientation="horizontal",
            )
            self.status_text = widgets.HTML(value="<b>Starting pipeline...</b>")
            display(self.progress_bar, self.status_text)
        except ImportError:
            # Fallback if widgets not available
            self.progress_bar = None
            self.status_text = None

    def update_progress(self, phase: int, message: str):
        """Update progress bar and status."""
        if self.progress_bar:
            self.progress_bar.value = phase
        if self.status_text:
            self.status_text.value = f"<b>Phase {phase}/5:</b> {message}"
        print(f"\n🔄 Phase {phase}/5: {message}")

    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        emoji = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}.get(
            level, "📝"
        )
        print(f"[{timestamp}] {emoji} {message}")

    def run_command(self, cmd, description, timeout=3600):
        """Run command with timeout and progress updates."""
        self.log(f"Starting: {description}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                check=False,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                self.log(f"Completed in {elapsed:.1f}s", "SUCCESS")
                return True, result.stdout
            else:
                self.log(f"Failed after {elapsed:.1f}s", "ERROR")
                self.log(f"Error: {result.stderr}", "ERROR")
                return False, result.stderr

        except subprocess.TimeoutExpired:
            self.log(f"Timeout after {timeout}s", "ERROR")
            return False, "Command timed out"
        except Exception as e:
            self.log(f"Exception: {e}", "ERROR")
            return False, str(e)

    def check_timeout(self):
        """Check if we've exceeded timeout."""
        elapsed = datetime.now() - self.start_time
        if elapsed > timedelta(hours=self.timeout_hours):
            self.log("Pipeline timeout exceeded!", "ERROR")
            return True
        return False

    def phase_1_verification(self):
        """Phase 1: Quick verification."""
        self.update_progress(1, "Quick verification")

        if self.check_timeout():
            return False

        # Test AOT-GAN
        success, _ = self.run_command(
            [
                "python",
                "scripts/test_subset_4.py",
                "--models",
                "aot-gan",
                "--timeout",
                "300",
            ],
            "Testing AOT-GAN",
            timeout=600,
        )

        if success:
            self.log("Phase 1 completed", "SUCCESS")
            return True
        else:
            self.log("Phase 1 failed", "ERROR")
            return False

    def phase_2_aot_gan(self):
        """Phase 2: AOT-GAN training."""
        self.update_progress(2, "AOT-GAN training")

        if self.check_timeout():
            return False

        # Check if already trained
        results_dir = project_root / "results" / "AOT-GAN" / "OAI" / "subset_4"
        if results_dir.exists() and any(results_dir.iterdir()):
            self.log("AOT-GAN already trained, skipping", "INFO")
            return True

        # Train AOT-GAN
        success, _ = self.run_command(
            [
                "python",
                "scripts/train.py",
                "--model",
                "aot-gan",
                "--config",
                "configs/oai_config.yml",
            ],
            "Training AOT-GAN",
            timeout=14400,  # 4 hours
        )

        if success:
            self.log("Phase 2 completed", "SUCCESS")
            return True
        else:
            self.log("Phase 2 failed", "ERROR")
            return False

    def phase_3_ict(self):
        """Phase 3: ICT training."""
        self.update_progress(3, "ICT training")

        if self.check_timeout():
            return False

        # Check if already trained
        results_dir = project_root / "results" / "ICT" / "OAI" / "subset_4"
        if results_dir.exists() and any(results_dir.iterdir()):
            self.log("ICT already trained, skipping", "INFO")
            return True

        # Train ICT
        success, _ = self.run_command(
            [
                "python",
                "scripts/train.py",
                "--model",
                "ict",
                "--config",
                "configs/oai_config.yml",
            ],
            "Training ICT",
            timeout=10800,  # 3 hours
        )

        if success:
            self.log("Phase 3 completed", "SUCCESS")
            return True
        else:
            self.log("Phase 3 failed", "ERROR")
            return False

    def phase_4_repaint(self):
        """Phase 4: RePaint inference."""
        self.update_progress(4, "RePaint inference")

        if self.check_timeout():
            return False

        # Check if already run
        results_dir = project_root / "results" / "RePaint" / "OAI" / "subset_4"
        if results_dir.exists() and any(results_dir.iterdir()):
            self.log("RePaint already run, skipping", "INFO")
            return True

        # Run RePaint
        success, _ = self.run_command(
            [
                "python",
                "scripts/test.py",
                "--model",
                "repaint",
                "--config",
                "configs/oai_config.yml",
            ],
            "Running RePaint",
            timeout=1800,  # 30 minutes
        )

        if success:
            self.log("Phase 4 completed", "SUCCESS")
            return True
        else:
            self.log("Phase 4 failed", "ERROR")
            return False

    def phase_5_evaluation(self):
        """Phase 5: Evaluation."""
        self.update_progress(5, "Comprehensive evaluation")

        if self.check_timeout():
            return False

        # Run evaluation
        success, _ = self.run_command(
            [
                "python",
                "scripts/evaluate.py",
                "--models",
                "all",
                "--output",
                "results/evaluation/",
            ],
            "Running evaluation",
            timeout=900,  # 15 minutes
        )

        if success:
            self.log("Phase 5 completed", "SUCCESS")
            return True
        else:
            self.log("Phase 5 failed", "ERROR")
            return False

    def run_pipeline(self):
        """Run the complete pipeline."""
        self.log("🚀 Starting OAI Inpainting Pipeline", "INFO")
        self.log(f"Timeout: {self.timeout_hours} hours", "INFO")

        self.setup_progress_ui()

        phases = [
            ("Verification", self.phase_1_verification),
            ("AOT-GAN Training", self.phase_2_aot_gan),
            ("ICT Training", self.phase_3_ict),
            ("RePaint Inference", self.phase_4_repaint),
            ("Evaluation", self.phase_5_evaluation),
        ]

        completed = 0
        total = len(phases)

        for phase_name, phase_func in phases:
            if self.check_timeout():
                break

            try:
                if phase_func():
                    completed += 1
                    self.log(f"✅ {phase_name} completed", "SUCCESS")
                else:
                    self.log(f"❌ {phase_name} failed", "ERROR")
                    break
            except Exception as e:
                self.log(f"❌ {phase_name} failed: {e}", "ERROR")
                break

        # Final summary
        elapsed = datetime.now() - self.start_time
        self.log("=" * 50, "INFO")
        self.log(f"Pipeline completed: {completed}/{total} phases", "INFO")
        self.log(f"Total time: {elapsed}", "INFO")

        if completed == total:
            self.log("🎉 ALL PHASES COMPLETED SUCCESSFULLY!", "SUCCESS")
        else:
            self.log(f"⚠️ Pipeline incomplete: {completed}/{total} phases", "WARNING")

        return completed == total


def run_full_pipeline(timeout_hours=8):
    """Run the complete pipeline (Colab-friendly function)."""
    pipeline = ColabPipeline(timeout_hours=timeout_hours)
    return pipeline.run_pipeline()


def run_phase(phase_num, timeout_hours=8):
    """Run a specific phase."""
    pipeline = ColabPipeline(timeout_hours=timeout_hours)

    phase_map = {
        1: pipeline.phase_1_verification,
        2: pipeline.phase_2_aot_gan,
        3: pipeline.phase_3_ict,
        4: pipeline.phase_4_repaint,
        5: pipeline.phase_5_evaluation,
    }

    if phase_num in phase_map:
        return phase_map[phase_num]()
    else:
        print(f"❌ Invalid phase number: {phase_num}")
        return False


if __name__ == "__main__":
    # For direct execution
    pipeline = ColabPipeline()
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)
