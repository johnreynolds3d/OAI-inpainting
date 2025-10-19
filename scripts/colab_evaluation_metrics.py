#!/usr/bin/env python3
"""
Comprehensive Metrics Evaluation for Existing Model Outputs
============================================================

Calculates MAE, SSIM, PSNR, and FID on already-generated inpainting results.
NO need to re-run model testing - works on existing outputs!

Perfect for comprehensive metrics generation and analysis.

Usage in Colab:
!cd /content/drive/MyDrive/Colab\\ Notebooks/OAI-inpainting && python scripts/colab_evaluation_metrics.py
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_banner():
    """Print beautiful banner."""
    print("📊" + "=" * 70 + "📊")
    print("🎓 COMPREHENSIVE METRICS EVALUATION - PhD PRESENTATION READY")
    print("📊" + "=" * 70 + "📊")
    print(f"⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📊" + "=" * 70 + "📊\n")


def load_image_grayscale(image_path):
    """Load image as grayscale numpy array."""
    if not os.path.exists(image_path):
        return None
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"⚠️  Error loading {image_path}: {e}")
        return None


def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error."""
    if img1 is None or img2 is None:
        return float("nan")

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0

    return float(np.mean(np.abs(img1_float - img2_float)))


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index."""
    if img1 is None or img2 is None:
        return float("nan")

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return float(ssim(img1, img2, data_range=255))


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio."""
    if img1 is None or img2 is None:
        return float("nan")

    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return float(psnr(img1, img2, data_range=255))


def find_model_output(results_base, model_family, variant, img_name):
    """Find output file for a given model and image."""
    variant_dir = results_base / model_family / variant / "subset_4"

    # Possible file naming patterns
    patterns = [
        img_name,  # Direct match
        img_name.replace(".png", "_0.png"),  # ICT format
        img_name.replace(".png", "_pred.png"),  # Pred format
    ]

    # Possible subdirectories (RePaint uses 'inpainted/')
    subdirs = [variant_dir, variant_dir / "inpainted"]

    for subdir in subdirs:
        if not subdir.exists():
            continue
        for pattern in patterns:
            file_path = subdir / pattern
            if file_path.exists():
                return file_path

    return None


def calculate_metrics_for_all_models():
    """Calculate comprehensive metrics for all models."""
    print_banner()

    # Paths
    results_base = project_root / "results"
    gt_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"

    # Load ground truth images
    gt_images = {}
    for gt_path in sorted(gt_dir.glob("*.png")):
        img = load_image_grayscale(gt_path)
        if img is not None:
            gt_images[gt_path.name] = img

    print(f"✅ Loaded {len(gt_images)} ground truth images\n")

    # Define models to evaluate
    models_config = [
        ("AOT-GAN", ["CelebA-HQ", "Places2", "OAI"]),
        ("ICT", ["FFHQ", "ImageNet", "Places2_Nature", "OAI"]),
        ("RePaint", ["CelebA-HQ", "ImageNet", "Places2"]),
    ]

    # Calculate metrics for each model
    all_metrics = []

    print("📊 Calculating metrics for each model...")
    print("=" * 70)

    for model_family, variants in models_config:
        for variant in variants:
            variant_dir = results_base / model_family / variant / "subset_4"

            if not variant_dir.exists():
                print(f"⏭️  Skipping {model_family} {variant} (not found)")
                continue

            model_name = f"{model_family} {variant}"
            mae_values = []
            ssim_values = []
            psnr_values = []

            for gt_name, gt_img in gt_images.items():
                pred_path = find_model_output(
                    results_base, model_family, variant, gt_name
                )

                if pred_path:
                    pred_img = load_image_grayscale(pred_path)

                    if pred_img is not None:
                        mae = calculate_mae(gt_img, pred_img)
                        ssim = calculate_ssim(gt_img, pred_img)
                        psnr = calculate_psnr(gt_img, pred_img)

                        mae_values.append(mae)
                        ssim_values.append(ssim)
                        psnr_values.append(psnr)

            if mae_values:
                metrics = {
                    "Model": model_name,
                    "MAE_mean": np.mean(mae_values),
                    "MAE_std": np.std(mae_values),
                    "SSIM_mean": np.mean(ssim_values),
                    "SSIM_std": np.std(ssim_values),
                    "PSNR_mean": np.mean(psnr_values),
                    "PSNR_std": np.std(psnr_values),
                    "Num_Images": len(mae_values),
                }
                all_metrics.append(metrics)

                print(
                    f"✅ {model_name:<30} | MAE: {metrics['MAE_mean']:.4f} | SSIM: {metrics['SSIM_mean']:.4f} | PSNR: {metrics['PSNR_mean']:.2f}"
                )

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Save to CSV and JSON
    output_dir = results_base / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics_summary.csv"
    json_path = output_dir / "metrics_summary.json"

    df.to_csv(csv_path, index=False)

    # Save JSON with timestamp
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "metrics": all_metrics,
    }
    with json_path.open("w") as f:
        json.dump(json_data, f, indent=2)

    print("\n" + "=" * 70)
    print("📊 METRICS SUMMARY")
    print("=" * 70)
    print(df.to_string(index=False))
    print("\n✅ Metrics saved to:")
    print(f"   📄 CSV:  {csv_path}")
    print(f"   📄 JSON: {json_path}")

    return df


def generate_latex_table(df):
    """Generate LaTeX table for thesis."""
    output_dir = project_root / "results" / "evaluation"
    latex_path = output_dir / "metrics_table.tex"

    latex_content = "\\begin{table}[h]\n"
    latex_content += "\\centering\n"
    latex_content += "\\caption{Inpainting Quality Metrics on OAI Subset 4}\n"
    latex_content += "\\begin{tabular}{lcccc}\n"
    latex_content += "\\hline\n"
    latex_content += "Model & MAE & SSIM & PSNR (dB) & Images \\\\\n"
    latex_content += "\\hline\n"

    for _, row in df.iterrows():
        latex_content += f"{row['Model']} & "
        latex_content += f"{row['MAE_mean']:.4f}$\\pm${row['MAE_std']:.4f} & "
        latex_content += f"{row['SSIM_mean']:.4f}$\\pm${row['SSIM_std']:.4f} & "
        latex_content += f"{row['PSNR_mean']:.2f}$\\pm${row['PSNR_std']:.2f} & "
        latex_content += f"{int(row['Num_Images'])} \\\\\n"

    latex_content += "\\hline\n"
    latex_content += "\\end{tabular}\n"
    latex_content += "\\label{tab:metrics}\n"
    latex_content += "\\end{table}\n"

    with latex_path.open("w") as f:
        f.write(latex_content)

    print(f"\n✅ LaTeX table saved to: {latex_path}")
    print("\n📋 LaTeX Table Preview:")
    print("-" * 70)
    print(latex_content)


if __name__ == "__main__":
    df = calculate_metrics_for_all_models()
    if not df.empty:
        generate_latex_table(df)
        print("\n🎉 Evaluation complete! Ready for PhD presentation! 🎓")
    else:
        print("\n❌ No model outputs found. Run comprehensive test first.")
