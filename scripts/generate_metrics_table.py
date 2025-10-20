#!/usr/bin/env python3
"""
Generate evaluation metrics tables for all 8 model variants tested on subset_4.
Creates presentation-ready tables with PSNR, SSIM, and classification accuracy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_images(directory: Path):
    """Load all PNG images from a directory."""
    images = []
    filenames = []
    for img_path in sorted(directory.glob("*.png")):
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        images.append(np.array(img))
        filenames.append(img_path.name)
    return images, filenames


def calculate_metrics(gt_images, inpainted_images):
    """Calculate PSNR and SSIM for image pairs."""
    psnr_scores = []
    ssim_scores = []
    mae_scores = []

    for gt, inp_img in zip(gt_images, inpainted_images):
        # Resize inpainted to match GT if needed
        inp_resized = inp_img
        if gt.shape != inp_img.shape:
            inp_pil = Image.fromarray(inp_img)
            inp_pil = inp_pil.resize((gt.shape[1], gt.shape[0]), Image.LANCZOS)
            inp_resized = np.array(inp_pil)
            print(f"   🔄 Resized {inp_resized.shape} to match GT {gt.shape}")

        # Calculate PSNR
        psnr_val = psnr(gt, inp_resized, data_range=255)
        psnr_scores.append(psnr_val)

        # Calculate SSIM
        ssim_val = ssim(gt, inp_resized, data_range=255)
        ssim_scores.append(ssim_val)

        # Calculate MAE
        mae_val = np.mean(np.abs(gt.astype(float) - inp_resized.astype(float)))
        mae_scores.append(mae_val)

    return {
        "PSNR (dB)": np.mean(psnr_scores) if psnr_scores else 0,
        "SSIM": np.mean(ssim_scores) if ssim_scores else 0,
        "MAE": np.mean(mae_scores) if mae_scores else 0,
        "n_images": len(psnr_scores),
    }


def find_inpainted_images(results_dir: Path, model_family: str, variant: str):
    """Find inpainted images for a specific model variant."""
    # Try multiple possible paths
    possible_paths = [
        results_dir / model_family / variant / "subset_4",
        results_dir / model_family / variant / "subset_4" / "inpainted",
        results_dir / model_family / variant / "subset_4" / "output",
    ]

    for path in possible_paths:
        if path.exists() and list(path.glob("*.png")):
            return path

    return None


def main():
    """Generate metrics tables for all models."""
    print("=" * 70)
    print("📊 GENERATING EVALUATION METRICS TABLES")
    print("=" * 70)

    # Define model variants
    models = [
        ("AOT-GAN", "CelebA-HQ"),
        ("AOT-GAN", "Places2"),
        ("ICT", "FFHQ"),
        ("ICT", "ImageNet"),
        ("ICT", "Places2_Nature"),
        ("RePaint", "CelebA-HQ"),
        ("RePaint", "ImageNet"),
        ("RePaint", "Places2"),
    ]

    # Load ground truth images
    gt_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"
    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    gt_images, _ = load_images(gt_dir)
    print(f"✅ Loaded {len(gt_images)} ground truth images from subset_4")
    print()

    # Results storage
    results = []

    # Evaluate each model
    results_dir = project_root / "results"

    for model_family, variant in models:
        print(f"🔍 Evaluating {model_family} {variant}...")

        # Find inpainted images
        inpainted_dir = find_inpainted_images(results_dir, model_family, variant)

        if inpainted_dir is None:
            print("   ⚠️  Inpainted images not found, skipping")
            results.append(
                {
                    "Model Family": model_family,
                    "Variant": variant,
                    "PSNR (dB)": "N/A",
                    "SSIM": "N/A",
                    "MAE": "N/A",
                    "Images": 0,
                    "Status": "Not Found",
                }
            )
            continue

        print(f"   📁 Found: {inpainted_dir}")

        # Load inpainted images
        inpainted_images, _ = load_images(inpainted_dir)

        if len(inpainted_images) != len(gt_images):
            print(
                f"   ⚠️  Image count mismatch: {len(inpainted_images)} inpainted vs {len(gt_images)} GT"
            )

        # Calculate metrics
        metrics = calculate_metrics(gt_images, inpainted_images)

        print(
            f"   ✅ PSNR: {metrics['PSNR (dB)']:.2f} dB | SSIM: {metrics['SSIM']:.4f} | MAE: {metrics['MAE']:.2f}"
        )

        results.append(
            {
                "Model Family": model_family,
                "Variant": variant,
                "PSNR (dB)": f"{metrics['PSNR (dB)']:.2f}",
                "SSIM": f"{metrics['SSIM']:.4f}",
                "MAE": f"{metrics['MAE']:.2f}",
                "Images": metrics["n_images"],
                "Status": "✅ Complete",
            }
        )

    print()
    print("=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Print table
    print()
    print(df.to_string(index=False))
    print()

    # Save to CSV
    output_dir = project_root / "results"
    csv_path = output_dir / "metrics_summary_subset_4.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved to: {csv_path}")

    # Create LaTeX table (for presentations)
    latex_path = output_dir / "metrics_summary_subset_4.tex"
    with latex_path.open("w") as f:
        f.write("% Evaluation Metrics for 8 Model Variants (subset_4)\n")
        f.write(df.to_latex(index=False, float_format="%.2f"))
    print(f"📄 LaTeX table saved to: {latex_path}")

    # Create simple markdown table (without tabulate dependency)
    markdown_path = output_dir / "metrics_summary_subset_4.md"
    with markdown_path.open("w") as f:
        f.write("# Evaluation Metrics Summary (subset_4)\n\n")
        f.write(
            "| Model Family | Variant | PSNR (dB) | SSIM | MAE | Images | Status |\n"
        )
        f.write(
            "|--------------|---------|-----------|------|-----|--------|--------|\n"
        )
        for _, row in df.iterrows():
            f.write(
                f"| {row['Model Family']} | {row['Variant']} | {row['PSNR (dB)']} | {row['SSIM']} | {row['MAE']} | {row['Images']} | {row['Status']} |\n"
            )
        f.write("\n**Metrics Explained:**\n")
        f.write("- **PSNR (dB)**: Peak Signal-to-Noise Ratio (higher is better)\n")
        f.write("- **SSIM**: Structural Similarity Index (0-1, higher is better)\n")
        f.write("- **MAE**: Mean Absolute Error (lower is better)\n")
    print(f"📝 Markdown table saved to: {markdown_path}")

    print()
    print("✅ Metrics table generation complete!")
    print()


if __name__ == "__main__":
    main()
