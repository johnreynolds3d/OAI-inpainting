#!/usr/bin/env python3
"""
Evaluation script for comparing inpainting results with ground truth.
Platform-agnostic and reproducible.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from classifier.resnet50 import BMDDataset, model


def calculate_inpainting_metrics(gt_images, inpainted_images, masks):
    """Calculate inpainting quality metrics."""
    metrics = {}

    # Convert to numpy arrays
    gt = np.array(gt_images)
    inpainted = np.array(inpainted_images)
    mask = np.array(masks)

    # Calculate PSNR
    mse = np.mean((gt - inpainted) ** 2)
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))

    # Calculate SSIM (simplified version)
    def ssim(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        c1 = 0.01**2
        c2 = 0.03**2

        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
        )
        return ssim_val

    ssim_val = ssim(gt, inpainted)

    metrics["psnr"] = psnr
    metrics["ssim"] = ssim_val

    return metrics


def evaluate_classification_performance(gt_images, inpainted_images, labels):
    """Evaluate classification performance on ground truth vs inpainted images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained classifier
    classifier = model.to(device)
    classifier.eval()

    results = {
        "gt_accuracy": 0,
        "inpainted_accuracy": 0,
        "gt_predictions": [],
        "inpainted_predictions": [],
        "gt_probabilities": [],
        "inpainted_probabilities": [],
    }

    # Evaluate on ground truth images
    with torch.no_grad():
        for img in gt_images:
            # Preprocess image
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)

            # Get prediction
            output = classifier(img_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0

            results["gt_predictions"].append(prediction)
            results["gt_probabilities"].append(probability)

    # Evaluate on inpainted images
    with torch.no_grad():
        for img in inpainted_images:
            # Preprocess image
            img_tensor = torch.tensor(img).unsqueeze(0).to(device)

            # Get prediction
            output = classifier(img_tensor)
            probability = torch.sigmoid(output).item()
            prediction = 1 if probability > 0.5 else 0

            results["inpainted_predictions"].append(prediction)
            results["inpainted_probabilities"].append(probability)

    # Calculate accuracy
    results["gt_accuracy"] = accuracy_score(labels, results["gt_predictions"])
    results["inpainted_accuracy"] = accuracy_score(
        labels, results["inpainted_predictions"]
    )

    return results


def generate_comparison_report(results, output_dir):
    """Generate a comprehensive comparison report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison DataFrame
    comparison_data = []
    for model_name, model_results in results.items():
        for metric_name, metric_value in model_results.items():
            comparison_data.append(
                {"Model": model_name, "Metric": metric_name, "Value": metric_value}
            )

    df = pd.DataFrame(comparison_data)

    # Save results
    df.to_csv(output_dir / "comparison_results.csv", index=False)

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Plot accuracy comparison
    plt.subplot(2, 2, 1)
    accuracy_data = df[df["Metric"].str.contains("accuracy")]
    sns.barplot(data=accuracy_data, x="Model", y="Value", hue="Metric")
    plt.title("Classification Accuracy Comparison")
    plt.ylabel("Accuracy")

    # Plot inpainting quality metrics
    plt.subplot(2, 2, 2)
    quality_data = df[df["Metric"].isin(["psnr", "ssim"])]
    sns.barplot(data=quality_data, x="Model", y="Value", hue="Metric")
    plt.title("Inpainting Quality Metrics")
    plt.ylabel("Score")

    # Plot probability distributions
    plt.subplot(2, 2, 3)
    for model_name, model_results in results.items():
        if "gt_probabilities" in model_results:
            plt.hist(
                model_results["gt_probabilities"], alpha=0.5, label=f"{model_name} (GT)"
            )
            plt.hist(
                model_results["inpainted_probabilities"],
                alpha=0.5,
                label=f"{model_name} (Inpainted)",
            )
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Probability Distribution Comparison")
    plt.legend()

    # Plot metric correlation
    plt.subplot(2, 2, 4)
    correlation_data = df.pivot(index="Model", columns="Metric", values="Value")
    sns.heatmap(correlation_data.corr(), annot=True, cmap="coolwarm", center=0)
    plt.title("Metric Correlation Matrix")

    plt.tight_layout()
    plt.savefig(output_dir / "comparison_report.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Comparison report saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate inpainting models on OAI dataset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["aot-gan", "ict", "repaint"],
        default=["aot-gan", "ict", "repaint"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--subset", type=str, default="subset_4", help="Dataset subset to evaluate on"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/evaluation",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth data
    gt_dir = project_root / "data" / "oai" / "test" / "img" / args.subset
    mask_dir = project_root / "data" / "oai" / "test" / "mask" / args.subset

    if not gt_dir.exists():
        print(f"❌ Ground truth directory not found: {gt_dir}")
        sys.exit(1)

    # Load ground truth images
    gt_images = []
    for img_path in sorted(gt_dir.glob("*.png")):
        img = Image.open(img_path).convert("RGB")
        gt_images.append(np.array(img))

    # Load labels (assuming you have a labels file)
    labels_file = project_root / "data" / "oai" / "test" / f"{args.subset}_info.csv"
    if labels_file.exists():
        labels_df = pd.read_csv(labels_file)
        labels = labels_df["is_osteo"].astype(int).tolist()
    else:
        print("⚠️  No labels file found, using dummy labels")
        labels = [0] * len(gt_images)

    # Evaluate each model
    results = {}
    for model_name in args.models:
        print(f"🔍 Evaluating {model_name}...")

        # Load inpainted images
        inpainted_dir = (
            project_root / "output" / model_name.upper() / "OAI" / args.subset
        )
        if not inpainted_dir.exists():
            print(f"⚠️  Inpainted images not found for {model_name}: {inpainted_dir}")
            continue

        inpainted_images = []
        for img_path in sorted(inpainted_dir.glob("*.png")):
            img = Image.open(img_path).convert("RGB")
            inpainted_images.append(np.array(img))

        # Calculate metrics
        inpainting_metrics = calculate_inpainting_metrics(
            gt_images, inpainted_images, []
        )
        classification_results = evaluate_classification_performance(
            gt_images, inpainted_images, labels
        )

        # Combine results
        results[model_name] = {**inpainting_metrics, **classification_results}

    # Generate comparison report
    generate_comparison_report(results, output_dir)

    print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()
