#!/usr/bin/env python3
"""
ResNet50 Classification Evaluation on Existing Inpainting Results
================================================================

Evaluates osteoporosis classification accuracy on:
- Ground Truth images
- All inpainted model outputs

Works on EXISTING results - no need to re-run inpainting!

Usage in Colab:
!cd /content/drive/MyDrive/Colab\\ Notebooks/OAI-inpainting && python scripts/colab_classification_evaluation.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_medical_resnet50(num_classes=2, dropout_rate=0.5):
    """
    Create ResNet50 model optimized for medical X-ray images.

    Architecture improvements for medical imaging:
    - Higher dropout for better generalization on small datasets
    - Additional fully connected layer for feature refinement
    - Binary classification output (normal vs osteoporotic)
    """
    # Load pretrained ResNet50
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace final layer with medical-optimized architecture
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes),  # 2 classes: Normal (0), Osteoporotic (1)
    )

    return model


def load_resnet50_model():
    """Load trained ResNet50 model for osteoporosis classification."""
    print("🤖 Loading ResNet50 classifier...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_medical_resnet50()

    # Try to find trained model
    possible_paths = [
        project_root / "models" / "classifier" / "resnet50_trained.pth",
        project_root / "models" / "classifier" / "best_resnet50_model.pth",
        project_root / "final_resnet50_model.pth",
        project_root / "best_resnet50_model.pth",
    ]

    model_loaded = False
    for model_path in possible_paths:
        if model_path.exists():
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"✅ Loaded trained model from: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"⚠️  Could not load {model_path}: {e}")
                continue

    if not model_loaded:
        print("⚠️  No trained model found - using pretrained ImageNet weights")
        print("💡 For accurate results, train ResNet50 on OAI data first")

    model.to(device)
    model.eval()

    return model, device


def preprocess_xray_for_resnet(img_gray):
    """
    Preprocess grayscale X-ray for ResNet50.

    Converts single-channel grayscale to 3-channel RGB format expected by ResNet.
    Applies ImageNet normalization for transfer learning.
    """
    # Convert grayscale to RGB (3 channels)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Resize to ResNet input size
    img_resized = cv2.resize(img_rgb, (224, 224))

    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img_resized)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def predict_osteoporosis(model, device, img_gray):
    """Predict osteoporosis classification for a single image."""
    if img_gray is None:
        return None, None

    # Preprocess
    img_tensor = preprocess_xray_for_resnet(img_gray).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence


def load_image_grayscale(image_path):
    """Load image as grayscale numpy array."""
    if not Path(image_path).exists():
        return None
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"⚠️  Error loading {image_path}: {e}")
        return None


def find_model_output(results_base, model_family, variant, img_name):
    """Find output file for a model variant."""
    variant_dir = results_base / model_family / variant / "subset_4"

    patterns = [
        img_name,
        img_name.replace(".png", "_0.png"),  # ICT
        img_name.replace(".png", "_pred.png"),
    ]

    subdirs = [variant_dir, variant_dir / "inpainted"]

    for subdir in subdirs:
        if not subdir.exists():
            continue
        for pattern in patterns:
            file_path = subdir / pattern
            if file_path.exists():
                return file_path
    return None


def evaluate_classification():
    """Run classification evaluation on all models."""
    print("🤖 RESNET50 CLASSIFICATION EVALUATION")
    print("=" * 70)

    # Load model
    model, device = load_resnet50_model()

    # Load ground truth labels
    info_file = project_root / "data" / "oai" / "test" / "subset_4_info.csv"
    if not info_file.exists():
        print(f"❌ subset_4_info.csv not found: {info_file}")
        return None

    df_labels = pd.read_csv(info_file)
    print(f"✅ Loaded labels for {len(df_labels)} images")

    # Create ground truth label dict
    gt_labels = {}
    for _, row in df_labels.iterrows():
        gt_labels[row["filename"]] = 1 if row["is_osteo"] else 0

    # Load GT images
    gt_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"
    gt_images = {}
    for img_name in gt_labels:
        img_path = gt_dir / img_name
        img = load_image_grayscale(img_path)
        if img is not None:
            gt_images[img_name] = img

    print(f"✅ Loaded {len(gt_images)} ground truth images\n")

    # Evaluate GT images
    print("📊 Evaluating Ground Truth images...")
    gt_predictions = []
    gt_true_labels = []

    for img_name, img in gt_images.items():
        pred, conf = predict_osteoporosis(model, device, img)
        if pred is not None:
            gt_predictions.append(pred)
            gt_true_labels.append(gt_labels[img_name])
            status = "osteoporotic" if pred == 1 else "normal"
            actual = "osteoporotic" if gt_labels[img_name] == 1 else "normal"
            match = "✅" if pred == gt_labels[img_name] else "❌"
            print(
                f"  {match} {img_name}: Pred={status} (conf={conf:.2f}), Actual={actual}"
            )

    gt_accuracy = np.mean(np.array(gt_predictions) == np.array(gt_true_labels))
    print(f"\n✅ Ground Truth Accuracy: {gt_accuracy:.2%}\n")

    # Evaluate model outputs
    results_base = project_root / "results"
    models_config = [
        ("AOT-GAN", ["CelebA-HQ", "Places2"]),
        ("ICT", ["FFHQ", "ImageNet", "Places2_Nature"]),
        ("RePaint", ["CelebA-HQ", "ImageNet", "Places2"]),
    ]

    classification_results = []

    print("=" * 70)
    print("📊 Evaluating Inpainted Images...")
    print("=" * 70)

    for model_family, variants in models_config:
        for variant in variants:
            variant_dir = results_base / model_family / variant / "subset_4"

            if not variant_dir.exists():
                print(f"⏭️  Skipping {model_family} {variant}")
                continue

            model_name = f"{model_family} {variant}"
            predictions = []
            true_labels = []

            for img_name, label in gt_labels.items():
                pred_path = find_model_output(
                    results_base, model_family, variant, img_name
                )

                if pred_path:
                    pred_img = load_image_grayscale(pred_path)
                    pred, conf = predict_osteoporosis(model, device, pred_img)

                    if pred is not None:
                        predictions.append(pred)
                        true_labels.append(label)

            if predictions:
                accuracy = np.mean(np.array(predictions) == np.array(true_labels))
                classification_results.append(
                    {
                        "Model": model_name,
                        "Accuracy": accuracy,
                        "Correct": np.sum(
                            np.array(predictions) == np.array(true_labels)
                        ),
                        "Total": len(predictions),
                    }
                )
                print(
                    f"✅ {model_name:<30} | Accuracy: {accuracy:.2%} ({np.sum(np.array(predictions) == np.array(true_labels))}/{len(predictions)})"
                )

    # Save classification results
    output_dir = project_root / "results" / "evaluation"

    df_class = pd.DataFrame(classification_results)
    class_csv = output_dir / "classification_results.csv"
    class_json = output_dir / "classification_results.json"

    df_class.to_csv(class_csv, index=False)

    json_data = {
        "timestamp": datetime.now().isoformat(),
        "ground_truth_accuracy": float(gt_accuracy),
        "model_results": classification_results,
    }
    with class_json.open("w") as f:
        json.dump(json_data, f, indent=2)

    print("\n" + "=" * 70)
    print("🤖 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(df_class.to_string(index=False))
    print(f"\n📊 Ground Truth Baseline: {gt_accuracy:.2%}")
    print("\n✅ Classification results saved to:")
    print(f"   📄 CSV:  {class_csv}")
    print(f"   📄 JSON: {class_json}")

    return df_class


if __name__ == "__main__":
    # Run classification evaluation
    print("\n" + "=" * 70)
    try:
        classification_df = evaluate_classification()
        print("\n🎉 Classification evaluation complete! Ready for tomorrow! 🎓")
    except Exception as e:
        print(f"\n⚠️  Classification evaluation failed: {e}")
        print("💡 This requires a trained ResNet50 model on OAI data")
        import traceback

        traceback.print_exc()
