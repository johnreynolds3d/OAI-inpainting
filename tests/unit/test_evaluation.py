"""Unit tests for evaluation functionality."""

import pytest
from pathlib import Path


def test_evaluation_script_exists(project_root):
    """Test that evaluation script exists."""
    eval_script = project_root / "scripts" / "evaluate.py"
    assert eval_script.exists(), "Evaluation script should exist"


def test_evaluation_dependencies(project_root):
    """Test that evaluation script dependencies are available."""
    # Check for required modules
    try:
        import pandas as pd
        import numpy as np
        from PIL import Image
        import cv2
    except ImportError as e:
        pytest.fail(f"Required evaluation dependency missing: {e}")


def test_metrics_calculation():
    """Test basic metrics calculation functions."""
    import numpy as np

    # Test PSNR calculation
    def calculate_psnr(img1, img2):
        """Calculate PSNR between two images."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * np.log10(255.0 / np.sqrt(mse))

    # Test with identical images
    img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    img2 = img1.copy()
    psnr = calculate_psnr(img1, img2)
    assert psnr == float("inf"), "PSNR should be infinity for identical images"

    # Test with different images
    img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    psnr = calculate_psnr(img1, img2)
    assert 0 <= psnr < float("inf"), "PSNR should be a finite positive value"


def test_ssim_calculation():
    """Test SSIM calculation function."""
    import numpy as np

    def calculate_ssim(img1, img2):
        """Calculate SSIM between two images."""
        # Convert to float for calculation
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        # Simplified SSIM calculation for testing
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

        c1 = 0.01**2
        c2 = 0.03**2

        # Avoid division by zero
        denominator = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
        if denominator == 0:
            return 1.0 if sigma1 == 0 and sigma2 == 0 else 0.0

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / denominator
        return max(0.0, min(1.0, ssim))  # Clamp to [0, 1]

    # Test with identical images
    img1 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    img2 = img1.copy()
    ssim = calculate_ssim(img1, img2)
    assert abs(ssim - 1.0) < 1e-6, "SSIM should be 1.0 for identical images"

    # Test with different images
    img2 = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    ssim = calculate_ssim(img1, img2)
    assert 0 <= ssim <= 1, "SSIM should be between 0 and 1"


def test_classification_metrics():
    """Test classification metrics calculation."""
    import numpy as np

    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy."""
        return np.mean(y_true == y_pred)

    def calculate_precision_recall(y_true, y_pred):
        """Calculate precision and recall."""
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return precision, recall

    # Test with perfect predictions
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])

    accuracy = calculate_accuracy(y_true, y_pred)
    assert accuracy == 1.0, "Accuracy should be 1.0 for perfect predictions"

    precision, recall = calculate_precision_recall(y_true, y_pred)
    assert precision == 1.0, "Precision should be 1.0 for perfect predictions"
    assert recall == 1.0, "Recall should be 1.0 for perfect predictions"

    # Test with imperfect predictions
    y_pred = np.array([1, 1, 0, 0, 1])
    accuracy = calculate_accuracy(y_true, y_pred)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"

    precision, recall = calculate_precision_recall(y_true, y_pred)
    assert 0 <= precision <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall <= 1, "Recall should be between 0 and 1"
