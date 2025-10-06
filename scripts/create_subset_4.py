#!/usr/bin/env python3
"""
Modernized subset_4 creator for OAI Dataset
Creates a 4-image test subset (2 osteoporotic + 2 normal) for quick testing.
Uses pathlib.Path for platform-agnostic path handling.
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import canny

from src.paths import get_oai_data_dir, get_project_root

# Constants
OSTEOPOROSIS_THRESHOLD = -2.5
SUBSET_SIZE = 2


def generate_mask_for_image(
    _image_path: Path, output_path: Path, image_size: int = 224
) -> None:
    """Generate a mask with one small square for a given image.

    Args:
        _image_path: Path to the input image (unused but kept for API compatibility)
        output_path: Path where the mask will be saved
        image_size: Size of the generated mask (default: 224)
    """
    # Create a white background image
    img = np.zeros([image_size, image_size, 3], np.uint8)
    img.fill(255)

    # Create a black mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Calculate the safe zone (avoiding outer 20% of boundaries)
    margin = int(image_size * 0.2)  # 20% margin
    safe_zone_min = margin
    safe_zone_max = image_size - margin

    # Fixed square size: 1/6 of image dimensions
    square_size = int(image_size / 6)  # 37x37 for 224x224 images

    # Random position within safe zone
    x1 = random.randint(safe_zone_min, safe_zone_max - square_size)
    y1 = random.randint(safe_zone_min, safe_zone_max - square_size)
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw white rectangle on the mask (white = masked area)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Save the mask
    cv2.imwrite(str(output_path), mask)


def generate_inverted_mask(mask_path: Path, output_path: Path) -> None:
    """Generate inverted mask (black = masked area)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    inverted_mask = 255 - mask
    cv2.imwrite(str(output_path), inverted_mask)


def generate_edge_map(image_path: Path, output_path: Path) -> None:
    """Generate edge map from image."""
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not load image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Generate edge map using Canny edge detection
    edges = canny(gray, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
    edges = (edges * 255).astype(np.uint8)

    # Save edge map
    cv2.imwrite(str(output_path), edges)


def load_bmd_data() -> pd.DataFrame:
    """Load BMD data from CSV file."""
    data_dir = get_oai_data_dir()
    csv_path = data_dir / "data.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"BMD data file not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=["BMD", "filename"])
    print(f"Total samples: {len(df)}")

    # Add osteoporotic classification (BMD < threshold)
    df["is_osteo"] = df["BMD"] < OSTEOPOROSIS_THRESHOLD
    osteo_count = df["is_osteo"].sum()
    normal_count = len(df) - osteo_count

    print(f"Osteoporotic samples: {osteo_count}")
    print(f"Normal samples: {normal_count}")

    return df


def create_subset_4() -> None:
    """Create subset_4 with 2 osteoporotic and 2 non-osteoporotic images."""
    print("🎯 Creating subset_4 with 2 osteoporotic + 2 non-osteoporotic images...")

    # Load BMD data
    df = load_bmd_data()

    # Get test data (assuming it exists from previous split)
    test_dir = get_oai_data_dir() / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    # Load test split info if available
    test_info_path = test_dir / "test_split_info.csv"
    if test_info_path.exists():
        test_df = pd.read_csv(test_info_path)
        print(f"Loaded test split info: {len(test_df)} samples")
    else:
        # If no test split info, use all data (fallback)
        test_df = df
        print("No test split info found, using all data")

    # Separate test data by class
    test_osteo = test_df[test_df["is_osteo"]].reset_index(drop=True)
    test_normal = test_df[~test_df["is_osteo"]].reset_index(drop=True)

    print(f"Test set: {len(test_osteo)} osteoporotic, {len(test_normal)} normal")

    # Check if we have enough samples
    if len(test_osteo) < SUBSET_SIZE or len(test_normal) < SUBSET_SIZE:
        print("❌ Not enough samples in test set for subset_4!")
        print(f"   Need: {SUBSET_SIZE} osteoporotic, {SUBSET_SIZE} normal")
        print(f"   Have: {len(test_osteo)} osteoporotic, {len(test_normal)} normal")
        return

    # Randomly select samples from each class
    random.seed(42)  # Use same seed for reproducibility
    selected_osteo = test_osteo.sample(n=SUBSET_SIZE, random_state=42).reset_index(
        drop=True
    )
    selected_normal = test_normal.sample(n=SUBSET_SIZE, random_state=42).reset_index(
        drop=True
    )

    # Combine selected samples
    subset_4_df = pd.concat([selected_osteo, selected_normal], ignore_index=True)

    print("Selected for subset_4:")
    for _, row in subset_4_df.iterrows():
        class_label = "osteoporotic" if row["is_osteo"] else "normal"
        print(f"   {row['filename']} ({class_label})")

    # Create subset_4 directories
    subset_4_dirs = [
        test_dir / "img" / "subset_4",
        test_dir / "mask" / "subset_4",
        test_dir / "mask_inv" / "subset_4",
        test_dir / "edge" / "subset_4",
    ]

    print("\n📁 Creating subset_4 directories...")
    for dir_path in subset_4_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Clean existing files to avoid conflicts
        for file in dir_path.glob("*"):
            file.unlink()
        print(f"   Created and cleaned: {dir_path.relative_to(get_project_root())}/")

    # Copy selected images and generate masks/edges
    print("\n📋 Processing subset_4 files...")
    copied_count = 0

    for _, row in subset_4_df.iterrows():
        filename = row["filename"]
        mask_filename = Path(filename).with_suffix(".png")

        # Source paths
        src_img = test_dir / "img" / filename
        dst_img = test_dir / "img" / "subset_4" / filename
        dst_mask = test_dir / "mask" / "subset_4" / mask_filename
        dst_mask_inv = test_dir / "mask_inv" / "subset_4" / mask_filename
        dst_edge = test_dir / "edge" / "subset_4" / mask_filename

        # Copy image
        if src_img.exists():
            shutil.copy2(src_img, dst_img)
            copied_count += 1

            # Generate mask
            generate_mask_for_image(dst_img, dst_mask)

            # Generate inverted mask
            generate_inverted_mask(dst_mask, dst_mask_inv)

            # Generate edge map
            generate_edge_map(dst_img, dst_edge)

            print(f"   ✅ {filename}")
        else:
            print(f"   ❌ {src_img} not found!")

    # Save subset_4 info
    subset_4_info_path = test_dir / "subset_4_info.csv"
    subset_4_df.to_csv(subset_4_info_path, index=False)

    print("\n✅ Subset_4 created successfully!")
    print(f"   📁 Location: {test_dir.relative_to(get_project_root())}/*/subset_4/")
    print(f"   📊 Files processed: {copied_count}")
    print(f"   📋 Info saved: {subset_4_info_path.relative_to(get_project_root())}")
    print("   🎯 Perfect balance: 2 osteoporotic + 2 normal images")


if __name__ == "__main__":
    create_subset_4()
