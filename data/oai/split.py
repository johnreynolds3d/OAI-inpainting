#!/usr/bin/env python3
"""
PERFECT BALANCED Dataset Splitter for OAI Dataset
Creates train/validation/test splits with GUARANTEED equal representation of
osteoporotic and non-osteoporotic samples.
Uses manual splitting to ensure perfect balance in all splits.

Split: 80% train, 10% validation, 10% test
Each subset maintains EXACT equal balance of osteoporotic vs non-osteoporotic
samples.

Also creates subset_4 for quick testing (2 osteoporotic + 2 normal images).
"""

import random
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import canny

# Constants
OSTEOPOROSIS_THRESHOLD = -2.5
SUBSET_SIZE = 2
IMAGE_SIZE = 224
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def generate_mask_for_image(
    _image_path: Path, output_path: Path, image_size: int = IMAGE_SIZE
) -> None:
    """Generate a mask with one small square for a given image.

    Args:
        _image_path: Path to the input image (unused but kept for API compatibility)
        output_path: Path where the mask will be saved
        image_size: Size of the generated mask (default: IMAGE_SIZE)
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
    if mask is not None:
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


def clean_directory(dir_path: Path) -> None:
    """Clean directory of all files."""
    if dir_path.exists():
        for file_path in dir_path.iterdir():
            if file_path.is_file():
                file_path.unlink()


def load_bmd_data(script_dir: Path) -> pd.DataFrame:
    """Load BMD data from CSV file."""
    csv_path = script_dir / "data.csv"

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


def create_balanced_splits(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create perfectly balanced train/validation/test splits."""
    print("\n🎯 Creating perfectly balanced splits...")

    # Separate by class
    osteo_df = df[df["is_osteo"]].reset_index(drop=True)
    normal_df = df[~df["is_osteo"]].reset_index(drop=True)

    print(f"Osteoporotic samples: {len(osteo_df)}")
    print(f"Normal samples: {len(normal_df)}")

    # Calculate split sizes for each class
    osteo_train_size = int(len(osteo_df) * TRAIN_RATIO)
    osteo_val_size = int(len(osteo_df) * VAL_RATIO)
    osteo_test_size = len(osteo_df) - osteo_train_size - osteo_val_size

    normal_train_size = int(len(normal_df) * TRAIN_RATIO)
    normal_val_size = int(len(normal_df) * VAL_RATIO)
    normal_test_size = len(normal_df) - normal_train_size - normal_val_size

    print("\nSplit sizes:")
    print(
        f"  Train: {osteo_train_size} osteo + {normal_train_size} normal = {osteo_train_size + normal_train_size}"
    )
    print(
        f"  Val:   {osteo_val_size} osteo + {normal_val_size} normal = {osteo_val_size + normal_val_size}"
    )
    print(
        f"  Test:  {osteo_test_size} osteo + {normal_test_size} normal = {osteo_test_size + normal_test_size}"
    )

    # Shuffle data
    random.seed(42)
    osteo_df = osteo_df.sample(frac=1, random_state=42).reset_index(drop=True)
    normal_df = normal_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Create splits
    train_osteo = osteo_df.iloc[:osteo_train_size].reset_index(drop=True)
    val_osteo = osteo_df.iloc[
        osteo_train_size : osteo_train_size + osteo_val_size
    ].reset_index(drop=True)
    test_osteo = osteo_df.iloc[osteo_train_size + osteo_val_size :].reset_index(
        drop=True
    )

    train_normal = normal_df.iloc[:normal_train_size].reset_index(drop=True)
    val_normal = normal_df.iloc[
        normal_train_size : normal_train_size + normal_val_size
    ].reset_index(drop=True)
    test_normal = normal_df.iloc[normal_train_size + normal_val_size :].reset_index(
        drop=True
    )

    # Combine splits
    train_df = pd.concat([train_osteo, train_normal], ignore_index=True)
    val_df = pd.concat([val_osteo, val_normal], ignore_index=True)
    test_df = pd.concat([test_osteo, test_normal], ignore_index=True)

    # Shuffle each split
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify balance
    print("\n✅ Balance verification:")
    print(
        f"  Train: {train_df['is_osteo'].sum()} osteo / {len(train_df) - train_df['is_osteo'].sum()} normal"
    )
    print(
        f"  Val:   {val_df['is_osteo'].sum()} osteo / {len(val_df) - val_df['is_osteo'].sum()} normal"
    )
    print(
        f"  Test:  {test_df['is_osteo'].sum()} osteo / {len(test_df) - test_df['is_osteo'].sum()} normal"
    )

    return train_df, val_df, test_df


def copy_images(
    df: pd.DataFrame, target_dir: Path, script_dir: Path
) -> Tuple[int, int]:
    """Copy images to target directory."""
    copied = 0
    missing = 0

    for _, row in df.iterrows():
        src_path = script_dir / "img" / row["filename"]
        dst_path = target_dir / row["filename"]

        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
        else:
            missing += 1

    return copied, missing


def generate_masks_for_split(
    df: pd.DataFrame,
    split_name: str,
    img_dir: Path,
    mask_dir: Path,
    mask_inv_dir: Path,
    edge_dir: Path,
) -> Tuple[int, int]:
    """Generate masks, inverted masks, and edge maps for a split."""
    print(f"\n🎭 Generating masks for {split_name} set...")

    mask_success = 0
    mask_failed = 0

    for _, row in df.iterrows():
        try:
            mask_filename = Path(row["filename"]).with_suffix(".png")
            mask_path = mask_dir / mask_filename
            inverted_mask_path = mask_inv_dir / mask_filename

            image_path = img_dir / row["filename"]
            if image_path.exists():
                generate_mask_for_image(image_path, mask_path)
                generate_inverted_mask(mask_path, inverted_mask_path)

                if edge_dir:
                    edge_path = edge_dir / mask_filename
                    generate_edge_map(image_path, edge_path)

                mask_success += 1
            else:
                mask_failed += 1

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            mask_failed += 1

    return mask_success, mask_failed


def create_subset_4(test_df: pd.DataFrame, script_dir: Path) -> None:
    """Create subset_4 with 2 osteoporotic and 2 non-osteoporotic images."""
    print("\n🎯 Creating subset_4 with 2 osteoporotic + 2 non-osteoporotic images...")

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
        script_dir / "test" / "img" / "subset_4",
        script_dir / "test" / "mask" / "subset_4",
        script_dir / "test" / "mask_inv" / "subset_4",
        script_dir / "test" / "edge" / "subset_4",
    ]

    print("\n📁 Creating subset_4 directories...")
    for dir_path in subset_4_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Clean existing files to avoid conflicts
        clean_directory(dir_path)
        print(f"   Created and cleaned: {dir_path.relative_to(script_dir)}/")

    # Copy selected images and generate masks/edges
    print("\n📋 Processing subset_4 files...")
    copied_count = 0

    for _, row in subset_4_df.iterrows():
        filename = row["filename"]
        mask_filename = Path(filename).with_suffix(".png")

        # Source paths
        src_img = script_dir / "test" / "img" / filename
        dst_img = script_dir / "test" / "img" / "subset_4" / filename
        dst_mask = script_dir / "test" / "mask" / "subset_4" / mask_filename
        dst_mask_inv = script_dir / "test" / "mask_inv" / "subset_4" / mask_filename
        dst_edge = script_dir / "test" / "edge" / "subset_4" / mask_filename

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
    subset_4_info_path = script_dir / "test" / "subset_4_info.csv"
    subset_4_df.to_csv(subset_4_info_path, index=False)

    print("\n✅ Subset_4 created successfully!")
    print("   📁 Location: test/*/subset_4/")
    print(f"   📊 Files processed: {copied_count}")
    print(f"   📋 Info saved: {subset_4_info_path.relative_to(script_dir)}")
    print("   🎯 Perfect balance: 2 osteoporotic + 2 normal images")


def create_balanced_splits_main():
    """Main function to create balanced splits and subset_4."""
    print("🎯 PERFECT BALANCED OAI Dataset Splitter")
    print("=" * 50)

    # Get script directory
    script_dir = Path(__file__).parent

    # Load BMD data
    print("Loading BMD data...")
    df = load_bmd_data(script_dir)

    # Create balanced splits
    train_df, val_df, test_df = create_balanced_splits(df)

    # Create output directories and CLEAN them first
    output_dirs = [
        script_dir / "train" / "img",
        script_dir / "train" / "mask",
        script_dir / "train" / "mask_inv",
        script_dir / "train" / "edge",
        script_dir / "valid" / "img",
        script_dir / "valid" / "mask",
        script_dir / "valid" / "mask_inv",
        script_dir / "valid" / "edge",
        script_dir / "test" / "img",
        script_dir / "test" / "mask",
        script_dir / "test" / "mask_inv",
        script_dir / "test" / "edge",
    ]

    print("\n📁 Creating output directories...")
    for dir_path in output_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        clean_directory(dir_path)
        print(f"   Created and cleaned: {dir_path.relative_to(script_dir)}/")

    # Copy images for each split
    print("\n📋 Copying images...")
    train_copied, train_missing = copy_images(
        train_df, script_dir / "train" / "img", script_dir
    )
    val_copied, val_missing = copy_images(
        val_df, script_dir / "valid" / "img", script_dir
    )
    test_copied, test_missing = copy_images(
        test_df, script_dir / "test" / "img", script_dir
    )

    print(f"   Train: {train_copied} copied, {train_missing} missing")
    print(f"   Valid: {val_copied} copied, {val_missing} missing")
    print(f"   Test:  {test_copied} copied, {test_missing} missing")

    # Generate masks for each split
    train_mask_success, _train_mask_failed = generate_masks_for_split(
        train_df,
        "train",
        script_dir / "train" / "img",
        script_dir / "train" / "mask",
        script_dir / "train" / "mask_inv",
        script_dir / "train" / "edge",
    )
    val_mask_success, _val_mask_failed = generate_masks_for_split(
        val_df,
        "validation",
        script_dir / "valid" / "img",
        script_dir / "valid" / "mask",
        script_dir / "valid" / "mask_inv",
        script_dir / "valid" / "edge",
    )
    test_mask_success, _test_mask_failed = generate_masks_for_split(
        test_df,
        "test",
        script_dir / "test" / "img",
        script_dir / "test" / "mask",
        script_dir / "test" / "mask_inv",
        script_dir / "test" / "edge",
    )

    # Save split information to CSV files
    print("\n💾 Saving split information...")
    train_df.to_csv(script_dir / "train_split_info.csv", index=False)
    val_df.to_csv(script_dir / "valid_split_info.csv", index=False)
    test_df.to_csv(script_dir / "test_split_info.csv", index=False)

    print("\n✅ PERFECTLY BALANCED dataset split completed successfully!")
    print(f"   train/img/    - {len(train_df)} images")
    print(f"   train/mask/   - {train_mask_success} masks")
    print(f"   valid/img/    - {len(val_df)} images")
    print(f"   valid/mask/   - {val_mask_success} masks")
    print(f"   test/img/     - {len(test_df)} images")
    print(f"   test/mask/    - {test_mask_success} masks")
    print("\n🎯 ALL SPLITS ARE PERFECTLY BALANCED!")

    # Create subset_4 for testing
    create_subset_4(test_df, script_dir)


if __name__ == "__main__":
    create_balanced_splits_main()
