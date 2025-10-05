#!/usr/bin/env python3
"""
PERFECT BALANCED Dataset Splitter for OAI Dataset
Creates train/validation/test splits with GUARANTEED equal representation of
osteoporotic and non-osteoporotic samples.
Uses manual splitting to ensure perfect balance in all splits.

Split: 80% train, 10% validation, 10% test
Each subset maintains EXACT equal balance of osteoporotic vs non-osteoporotic
samples.
"""

import os
import random
import shutil

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import canny


def generate_mask_for_image(image_path, output_path, image_size=224):
    """Generate a mask with one small square for a given image."""
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
    cv2.imwrite(output_path, mask)
    return mask


def create_inverted_mask(mask_path, inverted_mask_path):
    """Create an inverted mask for RePaint."""
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Invert mask: 0 becomes 255, 255 becomes 0
    inverted_mask = 255 - mask
    # Save inverted mask
    cv2.imwrite(inverted_mask_path, inverted_mask)
    return inverted_mask


def generate_edge_map(image_path, output_path, sigma=2, threshold=0.5):
    """Generate Canny edge detection map for an image (for ICT model)."""
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)
    # Convert to grayscale
    gray = rgb2gray(img_array)
    # Apply Canny edge detection
    edges = canny(
        gray, sigma=sigma, low_threshold=threshold * 0.5, high_threshold=threshold
    )
    # Convert back to RGB (3 channels)
    edges_rgb = gray2rgb(edges.astype(np.uint8) * 255)
    # Save edge map
    edge_img = Image.fromarray(edges_rgb)
    edge_img.save(output_path)
    return edges_rgb


def clean_directory(dir_path):
    """Clean a directory by removing all files."""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def manual_balanced_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Manually create perfectly balanced splits by ensuring equal numbers from each class.
    This guarantees perfect balance even with small numbers.
    """
    # Separate by class
    osteo_samples = df[df["is_osteo"]].reset_index(drop=True)
    normal_samples = df[~df["is_osteo"]].reset_index(drop=True)

    print(
        f"Available samples: {len(osteo_samples)} osteoporotic, "
        f"{len(normal_samples)} normal"
    )

    # Use the smaller class as the limiting factor
    min_class_size = min(len(osteo_samples), len(normal_samples))
    print(f"Using {min_class_size} samples from each class")

    # Calculate exact numbers for each split
    train_per_class = int(min_class_size * train_ratio)
    val_per_class = int(min_class_size * val_ratio)
    test_per_class = int(min_class_size * test_ratio)

    # Adjust if we don't have enough samples
    total_needed = (train_per_class + val_per_class + test_per_class) * 2
    if total_needed > min_class_size * 2:
        # Reduce proportionally
        scale_factor = (min_class_size * 2) / total_needed
        train_per_class = int(train_per_class * scale_factor)
        val_per_class = int(val_per_class * scale_factor)
        test_per_class = int(test_per_class * scale_factor)

    print(
        f"Split per class: Train={train_per_class}, "
        f"Val={val_per_class}, Test={test_per_class}"
    )

    # Set random seed for reproducibility
    np.random.seed(42)

    # Sample from each class
    osteo_shuffled = osteo_samples.sample(
        n=min_class_size, random_state=42
    ).reset_index(drop=True)
    normal_shuffled = normal_samples.sample(
        n=min_class_size, random_state=42
    ).reset_index(drop=True)

    # Create splits manually
    train_osteo = osteo_shuffled[:train_per_class]
    val_osteo = osteo_shuffled[train_per_class : train_per_class + val_per_class]
    test_osteo = osteo_shuffled[
        train_per_class + val_per_class : train_per_class
        + val_per_class
        + test_per_class
    ]

    train_normal = normal_shuffled[:train_per_class]
    val_normal = normal_shuffled[train_per_class : train_per_class + val_per_class]
    test_normal = normal_shuffled[
        train_per_class + val_per_class : train_per_class
        + val_per_class
        + test_per_class
    ]

    # Combine splits
    train_df = (
        pd.concat([train_osteo, train_normal], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat([val_osteo, val_normal], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    test_df = (
        pd.concat([test_osteo, test_normal], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return train_df, val_df, test_df


def create_balanced_splits():
    """Create perfectly balanced train/validation/test splits."""

    # Load the CSV data
    print("Loading BMD data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data.csv")
    df = pd.read_csv(csv_path, header=None, names=["BMD", "filename"])
    print(f"Total samples: {len(df)}")

    # Define osteoporosis threshold (using 25th percentile)
    osteo_threshold = df["BMD"].quantile(0.25)
    print(f"Osteoporosis threshold (25th percentile): {osteo_threshold:.4f}")

    # Classify samples
    df["is_osteo"] = df["BMD"] <= osteo_threshold
    df["class"] = df["is_osteo"].map({True: "osteoporotic", False: "normal"})

    # Check class distribution
    class_counts = df["class"].value_counts()
    print("\nClass distribution:")
    osteo_pct = class_counts["osteoporotic"] / len(df) * 100
    normal_pct = class_counts["normal"] / len(df) * 100
    print(f"Osteoporotic: {class_counts['osteoporotic']} ({osteo_pct:.1f}%)")
    print(f"Normal: {class_counts['normal']} ({normal_pct:.1f}%)")

    # Create perfectly balanced splits
    print("\n🎯 Creating PERFECTLY balanced splits...")
    train_df, val_df, test_df = manual_balanced_split(df)

    # Verify splits
    print("\nSplit sizes:")
    print(
        f"Train: {len(train_df)} ({len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100:.1f}%)"
    )
    print(
        f"Validation: {len(val_df)} ({len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100:.1f}%)"
    )
    print(
        f"Test: {len(test_df)} ({len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100:.1f}%)"
    )

    # Verify PERFECT class balance in each split
    for split_name, split_df in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        osteo_count = len(split_df[split_df["is_osteo"]])
        normal_count = len(split_df[~split_df["is_osteo"]])
        total = len(split_df)
        print(f"\n{split_name} set class balance:")
        osteo_pct = osteo_count / total * 100
        normal_pct = normal_count / total * 100
        print(f"  Osteoporotic: {osteo_count} ({osteo_pct:.1f}%)")
        print(f"  Normal: {normal_count} ({normal_pct:.1f}%)")

        # Verify perfect balance
        if osteo_count == normal_count:
            print("  ✅ PERFECTLY BALANCED!")
        else:
            diff = abs(osteo_count - normal_count)
            print(f"  ❌ IMBALANCED (difference: {diff})")

    # Create output directories and CLEAN them first
    output_dirs = [
        os.path.join(script_dir, "train/img"),
        os.path.join(script_dir, "train/mask"),
        os.path.join(script_dir, "train/mask_inv"),
        os.path.join(script_dir, "train/edge"),
        os.path.join(script_dir, "valid/img"),
        os.path.join(script_dir, "valid/mask"),
        os.path.join(script_dir, "valid/mask_inv"),
        os.path.join(script_dir, "valid/edge"),
        os.path.join(script_dir, "test/img"),
        os.path.join(script_dir, "test/mask"),
        os.path.join(script_dir, "test/mask_inv"),
        os.path.join(script_dir, "test/edge"),
    ]

    print("\n🧹 Cleaning existing directories...")
    for dir_name in output_dirs:
        os.makedirs(dir_name, exist_ok=True)
        clean_directory(dir_name)
        rel_path = os.path.relpath(dir_name, script_dir)
        print(f"Cleaned: {rel_path}/")

    # Copy images to respective directories
    print("\n📁 Copying images...")

    def copy_images(df, target_dir):
        """Copy images from the dataframe to the target directory"""
        copied_count = 0
        missing_count = 0

        for _, row in df.iterrows():
            src_path = os.path.join(script_dir, "img", row["filename"])
            dst_path = os.path.join(target_dir, row["filename"])

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: {src_path} not found!")
                missing_count += 1

        return copied_count, missing_count

    # Copy images for each split
    train_copied, train_missing = copy_images(
        train_df, os.path.join(script_dir, "train/img")
    )
    val_copied, val_missing = copy_images(val_df, os.path.join(script_dir, "valid/img"))
    test_copied, test_missing = copy_images(
        test_df, os.path.join(script_dir, "test/img")
    )

    print("\nCopy results:")
    print(f"Train: {train_copied} copied, {train_missing} missing")
    print(f"Validation: {val_copied} copied, {val_missing} missing")
    print(f"Test: {test_copied} copied, {test_missing} missing")

    # Generate masks and edge maps for each split
    print("\n🎭 Generating masks and edge maps for all splits...")

    def generate_masks_for_split(
        split_df, split_name, img_dir, mask_dir, mask_inv_dir, edge_dir=None
    ):
        """Generate masks, inverted masks, and edge maps for all images in a split."""
        print(f"\n🎭 Generating masks for {split_name} set...")

        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(mask_inv_dir, exist_ok=True)
        if edge_dir:
            os.makedirs(edge_dir, exist_ok=True)

        successful = 0
        failed = 0

        for _, row in split_df.iterrows():
            try:
                mask_filename = os.path.splitext(row["filename"])[0] + ".png"
                mask_path = os.path.join(mask_dir, mask_filename)
                inverted_mask_path = os.path.join(mask_inv_dir, mask_filename)

                image_path = os.path.join(img_dir, row["filename"])
                if os.path.exists(image_path):
                    generate_mask_for_image(image_path, mask_path)
                    create_inverted_mask(mask_path, inverted_mask_path)

                    if edge_dir:
                        edge_path = os.path.join(edge_dir, mask_filename)
                        generate_edge_map(image_path, edge_path)

                    successful += 1
                else:
                    print(f"Warning: {image_path} not found!")
                    failed += 1

            except Exception as e:
                print(f"Error processing {row['filename']}: {e!s}")
                failed += 1

        print(f"   {split_name} masks: {successful} successful, {failed} failed")
        return successful, failed

    # Generate masks and edge maps for each split
    train_mask_success, train_mask_failed = generate_masks_for_split(
        train_df,
        "train",
        os.path.join(script_dir, "train/img"),
        os.path.join(script_dir, "train/mask"),
        os.path.join(script_dir, "train/mask_inv"),
        os.path.join(script_dir, "train/edge"),
    )
    val_mask_success, val_mask_failed = generate_masks_for_split(
        val_df,
        "validation",
        os.path.join(script_dir, "valid/img"),
        os.path.join(script_dir, "valid/mask"),
        os.path.join(script_dir, "valid/mask_inv"),
        os.path.join(script_dir, "valid/edge"),
    )
    test_mask_success, test_mask_failed = generate_masks_for_split(
        test_df,
        "test",
        os.path.join(script_dir, "test/img"),
        os.path.join(script_dir, "test/mask"),
        os.path.join(script_dir, "test/mask_inv"),
        os.path.join(script_dir, "test/edge"),
    )

    # Save split information to CSV files
    print("\n💾 Saving split information...")
    train_df.to_csv(os.path.join(script_dir, "train_split_info.csv"), index=False)
    val_df.to_csv(os.path.join(script_dir, "valid_split_info.csv"), index=False)
    test_df.to_csv(os.path.join(script_dir, "test_split_info.csv"), index=False)

    print("\n✅ PERFECTLY BALANCED dataset split completed successfully!")
    print("📁 Directory structure:")
    print(f"   train/img/   - {len(train_df)} images")
    print(f"   train/mask/  - {train_mask_success} masks")
    print(f"   valid/img/   - {len(val_df)} images")
    print(f"   valid/mask/  - {val_mask_success} masks")
    print(f"   test/img/    - {len(test_df)} images")
    print(f"   test/mask/   - {test_mask_success} masks")
    print("\n🎯 ALL SPLITS ARE PERFECTLY BALANCED!")

    # Create subset_4 for testing
    create_subset_4(test_df, script_dir)


def create_subset_4(test_df, script_dir):
    """Create subset_4 with 2 osteoporotic and 2 non-osteoporotic images from test set."""
    print("\n🎯 Creating subset_4 with 2 osteoporotic + 2 non-osteoporotic images...")

    # Separate test data by class
    test_osteo = test_df[test_df["is_osteo"]].reset_index(drop=True)
    test_normal = test_df[~test_df["is_osteo"]].reset_index(drop=True)

    print(
        f"Available in test set: {len(test_osteo)} osteoporotic, "
        f"{len(test_normal)} normal"
    )

    # Check if we have enough samples
    if len(test_osteo) < 2 or len(test_normal) < 2:
        print("❌ Not enough samples in test set for subset_4!")
        print("   Need: 2 osteoporotic, 2 normal")
        print(f"   Have: {len(test_osteo)} osteoporotic, {len(test_normal)} normal")
        return

    # Randomly select 2 from each class
    np.random.seed(42)  # Use same seed for reproducibility
    selected_osteo = test_osteo.sample(n=2, random_state=42).reset_index(drop=True)
    selected_normal = test_normal.sample(n=2, random_state=42).reset_index(drop=True)

    # Combine selected samples
    subset_4_df = pd.concat([selected_osteo, selected_normal], ignore_index=True)

    print("Selected for subset_4:")
    for _, row in subset_4_df.iterrows():
        class_label = "osteoporotic" if row["is_osteo"] else "normal"
        print(f"   {row['filename']} ({class_label})")

    # Create subset_4 directories (only subset_4 subdirectories, not main test dirs)
    subset_4_dirs = [
        os.path.join(script_dir, "test/img/subset_4"),
        os.path.join(script_dir, "test/mask/subset_4"),
        os.path.join(script_dir, "test/mask_inv/subset_4"),
        os.path.join(script_dir, "test/edge/subset_4"),
    ]

    # Safety check: ensure we're only working with subset_4 subdirectories
    for dir_path in subset_4_dirs:
        if not dir_path.endswith("/subset_4"):
            raise ValueError(
                f"Safety check failed: {dir_path} is not a subset_4 directory!"
            )

    print("\n📁 Creating subset_4 directories...")
    for dir_path in subset_4_dirs:
        os.makedirs(dir_path, exist_ok=True)
        # Clean existing files to avoid conflicts
        clean_directory(dir_path)
        rel_path = os.path.relpath(dir_path, script_dir)
        print(f"   Created and cleaned: {rel_path}/")

    # Copy selected images and their masks
    print("\n📋 Copying subset_4 files...")
    copied_count = 0

    for _, row in subset_4_df.iterrows():
        filename = row["filename"]
        mask_filename = os.path.splitext(filename)[0] + ".png"

        # Source paths
        src_img = os.path.join(script_dir, "test/img", filename)
        src_mask = os.path.join(script_dir, "test/mask", mask_filename)
        src_mask_inv = os.path.join(script_dir, "test/mask_inv", mask_filename)
        src_edge = os.path.join(script_dir, "test/edge", mask_filename)

        # Destination paths
        dst_img = os.path.join(script_dir, "test/img/subset_4", filename)
        dst_mask = os.path.join(script_dir, "test/mask/subset_4", mask_filename)
        dst_mask_inv = os.path.join(script_dir, "test/mask_inv/subset_4", mask_filename)
        dst_edge = os.path.join(script_dir, "test/edge/subset_4", mask_filename)

        # Copy files
        files_to_copy = [
            (src_img, dst_img),
            (src_mask, dst_mask),
            (src_mask_inv, dst_mask_inv),
            (src_edge, dst_edge),
        ]

        for src, dst in files_to_copy:
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1
            else:
                print(f"   Warning: {src} not found!")

    # Save subset_4 info
    subset_4_df.to_csv(os.path.join(script_dir, "test/subset_4_info.csv"), index=False)

    print("\n✅ Subset_4 created successfully!")
    print("   📁 Location: test/*/subset_4/")
    print(f"   📊 Files copied: {copied_count}")
    print("   📋 Info saved: test/subset_4_info.csv")
    print("   🎯 Perfect balance: 2 osteoporotic + 2 normal images")


if __name__ == "__main__":
    create_balanced_splits()
