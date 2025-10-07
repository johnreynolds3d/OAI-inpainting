#!/usr/bin/env python3
"""
Generate Comparison Strips for Model Evaluation
Creates horizontal strips showing: GT, GT+Mask, and all model variant outputs
Optimized for visual comparison and thesis inclusion.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_gt_with_mask(gt_img: np.ndarray, mask_img: np.ndarray) -> np.ndarray:
    """Create GT with red mask overlay."""
    # Convert grayscale to RGB for red overlay
    if len(gt_img.shape) == 2:
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
    else:
        gt_rgb = gt_img.copy()

    # Create red overlay
    red_overlay = np.zeros_like(gt_rgb)
    red_overlay[:, :, 2] = 255  # Red channel (BGR format)

    # Apply mask (white areas in mask)
    mask_bool = mask_img > 127
    gt_with_mask = gt_rgb.copy()
    gt_with_mask[mask_bool] = red_overlay[mask_bool]

    return gt_with_mask


def load_model_outputs(img_name: str, results_base: Path) -> dict:
    """Load all model variant outputs for an image."""
    model_outputs = {}
    
    # Model families and their variants
    models_to_check = [
        ("AOT-GAN", ["CelebA-HQ", "Places2", "OAI"]),
        ("ICT", ["FFHQ", "ImageNet", "Places2_Nature", "OAI"]),
        ("RePaint", ["CelebA-HQ", "ImageNet", "Places2"]),
    ]
    
    for model_family, variants in models_to_check:
        for variant in variants:
            variant_dir = results_base / model_family / variant / "subset_4"
            
            if not variant_dir.exists():
                continue
            
            # Look for the image in various possible locations/names
            possible_patterns = [
                img_name,
                img_name.replace(".png", "_pred.png"),
                img_name.replace(".png", "_comp.png"),
                img_name.replace(".png", "_output.png"),
                img_name.replace(".png", "_0.png"),  # ICT format
            ]
            
            # Also check in subdirectories (RePaint uses 'inpainted/')
            subdirs_to_check = [".", "inpainted", "results", "output"]
            
            found = False
            for subdir in subdirs_to_check:
                if found:
                    break
                check_dir = variant_dir / subdir if subdir != "." else variant_dir
                if not check_dir.exists():
                    continue
                    
                for pattern in possible_patterns:
                    img_path = check_dir / pattern
                    if img_path.exists():
                        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            model_outputs[f"{model_family} {variant}"] = img
                            found = True
                            break
    
    return model_outputs


def add_labels_to_strip(
    strip: np.ndarray, labels: list, image_width: int
) -> np.ndarray:
    """Add text labels under each image in the strip."""
    strip_height, strip_width = strip.shape[:2]
    label_height = 40
    total_height = strip_height + label_height

    # Create new image with space for labels
    strip_with_labels = np.ones((total_height, strip_width, 3), dtype=np.uint8) * 255
    strip_with_labels[:strip_height, :] = strip

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (0, 0, 0)  # Black

    for i, label in enumerate(labels):
        x_start = i * image_width
        x_center = x_start + image_width // 2
        
        # Get text size for centering
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x = x_center - text_size[0] // 2
        text_y = strip_height + 25

        cv2.putText(
            strip_with_labels,
            label,
            (text_x, text_y),
            font,
            font_scale,
            font_color,
            font_thickness,
        )

    return strip_with_labels


def add_filename_label(strip: np.ndarray, img_name: str) -> np.ndarray:
    """Add filename label above the strip."""
    filename_height = 50
    strip_height, strip_width = strip.shape[:2]
    total_height = strip_height + filename_height

    strip_with_filename = np.ones((total_height, strip_width, 3), dtype=np.uint8) * 255
    strip_with_filename[filename_height:, :] = strip

    # Add filename
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (0, 0, 0)

    text_size = cv2.getTextSize(img_name, font, font_scale, font_thickness)[0]
    text_x = (strip_width - text_size[0]) // 2
    text_y = 35

    cv2.putText(
        strip_with_filename,
        img_name,
        (text_x, text_y),
        font,
        font_scale,
        font_color,
        font_thickness,
    )

    return strip_with_filename


def create_comparison_strip(
    img_name: str,
    gt_dir: Path,
    mask_dir: Path,
    results_base: Path,
    output_dir: Path,
) -> Path:
    """Create a horizontal comparison strip for one image."""
    print(f"🖼️  Creating strip for {img_name}")

    # Load GT and mask
    gt_path = gt_dir / img_name
    mask_path = mask_dir / img_name

    if not gt_path.exists() or not mask_path.exists():
        print(f"  ❌ Missing GT or mask for {img_name}")
        return None

    gt_img = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if gt_img is None or mask_img is None:
        print(f"  ❌ Failed to load GT or mask for {img_name}")
        return None

    # Create GT with red mask overlay
    gt_with_mask = create_gt_with_mask(gt_img, mask_img)

    # Load all model outputs
    model_outputs = load_model_outputs(img_name, results_base)

    # Collect images for strip
    strip_images = []
    strip_labels = []

    # Add GT (grayscale)
    strip_images.append(gt_img)
    strip_labels.append("GT")

    # Add GT+Mask (color with red overlay)
    strip_images.append(gt_with_mask)
    strip_labels.append("GT+Mask")

    # Add model outputs in order
    model_order = [
        "AOT-GAN CelebA-HQ",
        "AOT-GAN Places2",
        "AOT-GAN OAI",
        "ICT FFHQ",
        "ICT ImageNet",
        "ICT Places2_Nature",
        "ICT OAI",
        "RePaint CelebA-HQ",
        "RePaint ImageNet",
        "RePaint Places2",
    ]

    for model_name in model_order:
        if model_name in model_outputs:
            strip_images.append(model_outputs[model_name])
            strip_labels.append(model_name)

    print(f"  ✅ Found {len(strip_images)} images for strip")

    # Resize all to 256x256
    target_size = (256, 256)
    resized_images = []

    for img in strip_images:
        resized = cv2.resize(img, target_size)
        # Convert to RGB for consistent concatenation
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        resized_images.append(resized)

    # Concatenate horizontally
    strip = np.hstack(resized_images)

    # Add labels
    strip_with_labels = add_labels_to_strip(strip, strip_labels, target_size[0])

    # Add filename
    strip_with_filename = add_filename_label(strip_with_labels, img_name)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = img_name.replace(".png", "")
    strip_path = output_dir / f"comparison_{base_name}.png"
    
    cv2.imwrite(str(strip_path), cv2.cvtColor(strip_with_filename, cv2.COLOR_RGB2BGR))
    print(f"  ✅ Saved: {strip_path.name}")

    return strip_path


def create_summary_figure(strip_paths: list, output_dir: Path) -> Path:
    """Create summary figure with all strips vertically stacked."""
    print("\n📊 Creating summary figure...")

    if not strip_paths:
        print("  ❌ No strips to create summary")
        return None

    # Load all strips
    strips = []
    for strip_path in strip_paths:
        if strip_path and strip_path.exists():
            strip_img = cv2.imread(str(strip_path))
            if strip_img is not None:
                strips.append(strip_img)

    if not strips:
        print("  ❌ No valid strips loaded")
        return None

    # Pad to same width
    max_width = max(s.shape[1] for s in strips)
    padded_strips = []

    for strip in strips:
        if strip.shape[1] < max_width:
            padding_width = max_width - strip.shape[1]
            padding = np.ones((strip.shape[0], padding_width, 3), dtype=np.uint8) * 255
            padded_strip = np.hstack([strip, padding])
            padded_strips.append(padded_strip)
        else:
            padded_strips.append(strip)

    # Stack vertically
    summary_img = np.vstack(padded_strips)

    # Add title
    title_height = 60
    total_height = summary_img.shape[0] + title_height
    summary_with_title = np.ones(
        (total_height, summary_img.shape[1], 3), dtype=np.uint8
    ) * 255
    summary_with_title[title_height:, :] = summary_img

    # Add title text
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "OAI Inpainting: Model Comparison on subset_4"
    text_size = cv2.getTextSize(title_text, font, 1.2, 3)[0]
    text_x = (summary_img.shape[1] - text_size[0]) // 2
    text_y = 40

    cv2.putText(
        summary_with_title,
        title_text,
        (text_x, text_y),
        font,
        1.2,
        (0, 0, 0),
        3,
    )

    # Save
    summary_path = output_dir / "all_comparisons_summary.png"
    cv2.imwrite(str(summary_path), summary_with_title)
    print(f"  ✅ Summary saved: {summary_path.name}")

    return summary_path


def main():
    """Generate comparison strips for all subset_4 images."""
    print("=" * 60)
    print("🎨 COMPARISON STRIP GENERATOR")
    print("=" * 60)

    # Paths
    data_dir = project_root / "data" / "oai" / "test"
    gt_dir = data_dir / "img" / "subset_4"
    mask_dir = data_dir / "mask" / "subset_4"
    results_base = project_root / "results"
    output_dir = project_root / "results" / "comparison_strips"

    print(f"📁 GT images: {gt_dir}")
    print(f"📁 Masks: {mask_dir}")
    print(f"📁 Results: {results_base}")
    print(f"📁 Output: {output_dir}")

    # Check if data exists
    if not gt_dir.exists():
        print(f"\n❌ GT directory not found: {gt_dir}")
        print("💡 Run split.py first to generate subset_4")
        sys.exit(1)

    # Get all images from subset_4
    test_images = sorted(list(gt_dir.glob("*.png")))

    if not test_images:
        print(f"\n❌ No images found in {gt_dir}")
        sys.exit(1)

    print(f"\n✅ Found {len(test_images)} test images")

    # Create strips
    strip_paths = []
    for img_path in test_images:
        img_name = img_path.name
        strip_path = create_comparison_strip(
            img_name, gt_dir, mask_dir, results_base, output_dir
        )
        if strip_path:
            strip_paths.append(strip_path)

    # Create summary
    summary_path = create_summary_figure(strip_paths, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Created {len(strip_paths)} comparison strips")
    print(f"✅ Summary figure: {summary_path.name if summary_path else 'N/A'}")
    print(f"📁 Output: {output_dir}")

    return strip_paths


if __name__ == "__main__":
    main()

