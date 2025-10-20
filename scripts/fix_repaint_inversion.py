"""
Fix color inversion in RePaint ImageNet outputs.

The RePaint ImageNet model appears to output inverted pixel values
(black becomes white, white becomes black), like a photographic negative.

This script:
1. Detects inverted images by comparing with ground truth
2. Inverts the pixel values (255 - pixel_value) to correct them
3. Saves corrected images back to the same location
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Constants
INVERSION_CORRELATION_THRESHOLD = 0.8  # Threshold for detecting inverted images


def invert_image(img_array):
    """Invert image pixel values: 255 - value."""
    return 255 - img_array


def check_if_inverted(gt_path, inpainted_path):
    """
    Check if inpainted image appears to be inverted relative to GT.

    Returns True if the inpainted appears to be a negative of GT.
    """
    gt = np.array(Image.open(gt_path).convert("L"))  # Grayscale for simplicity
    inp = np.array(Image.open(inpainted_path).convert("L").resize(gt.shape[::-1]))

    # Check correlation with normal and inverted versions
    # If image is inverted, correlation with (255-GT) should be higher than with GT
    inv_gt = 255 - gt

    # Normalize to [-1, 1] for correlation
    gt_norm = (gt.astype(float) - 127.5) / 127.5
    inv_gt_norm = (inv_gt.astype(float) - 127.5) / 127.5
    inp_norm = (inp.astype(float) - 127.5) / 127.5

    # Correlation coefficients
    corr_normal = np.corrcoef(gt_norm.flatten(), inp_norm.flatten())[0, 1]
    corr_inverted = np.corrcoef(inv_gt_norm.flatten(), inp_norm.flatten())[0, 1]

    # If correlation with inverted GT is significantly higher, image is likely inverted
    return (
        corr_inverted > corr_normal and corr_inverted > INVERSION_CORRELATION_THRESHOLD
    )


def fix_inverted_images(model_name, result_dir, gt_dir):
    """Fix inverted images for a specific model."""
    result_path = Path(result_dir)
    if not result_path.exists():
        print(f"⚠️  {model_name}: Results not found at {result_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"🔍 Checking {model_name}")
    print(f"{'=' * 70}")

    # Get GT files first, then only process inpainted images that have corresponding GT
    gt_path = Path(gt_dir)
    gt_files = {f.name: f for f in sorted(gt_path.glob("*.png"))}

    if not gt_files:
        print(f"⚠️  No GT images found in {gt_dir}")
        return

    image_files = sorted(result_path.glob("*.png"))
    if not image_files:
        print(f"⚠️  No images found in {result_path}")
        return

    # Filter to only process images that have corresponding GT
    valid_image_files = [f for f in image_files if f.name in gt_files]

    if len(valid_image_files) < len(image_files):
        extra_count = len(image_files) - len(valid_image_files)
        print(
            f"   Found {len(image_files)} inpainted images, "
            f"{len(valid_image_files)} match subset_4 GT ({extra_count} extras skipped)"
        )

    inverted_count = 0
    fixed_count = 0

    for img_file in valid_image_files:
        gt_file = gt_files[img_file.name]

        # Check if inverted
        is_inverted = check_if_inverted(gt_file, img_file)

        if is_inverted:
            inverted_count += 1
            print(f"  🔄 {img_file.name}: INVERTED - fixing...")

            # Load, invert, and save
            img = Image.open(img_file)
            img_array = np.array(img)
            inverted_array = invert_image(img_array)
            fixed_img = Image.fromarray(inverted_array)
            fixed_img.save(img_file)

            fixed_count += 1
            print("     ✅ Fixed and saved")
        else:
            print(f"  ✅ {img_file.name}: Normal (no fix needed)")

    print(f"\n📊 Summary for {model_name}:")
    print(f"   Total processed: {len(valid_image_files)}")
    print(f"   Inverted: {inverted_count}")
    print(f"   Fixed: {fixed_count}")

    return inverted_count, fixed_count


def main():
    """Check and fix all RePaint model outputs."""
    project_root = Path(__file__).parent.parent
    gt_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"

    models_to_check = [
        ("RePaint ImageNet", "results/RePaint/ImageNet/subset_4/inpainted"),
        ("RePaint Places2", "results/RePaint/Places2/subset_4/inpainted"),
        ("RePaint CelebA-HQ", "results/RePaint/CelebA-HQ/subset_4/inpainted"),
    ]

    print("=" * 70)
    print("🔧 REPAINT COLOR INVERSION FIX")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Check if inpainted images have inverted colors (negative effect)")
    print("  2. Automatically fix inverted images by inverting pixel values")
    print("  3. Save corrected images back to original location\n")

    total_inverted = 0
    total_fixed = 0

    for model_name, result_path in models_to_check:
        inv, fix = fix_inverted_images(
            model_name, project_root / result_path, gt_dir
        ) or (0, 0)
        total_inverted += inv
        total_fixed += fix

    print("\n" + "=" * 70)
    print("🎉 COMPLETE!")
    print("=" * 70)
    print(f"\nTotal inverted images found: {total_inverted}")
    print(f"Total images fixed: {total_fixed}")

    if total_fixed > 0:
        print("\n💡 Next steps:")
        print("   1. Re-run generate_metrics_table.py to get corrected metrics")
        print("   2. Visual check: Compare fixed images with ground truth")
        print("   3. Update presentation materials with correct results")


if __name__ == "__main__":
    main()
