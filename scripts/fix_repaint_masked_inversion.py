"""
Fix localized brightness inversion in RePaint outputs.

The RePaint ImageNet model outputs images where ONLY the inpainted/masked
region has inverted brightness (like a photographic negative), while the
preserved region looks normal.

This script:
1. Loads inpainted image, ground truth, and mask
2. Inverts brightness ONLY in the masked region: masked_region = 255 - masked_region
3. Keeps preserved region unchanged
4. Saves the corrected image
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Constants
MASK_THRESHOLD = 127  # Threshold for binary mask (0-255)


def fix_masked_inversion(inpainted_path, mask_path, output_path):
    """
    Fix brightness inversion in masked region only.

    Args:
        inpainted_path: Path to inpainted image
        mask_path: Path to mask (white=inpainted region, black=preserved)
        output_path: Path to save corrected image
    """
    # Load images
    inpainted = np.array(Image.open(inpainted_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    # Resize mask to match inpainted if needed
    if mask.shape != inpainted.shape[:2]:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(
            (inpainted.shape[1], inpainted.shape[0]), Image.LANCZOS
        )
        mask = np.array(mask_pil)

    # Create binary mask (True = inpainted region to fix)
    mask_binary = mask > MASK_THRESHOLD

    # Create 3-channel mask for RGB
    mask_3d = np.stack([mask_binary] * 3, axis=-1)

    # Invert brightness ONLY in masked region
    corrected = inpainted.copy()
    corrected[mask_3d] = 255 - inpainted[mask_3d]

    # Save corrected image
    Image.fromarray(corrected).save(output_path)


def process_model(model_name, inpainted_dir, mask_dir, output_dir=None):
    """Process all images for a specific model."""
    inpainted_path = Path(inpainted_dir)
    mask_path = Path(mask_dir)

    if not inpainted_path.exists():
        print(f"⚠️  {model_name}: Inpainted directory not found at {inpainted_path}")
        return 0

    if not mask_path.exists():
        print(f"⚠️  {model_name}: Mask directory not found at {mask_path}")
        return 0

    print(f"\n{'=' * 70}")
    print(f"🔧 Fixing {model_name}")
    print(f"{'=' * 70}")

    # If no output_dir specified, overwrite original
    if output_dir is None:
        output_path = inpainted_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    inpainted_files = sorted(inpainted_path.glob("*.png"))
    if not inpainted_files:
        print(f"⚠️  No images found in {inpainted_path}")
        return 0

    fixed_count = 0

    for inp_file in inpainted_files:
        mask_file = mask_path / inp_file.name

        if not mask_file.exists():
            print(f"  ⚠️  Mask not found for {inp_file.name}, skipping")
            continue

        output_file = output_path / inp_file.name

        try:
            fix_masked_inversion(inp_file, mask_file, output_file)
            print(f"  ✅ Fixed: {inp_file.name}")
            fixed_count += 1
        except Exception as e:
            print(f"  ❌ Error processing {inp_file.name}: {e}")

    print(f"\n📊 Summary: Fixed {fixed_count}/{len(inpainted_files)} images")
    return fixed_count


def main():
    """Fix masked inversion for all RePaint models."""
    project_root = Path(__file__).parent.parent
    mask_dir = project_root / "data" / "oai" / "test" / "mask" / "subset_4"

    print("=" * 70)
    print("🔧 REPAINT MASKED REGION BRIGHTNESS INVERSION FIX")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load inpainted images and masks")
    print("  2. Invert brightness (255-value) ONLY in masked regions")
    print("  3. Keep preserved regions unchanged")
    print("  4. Overwrite original images with corrected versions\n")

    input("Press Enter to continue or Ctrl+C to cancel...")

    models_to_fix = [
        ("RePaint ImageNet", "results/RePaint/ImageNet/subset_4/inpainted"),
        ("RePaint Places2", "results/RePaint/Places2/subset_4/inpainted"),
        ("RePaint CelebA-HQ", "results/RePaint/CelebA-HQ/subset_4/inpainted"),
    ]

    total_fixed = 0

    for model_name, inpainted_path in models_to_fix:
        fixed = process_model(model_name, project_root / inpainted_path, mask_dir)
        total_fixed += fixed

    print("\n" + "=" * 70)
    print("🎉 COMPLETE!")
    print("=" * 70)
    print(f"\nTotal images fixed: {total_fixed}")

    if total_fixed > 0:
        print("\n💡 Next steps:")
        print("   1. Visually inspect corrected images")
        print("   2. Re-run generate_metrics_table.py to get corrected metrics")
        print(
            "   3. RePaint ImageNet metrics should drop to realistic levels (~17-20 dB)"
        )


if __name__ == "__main__":
    main()
