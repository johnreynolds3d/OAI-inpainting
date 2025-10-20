"""
Create inverted masks for RePaint models.

RePaint uses the opposite mask convention:
- White (255) = KEEP/PRESERVE this region
- Black (0) = INPAINT this region

Our standard masks are:
- White (255) = INPAINT this region
- Black (0) = KEEP/PRESERVE this region

This script creates inverted masks specifically for RePaint.
"""

from pathlib import Path

import numpy as np
from PIL import Image


def invert_mask(mask_path, output_path):
    """Invert a mask image (255 - pixel_value)."""
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    inverted_array = 255 - mask_array
    inverted_mask = Image.fromarray(inverted_array)
    inverted_mask.save(output_path)


def main():
    """Create inverted masks for RePaint in all test splits."""
    project_root = Path(__file__).parent.parent

    # Process all test splits
    splits = ["subset_4", "test", "val", "train"]

    for split in splits:
        if split == "subset_4":
            mask_dir = project_root / "data" / "oai" / "test" / "mask" / "subset_4"
            output_dir = (
                project_root / "data" / "oai" / "test" / "mask_repaint" / "subset_4"
            )
        else:
            mask_dir = project_root / "data" / "oai" / split / "mask"
            output_dir = project_root / "data" / "oai" / split / "mask_repaint"

        if not mask_dir.exists():
            print(f"⚠️  Skipping {split}: {mask_dir} not found")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)

        mask_files = sorted(mask_dir.glob("*.png"))
        if not mask_files:
            print(f"⚠️  No masks found in {mask_dir}")
            continue

        print(f"\n📁 Processing {split}:")
        print(f"   Source: {mask_dir}")
        print(f"   Output: {output_dir}")

        for mask_file in mask_files:
            output_file = output_dir / mask_file.name
            invert_mask(mask_file, output_file)
            print(f"   ✅ {mask_file.name}")

        print(f"   ✅ Created {len(mask_files)} inverted masks for {split}")

    print("\n" + "=" * 70)
    print("✅ Inverted mask generation complete!")
    print("=" * 70)
    print("\n💡 Next steps:")
    print("   1. Update RePaint config files to use mask_repaint/ instead of mask/")
    print("   2. Re-run RePaint models with inverted masks")
    print("   3. Compare results - inpainting should now happen in correct region!")


if __name__ == "__main__":
    main()
