"""
Diagnostic script to check if RePaint ImageNet has a mask inversion issue.

This checks if the "inpainted" images are suspiciously identical to ground truth,
which would indicate the model is preserving the wrong regions.
"""

from pathlib import Path

import numpy as np
from PIL import Image

# Constants
MASK_THRESHOLD = 0.5  # Threshold for binary mask
SUSPICIOUS_DIFF_THRESHOLD = (
    5  # Mean pixel difference threshold for suspicious similarity
)


def check_image_similarity(gt_path, inpainted_path, mask_path):
    """Compare GT, inpainted, and masked regions."""
    gt = np.array(Image.open(gt_path).convert("RGB"))
    inp = np.array(
        Image.open(inpainted_path).convert("RGB").resize((gt.shape[1], gt.shape[0]))
    )
    mask = np.array(
        Image.open(mask_path).convert("L").resize((gt.shape[1], gt.shape[0]))
    )

    # Normalize mask to 0-1
    mask_norm = mask / 255.0

    # Calculate differences
    total_diff = np.abs(gt.astype(float) - inp.astype(float))
    mean_diff = total_diff.mean()
    max_diff = total_diff.max()

    # Check difference in masked region (where inpainting should happen)
    # Mask convention: 255 = inpaint this region, 0 = keep original
    masked_region = mask_norm > MASK_THRESHOLD
    diff_in_masked = total_diff[masked_region].mean() if masked_region.any() else 0

    # Check difference in preserved region (should be identical to GT)
    preserved_region = mask_norm <= MASK_THRESHOLD
    diff_in_preserved = (
        total_diff[preserved_region].mean() if preserved_region.any() else 0
    )

    return {
        "mean_diff": mean_diff,
        "max_diff": max_diff,
        "diff_in_masked_region": diff_in_masked,
        "diff_in_preserved_region": diff_in_preserved,
        "mask_coverage": masked_region.sum() / masked_region.size,
    }


def main():
    """Check all models for potential mask inversion."""
    project_root = Path(__file__).parent.parent
    gt_dir = project_root / "data" / "oai" / "test" / "img" / "subset_4"
    mask_dir = project_root / "data" / "oai" / "test" / "mask" / "subset_4"

    models = [
        ("AOT-GAN", "CelebA-HQ", "results/AOT-GAN/CelebA-HQ/subset_4"),
        ("AOT-GAN", "Places2", "results/AOT-GAN/Places2/subset_4"),
        ("ICT", "FFHQ", "results/ICT/FFHQ/subset_4"),
        ("ICT", "ImageNet", "results/ICT/ImageNet/subset_4"),
        ("ICT", "Places2_Nature", "results/ICT/Places2_Nature/subset_4"),
        ("RePaint", "CelebA-HQ", "results/RePaint/CelebA-HQ/subset_4/inpainted"),
        ("RePaint", "ImageNet", "results/RePaint/ImageNet/subset_4/inpainted"),
        ("RePaint", "Places2", "results/RePaint/Places2/subset_4/inpainted"),
    ]

    print("=" * 80)
    print("🔍 MASK INVERSION DIAGNOSTIC")
    print("=" * 80)
    print("\nChecking if inpainted images are suspiciously similar to ground truth...")
    print("(High similarity suggests mask inversion or model failure)\n")

    gt_files = sorted(gt_dir.glob("*.png"))

    for family, variant, result_path in models:
        result_dir = project_root / result_path
        if not result_dir.exists():
            print(f"⚠️  {family} {variant}: Results not found")
            continue

        print(f"\n{'─' * 80}")
        print(f"🔍 {family} {variant}")
        print(f"{'─' * 80}")

        similarities = []
        for gt_file in gt_files:
            inpainted_file = result_dir / gt_file.name
            mask_file = mask_dir / gt_file.name

            if not inpainted_file.exists():
                print(f"  ⚠️  Missing: {gt_file.name}")
                continue

            if not mask_file.exists():
                print(f"  ⚠️  Mask missing: {gt_file.name}")
                continue

            stats = check_image_similarity(gt_file, inpainted_file, mask_file)
            similarities.append(stats)

            print(f"\n  📄 {gt_file.name}:")
            print(f"     Mean diff (overall): {stats['mean_diff']:.2f}")
            print(f"     Max diff: {stats['max_diff']:.2f}")
            print(f"     Diff in MASKED region: {stats['diff_in_masked_region']:.2f}")
            print(
                f"     Diff in PRESERVED region: {stats['diff_in_preserved_region']:.2f}"
            )
            print(f"     Mask coverage: {stats['mask_coverage'] * 100:.1f}%")

            # Flag suspicious cases
            if stats["mean_diff"] < SUSPICIOUS_DIFF_THRESHOLD:
                print("     🚨 SUSPICIOUS: Nearly identical to GT!")
            if stats["diff_in_preserved_region"] > stats["diff_in_masked_region"]:
                print("     🚨 POSSIBLE MASK INVERSION: Preserved region changed more!")

        if similarities:
            avg_masked = np.mean([s["diff_in_masked_region"] for s in similarities])
            avg_preserved = np.mean(
                [s["diff_in_preserved_region"] for s in similarities]
            )
            avg_overall = np.mean([s["mean_diff"] for s in similarities])

            print(f"\n  📊 Average across {len(similarities)} images:")
            print(f"     Overall diff: {avg_overall:.2f}")
            print(f"     Masked region diff: {avg_masked:.2f}")
            print(f"     Preserved region diff: {avg_preserved:.2f}")

            if avg_overall < SUSPICIOUS_DIFF_THRESHOLD:
                print("\n  🚨 ALERT: Suspiciously low difference from GT!")
                print("     → Model may be doing minimal inpainting")
            if avg_preserved > avg_masked:
                print("\n  🚨 ALERT: Preserved region changed MORE than masked!")
                print("     → Possible mask inversion issue")

    print("\n" + "=" * 80)
    print("✅ Diagnostic complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
