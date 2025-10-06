#!/usr/bin/env python3
"""
Colab-specific script to create subset_4 data
Runs the subset_4 creation process in Google Colab environment
"""

import sys
from pathlib import Path

from create_subset_4 import create_subset_4


def setup_subset_4_colab():
    """Setup subset_4 data in Colab environment."""
    print("🔄 Setting up subset_4 data for Colab...")

    try:
        # Run the subset_4 creation
        create_subset_4()

        # Verify creation
        oai_data_dir = Path("data/oai")
        subset_4_path = oai_data_dir / "test" / "img" / "subset_4"

        if subset_4_path.exists():
            file_count = len(list(subset_4_path.glob("*.png")))
            print(f"✅ subset_4 verified: {file_count} images found")
            return True
        else:
            print("❌ subset_4 verification failed")
            return False

    except Exception as e:
        print(f"❌ Error creating subset_4: {e}")
        return False


if __name__ == "__main__":
    success = setup_subset_4_colab()
    if success:
        print("🎉 Subset_4 setup completed successfully!")
    else:
        print("💥 Subset_4 setup failed!")
        sys.exit(1)
