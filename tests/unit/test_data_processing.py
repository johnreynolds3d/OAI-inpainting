"""Unit tests for data processing functionality."""

import pytest
from pathlib import Path


def test_subset_4_exists(subset_4_data):
    """Test that subset_4 data exists and has expected structure."""
    # Check that all required directories exist
    assert subset_4_data["images"].exists(), "Images directory should exist"
    assert subset_4_data["masks"].exists(), "Masks directory should exist"
    assert subset_4_data["edges"].exists(), "Edges directory should exist"
    assert subset_4_data["info"].exists(), "Info CSV should exist"


def test_subset_4_image_count(subset_4_data):
    """Test that subset_4 has exactly 4 images."""
    image_files = list(subset_4_data["images"].glob("*.png"))
    assert len(image_files) == 4, f"Expected 4 images, found {len(image_files)}"


def test_subset_4_mask_count(subset_4_data):
    """Test that subset_4 has exactly 4 masks."""
    mask_files = list(subset_4_data["masks"].glob("*.png"))
    assert len(mask_files) == 4, f"Expected 4 masks, found {len(mask_files)}"


def test_subset_4_edge_count(subset_4_data):
    """Test that subset_4 has exactly 4 edge files."""
    edge_files = list(subset_4_data["edges"].glob("*.png"))
    assert len(edge_files) == 4, f"Expected 4 edge files, found {len(edge_files)}"


def test_subset_4_info_csv(subset_4_data):
    """Test that subset_4_info.csv has expected structure."""
    import pandas as pd

    df = pd.read_csv(subset_4_data["info"])

    # Check required columns
    required_columns = ["BMD", "filename", "is_osteo", "class"]
    for col in required_columns:
        assert col in df.columns, f"Required column '{col}' not found"

    # Check row count
    assert len(df) == 4, f"Expected 4 rows in info CSV, found {len(df)}"

    # Check that filenames match image files
    image_files = [f.name for f in subset_4_data["images"].glob("*.png")]
    csv_filenames = df["filename"].tolist()

    for filename in csv_filenames:
        assert (
            filename in image_files
        ), f"Filename '{filename}' not found in images directory"


def test_subset_4_filename_consistency(subset_4_data):
    """Test that image, mask, and edge filenames are consistent."""
    image_files = set(f.stem for f in subset_4_data["images"].glob("*.png"))
    mask_files = set(f.stem for f in subset_4_data["masks"].glob("*.png"))
    edge_files = set(f.stem for f in subset_4_data["edges"].glob("*.png"))

    assert image_files == mask_files, "Image and mask filenames don't match"
    assert image_files == edge_files, "Image and edge filenames don't match"
