"""Shared test fixtures and configuration for pytest."""

import pytest
from pathlib import Path


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def subset_4_data(project_root):
    """Return paths to subset_4 test data."""
    return {
        "images": project_root / "data" / "oai" / "test" / "img" / "subset_4",
        "masks": project_root / "data" / "oai" / "test" / "mask" / "subset_4",
        "edges": project_root / "data" / "oai" / "test" / "edge" / "subset_4",
        "info": project_root / "data" / "oai" / "test" / "subset_4_info.csv",
    }


@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "model": "aotgan",
        "pre_train": "test_model.pt",
        "dir_image": "test_images/",
        "dir_mask": "test_masks/",
        "outputs": "test_outputs/",
    }
