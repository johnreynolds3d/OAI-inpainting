"""
Pytest configuration and fixtures for OAI Inpainting project.
"""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample image for testing."""
    return Image.new("RGB", (224, 224), color="white")


@pytest.fixture
def sample_mask() -> Image.Image:
    """Create a sample mask for testing."""
    mask = Image.new("L", (224, 224), color=255)  # White background
    # Add a black square in the center
    mask_array = np.array(mask)
    mask_array[100:124, 100:124] = 0  # Black square
    return Image.fromarray(mask_array)


@pytest.fixture
def sample_data_dir(temp_dir: Path) -> Path:
    """Create a sample data directory structure."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()

    # Create subdirectories
    (data_dir / "img").mkdir()
    (data_dir / "mask").mkdir()
    (data_dir / "mask_inv").mkdir()
    (data_dir / "edge").mkdir()

    # Create sample files
    for i in range(4):
        img = Image.new("RGB", (224, 224), color="white")
        img.save(data_dir / "img" / f"sample_{i:02d}.png")

        mask = Image.new("L", (224, 224), color=255)
        mask_array = np.array(mask)
        mask_array[100:124, 100:124] = 0
        mask_img = Image.fromarray(mask_array)
        mask_img.save(data_dir / "mask" / f"sample_{i:02d}.png")

        # Inverted mask
        inv_mask = Image.fromarray(255 - mask_array)
        inv_mask.save(data_dir / "mask_inv" / f"sample_{i:02d}.png")

        # Edge map
        edge = Image.new("RGB", (224, 224), color="black")
        edge.save(data_dir / "edge" / f"sample_{i:02d}.png")

    return data_dir


@pytest.fixture
def sample_config() -> dict:
    """Create a sample configuration dictionary."""
    return {
        "model": {
            "name": "test_model",
            "type": "aot-gan",
            "pretrained_path": "test_path.pt",
        },
        "data": {
            "img_dir": "test_img",
            "mask_dir": "test_mask",
            "output_dir": "test_output",
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "learning_rate": 0.001,
        },
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""

    class MockModel:
        def __init__(self):
            self.training = True

        def train(self):
            self.training = True

        def eval(self):
            self.training = False

        def __call__(self, x):
            return torch.randn_like(x)

    return MockModel()


@pytest.fixture
def device():
    """Get the device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def session_temp_dir() -> Generator[Path, None, None]:
    """Create a session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "data: marks tests that require data")
    config.addinivalue_line("markers", "model: marks tests that require models")
