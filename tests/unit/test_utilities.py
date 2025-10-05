"""Unit tests for utility functions."""

import tempfile
from pathlib import Path

import pytest


def test_path_utilities():
    """Test path utility functions."""
    # Test Path operations
    test_path = Path("/tmp/test/path")

    # Test path joining
    joined_path = test_path / "subdir" / "file.txt"
    assert str(joined_path) == "/tmp/test/path/subdir/file.txt"

    # Test path parts
    assert joined_path.parent == test_path / "subdir"
    assert joined_path.name == "file.txt"
    assert joined_path.suffix == ".txt"
    assert joined_path.stem == "file"


def test_file_operations():
    """Test basic file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Test file creation
        test_file = temp_path / "test.txt"
        test_file.write_text("Hello, World!")
        assert test_file.exists()
        assert test_file.read_text() == "Hello, World!"

        # Test directory creation
        test_dir = temp_path / "test_dir"
        test_dir.mkdir()
        assert test_dir.exists()
        assert test_dir.is_dir()


def test_directory_operations():
    """Test directory operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create nested directories
        nested_dir = temp_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True)
        assert nested_dir.exists()

        # Create files in nested directory
        test_file = nested_dir / "test.txt"
        test_file.write_text("Nested file")
        assert test_file.exists()

        # Test globbing
        all_files = list(temp_path.rglob("*.txt"))
        assert len(all_files) == 1
        assert all_files[0] == test_file


def test_config_parsing():
    """Test configuration parsing utilities."""
    import yaml

    # Test YAML parsing
    test_config = {
        "model": "aotgan",
        "pre_train": "model.pt",
        "dir_image": "images/",
        "dir_mask": "masks/",
        "outputs": "outputs/",
    }

    # Convert to YAML string
    yaml_str = yaml.dump(test_config)

    # Parse back
    parsed_config = yaml.safe_load(yaml_str)
    assert parsed_config == test_config

    # Test nested config
    nested_config = {
        "model": {
            "name": "aotgan",
            "params": {"learning_rate": 0.001, "batch_size": 4},
        },
        "data": {"train": "train/", "val": "val/"},
    }

    yaml_str = yaml.dump(nested_config)
    parsed_config = yaml.safe_load(yaml_str)
    assert parsed_config == nested_config


def test_image_processing_utilities():
    """Test image processing utility functions."""
    import numpy as np
    from PIL import Image

    # Create test image
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(test_image)
    assert pil_image.size == (100, 100)
    assert pil_image.mode == "RGB"

    # Convert back to numpy
    numpy_image = np.array(pil_image)
    assert numpy_image.shape == (100, 100, 3)
    assert np.array_equal(numpy_image, test_image)

    # Test image resizing
    resized = pil_image.resize((50, 50))
    assert resized.size == (50, 50)

    # Test image saving and loading
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        pil_image.save(tmp_file.name)
        loaded_image = Image.open(tmp_file.name)
        assert loaded_image.size == pil_image.size
        assert loaded_image.mode == pil_image.mode


def test_data_validation():
    """Test data validation utilities."""
    import numpy as np
    import pandas as pd

    # Test DataFrame validation
    test_data = {
        "filename": ["image1.png", "image2.png", "image3.png"],
        "BMD": [0.8, 0.9, 1.0],
        "is_osteo": [True, False, False],
        "class": ["osteoporotic", "normal", "normal"],
    }

    df = pd.DataFrame(test_data)

    # Validate required columns
    required_columns = ["filename", "BMD", "is_osteo", "class"]
    for col in required_columns:
        assert col in df.columns, f"Required column {col} missing"

    # Validate data types
    assert df["BMD"].dtype in [np.float64, np.float32], "BMD should be numeric"
    assert df["is_osteo"].dtype == bool, "is_osteo should be boolean"

    # Validate data ranges
    assert df["BMD"].min() >= 0, "BMD should be non-negative"
    assert df["BMD"].max() <= 2, "BMD should be reasonable"

    # Validate unique filenames
    assert df["filename"].nunique() == len(df), "Filenames should be unique"


def test_error_handling():
    """Test error handling utilities."""
    # Test file not found handling
    non_existent_file = Path("/non/existent/file.txt")
    assert not non_existent_file.exists()

    # Test directory not found handling
    non_existent_dir = Path("/non/existent/directory")
    assert not non_existent_dir.exists()

    # Test invalid file operations
    with pytest.raises(FileNotFoundError), open(non_existent_file) as f:
        f.read()

    # Test invalid YAML
    import yaml

    with pytest.raises(yaml.YAMLError):
        yaml.safe_load("invalid: yaml: content: [")

    # Test invalid JSON
    import json

    with pytest.raises(json.JSONDecodeError):
        json.loads("invalid json content")


def test_memory_management():
    """Test memory management utilities."""
    import gc

    # Test garbage collection
    initial_objects = len(gc.get_objects())

    # Create some objects
    large_list = list(range(10000))
    large_dict = {i: i**2 for i in range(1000)}

    # Delete references
    del large_list
    del large_dict

    # Force garbage collection
    gc.collect()

    # Check that objects were cleaned up
    final_objects = len(gc.get_objects())
    # Note: This is a rough check, exact numbers may vary
    assert final_objects <= initial_objects + 100, "Memory should be cleaned up"


def test_logging_utilities():
    """Test logging utility functions."""
    import logging

    # Test logger creation
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)

    # Test log message formatting
    test_message = "Test log message"

    # Create a string handler to capture log output
    import io

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Log a message
    logger.info(test_message)

    # Check that the message was logged
    log_output = log_capture.getvalue()
    assert test_message in log_output
    assert "INFO" in log_output

    # Clean up
    logger.removeHandler(handler)
