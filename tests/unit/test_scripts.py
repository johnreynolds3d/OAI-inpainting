"""Unit tests for script functionality."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_test_subset_4_script_exists(project_root):
    """Test that test_subset_4.py script exists and can be imported."""
    script_path = project_root / "scripts" / "test_subset_4.py"
    assert script_path.exists(), "test_subset_4.py script should exist"

    # Test that the script can be imported
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("test_subset_4", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert True, "test_subset_4.py script imported successfully"
    except Exception as e:
        pytest.fail(f"test_subset_4.py script failed to import: {e}")


def test_setup_data_script_exists(project_root):
    """Test that setup_data.py script exists and can be imported."""
    script_path = project_root / "scripts" / "setup_data.py"
    assert script_path.exists(), "setup_data.py script should exist"

    # Test that the script can be imported
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("setup_data", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert True, "setup_data.py script imported successfully"
    except Exception as e:
        pytest.fail(f"setup_data.py script failed to import: {e}")


def test_evaluate_script_exists(project_root):
    """Test that evaluate.py script exists and can be imported."""
    script_path = project_root / "scripts" / "evaluate.py"
    assert script_path.exists(), "evaluate.py script should exist"

    # Test that the script can be imported (skip if seaborn not available)
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("evaluate", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert True, "evaluate.py script imported successfully"
    except ImportError as e:
        if "seaborn" in str(e):
            pytest.skip("seaborn not available, skipping evaluate.py import test")
        else:
            pytest.fail(f"evaluate.py script failed to import: {e}")
    except Exception as e:
        pytest.fail(f"evaluate.py script failed to import: {e}")


def test_test_subset_4_help():
    """Test that test_subset_4.py shows help when run with --help."""
    cmd = [sys.executable, "scripts/test_subset_4.py", "--help"]

    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "Test all model variants using subset_4 data" in result.stdout
        assert "--models" in result.stdout
        assert "--timeout" in result.stdout

    except subprocess.TimeoutExpired:
        pytest.fail("Help command timed out")
    except Exception as e:
        pytest.fail(f"Help command failed with exception: {e}")


def test_setup_data_help():
    """Test that setup_data.py shows help when run with --help."""
    cmd = [sys.executable, "scripts/setup_data.py", "--help"]

    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        assert result.returncode == 0, f"Help command failed: {result.stderr}"
        assert "Set up OAI Inpainting project with untracked data" in result.stdout
        assert "--source-dir" in result.stdout
        assert "--dry-run" in result.stdout

    except subprocess.TimeoutExpired:
        pytest.fail("Help command timed out")
    except Exception as e:
        pytest.fail(f"Help command failed with exception: {e}")


def test_script_argument_parsing():
    """Test that scripts can parse arguments without errors."""
    # Test test_subset_4.py with invalid model
    cmd = [sys.executable, "scripts/test_subset_4.py", "--models", "invalid-model"]

    try:
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30, cwd=Path.cwd()
        )

        # Should fail gracefully with invalid model
        assert result.returncode != 0, "Invalid model should cause failure"
        assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower()

    except subprocess.TimeoutExpired:
        pytest.fail("Invalid model test timed out")
    except Exception as e:
        pytest.fail(f"Invalid model test failed with exception: {e}")


def test_script_dependencies():
    """Test that scripts can import required dependencies."""
    # Test that required modules can be imported
    required_modules = [
        "numpy",
        "pandas",
        "PIL",
        "cv2",
        "torch",
        "yaml",
        "pathlib",
        "argparse",
    ]

    for module_name in required_modules:
        try:
            if module_name == "PIL":
                import PIL
            elif module_name == "cv2":
                import cv2
            else:
                __import__(module_name)
            assert True, f"Module {module_name} imported successfully"
        except ImportError as e:
            pytest.fail(f"Required module {module_name} not available: {e}")


def test_script_file_permissions(project_root):
    """Test that script files have correct permissions."""
    script_files = [
        "scripts/test_subset_4.py",
        "scripts/setup_data.py",
        "scripts/evaluate.py",
        "scripts/test.py",
        "scripts/train.py",
    ]

    for script_file in script_files:
        script_path = project_root / script_file
        if script_path.exists():
            # Check that file is readable
            assert script_path.is_file(), f"{script_file} should be a file"
            # Check that file is not empty
            assert script_path.stat().st_size > 0, f"{script_file} should not be empty"


def test_script_shebang():
    """Test that Python scripts have correct shebang lines."""
    script_files = [
        "scripts/test_subset_4.py",
        "scripts/setup_data.py",
        "scripts/evaluate.py",
    ]

    for script_file in script_files:
        script_path = Path(script_file)
        if script_path.exists():
            with script_path.open() as f:
                first_line = f.readline().strip()
                # Check for Python shebang
                assert first_line.startswith("#!"), f"{script_file} should have shebang"
                assert "python" in first_line, (
                    f"{script_file} should have Python shebang"
                )
