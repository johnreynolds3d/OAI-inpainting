"""
Platform-agnostic path utilities for the OAI inpainting project.
Ensures reproducibility across different operating systems.
"""

from pathlib import Path
from typing import List, Optional, Union

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def get_data_dir() -> Path:
    """Get the data directory."""
    return PROJECT_ROOT / "data"


def get_config_dir() -> Path:
    """Get the config directory."""
    return PROJECT_ROOT / "configs"


def get_results_dir() -> Path:
    """Get the results directory."""
    return PROJECT_ROOT / "results"


def get_output_dir() -> Path:
    """Get the output directory."""
    return PROJECT_ROOT / "output"


def get_models_dir() -> Path:
    """Get the models directory."""
    return PROJECT_ROOT / "models"


def get_scripts_dir() -> Path:
    """Get the scripts directory."""
    return PROJECT_ROOT / "scripts"


def get_utils_dir() -> Path:
    """Get the utils directory."""
    return PROJECT_ROOT / "utils"


def get_docs_dir() -> Path:
    """Get the docs directory."""
    return PROJECT_ROOT / "docs"


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_relative_path(
    path: Union[str, Path], base: Optional[Union[str, Path]] = None
) -> Path:
    """Get relative path from base directory."""
    if base is None:
        base = PROJECT_ROOT
    return Path(path).relative_to(Path(base))


def get_absolute_path(
    path: Union[str, Path], base: Optional[Union[str, Path]] = None
) -> Path:
    """Get absolute path from base directory."""
    if base is None:
        base = PROJECT_ROOT
    return Path(base) / path


def find_files(
    pattern: str, directory: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Find files matching pattern in directory."""
    if directory is None:
        directory = PROJECT_ROOT
    return list(Path(directory).glob(pattern))


def find_directories(
    pattern: str, directory: Optional[Union[str, Path]] = None
) -> List[Path]:
    """Find directories matching pattern in directory."""
    if directory is None:
        directory = PROJECT_ROOT
    return [p for p in Path(directory).glob(pattern) if p.is_dir()]


# OAI-specific paths
def get_oai_data_dir() -> Path:
    """Get OAI data directory."""
    return get_data_dir() / "oai"


def get_oai_train_dir() -> Path:
    """Get OAI training data directory."""
    return get_oai_data_dir() / "train"


def get_oai_val_dir() -> Path:
    """Get OAI validation data directory."""
    return get_oai_data_dir() / "valid"


def get_oai_test_dir() -> Path:
    """Get OAI test data directory."""
    return get_oai_data_dir() / "test"


def get_oai_subset_dir(subset: str) -> Path:
    """Get OAI subset directory (e.g., subset_4)."""
    return get_oai_test_dir() / "img" / subset


def get_pretrained_dir() -> Path:
    """Get pretrained models directory."""
    return get_data_dir() / "pretrained"


# Model-specific paths
def get_model_dir(model_name: str) -> Path:
    """Get model-specific directory."""
    return PROJECT_ROOT / model_name


def get_model_config_dir(model_name: str) -> Path:
    """Get model-specific config directory."""
    return get_config_dir() / model_name


def get_model_output_dir(model_name: str) -> Path:
    """Get model-specific output directory."""
    return get_output_dir() / model_name.upper() / "OAI"


def get_model_results_dir(model_name: str) -> Path:
    """Get model-specific results directory."""
    return get_results_dir() / "logs" / model_name


# Platform-agnostic path operations
def normalize_path(path: Union[str, Path]) -> Path:
    """Normalize path for current platform."""
    return Path(path).resolve()


def join_paths(*paths: Union[str, Path]) -> Path:
    """Join paths in a platform-agnostic way."""
    return Path(*paths)


def get_filename(path: Union[str, Path]) -> str:
    """Get filename from path."""
    return Path(path).name


def get_stem(path: Union[str, Path]) -> str:
    """Get filename without extension."""
    return Path(path).stem


def get_suffix(path: Union[str, Path]) -> str:
    """Get file extension."""
    return Path(path).suffix


def change_extension(path: Union[str, Path], new_ext: str) -> Path:
    """Change file extension."""
    return Path(path).with_suffix(new_ext)
