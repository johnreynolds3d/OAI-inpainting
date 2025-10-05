"""
OAI Inpainting Package

X-ray image inpainting using multiple state-of-the-art deep learning approaches.
"""

__version__ = "0.1.0"
__author__ = "OAI Inpainting Team"
__email__ = "team@example.com"

from .config import Config, HardwareConfig, LossConfig, PathsConfig
from .data import OAIDataset
from .paths import get_data_dir, get_output_dir, get_project_root

__all__ = [
    "Config",
    "HardwareConfig",
    "LossConfig",
    "OAIDataset",
    "PathsConfig",
    "get_data_dir",
    "get_output_dir",
    "get_project_root",
]
