"""
Configuration utilities for the OAI inpainting project.
Platform-agnostic configuration management.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Union, Optional
from dataclasses import dataclass, asdict

from .paths import get_config_dir, get_project_root


@dataclass
class BaseConfig:
    """Base configuration class."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "BaseConfig":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "BaseConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class DataConfig(BaseConfig):
    """Data configuration."""

    train_images: str = "./data/oai/train/img"
    train_masks: str = "./data/oai/train/mask"
    val_images: str = "./data/oai/valid/img"
    val_masks: str = "./data/oai/valid/mask"
    test_images: str = "./data/oai/test/img"
    test_masks: str = "./data/oai/test/mask"

    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        project_root = get_project_root()
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str) and field_value.startswith("./"):
                setattr(self, field_name, str(project_root / field_value[2:]))


@dataclass
class ModelConfig(BaseConfig):
    """Model configuration."""

    name: str = "aotgan"
    block_num: int = 8
    rates: str = "1+2+4+8"
    gan_type: str = "smgan"


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""

    batch_size: int = 8
    image_size: int = 512
    mask_type: str = "pconv"
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    num_workers: int = 4
    seed: int = 2021


@dataclass
class HardwareConfig(BaseConfig):
    """Hardware configuration."""

    distributed: bool = False
    local_rank: int = 0
    tensorboard: bool = True
    device: str = "cuda"


@dataclass
class PathsConfig(BaseConfig):
    """Paths configuration."""

    save_dir: str = "./results/logs"
    outputs: str = "./output"
    resume: Optional[str] = None

    def __post_init__(self):
        """Convert relative paths to absolute paths."""
        project_root = get_project_root()
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str) and field_value.startswith("./"):
                setattr(self, field_name, str(project_root / field_value[2:]))


@dataclass
class LossConfig(BaseConfig):
    """Loss configuration."""

    l1: float = 1.0
    style: float = 1.0
    perceptual: float = 1.0


@dataclass
class AOTGANConfig(BaseConfig):
    """AOT-GAN configuration."""

    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    hardware: HardwareConfig = None
    paths: PathsConfig = None
    loss: LossConfig = None

    def __post_init__(self):
        """Initialize default configs if None."""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.hardware is None:
            self.hardware = HardwareConfig()
        if self.paths is None:
            self.paths = PathsConfig()
        if self.loss is None:
            self.loss = LossConfig()


def load_config(
    config_path: Union[str, Path], config_type: str = "aot-gan"
) -> BaseConfig:
    """Load configuration from file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load configuration data
    with open(config_path, "r") as f:
        if (
            config_path.suffix.lower() == ".yaml"
            or config_path.suffix.lower() == ".yml"
        ):
            data = yaml.safe_load(f)
        elif config_path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )

    # Create appropriate config object
    if config_type == "aot-gan":
        return AOTGANConfig(**data)
    else:
        raise ValueError(f"Unsupported configuration type: {config_type}")


def save_config(config: BaseConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if config_path.suffix.lower() == ".yaml" or config_path.suffix.lower() == ".yml":
        config.to_yaml(config_path)
    elif config_path.suffix.lower() == ".json":
        config.to_json(config_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def get_default_config_path(model_name: str) -> Path:
    """Get default configuration file path for model."""
    config_dir = get_config_dir()
    return config_dir / model_name / "oai_config.yml"


def create_default_configs() -> None:
    """Create default configuration files for all models."""
    config_dir = get_config_dir()

    # Create AOT-GAN config
    aot_gan_config = AOTGANConfig()
    aot_gan_path = config_dir / "aot-gan" / "oai_config.yml"
    save_config(aot_gan_config, aot_gan_path)

    print(f"✅ Created default config: {aot_gan_path}")


def validate_config(config: BaseConfig) -> bool:
    """Validate configuration."""
    try:
        # Check if all required fields are present
        if hasattr(config, "data") and config.data:
            data_config = config.data
            required_data_fields = [
                "train_images",
                "train_masks",
                "val_images",
                "val_masks",
                "test_images",
                "test_masks",
            ]
            for field in required_data_fields:
                if not hasattr(data_config, field) or not getattr(data_config, field):
                    print(f"❌ Missing required data field: {field}")
                    return False

        # Check if paths exist
        if hasattr(config, "paths") and config.paths:
            paths_config = config.paths
            if hasattr(paths_config, "save_dir"):
                save_dir = Path(paths_config.save_dir)
                if not save_dir.parent.exists():
                    print(f"❌ Save directory parent does not exist: {save_dir.parent}")
                    return False

        print("✅ Configuration validation passed")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False
