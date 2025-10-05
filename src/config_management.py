"""
Modern configuration management using Hydra and Pydantic.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration."""

    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (aot-gan, ict, repaint)")
    pretrained_path: Optional[str] = Field(None, description="Path to pretrained model")
    input_size: int = Field(256, description="Input image size")
    batch_size: int = Field(1, description="Batch size")

    @validator("type")
    def validate_model_type(cls, v):
        allowed_types = ["aot-gan", "ict", "repaint"]
        if v not in allowed_types:
            raise ValueError(f"Model type must be one of {allowed_types}")
        return v


class DataConfig(BaseModel):
    """Data configuration."""

    img_dir: str = Field(..., description="Image directory path")
    mask_dir: str = Field(..., description="Mask directory path")
    output_dir: str = Field(..., description="Output directory path")
    subset: str = Field("subset_4", description="Data subset to use")
    num_samples: int = Field(4, description="Number of samples")

    @validator("img_dir", "mask_dir", "output_dir")
    def validate_paths(cls, v):
        if not v:
            raise ValueError("Path cannot be empty")
        return v


class TrainingConfig(BaseModel):
    """Training configuration."""

    epochs: int = Field(100, description="Number of training epochs")
    learning_rate: float = Field(0.001, description="Learning rate")
    batch_size: int = Field(8, description="Training batch size")
    save_frequency: int = Field(10, description="Model save frequency")
    validation_frequency: int = Field(5, description="Validation frequency")

    @validator("learning_rate")
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v


class InferenceConfig(BaseModel):
    """Inference configuration."""

    sample_num: int = Field(1, description="Number of samples to generate")
    jump_length: int = Field(10, description="Jump length for RePaint")
    jump_n_sample: int = Field(10, description="Jump n sample for RePaint")
    n_steps: int = Field(1000, description="Number of diffusion steps")
    use_fp16: bool = Field(False, description="Use half precision")

    @validator("sample_num", "jump_length", "jump_n_sample", "n_steps")
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class ExperimentConfig(BaseModel):
    """Experiment configuration."""

    name: str = Field(..., description="Experiment name")
    description: str = Field("", description="Experiment description")
    tags: List[str] = Field(default_factory=list, description="Experiment tags")
    use_wandb: bool = Field(True, description="Use Weights & Biases")
    use_mlflow: bool = Field(True, description="Use MLflow")
    tracking_uri: Optional[str] = Field(None, description="MLflow tracking URI")


class Config(BaseModel):
    """Main configuration class."""

    model: ModelConfig = Field(..., description="Model configuration")
    data: DataConfig = Field(..., description="Data configuration")
    training: Optional[TrainingConfig] = Field(
        None, description="Training configuration"
    )
    inference: Optional[InferenceConfig] = Field(
        None, description="Inference configuration"
    )
    experiment: ExperimentConfig = Field(..., description="Experiment configuration")

    # Global settings
    device: str = Field("cuda", description="Device to use (cuda/cpu)")
    seed: int = Field(42, description="Random seed")
    log_level: str = Field("INFO", description="Logging level")

    @validator("device")
    def validate_device(cls, v):
        allowed_devices = ["cuda", "cpu"]
        if v not in allowed_devices:
            raise ValueError(f"Device must be one of {allowed_devices}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of {allowed_levels}")
        return v.upper()


class ConfigManager:
    """Configuration manager using Hydra."""

    def __init__(self, config_dir: Path = Path("configs")):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir.resolve()
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Hydra
        if not GlobalHydra().is_initialized():
            initialize_config_dir(config_dir=str(self.config_dir))

    def load_config(
        self,
        config_name: str = "default",
        overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> Config:
        """
        Load configuration from file.

        Args:
            config_name: Name of the configuration file
            overrides: List of configuration overrides
            **kwargs: Additional configuration overrides

        Returns:
            Configuration object
        """
        try:
            # Compose configuration with Hydra
            cfg = compose(config_name=config_name, overrides=overrides or [])

            # Convert to dictionary
            config_dict = self._hydra_to_dict(cfg)

            # Apply additional overrides
            if kwargs:
                config_dict.update(kwargs)

            # Validate and create config object
            return Config(**config_dict)

        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")

    def save_config(self, config: Config, config_name: str) -> Path:
        """
        Save configuration to file.

        Args:
            config: Configuration object
            config_name: Name for the configuration file

        Returns:
            Path to saved configuration file
        """
        config_path = self.config_dir / f"{config_name}.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config.dict(), f, default_flow_style=False, indent=2)

        return config_path

    def list_configs(self) -> List[str]:
        """List available configuration files."""
        config_files = list(self.config_dir.glob("*.yaml")) + list(
            self.config_dir.glob("*.yml")
        )
        return [f.stem for f in config_files]

    def _hydra_to_dict(self, cfg) -> Dict[str, Any]:
        """Convert Hydra config to dictionary."""

        def _convert(obj):
            if hasattr(obj, "_content"):
                return {k: _convert(v) for k, v in obj._content.items()}
            elif isinstance(obj, list):
                return [_convert(item) for item in obj]
            else:
                return obj

        return _convert(cfg)

    def create_default_configs(self) -> None:
        """Create default configuration files."""
        # AOT-GAN config
        aotgan_config = Config(
            model=ModelConfig(
                name="aot-gan",
                type="aot-gan",
                pretrained_path="data/pretrained/aot-gan/places2/G0000000.pt",
            ),
            data=DataConfig(
                img_dir="data/oai/test/img/subset_4",
                mask_dir="data/oai/test/mask/subset_4",
                output_dir="output/AOT-GAN/Places2/subset_4",
            ),
            experiment=ExperimentConfig(name="aot-gan-test"),
        )
        self.save_config(aotgan_config, "aot-gan")

        # ICT config
        ict_config = Config(
            model=ModelConfig(
                name="ict",
                type="ict",
                pretrained_path="data/pretrained/ict/Upsample/Places2_Nature",
            ),
            data=DataConfig(
                img_dir="data/oai/test/img/subset_4",
                mask_dir="data/oai/test/mask/subset_4",
                output_dir="output/ICT/Places2_Nature/subset_4",
            ),
            experiment=ExperimentConfig(name="ict-test"),
        )
        self.save_config(ict_config, "ict")

        # RePaint config
        repaint_config = Config(
            model=ModelConfig(
                name="repaint",
                type="repaint",
                pretrained_path="data/pretrained/repaint/places256_300000.pt",
            ),
            data=DataConfig(
                img_dir="data/oai/test/img/subset_4",
                mask_dir="data/oai/test/mask/subset_4",
                output_dir="output/RePaint/Places2/subset_4",
            ),
            inference=InferenceConfig(
                sample_num=1, jump_length=3, jump_n_sample=3, n_steps=25, use_fp16=False
            ),
            experiment=ExperimentConfig(name="repaint-test"),
        )
        self.save_config(repaint_config, "repaint")


def load_config_from_file(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from a specific file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration object
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return Config(**config_dict)


def create_config_from_dict(config_dict: Dict[str, Any]) -> Config:
    """
    Create configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Configuration object
    """
    return Config(**config_dict)
