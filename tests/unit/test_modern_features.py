"""
Unit tests for modern ML features.
"""

from unittest.mock import patch

from src.data_versioning import DataLineage, DataVersioning
from src.experiment_tracking import ExperimentTracker, ModelRegistry
from src.logging_config import get_logger, setup_logging


class TestExperimentTracker:
    """Test experiment tracking functionality."""

    def test_experiment_tracker_init(self):
        """Test experiment tracker initialization."""
        with patch("src.experiment_tracking.wandb"), patch(
            "src.experiment_tracking.MlflowClient"
        ):
            tracker = ExperimentTracker(
                "test_experiment", use_wandb=False, use_mlflow=False
            )
            assert tracker.experiment_name == "test_experiment"

    def test_log_params(self):
        """Test parameter logging."""
        with patch("src.experiment_tracking.wandb") as mock_wandb, patch(
            "src.experiment_tracking.log_params"
        ) as mock_log_params:
            tracker = ExperimentTracker(
                "test_experiment", use_wandb=True, use_mlflow=True
            )
            params = {"lr": 0.001, "batch_size": 32}

            tracker.log_params(params)

            mock_log_params.assert_called_once_with(params)
            mock_wandb.config.update.assert_called_once_with(params)

    def test_log_metrics(self):
        """Test metrics logging."""
        with patch("src.experiment_tracking.wandb") as mock_wandb, patch(
            "src.experiment_tracking.log_metric"
        ) as mock_log_metric:
            tracker = ExperimentTracker(
                "test_experiment", use_wandb=True, use_mlflow=True
            )
            metrics = {"accuracy": 0.95, "loss": 0.1}

            tracker.log_metrics(metrics, step=100)

            assert mock_log_metric.call_count == 2
            mock_wandb.log.assert_called_once_with(metrics, step=100)


class TestModelRegistry:
    """Test model registry functionality."""

    def test_model_registry_init(self, temp_dir):
        """Test model registry initialization."""
        registry = ModelRegistry(temp_dir)
        assert registry.registry_path == temp_dir
        assert registry.registry == {}

    def test_register_model(self, temp_dir):
        """Test model registration."""
        registry = ModelRegistry(temp_dir)

        model_path = temp_dir / "test_model.pt"
        model_path.touch()

        metrics = {"accuracy": 0.95, "loss": 0.1}
        metadata = {"model_type": "test", "dataset": "test_data"}

        model_id = registry.register_model("test_model", model_path, metrics, metadata)

        assert model_id.startswith("test_model_")
        assert model_id in registry.registry
        assert registry.registry[model_id]["metrics"] == metrics
        assert registry.registry[model_id]["metadata"] == metadata

    def test_get_best_model(self, temp_dir):
        """Test getting best model by metric."""
        registry = ModelRegistry(temp_dir)

        # Register multiple models
        for i in range(3):
            model_path = temp_dir / f"model_{i}.pt"
            model_path.touch()

            metrics = {"accuracy": 0.9 + i * 0.02, "loss": 0.1 - i * 0.01}
            metadata = {"model_type": "test"}

            registry.register_model("test_model", model_path, metrics, metadata)

        best_model = registry.get_best_model("test_model", "accuracy")
        assert best_model is not None
        assert best_model["metrics"]["accuracy"] == 0.94  # Highest accuracy


class TestDataVersioning:
    """Test data versioning functionality."""

    def test_data_versioning_init(self, temp_dir):
        """Test data versioning initialization."""
        versioning = DataVersioning(temp_dir)
        assert versioning.version_dir == temp_dir
        assert versioning.versions == {}

    def test_create_version(self, temp_dir, sample_data_dir):
        """Test creating data version."""
        versioning = DataVersioning(temp_dir)

        version_id = versioning.create_version(
            sample_data_dir, "test_version", "Test data version", {"source": "test"}
        )

        assert version_id.startswith("test_version_")
        assert version_id in versioning.versions
        assert versioning.versions[version_id]["version_name"] == "test_version"
        assert versioning.versions[version_id]["description"] == "Test data version"

    def test_verify_version(self, temp_dir, sample_data_dir):
        """Test version verification."""
        versioning = DataVersioning(temp_dir)

        version_id = versioning.create_version(sample_data_dir, "test_version")

        # Should pass verification
        assert versioning.verify_version(version_id, sample_data_dir)

        # Modify data and verify it fails
        (sample_data_dir / "img" / "new_file.png").touch()
        assert not versioning.verify_version(version_id, sample_data_dir)


class TestDataLineage:
    """Test data lineage functionality."""

    def test_data_lineage_init(self, temp_dir):
        """Test data lineage initialization."""
        lineage = DataLineage(temp_dir)
        assert lineage.lineage_dir == temp_dir
        assert lineage.lineage == {}

    def test_add_transformation(self, temp_dir):
        """Test adding data transformation."""
        lineage = DataLineage(temp_dir)

        input_paths = [temp_dir / "input1.txt", temp_dir / "input2.txt"]
        output_path = temp_dir / "output.txt"

        for path in input_paths:
            path.touch()
        output_path.touch()

        transformation_id = lineage.add_transformation(
            input_paths,
            output_path,
            "merge",
            {"method": "concatenate"},
            {"author": "test"},
        )

        assert transformation_id.startswith("merge_")
        assert transformation_id in lineage.lineage
        assert lineage.lineage[transformation_id]["transformation_type"] == "merge"

    def test_get_lineage(self, temp_dir):
        """Test getting data lineage."""
        lineage = DataLineage(temp_dir)

        input_path = temp_dir / "input.txt"
        output_path = temp_dir / "output.txt"

        input_path.touch()
        output_path.touch()

        lineage.add_transformation([input_path], output_path, "transform", {})

        # Get lineage for output path
        output_lineage = lineage.get_lineage(output_path)
        assert len(output_lineage) == 1
        assert output_lineage[0]["output_path"] == str(output_path)

        # Get lineage for input path
        input_lineage = lineage.get_lineage(input_path)
        assert len(input_lineage) == 1
        assert str(input_path) in input_lineage[0]["input_paths"]


class TestLoggingConfig:
    """Test logging configuration."""

    def test_setup_logging(self, temp_dir):
        """Test logging setup."""
        log_file = temp_dir / "test.log"

        setup_logging(level="DEBUG", log_file=log_file, structured=False)

        logger = get_logger("test")
        logger.info("Test message")

        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_structured_logging(self, temp_dir):
        """Test structured logging."""
        log_file = temp_dir / "structured.log"

        setup_logging(level="INFO", log_file=log_file, structured=True)

        logger = get_logger("test")
        logger.info("Structured message", extra={"key": "value"})

        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Structured message" in log_content
        assert "key" in log_content
        assert "value" in log_content
