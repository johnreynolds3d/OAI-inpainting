"""
Experiment tracking utilities for OAI Inpainting project.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import wandb
from mlflow import MlflowClient, log_artifact, log_metric, log_params


class ExperimentTracker:
    """Unified experiment tracking interface."""

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        use_wandb: bool = True,
        use_mlflow: bool = True,
    ):
        """
        Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow tracking URI
            use_wandb: Whether to use Weights & Biases
            use_mlflow: Whether to use MLflow
        """
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

        # Initialize MLflow
        if self.use_mlflow:
            if tracking_uri:
                import mlflow

                mlflow.set_tracking_uri(tracking_uri)

            self.mlflow_client = MlflowClient()
            try:
                experiment_id = self.mlflow_client.create_experiment(experiment_name)
            except Exception:
                experiment = self.mlflow_client.get_experiment_by_name(experiment_name)
                experiment_id = experiment.experiment_id

            self.experiment_id = experiment_id

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                project="oai-inpainting",
                name=experiment_name,
                config={"experiment_name": experiment_name},
            )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.use_mlflow:
            log_params(params)

        if self.use_wandb:
            wandb.config.update(params)

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        if self.use_mlflow:
            for key, value in metrics.items():
                log_metric(key, value, step=step)

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: Path) -> None:
        """Log an artifact."""
        if self.use_mlflow:
            log_artifact(str(artifact_path))

        if self.use_wandb:
            wandb.save(str(artifact_path))

    def log_model_info(self, model_info: Dict[str, Any]) -> None:
        """Log model information."""
        model_info["timestamp"] = datetime.now().isoformat()

        if self.use_mlflow:
            log_params({"model_info": json.dumps(model_info)})

        if self.use_wandb:
            wandb.config.update({"model_info": model_info})

    def finish(self) -> None:
        """Finish the experiment."""
        if self.use_wandb:
            wandb.finish()


class ModelRegistry:
    """Model registry for tracking trained models."""

    def __init__(self, registry_path: Path):
        """
        Initialize model registry.

        Args:
            registry_path: Path to store model registry
        """
        self.registry_path = registry_path
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / "model_registry.json"

        if self.registry_file.exists():
            with self.registry_file.open() as f:
                self.registry = json.load(f)
        else:
            self.registry = {}

    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> str:
        """
        Register a new model.

        Args:
            model_name: Name of the model
            model_path: Path to the model file
            metrics: Model performance metrics
            metadata: Additional model metadata

        Returns:
            Model ID
        """
        model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.registry[model_id] = {
            "model_name": model_name,
            "model_path": str(model_path),
            "metrics": metrics,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }

        self._save_registry()
        return model_id

    def get_best_model(
        self, model_name: str, metric: str = "accuracy"
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best model by metric.

        Args:
            model_name: Name of the model
            metric: Metric to optimize

        Returns:
            Best model information or None
        """
        models = [
            model
            for model in self.registry.values()
            if model["model_name"] == model_name
        ]

        if not models:
            return None

        return max(models, key=lambda x: x["metrics"].get(metric, 0))

    def list_models(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        List all models or models of a specific type.

        Args:
            model_name: Optional model name filter

        Returns:
            Dictionary of models
        """
        if model_name:
            return {
                k: v for k, v in self.registry.items() if v["model_name"] == model_name
            }

        return self.registry

    def _save_registry(self) -> None:
        """Save registry to file."""
        with self.registry_file.open("w") as f:
            json.dump(self.registry, f, indent=2)
