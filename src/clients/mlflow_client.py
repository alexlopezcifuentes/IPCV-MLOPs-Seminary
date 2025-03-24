import os

import matplotlib.pyplot as plt
import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig

from src.utils import AverageMeter

import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

class MLFlowClient:
    """
    Implements an MLFlow Client to manage MLFlow logging and model management.
    """

    def __init__(self, cfg):
        """
        Initialize the MLFlow Client in local mode.
        """
        self.cfg = cfg

        # Set MLFlow Experiment
        mlflow.set_experiment(self.cfg.experiment_name)

        self.start_run()

        # Fill MLFlow Tags
        self.set_tags({"Dataset": cfg.dataset.name})

        # Enable system metrics logging
        mlflow.system_metrics.enable_system_metrics_logging()

    def start_run(self):
        """
        Start a new MLFlow run.
        """
        self.run = mlflow.start_run()

    def end_run(self):
        """
        End the current MLFlow run.
        """
        mlflow.end_run()

    def set_tags(self, tags: dict):
        """
        Set the tags for the current run.
        """
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    def log_param(self, key: str, value: str):
        """
        Log a parameter to MLFlow.
        """
        mlflow.log_param(key, value)

    def log_config_mlflow(self, config: DictConfig = None, parent_key: str = "") -> None:
        """
        Recursive function to log a Hydra configuration structure to MLFlow as parameters.

        In addition to logging the parameters, it also logs the config folder as artifact.

        :param config: Hydra configuration to log.
        :param parent_key: Parent key used for recursive calls.
        """
        # Iterate over the key-value pairs in the config dictionary
        for key, value in config.items():
            # Create the full key path by appending the current key to the parent key
            full_key = f"{parent_key}.{key}" if parent_key else key

            # If the value is a dictionary, recursively call log_config_mlflow with the nested config
            if isinstance(value, DictConfig):
                self.log_config_mlflow(value, full_key)
            else:
                # Log the key-value pair using mlflow
                mlflow.log_param(full_key, value)

        # Log the config folder as artifact
        self.log_artifacts(self.cfg.config_path, "config", verbose=False)

    @staticmethod
    def log_metric_mlflow(metric_name: str, value: int | float, step: int, decimals: int = 2) -> None:
        """
        Logs a single metric to MLFlow.

        :param metric_name: Metric name.
        :param value: Metric value.
        :param step: Step number.
        """
        # Check if the value is a tensor
        if isinstance(value, torch.Tensor):
            value = value.item()
        # Limit the number of decimals to 3
        value = round(value, decimals)
        # Log metric
        mlflow.log_metric(metric_name, value, step=step)

    def log_metrics_mlflow(self, metrics: dict, epoch: int, decimals: int = 2) -> None:
        """
        Logs a set of metrics to MLFlow.

        :param metrics: Metrics to log to MLFlow in the form of a dictionary.
        :param epoch: Epoch number.
        :param decimals: Number of decimals to round the metrics.
        """

        # Sort metrics by name
        metrics = dict(sorted(metrics.items()))

        for metric_name, sets in metrics.items():
            if isinstance(sets, dict):
                for set_name, value in sets.items():
                    if isinstance(value, AverageMeter):
                        value = value.avg
                    self.log_metric_mlflow(f"{metric_name}_{set_name}", value, step=epoch, decimals=decimals)
            else:
                if isinstance(sets, AverageMeter):
                    sets = sets.avg
                self.log_metric_mlflow(metric_name, sets, step=epoch, decimals=decimals)

    @staticmethod
    def log_artifacts(local_path: str, artifact_path: str = None, verbose: bool = True) -> None:
        """
        Logs a set of artifacts to MLFlow.

        :param local_path: Path of the artifacts to log.
        :param artifact_path: MLFlow artifact path.
        :param verbose: Boolean to control verbosity.
        """
        if verbose:
            logger.info(f"Logging artifacts from {local_path} to {artifact_path}")
        mlflow.log_artifacts(local_path, artifact_path)

    def log_figure_mlflow(self, figure: plt.figure, figure_name: str, figure_path: str = None) -> None:
        """
        Logs a figure to MLFlow.

        :param figure: Figure to log.
        :param figure_name: Figure name.
        :param figure_path: Figure path.
        """
        if figure_path:
            figure_name = os.path.join(figure_path, figure_name)
        mlflow.log_figure(figure, figure_name)

    @staticmethod
    def log_model_mlflow(model: torch.nn.Module, model_name: str) -> None:
        """
        Logs a PyTorch model to MLFlow.

        :param model: PyTorch model to lo. It must be a subclass of nn.Module.
        :param model_name: Model name.
        """
        mlflow.pytorch.log_model(model, model_name, pip_requirements="requirements.txt")
