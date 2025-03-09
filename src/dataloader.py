from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from src.clients.mlflow_client import MLFlowClient
from src.datasets import ImageDataset

DATASETS = {"cifar10": ImageDataset}


class Dataloader(DataLoader):
    def __init__(self, cfg: DictConfig, stage: str, mlflow_client: MLFlowClient) -> None:
        logger.info(f"Initiating Dataloader for stage: {stage}")

        self.cfg = cfg
        self.stage = stage
        self.mlflow_client = mlflow_client
        self.dataset = self.set_dataset()

        super().__init__(
            self.dataset,
            batch_size=(self.cfg.training.batch_size if self.stage == "train" else self.cfg.training.batch_size),
            shuffle=True if self.stage == "train" else False,
            num_workers=self.cfg.training.n_workers,
        )

    def set_dataset(self) -> Dataset:
        try:
            logger.info(f"Using {self.cfg.dataset.name} dataset")
            return DATASETS[self.cfg.dataset.name](self.cfg.dataset, self.stage, self.mlflow_client)
        except KeyError:
            logger.error(f"Unknown dataset: {self.cfg.dataset.name}")
            raise KeyError(f"Unknown dataset: {self.cfg.dataset.name}")
