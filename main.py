import os

import hydra

from src.clients.mlflow_client import MLFlowClient
from src.dataloader import Dataloader
from src.loss import Loss
from src.model import Model
from src.optimizer import Optimizer
from src.runners import ClassificationRunner


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Initialize MLFlow Client
    mlflow_client = MLFlowClient(cfg)

    # Update dataset location
    # Create path key if it doesn't exist
    cfg.dataset.path = os.path.join("datasets", cfg.dataset.name)

    # Log config to MLFlow
    mlflow_client.log_config_mlflow(cfg)

    # Define Dataset and Dataloader
    train_loader = Dataloader(cfg, stage="train", mlflow_client=mlflow_client)
    val_loader = Dataloader(cfg, stage="val", mlflow_client=mlflow_client)

    # Define Model
    model = Model(cfg, classes=train_loader.dataset.classes, transform=train_loader.dataset.transform)

    # Define Loss
    loss_fn = Loss(cfg.optimizer.loss, device=model.device, ignore_index=cfg.dataset.unknown_index)

    # Define Optimizer & Scheduler
    optimizer = Optimizer(cfg.optimizer, model)

    # Define Runner
    runner = ClassificationRunner(cfg, train_loader, val_loader, val_loader, model, loss_fn, optimizer, mlflow_client)

    # Train
    runner.complete_train()

    mlflow_client.end_run()

    return


if __name__ == "__main__":
    main()
