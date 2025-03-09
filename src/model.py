import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig


class AlexNet(nn.Module):
    """
    AlexNet class
    """

    def __init__(self, n_classes) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc = nn.Linear(in_features=576, out_features=self.n_classes)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc(x))
        return x


# Add ResNet class
class ResNet(nn.Module):
    """
    ResNet class that agglutinates all the ResNet models from torchvision.
    """

    def __init__(self, model_name, n_classes) -> None:
        """
        Initialize the ResNet class.

        :param model_name: Name of the ResNet model to be used
        :param n_classes: Number of classes to be predicted
        """
        super().__init__()
        self.n_classes = n_classes
        self.model_name = model_name

        # Get possible pretrained weights
        weight_enum = list(torch.hub.load("pytorch/vision", "get_model_weights", name=self.model_name))
        # Get the ResNet model from torchvision
        model = torch.hub.load("pytorch/vision", self.model_name, weights=weight_enum[-1])

        # Take only the layers before the last one
        self.features = nn.Sequential(*list(model.children())[:-1])
        # Define the last layer
        self.fc = nn.Linear(model.fc.in_features, self.n_classes)
        # Add a softmax layer to output probabilities
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Forward pass of the ResNet class.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


MODELS = {"alexnet": AlexNet, "resnet": ResNet}


class Model(nn.Module):
    """
    General Model class.
    """

    def __init__(self, cfg: DictConfig, **kwargs) -> None:
        super().__init__()
        logger.info("Instantiating Model")

        self.cfg = cfg
        self.kwargs = kwargs
        if cfg.training.device == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"{cfg['training']['device']}:{cfg['training']['gpu_id']}" if torch.cuda.is_available() else "cpu"
            )
        logger.info(f"Device to be used: {self.device}")

        # Save classes as a list in model
        if "classes" in kwargs.keys():
            self.classes = kwargs["classes"]
            cfg.model.n_classes = len(self.classes)

        self.n_classes = cfg.model.n_classes
        # Save required transformations
        if "transform" in kwargs.keys():
            self.transform = kwargs["transform"]

        # Set the model
        self.model = self.set_model()
        self.model = self.model.to(self.device)


    def set_model(self) -> nn.Module:
        try:
            logger.info(f"Initiating Model: {self.cfg.model.name}")
            if "regression" in self.cfg.model.name:
                self.kwargs["n_regression_layers"] = self.cfg.model["n_regression_layers"]
                return MODELS["regression_resnet"](self.cfg.model.name, self.n_classes, **self.kwargs)
            elif "resnet" in self.cfg.model.name:
                return MODELS["resnet"](self.cfg.model.name, self.n_classes)
            else:
                return MODELS[self.cfg.model.name](self.n_classes)
        except KeyError:
            logger.error(f"Unknown model: {self.cfg.model.name}")
            raise KeyError(f"Unknown model: {self.cfg.model.name}")

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def save_model(self, metrics, mlflow_client, max_metric, mode="accuracy"):
        self.model.to('cpu')
        if mode == "accuracy":
            if metrics["accuracy"]["val"].avg > max_metric:
                logger.info(f"Accuracy {metrics['accuracy']['val'].avg:.3f}% > {max_metric:.3f}% saving model...")
                mlflow_client.log_model_mlflow(self, "model")
                max_metric = metrics["accuracy"]["val"].avg
                logger.info("Model saved")
        elif mode == "error":
            if metrics["error"]["val"].avg < max_metric:
                logger.info(f"Error {metrics['error']['val'].avg:.3f} < {max_metric:.3f} saving model...")
                pipe_logger.mlflow_client.log_model_mlflow(self, "model")
                max_metric = metrics["error"]["val"].avg
                logger.info("Model saved")
        elif mode == "loss":
            if metrics["loss"]["val"].avg < max_metric:
                logger.info(f"Loss {metrics['loss']['val'].avg:.3f} < {max_metric:.3f} saving model...")
                pipe_logger.mlflow_client.log_model_mlflow(self, "model")
                max_metric = metrics["loss"]["val"].avg
                logger.info("Model saved")
        else:
            logger.error(f"Unknown mode: {mode} in save_model")
            raise ValueError(f"Unknown mode: {mode} in save_model")
        self.model.to(self.device)
        return max_metric
