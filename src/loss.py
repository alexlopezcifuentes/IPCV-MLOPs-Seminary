import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig


def get_CrossEntropyLoss(cfg: DictConfig, **kwargs) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=kwargs["ignore_index"])


def get_MSELoss(cfg: DictConfig, **kwargs) -> nn.Module:
    return nn.MSELoss()


def get_MAE(cfg: DictConfig, **kwargs) -> nn.Module:
    return nn.L1Loss()


LOSS = {"crossentropy": get_CrossEntropyLoss, "mean_square_error": get_MSELoss, "mean_absolute_error": get_MAE}


class Loss(nn.Module):
    def __init__(self, cfg: DictConfig, **kwargs):
        logger.info("Instantiating Loss")
        super().__init__()
        self.cfg = cfg
        self.device = kwargs["device"]    

        # Decode ignore_index
        self.ignore_index = kwargs["ignore_index"]
        if self.ignore_index == 'None':
            logger.info("Setting ignore_index to -100 so that none of the classes are ignored")
            self.ignore_index = -100
        # check if ignore_index is and integer
        elif type(self.ignore_index) != int:
            raise ValueError(f"ignore_index must be an integer, got {self.ignore_index}")
        else:
            self.ignore_index = int(self.ignore_index)
            logger.info(f"Setting ignore_index to {self.ignore_index}")

        del kwargs["ignore_index"]

        self.loss = self.set_loss(**kwargs)

    def set_loss(self, **kwargs):
        try:
            logger.info(f"Initiating loss: {self.cfg.name}")
            return LOSS[self.cfg.name](self.cfg, ignore_index=self.ignore_index, **kwargs)
        except KeyError:
            logger.error(f"Unknown loss: {self.cfg.name}")
            raise KeyError(f"Unknown loss: {self.cfg.name}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(x, y)  
