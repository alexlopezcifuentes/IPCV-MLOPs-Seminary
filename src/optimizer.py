import torch
import torch.optim as optim
from loguru import logger
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


# Optimizers
def get_SGD(cfg, net):
    return optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def get_Adam(cfg, net):
    return optim.Adam(net.parameters(), lr=cfg.lr)


OPTIMIZERS = {"sgd": get_SGD, "adam": get_Adam}


# Schedulers
def set_StepLR(optimizer: torch.optim.Optimizer, cfg: DictConfig, warmup) -> StepLR:
    return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.lr_step, gamma=cfg.lr_factor)


def set_ReduceLROnPlateau(optimizer: torch.optim.Optimizer, cfg: DictConfig, warmup) -> ReduceLROnPlateau:
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, cfg.lr_metric, patience=cfg.patience, verbose=True)


def set_CosineAnnealingLR(optimizer: torch.optim.Optimizer, cfg: DictConfig, warmup) -> CosineAnnealingLR:
    if warmup:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs - cfg.warmup.warmup_epochs, eta_min=0.0
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0.0)
    return scheduler


SCHEDULERS = {"step": set_StepLR, "plateau": set_ReduceLROnPlateau, "cosine_annealing": set_CosineAnnealingLR}


class Optimizer(optim.Optimizer):
    def __init__(self, cfg, net, **kwargs):
        self.cfg = cfg
        self.net = net

        self.warmup = False
        """ # Check if warmup is needed
        if cfg.scheduler.warmup.warmup_epochs > 0:
            self.warmup = True
            self.len_loader = kwargs["len_loader"]
        else:
            self.warmup = False """

        # Set optimizer and scheduler
        self.optimizer = self.set_optimizer()
        self.scheduler = self.set_scheduler()

    def set_optimizer(self):
        try:
            logger.info(f"Initiating optimizer: {self.cfg.name}")
            return OPTIMIZERS[self.cfg.name](self.cfg, self.net)
        except KeyError:
            logger.error(f"Unknown optimizer: {self.cfg.name}")
            raise KeyError(f"Unknown optimizer: {self.cfg.name}")

    def set_scheduler(self):
        try:
            logger.info(f"Initiating scheduler: {self.cfg.scheduler.name}")
            scheduler = SCHEDULERS[self.cfg.scheduler.name](self.optimizer, self.cfg.scheduler, self.warmup)
            if self.warmup:
                return 1
            else:
                return scheduler
        except KeyError:
            logger.error(f"Unknown scheduler: {self.cfg.scheduler.name}")
            raise KeyError(f"Unknown scheduler: {self.cfg.scheduler.name}")
