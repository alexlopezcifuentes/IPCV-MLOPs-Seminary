import os
import shutil
from omegaconf import DictConfig
from loguru import logger
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        """
        Initialize the AverageMeter class.
        """
        self.reset()

    def reset(self) -> None:
        """
        Reset the AverageMeter class.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: int | float, n: int = 1) -> None:
        """
        Update the AverageMeter class with a given value.

        :param val: Value to update the class with
        :param n: Number of times to update the class
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count