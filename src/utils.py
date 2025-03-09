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


def init_results_path(cfg: DictConfig) -> str:
    """
    Initialize the results path.
    """
    results_path = os.path.join(cfg.results_path)

    logger.info(f"Initializing results path: {results_path}")

    # Check if the results path exists
    if os.path.exists(results_path):
        # If it exists, delete it
        shutil.rmtree(results_path)

    # Create the results path
    os.makedirs(results_path, exist_ok=True)
    
    return results_path