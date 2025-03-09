import os


import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from src.clients.mlflow_client import MLFlowClient
from torchvision import transforms


def set_normalization(settings):
    return transforms.Normalize(mean=settings["mean"], std=settings["std"])


def set_resize(settings):
    return transforms.Resize(settings, antialias=True)


def set_to_tensor(_):
    return transforms.ToTensor()


def set_center_crop(settings):
    return transforms.CenterCrop(size=settings)


def set_random_crop(settings):
    return transforms.RandomCrop(size=settings)


def set_random_horizontal_flip(settings):
    return transforms.RandomHorizontalFlip(p=settings)


def set_random_vertical_flip(settings):
    return transforms.RandomVerticalFlip(p=settings)


TRANSFORMS = {
    "normalization": set_normalization,
    "resize": set_resize,
    "to_tensor": set_to_tensor,
    "center_crop": set_center_crop,
    "random_crop": set_random_crop,
    "random_horizontal_flip": set_random_horizontal_flip,
    "random_vertical_flip": set_random_vertical_flip,
}


class ImageDataset(VisionDataset):
    """
    Basic dataset class for images and labels.

    """

    def __init__(self, cfg: DictConfig, stage: str, mlflow_client: MLFlowClient):
        super().__init__(root="")
        self.cfg = cfg
        self.stage = stage
        self.results_path = cfg.results_path
        self.mlflow_client = mlflow_client
        # Set root
        self.root = cfg.path

        # Set transforms
        self.transform = self.set_transform()

        # Set Classes
        self.classes = self.set_classes()
        self.cfg.n_classes = len(self.classes)

        # Set images and labels
        self.images, self.labels = self.decode_imgs_labels()

        # Compute histogram
        self.compute_histogram()

    def __len__(self):
        return len(self.images)

    def set_transform(self) -> transforms.Compose:
        """
        Set transforms for dataset.

        :return: Composed transforms.
        """

        transforms_list = []
        for transform_stage, transforms_cfg in self.cfg.transforms.items():
            if transform_stage in ["common", self.stage]:
                # Iterate over transforms
                for name, settings in transforms_cfg.items():
                    if settings:
                        try:
                            transforms_list.append(TRANSFORMS[name](settings))
                        except KeyError:
                            raise KeyError(f"Unknown transform: {name}")

        return transforms.Compose(transforms_list)

    def set_classes(self):
        """
        Set classes for dataset.

        This function supports that there is a classes.txt file in the dataset or that the dataset is organized in
        class folders.
        """

        # First check if there is a classes.txt file in the dataset
        if os.path.exists(os.path.join(self.root, "classes.txt")):
            classes = []
            with open(os.path.join(self.root, "classes.txt")) as f:
                for line in f:
                    classes.append(line.strip())
        else:
            # If there is no classes.txt file, get the classes from the folders
            classes = os.listdir(os.path.join(self.root, self.stage))
            classes.sort()

        return classes

    def decode_imgs_labels(self):
        """
        Decode images and labels from dataset. This function supports that the dataset is organized in superclasses
        depending on how images where obtained. For example, if we have a dataset of cooked fries, we might have
        different folders for each fries type (e.g. natural, frozen, etc.).

        In that case we might want to train a model on all the fries types or only on a specific one.

        """

        images, labels = [], []

        with open(os.path.join(self.root, f"{self.stage}.txt"), "r") as f:
            for line in f:
                cls, img = line.split("/")
                # Remove the last character (\n) from the image path
                img = img[:-1]

                # Append image and label to lists
                images.append(os.path.join(self.root, self.stage, cls, img))
                labels.append(self.classes.index(cls))


        return images, labels


    def compute_histogram(self):
        """
        Computes a histogram of classes of the dataset.
        """
        # Create figure with larger size
        plt.figure(figsize=(12, 6))

        # Compute histogram
        self.hist = plt.hist(self.labels, bins=len(self.classes))

        plt.title("Sample Histogram")
        plt.xlabel("Class")
        plt.ylabel("Frequency")

        # Modify the x-axis to show the classes with better formatting
        plt.xticks(np.arange(len(self.classes)), self.classes, rotation=45, ha='right')

        # Add padding to prevent label cutoff
        plt.tight_layout()
        self.mlflow_client.log_figure_mlflow(plt.gcf(), f"{self.stage}_histogram.png", "histograms")
        plt.close()

    def plot_sample_images(self, images, labels, epoch, predictions=None):
        """
        Plot a sample of images from the dataset.

        :param images: PyTorch tensor with images.
        :param labels: PyTorch tensor with labels.
        :param epoch: Current epoch number.
        :param predictions: PyTorch tensor with predicted class indices (optional).
        """

        plt.figure(figsize=(15, 15))

        # Convert labels to class names
        label_names = [self.classes[label.item()] for label in labels]

        # Convert predictions to class names if provided
        if predictions is not None:
            pred_names = [self.classes[pred.item()] for pred in predictions]

        # Create a grid of images
        grid_size = min(16, len(images))  # Show max 16 images
        for idx in range(grid_size):
            plt.subplot(4, 4, idx + 1)  # Create a 4x4 grid

            # Convert tensor to image
            img = images[idx].cpu().permute(1, 2, 0)  # Change from CxHxW to HxWxC

            # Denormalize if necessary
            mean = torch.tensor(self.transform.transforms[1].mean)
            std = torch.tensor(self.transform.transforms[1].std)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)  # Ensure values are in [0,1] range

            plt.imshow(img)

            # Create title with true label and prediction (if available)
            if predictions is not None:
                # Color code the title based on whether prediction is correct
                is_correct = label_names[idx] == pred_names[idx]
                color = 'green' if is_correct else 'red'
                title = f'GT: {label_names[idx]}\nPred: {pred_names[idx]}'
                plt.title(title, color=color)
            else:
                plt.title(label_names[idx])

            plt.axis('off')  # Hide axes

        plt.tight_layout()
        self.mlflow_client.log_figure_mlflow(plt.gcf(), f"{self.stage}_{epoch}_sample_images.png", "sample_images")
        plt.close()

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.images[idx], self.labels[idx]
        
        # Check if image exists
        if not os.path.isfile(img):
            logger.error(f"Image not found: {img}")
            raise FileNotFoundError(f"Image not found: {img}")

        # Read image
        img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        sample = {"images": img, "labels": target}
        return sample

