import random

import torch
import torchmetrics
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.classification import Accuracy

from src.utils import AverageMeter


class ClassificationRunner:
    def __init__(
        self,
        cfg,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        model=None,
        loss=None,
        optimizer=None,
        mlflow_client=None,
    ):
        logger.info("Instantiating Runner")
        self.cfg = cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.classes = self.train_loader.dataset.classes
        self.device = self.model.device
        self.epochs = self.cfg.training.epochs
        self.batch_size = self.cfg.training.batch_size
        self.print_freq = self.cfg.training.print_freq
        self.mlflow_client = mlflow_client

        # Metrics
        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes))
        self.max_acc = 0
        losses = {"train": AverageMeter(), "val": AverageMeter(), "test": AverageMeter()}
        accuracies = {"train": AverageMeter(), "val": AverageMeter(), "test": AverageMeter()}
        self.metrics = {"loss": losses, "accuracy": accuracies}

    def complete_train(self):
        logger.info("Starting complete training...")

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1} of {self.epochs}")

            # Train
            logger.info("Training step...")
            self.epoch_train(self.train_loader, epoch)

            # Simple Validation
            logger.info("Simple validation step...")
            self.simple_validation(self.val_loader, epoch)

            # Scheduler Step
            logger.info("Scheduler step...")
            self.optimizer.scheduler.step()
            logger.info(f"Learning rate: {self.optimizer.scheduler.get_last_lr()[0]}")
            self.metrics["lr"] = self.optimizer.scheduler.get_last_lr()[0]

            # Log Metrics
            self.mlflow_client.log_metrics_mlflow(self.metrics, epoch)

            # Save Model
            self.max_acc = self.model.save_model(self.metrics, self.mlflow_client, self.max_acc)

        # Complete Validation
        logger.info("Complete validation step...")
        self.complete_validation(self.test_loader)
        # Log Test Metrics
        self.mlflow_client.log_metrics_mlflow(self.metrics, epoch)
        logger.info("Finished Training")

    def forward_pass(self, images, labels, stage):
        # Forward Pass
        outputs = self.model(images)
        # Loss
        loss = self.loss(outputs, labels)
        # Accuracy
        # TODO: Compute accuracy with torchmetrics. They even have accumulation implemented.
        acc = self.accuracy(outputs, labels)
        # Save Losses and Accuracies
        self.metrics["loss"][stage].update(loss.item(), self.batch_size)
        self.metrics["accuracy"][stage].update(acc.item(), self.batch_size)

        return outputs, loss, acc

    def epoch_train(self, loader, epoch, stage="train"):
        # Reset metrics
        self.metrics["loss"][stage].reset()
        self.metrics["accuracy"][stage].reset()
        for i, data in enumerate(loader, 0):
            # Extract data
            images, labels = data["images"].to(self.device), data["labels"].to(self.device)

            # Zero the parameter gradients
            self.optimizer.optimizer.zero_grad()

            # Forward Pass
            outputs, loss, acc = self.forward_pass(images, labels, stage)
            _, predictions = torch.max(outputs, 1)

            # Draw a grid of images where each image should have the label as the title
            if i == 0:
                loader.dataset.plot_sample_images(images, labels, epoch, predictions)

            # Backward Pass
            loss.backward()
            self.optimizer.optimizer.step()

            # Print statistics
            if i % self.print_freq == 0:
                logger.info(
                    f"[Epoch {epoch + 1}/{self.epochs}, {i + 1:5d}/{len(loader)}]. "
                    f"Loss: {self.metrics['loss'][stage].avg:.3f}. "
                    f"Accuracy: {self.metrics['accuracy'][stage].avg:.3f}"
                )

    def simple_validation(self, loader, epoch, stage="val"):
        # Reset metrics
        self.metrics["loss"][stage].reset()
        self.metrics["accuracy"][stage].reset()

        # Select a random batch to plot. We dont consider the last batch to obtain a full batch of images.
        batch_to_plot = random.randint(0, len(loader) - 1)

        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                # Extract data
                images, labels = data["images"].to(self.device), data["labels"].to(self.device)

                # Forward Pass
                outputs, loss, acc = self.forward_pass(images, labels, stage)
                _, predictions = torch.max(outputs, 1)

                # Draw a grid of images where each image should have the label as the title
                if i == batch_to_plot:
                    loader.dataset.plot_sample_images(images, labels, epoch, predictions)

                if i % self.print_freq == 0:
                    logger.info(
                        f"[Epoch {epoch + 1}/{self.epochs}, {i + 1:5d}/{len(loader)}]. "
                        f"Loss: {self.metrics['loss'][stage].avg:.3f}. "
                        f"Accuracy: {self.metrics['accuracy'][stage].avg:.3f}"
                    )

        logger.info(
            f"Accuracy of the network on the {len(loader) * loader.batch_size} images: "
            f"{self.metrics['accuracy'][stage].avg:.3f} %"
        )

    def complete_validation(self, loader, stage="test"):
        # Reset metrics
        self.metrics["loss"][stage].reset()
        self.metrics["accuracy"][stage].reset()

        # Prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in loader.dataset.classes}
        total_pred = {classname: 0 for classname in loader.dataset.classes}
        self.metrics["class_accuracy"] = {classname: 0 for classname in loader.dataset.classes}
        predictions_list = []
        labels_list = []

        # again no gradients needed
        with torch.no_grad():
            for data in loader:
                # Extract data
                images, labels = data["images"].to(self.device), data["labels"].to(self.device)

                # Forward Pass
                outputs, loss, acc = self.forward_pass(images, labels, stage)

                _, predictions = torch.max(outputs, 1)

                # Save Predictions and Labels
                predictions_list.extend(predictions)
                labels_list.extend(labels)

                # Collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[loader.dataset.classes[label]] += 1
                    # TODO: Draw error images and labels and log them to MLFlow
                    total_pred[loader.dataset.classes[label]] += 1

        logger.info(
            f"Accuracy of the network on the {len(loader) * loader.batch_size} images: "
            f"{self.metrics['accuracy'][stage].avg:.3f} %"
        )

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            if total_pred[classname] > 0:
                cls_accuracy = 100 * float(correct_count) / total_pred[classname]
            else:
                logger.warning(f"Class {classname} does not have images in validation set. Class accuracy set to 100.")
                cls_accuracy = 100.0
            self.metrics["class_accuracy"][classname] = cls_accuracy
            logger.info(f"Accuracy for class: {classname:5s} is {self.metrics['class_accuracy'][classname]:.1f} %")

        self.metrics["MCA"] = sum(self.metrics["class_accuracy"].values()) / len(
            self.metrics["class_accuracy"].values()
        )
        logger.info(f"Mean Class Accuracy: {self.metrics['MCA']:.3f} %")

        confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            len(loader.dataset.classes), ignore_index=None, normalize=None, validate_args=True
        )

        cm = confmat(torch.tensor(predictions_list), torch.tensor(labels_list))

        plt.figure(figsize=(24, 20))  # Increase figure size
        disp = ConfusionMatrixDisplay(confusion_matrix=cm.numpy(), display_labels=loader.dataset.classes)
        disp.plot(xticks_rotation=45, values_format="d")  # Rotate x-axis labels 45 degrees  # Show values as integers
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        self.mlflow_client.log_figure_mlflow(plt.gcf(), "confusion_matrix.png")
        plt.close()
