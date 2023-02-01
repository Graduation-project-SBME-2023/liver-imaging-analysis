"""
a module contains the fixed structure of the core of our code

"""
import os
import random

import dataloader
import losses
import models
import numpy as np
import torch
import torch.optim.lr_scheduler
from config import config
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset
from monai.losses import DiceLoss as monaiDiceLoss
from torchmetrics import Accuracy, Dice, JaccardIndex
from utils import progress_bar


class Engine:
    """
    Class that implements the basic PyTorch methods for deep learning tasks
    tasks should inherit this class
    """

    def __init__(self):

        self.device = config.device
        print("Used Device: ", self.device)

        self.loss = self.get_loss(
            loss_name=config.training["loss_name"],
            **config.training["loss_parameters"],
        )

        self.network = self.get_network(
            network_name=config.network_name, **config.network_parameters
        ).to(self.device)

        self.optimizer = self.get_optimizer(
            optimizer_name=config.training["optimizer"],
            **config.training["optimizer_parameters"],
        )

        self.scheduler = self.get_scheduler(
            scheduler_name=config.training["lr_scheduler"],
            **config.training["scheduler_parameters"],
        )
        self.metrics = self.get_metrics(
            metrics_name=config.training["metrics"],
            **config.training["metrics_parameters"],
        ).to(self.device)
        self.load_data()

    def get_optimizer(self, optimizer_name, **kwargs):
        """
        internally used to load optimizer.
        Parameters
        ----------
            optimizer_name: str
                name of optimizer to fetch from dictionary
                should be chosen from: 'Adam','SGD'
            **kwargs: dict
                parameters of optimizer, if exist.
        """
        optimizers = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }
        return optimizers[optimizer_name](self.network.parameters(), **kwargs)

    def get_scheduler(self, scheduler_name, **kwargs):
        """
        internally used to load lr scheduler.
        Parameters
        ----------
            scheduler_name: str
                name of scheduler to fetch from dictionary
                should be chosen from: 'StepLR','CyclicLR','...'
            kwargs: dict
                parameters of optimizer, if exist.
        """
        schedulers = {
            "StepLR": torch.optim.lr_scheduler.StepLR,
            "CyclicLR": torch.optim.lr_scheduler.CyclicLR,
        }
        return schedulers[scheduler_name](self.optimizer, **kwargs)

    def get_network(self, network_name, **kwargs):
        """
        internally used to load network.
        Parameters
        ----------
            network_name: str
                name of network to fetch from dictionary
                should be chosen from: '3DUNet','3DResNet','2DUNet'
            **kwargs: dict
                parameters of network, if exist.
        """
        networks = {
            "3DUNet": models.UNet3D,
            "3DResNet": models.ResidualUNet3D,
            "2DUNet": models.UNet2D,
        }
        return networks[network_name](**kwargs)

    def get_loss(self, loss_name, **kwargs):
        """
        internally used to load loss function.
        Parameters
        ----------
            loss_name: str
                name of loss function to fetch from dictionary
                should be chosen from: 'dice_loss','monai_dice'
            **kwargs: dict
                parameters of loss function, if exist.
        """
        loss_functions = {
            "dice_loss": losses.DiceLoss,
            "monai_dice": monaiDiceLoss,
            "bce_dice": losses.BCEDiceLoss,
        }
        return loss_functions[loss_name](**kwargs)

    def get_metrics(self, metrics_name, **kwargs):
        """
        internally used to load metrics.
        Parameters
        ----------
            metrics_name: str
                name of metrics to be fetched from dictionary
                should be chosen from: 'accuracy','dice',
            **kwargs: dict
                parameters of metrics, if exist.
        """
        metrics = {
            "accuracy": Accuracy,
            "dice": Dice,
            "jaccard": JaccardIndex,
        }
        return metrics[metrics_name](**kwargs)

    def get_pretraining_transforms(self):
        """
        Should be Implemented by user in liver_segmentation module.
        Transforms to be applied on training set before training.
        Expected to return a monai Compose object with desired transforms.
        """
        raise NotImplementedError()

    def get_pretesting_transforms(self):
        """
        Should be Implemented by user in liver_segmentation module.
        Transforms to be applied on testing set before evaluation.
        Expected to return a monai Compose object with desired transforms.
        """
        raise NotImplementedError()

    def load_data(
        self,
    ):
        """
        Internally used to load and save the data to the data attributes.
        Uses the parameter values in config.
        """

        self.train_dataloader = []
        self.val_dataloader = []
        self.test_dataloader = []
        self.keys = (config.transforms["img_key"], config.transforms["label_key"])
        self.batch_size = config.training["batch_size"]
        self.train_transform = self.get_pretraining_transforms(
            config.transforms["train_transform"], self.keys
        )
        self.test_transform = self.get_pretesting_transforms(
            config.transforms["test_transform"], self.keys
        )

        trainloader = dataloader.DataLoader(
            dataset_path=config.dataset["training"],
            batch_size=config.training["batch_size"],
            train_transforms=self.train_transform,
            test_transforms=self.test_transform,
            num_workers=0,
            pin_memory=False,
            test_size=config.training["train_valid_split"],
            keys=self.keys,
            mode=config.dataset["mode"],
        )
        testloader = dataloader.DataLoader(
            dataset_path=config.dataset["testing"],
            batch_size=config.training["batch_size"],
            train_transforms=self.train_transform,
            test_transforms=self.test_transform,
            num_workers=0,
            pin_memory=False,
            test_size=1,  # testing set should all be set as evaluation (no training)
            keys=self.keys,
            mode=config.dataset["mode"],
        )
        self.train_dataloader = trainloader.get_training_data()
        self.val_dataloader = trainloader.get_testing_data()
        self.test_dataloader = testloader.get_testing_data()

    def data_status(self):
        """
        Prints the shape and data type of a training batch
        and a testing batch, if exists.
        """

        img_key = config.transforms["img_key"]
        label_key = config.transforms["label_key"]

        dataloader_iterator = iter(self.train_dataloader)
        try:
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Training Features:"
                f" {batch[img_key].shape} {batch[img_key].dtype}"
            )
            print(
                f"Batch Shape of Training Labels:"
                f" {batch[label_key].shape} {batch[label_key].dtype}"
            )
        except StopIteration:
            print("No Training Set")

        dataloader_iterator = iter(self.val_dataloader)
        try:
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Validation Features:"
                f" {batch[img_key].shape} {batch[img_key].dtype}"
            )
            print(
                f"Batch Shape of Validation Labels:"
                f" {batch[label_key].shape} {batch[label_key].dtype}"
            )
        except StopIteration:
            print("No Validation Set")

        dataloader_iterator = iter(self.test_dataloader)
        try:
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Testing Features:"
                f" {batch[img_key].shape} {batch[img_key].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch[label_key].shape} {batch[label_key].dtype}"
            )
        except StopIteration:
            print("No Testing Set")

    def save_checkpoint(self, path=config.save["potential_checkpoint"]):
        """
        Saves current checkpoint to a specific path. (Default is the model path in config)

        Parameters
        ----------
        path: str
            The path where the checkpoint will be saved at
        """
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path=config.save["model_checkpoint"]):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: str
            The path of the checkpoint. (Default is the model path in config)
        """
        self.network.load_state_dict(
            torch.load(path, map_location=torch.device(self.device))
        )

    def compile_status(self):
        """
        Prints the loss function and optimizer to be used.
        """
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")

    def fit(
        self,
        epochs=config.training["epochs"],
        evaluate_epochs=1,
        batch_callback_epochs=None,
        save_weight=False,
        save_path=config.save["potential_checkpoint"],
        per_batch_callback=None,
        per_epoch_callback=None,
    ):
        """
        train the model using the stored training set

        Parameters
        ----------
        epochs: int
            the number of iterations for fitting. (Default is the value in config)
        evaluate_epochs: int
            the number of epochs to evaluate model after. (Default is 1)
        batch_callback_epochs: int
            the number of epochs to visualize gifs after, if exists. (Default is None)
        save_weight: bool
            flag to save best weights. (Default is False)
        save_path: str
            directory to save best weights at. (Default is the potential path in config)
        per_batch_callback: method
            a function that contains the code to be executed after each batch
        per_epoch_callback: method
            a function that contains the code to be executed after each epoch
        """
        best_valid_loss = float("inf")  # initialization with largest possible number

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
            training_loss = 0

            self.network.train()
            progress_bar(0, len(self.train_dataloader))  # batch progress bar
            for batch_num, batch in enumerate(self.train_dataloader):
                progress_bar(batch_num, len(self.train_dataloader))
                volume, mask = batch["image"].to(self.device), batch["label"].to(
                    self.device
                )
                pred = self.network(volume)
                loss = self.loss(pred, mask)
                # batch_metric = self.metrics(pred, mask.int())
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                training_loss += loss.item()
                if batch_callback_epochs is not None:
                    if (epoch + 1) % batch_callback_epochs == 0:
                        if per_batch_callback is not None:
                            per_batch_callback(
                                batch_num,
                                batch["image"],
                                batch["label"],
                                (
                                    torch.sigmoid(pred) > 0.5
                                ).float(),  # predicted mask after thresholding
                            )
            training_loss = training_loss / len(
                self.train_dataloader
            )  # normalize loss over batch size
            training_metric = self.metrics.compute()  # total epoch metric

            if (
                epoch + 1
            ) % evaluate_epochs == 0:  # every evaluate_epochs, test model on test set
                valid_loss, valid_metric = self.test(self.test_dataloader)
                if save_weight:  # save model if performance improved on validation set
                    if valid_loss <= best_valid_loss:
                        best_valid_loss = valid_loss
                        self.save_checkpoint(save_path)
            else:
                valid_loss = None
                valid_metric = None
            if per_epoch_callback is not None:
                per_epoch_callback(
                    epoch,
                    training_loss,
                    valid_loss,
                    training_metric.item(),
                    valid_metric,
                )

            self.metrics.reset()

    def test(self, dataloader):
        """
        calculates loss on input dataset

        Parameters
        ----------
        dataloader: dict
                the dataset to evaluate on
        """
        num_batches = len(dataloader)
        test_loss = 0
        self.metrics.reset()
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = batch["image"].to(self.device), batch["label"].to(
                    self.device
                )
                pred = self.network(volume)
                test_loss += self.loss(pred, mask).item()
                # metric = self.metrics(pred, mask.int())
            test_metric = self.metrics.compute()

            test_loss /= num_batches

        return test_loss, test_metric.item()

    def predict(self, data_dir):
        """
        predict the label of the given input
        Parameters
        ----------
        volume_path: str
            path of the input directory. expects nifti or png files.
        Returns
        -------
        tensor
            tensor of the predicted labels
        """

        volume_names = os.listdir(data_dir)
        volume_paths = [os.path.join(data_dir, file_name) for file_name in volume_names]
        predict_files = [{"image": image_name} for image_name in volume_paths]
        predict_set = Dataset(data=predict_files, transform=self.test_transform)
        predict_loader = MonaiLoader(
            predict_set,
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=False,
        )
        prediction_list = []
        with torch.no_grad():
            for batch in predict_loader:
                volume = batch["image"].to(self.device)
                pred = self.network(volume)
                pred = (torch.sigmoid(pred) > 0.5).float()
                prediction_list.append(pred)
            prediction_list = torch.cat(prediction_list, dim=0)
        return prediction_list


def set_seed(self):
    """
    function to set seed for all randomized attributes of the packages and modules
    """
    seed = config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
