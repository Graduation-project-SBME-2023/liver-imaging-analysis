from typing import Dict
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import dataloader
import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import plot_2d_or_3d_image
import models
import losses
from monai.transforms import (
LoadImageD,
ForegroundMaskD,
EnsureChannelFirstD,
AddChannelD,
ScaleIntensityD,
ToTensorD,
Compose,
NormalizeIntensityD,
AsDiscreteD,
SpacingD,
OrientationD,
ResizeD,

RandSpatialCropd,
Spacingd,
RandFlipd,
RandScaleIntensityd,
RandShiftIntensityd,
RandRotated,
SqueezeDimd,
CenterSpatialCropD,
)
class Engine():
    """Class that implements the basic PyTorch methods for neural network
    Neural Networks should inherit from this class
    Parameters
    ----------
        device: str
            the device to be used by the model.
            either "cpu" or "cuda"
        loss: class
            the loss function to be used,
            should be imported from diceloss class
        optimizer: str
            the optimizer to be used, should be imported from optimizers.py
        metrics: array_like
            the metrics calculated for each batch per epoch during training,
            and for the whole data during evaluating expects an array of string
            of one or more of: 'loss', 'dice_score'. eg; ['loss']
        training_data_path: str
              String containing paths of the training volumes at the folder
              Path and masks at the folder Path2.
        testing_data_path: str
              String containing paths of the testing volumes at the folder
              Path and masks at the folder Path2.
        transformation_flag: bool
              indicates if data preprocessing should be performed or not
              False will ignore "transformation" argument
        data_size: array_like
              an array of the shape data will be transformed into.
              eg; [64,512,512]
        batchsize: int
            the number of features to be loaded in each batch.(Default: 1)
        train_valid_split: float
            a fraction between 0-1 that indicate the portion of dataset to be
            loaded to the validation set.( Default: 0)
        learning_rate: float
            learning rate to be used by the optimizer
    """

    def __init__(
        self,
        config, 
        device,
        metrics,
        training_data_path,
        testing_data_path,
        transformation_flag,
        data_size,
        batchsize=1,
        train_valid_split=0,
        ):
        # super().__init__()
        self.device = device
        self.loss = self._get_loss(
            loss_name=config["training"]["loss_name"],
            **config["training"]["loss_params"]
        )
        self.network = self._get_network(
            network_name=config["network_name"],
            **config["network_params"]
            ).to(device)

        self.optimizer = self._get_optimizer(
            optimizer_name=config['training']['optimizer'],
            **config['training']['optimizer_params']
            )

        self.metrics = metrics

        self._load_data(training_data_path=training_data_path,\
                        testing_data_path=testing_data_path,\
                        transformation_flag=transformation_flag,\
                        data_size=data_size,\
                        batchsize=batchsize,train_valid_split=train_valid_split)
    

    def _get_optimizer(self,optimizer_name,**kwargs):        
        optimizers={
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
        }
        return optimizers[optimizer_name](self.network.parameters(), **kwargs)

    def _get_network(self,network_name,**kwargs):
        networks = {
            '3D UNet': models.UNet3D,
            '3D ResNet': models.ResidualUNet3D
        }
        return networks[network_name](**kwargs)

    def _get_loss(self,loss_name,**kwargs):
        loss_functions= {
            'Dice Loss': losses.DiceLoss,
        }        
        return loss_functions[loss_name](**kwargs)


    def get_pretraining_transforms(self,transform_name,keys,size):
        transforms= {
            '3DUnet': Compose(
            [
                LoadImageD(keys),
                EnsureChannelFirstD(keys),
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                ResizeD(keys, size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                # normalize intensity to have mean = 0 and std = 1.
                ToTensorD(keys),
            ]
        ),
        '2DUnet': Compose(
            [
                LoadLoadImageD(keys),
                EnsureChannelFirstD(keys),
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                ResizeD(keys, size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                # normalize intensity to have mean = 0 and std = 1.
                ToTensorD(keys),
            ]
        )
        } 

        return transforms[transform_name]     


    def _load_data(
        self,
        training_data_path,
        testing_data_path,
        transformation_flag,
        data_size,
        batchsize=1,
        train_valid_split=0,
    ):
        """
        Internally used to load and save the data to the data attribute

        Parameters
        ----------
        training_data_path: str
              String containing paths of the training volumes at the folder
              Path and masks at the folder Path2.
        testing_data_path: str
              String containing paths of the testing volumes at the folder
              Path and masks at the folder Path2.
        transformation_flag: bool
              indicates if data preprocessing should be performed or not
              False will ignore "transformation" argument
        data_size: array_like
              an array of the shape data will be transformed into.
              eg; [64,512,512]
        batchsize: int
            the number of features to be loaded in each batch.(Default: 1)
        train_valid_split: float
            a fraction between 0-1 that indicate the portion of training dataset to be
            loaded to the validation set.( Default: 0)
        """

        self.data_size = data_size
        # self.expand_flag = not transformation_flag
        self.train_dataloader = []
        self.val_dataloader = []
        self.test_dataloader = []
        self.keys = (self.config["transforms"]['key1'],self.config["transforms"]['key2'])
        self.transform = self.get_pretraining_transforms(self.config["transforms"]['transform_name'], self.keys, data_size)

        trainloader = dataloader.DataLoader(
            dataset_path=training_data_path,
            batch_size=batchsize,
            transforms=self.transform,
            num_workers=0,
            pin_memory=False,
            test_size=train_valid_split,
            keys=self.keys,
        )
        testloader = dataloader.DataLoader(
            dataset_path=testing_data_path,
            batch_size=batchsize,
            transforms=self.transform,
            num_workers=0,
            pin_memory=False,
            test_size=0, #testing set shouldn't be divided
            keys=self.keys,
        )
        self.train_dataloader = trainloader.get_training_data()
        self.val_dataloader = trainloader.get_testing_data()
        self.test_dataloader = testloader.get_training_data()

    def data_status(self):
        """
        Prints the shape and data type of the first batch of the training set
        and testing set, if exists
        """
        for batch in self.train_dataloader:
            print(
                f"Batch Shape of Training Features:"
                f" {batch['image'].shape} {batch['image'].dtype}"
            )
            print(
                f"Batch Shape of Training Labels:"
                f" {batch['label'].shape} {batch['label'].dtype}"
            )
            break
        for batch in self.val_dataloader:
            print(
                f"Batch Shape of Testing Features:"
                f" {batch['image'].shape} {batch['image'].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch['label'].shape} {batch['label'].dtype}"
            )
            break

    def save_checkpoint(self, path):
        """
        Saves current checkpoint to a specific path

        Parameters
        ----------
        path: int
            The path where the checkpoint will be saved at
        """
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: int
            The path of the checkpoint
        """
        self.network.load_state_dict(
            torch.load(path, map_location=torch.device(self.device))
        )  # if working with CUDA remove torch.device('cpu')


    def compile_status(self):
        """
        Prints the stored loss function, optimizer, and metrics.
        """
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")
        print(f"Metrics= {self.metrics} \n")

    def fit(
        self,
        epochs=1,
        evaluation_set=None,
        evaluate_epochs=1,
        visualize_epochs=1,
        save_flag=True,
        save_path="best_epoch"
        ):
        """
        train the model using the stored training set

        Parameters
        ----------
        epochs: int
            the number of iterations for fitting. (Default = 1)
        evaluation_set: dict
            the dataset to be used for evaluation
        evaluate_epochs: int
            the number of epochs to evaluate model after
        visualize_epochs: int
            the number of epochs to visualize gifs after
        save_flag: bool
            flag to save best weights
        save_path: str
            directory to save best weights at

        """
        tb = SummaryWriter()    
        best_valid_loss=float('inf') #initialization with largest possible number
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
            training_loss = 0
            self.network.train()  
            for batch_num, batch in enumerate(self.train_dataloader):
                print(f"Batch {batch_num+1}/{len(self.train_dataloader)}")
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                pred = self.network(volume)
                loss = self.loss(pred, mask)                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()
                # Print Progress
                if ((epoch+1)%visualize_epochs==0): #every visualize_epochs create gifs
                    plot_2d_or_3d_image(data=batch['image'],step=0,writer=tb,
                                        frame_dim=-1,tag=f"Batch{batch_num}:Volume")
                    plot_2d_or_3d_image(data=batch['label'],step=0,writer=tb,
                                        frame_dim=-1,tag=f"Batch{batch_num}:Mask")
                    plot_2d_or_3d_image(data=(torch.sigmoid(pred)>0.5).float(),step=0,writer=tb,
                                        frame_dim=-1,tag=f"Batch{batch_num}:Prediction")                  
            training_loss = training_loss / len(self.train_dataloader)
            print("Training Loss=",training_loss)
            tb.add_scalar("Training Loss", training_loss, epoch)
            if ((epoch+1)%evaluate_epochs==0): #every evaluate_epochs test model on test set
                if evaluation_set != None:
                    valid_loss=self.test(evaluation_set)
                    print(f"Validation Loss={valid_loss}")
                    tb.add_scalar("Validation Loss", valid_loss, epoch)
                if save_flag: #save model if performance improved on validation set
                    if valid_loss <= best_valid_loss:
                        best_valid_loss=valid_loss
                        self.save_checkpoint(save_path)


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
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                pred = self.network(volume)
                test_loss += self.loss(pred, mask).item()
            test_loss /= num_batches
        return test_loss


    def predict(self, volume_path):
        """
        predict the label of the given input using the current weights

        Parameters
        ----------
        volume_path: str
                  path of the input feature. expects a nifti file.
        Returns
        -------
        tensor
            tensor of the predicted label
        """
        dict_loader = dataloader.LoadImageD(keys=("image", "label"))
        data_dict = dict_loader({"image": volume_path, "label": volume_path})
        preprocess = dataloader.Preprocessing(("image", "label"),
                                              self.data_size)
        data_dict_processed = preprocess(data_dict)
        volume = data_dict_processed["image"]
        volume = volume.expand(
            1, volume.shape[0], volume.shape[1],
            volume.shape[2], volume.shape[3]
        )       
        with torch.no_grad():
            pred = self.network(volume.to(self.device))
        return pred
