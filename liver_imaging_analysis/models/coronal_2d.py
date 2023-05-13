"""
This module provides classes and functions for
training and testing a liver segmentation model
in coronal plane using Monai.

Classes
-------
CoronalSegmentation2D
    A class representing a liver segmentation engine
    that can be used to train and test a 2D liver segmentation model
    for coronal view.

Functions
---------
per_batch_callback
    A function that plots the results of a batch in TensorBoard.
per_epoch_callback
    A function that writes the results of an epoch in TensorBoard.
"""
import shutil
import os
import SimpleITK
import cv2
import torch
import natsort
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader as MonaiLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImageD,
    NormalizeIntensityD,
    OrientationD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    ActivationsD,
    AsDiscreteD,
    KeepLargestConnectedComponentD,
    FillHolesD,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from config import config
from engine import Engine

summary_writer = SummaryWriter(config.save["tensorboard"])
dice = DiceMetric(ignore_empty=False)


class CoronalSegmentation2D(Engine):
    """
    A class that must be used when you want to run the
     liver segmentation engine,contains the transforms required
      by the user and the function that is used to start training
    """

    def __init__(self):
        config.network_parameters["channels"] = [64, 128, 256, 512, 1024, 2048]
        config.network_parameters["strides"] = [2, 2, 2, 2, 2]
        config.network_parameters["dropout"] = 0.25
        config.network_parameters["num_res_units"] = 4
        config.network_parameters["norm"] = "INSTANCE"
        config.network_parameters["bias"] = 1
        config.training["batch_size"] = 8
        config.training["optimizer_parameters"]["lr"] = 0.001
        config.training["scheduler_parameters"]["step_size"] = 20000
        config.transforms["mode"] = "3D"
        config.transforms["post_transform"] = "3DUnet_transform"

        super().__init__()

    def load_checkpoint(self, path=config.save["model_checkpoint"]):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: str
            The path of the checkpoint. (Default is the model path in config)
        """
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        self.network.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_pretraining_transforms(self, transform_name, keys):
        """
        Function used to define the needed transforms for the training data

        Parameters
        ----------
        transform_name : str
            The name of the required set of transforms.
        keys : List[str]
            The keys of the corresponding items to be transformed.

        Returns
        -------
        transforms : Compose
            A Compose object representing the set of transforms selected.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(
                        keys,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        prob=0.5,
                        keep_size=True,
                    ),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest")),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(keys, range_x=1.5,
                                range_y=0, range_z=0, prob=0.5),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name, keys):
        """
        Function used to define the needed transforms for the testing data

        Parameters
        ----------
        transform_name : str
            The name of the required set of transforms.
        keys : List[str]
            The keys of the corresponding items to be transformed.

        Returns
        -------
        transforms : Compose
            A Compose object representing the set of transforms selected.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    ResizeD(
                        keys,
                        resize_size,
                        mode=("bilinear", "nearest"),
                        allow_missing_keys=True,
                    ),
                    # RandFlipd(keys, prob=0.5,
                    # spatial_axis=1, allow_missing_keys=True),
                    # RandRotated(keys,range_x=1.5,
                    # range_y=0, range_z=0, prob=0.5, allow_missing_keys=True),
                    NormalizeIntensityD(
                        keys=keys[0], channel_wise=True,
                        allow_missing_keys=True
                    ),
                    ForegroundMaskD(
                        keys[1], threshold=0.5,
                        invert=True, allow_missing_keys=True
                    ),
                    ToTensorD(keys, allow_missing_keys=True),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

    def get_postprocessing_transforms(self, transform_name, keys):
        """
        Function used to define the needed post-processing
         transforms for prediction correction

        Parameters
        ----------
        transform_name : str
            The name of the required set of transforms.
        keys : List[str]
            The keys of the corresponding items to be transformed.

        Returns
        -------
        transforms : Compose
            A Compose object representing the set of transforms selected.
        """

        transforms = {
            "3DUnet_transform": Compose(
                [
                    ActivationsD(keys[1], sigmoid=True),
                    AsDiscreteD(keys[1], threshold=0.5),
                    FillHolesD(keys[1]),
                    KeepLargestConnectedComponentD(keys[1]),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    ActivationsD(keys[1], sigmoid=True),
                    AsDiscreteD(keys[1], threshold=0.5),
                    FillHolesD(keys[1]),
                    KeepLargestConnectedComponentD(keys[1]),
                ]
            ),
        }
        return transforms[transform_name]

    def predict_2dto3d(self, volume_path, temp_path="temp/"):
        """
        predicts the label of a 3D volume using a 2D network

        Parameters
        ----------
        volume_path: str
            path of the input file. expects a 3D nifti file.
        temp_path: str
            a temporary path to save 3d volume as 2d png slices.
             default is "temp/" and is automatically
             deleted before returning the prediction

        Returns
        -------
        tensor
            tensor of the predicted labels
            with shape (1,channel,length,width,depth)
        """
        # read volume
        img_volume = SimpleITK.ReadImage(volume_path)
        img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        number_of_slices = img_volume_array.shape[1]
        # create temporary folder to store 2d png files
        if os.path.exists(temp_path) is False:
            os.mkdir(temp_path)
        # write volume slices as 2d png files
        for slice_number in range(number_of_slices):
            volume_silce = img_volume_array[:, slice_number, :]
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[
                -1
            ]  # delete extension from filename
            volume_png_path = (
                os.path.join(temp_path,
                             volume_file_name + "_" + str(slice_number))
                + ".png"
            )
            cv2.imwrite(volume_png_path, volume_silce)
        # predict slices individually then reconstruct 3d prediction
        prediction = self.predict(temp_path)
        # transform shape from (batch,channel,length,width)
        # to (1,channel,length,width,batch)
        prediction = prediction.permute(1, 0, 2, 3).unsqueeze(dim=0)
        # delete temporary folder
        shutil.rmtree(temp_path)
        return prediction

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
        keys = (config.transforms["img_key"], config.transforms["pred_key"])
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(data_dir))
            volume_paths = [
                os.path.join(data_dir, file_name) for file_name in volume_names
            ]
            predict_files = [
                {keys[0]: image_name}
                for image_name in volume_paths
            ]
            predict_set = Dataset(data=predict_files,
                                  transform=self.test_transform)
            predict_loader = MonaiLoader(
                predict_set,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,
            )
            prediction_list = []
            for batch in predict_loader:
                volume = batch[keys[0]].to(self.device)
                batch[keys[1]] = self.network(volume)
                # Apply post processing transforms on 2D prediction
                if config.transforms["mode"] == "2D":
                    batch = self.post_process(batch, keys[1])
                prediction_list.append(batch[keys[1]])
            prediction_list = torch.cat(prediction_list, dim=0)

        prediction_list = self.post_process(
            {keys[1]: prediction_list},
            keys[1]
        )
        prediction_list = prediction_list[keys[1]]

        return prediction_list


def per_batch_callback(batch_num, image, label, prediction):
    """
    A function which is called after every
     batch to plot the results in tensorboard

    Parameters
    ----------
    batch_num : int
        The batch number.
    image : torch.Tensor
        The input image tensor.
    label : torch.Tensor
        The label tensor.
    prediction : torch.Tensor
        The output prediction tensor.

    Returns
    -------
    None
    """
    score = dice(prediction.bool(), label.bool())[
        0
    ]  # first slice as default in plot_2d_or_3d_image
    plot_2d_or_3d_image(
        data=image,
        step=0,
        writer=summary_writer,
        frame_dim=-1,
        tag=f"Batch{batch_num}:Volume",
    )
    plot_2d_or_3d_image(
        data=label,
        step=0,
        writer=summary_writer,
        frame_dim=-1,
        tag=f"Batch{batch_num}:Mask",
    )
    plot_2d_or_3d_image(
        data=prediction,
        step=0,
        writer=summary_writer,
        frame_dim=-1,
        tag=f"Batch{batch_num}:Prediction:Dice Score={score}",
    )


def per_epoch_callback(epoch, training_loss,
                       valid_loss, training_metric, valid_metric):
    """
    A function which is called every epoch to write the results in tensorboard

    Parameters
    ----------
    epoch : int
        The epoch number.
    training_loss : float
        The training loss for the epoch.
    valid_loss : float or None
        The validation loss for the epoch, or None if validation was not done.
    training_metric : float
        The training metric for the epoch.
    valid_metric : float or None
        The validation metric for the epoch,
         or None if validation was not done.

    Returns
    -------
    None
    """
    print("\nTraining Loss=", training_loss)
    print("Training Metric=", training_metric)

    summary_writer.add_scalar("Training Loss", training_loss, epoch)
    summary_writer.add_scalar("Training Metric", training_metric, epoch)

    if valid_loss is not None:
        print(f"Validation Loss={valid_loss}")
        print(f"Validation Metric={valid_metric}")

        summary_writer.add_scalar("Validation Loss", valid_loss, epoch)
        summary_writer.add_scalar("Validation Metric", valid_metric, epoch)
