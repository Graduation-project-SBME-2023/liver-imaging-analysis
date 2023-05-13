"""
This module provides classes and functions for
training and testing a liver segmentation model
in axial plane using Monai.

Classes
-------
AxialSegmentation2D
    A class representing a liver segmentation engine
    that can be used to train and test a 2D liver segmentation model
    for axial view.

Functions
---------
segment_liver
    A functin segments the liver in 2D images using a 2D liver model.
segment_liver_3d
    A function used to segment the liver of a 3d volume using a 2D liver model
train_liver
    A function used to start the training of liver segmentation
set_configs
    A function that sets configurations given different modes
"""

import os
import shutil
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImageD,
    NormalizeIntensityD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    RandAdjustContrastd,
    RandZoomd,
    ActivationsD,
    AsDiscreteD,
    KeepLargestConnectedComponentD,
    FillHolesD
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric

from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset
import torch
import SimpleITK
import cv2
import natsort
from config import config
from engine import Engine, set_seed

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric = DiceMetric(ignore_empty=True, include_background=True)


def set_configs(mode="2D"):
    """
    Sets configurations for different modes.

    Parameters
    ----------
    mode : str
        The mode for which to set the configurations.

    Returns:
    -------
    None
    """

    if mode == "2D":
        config.training["scheduler_parameters"] = {
            "step_size": 20,
            "gamma": 0.5,
            "verbose": False,
        }
        config.dataset["prediction"] = "test cases/sample_image"
        config.training["batch_size"] = 8
        config.network_parameters["dropout"] = 0
        config.network_parameters["channels"] = [64, 128, 256, 512]
        config.network_parameters["strides"] = [2, 2, 2]
        config.network_parameters["num_res_units"] = 4
        config.network_parameters["norm"] = "INSTANCE"
        config.network_parameters["bias"] = 1
        config.save["liver_checkpoint"] = "liver_cp"
        config.transforms["mode"] = "3D"

    elif mode == "sliding_window":
        config.dataset["prediction"] = "test cases/sample_volume"
        config.training["batch_size"] = 1
        config.training["scheduler_parameters"] = {
            "step_size": 20,
            "gamma": 0.5,
            "verbose": False,
        }
        config.network_parameters["dropout"] = 0
        config.network_parameters["channels"] = [64, 128, 256, 512]
        config.network_parameters["spatial_dims"] = 3
        config.network_parameters["strides"] = [2, 2, 2]
        config.network_parameters["num_res_units"] = 6
        config.network_parameters["norm"] = "BATCH"
        config.network_parameters["bias"] = False
        config.save["liver_checkpoint"] = "liver_cp_sliding_window"
        config.transforms["mode"] = "3D"
        config.transforms["test_transform"] = "3DUnet_transform"
        config.transforms["post_transform"] = "3DUnet_transform"


class AxialSegmentation2D(Engine):
    """
    A class that must be used when you want to run the
     liver segmentation engine,contains the transforms required
      by the user and the function that is used to start training
    """

    def __init__(self):
        set_configs(mode="2D")
        super().__init__()

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
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(
                        keys[1], threshold=0.5,
                        invert=True, allow_missing_keys=True
                    ),
                    ToTensorD(keys, allow_missing_keys=True),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest")),
                    RandZoomd(keys, prob=0.5, min_zoom=0.8, max_zoom=1.2),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(keys, range_x=1.5, range_y=0,
                                range_z=0, prob=0.5),
                    RandAdjustContrastd(keys[0], prob=0.25),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(
                        keys[1], threshold=0.5, invert=True
                    ),  # remove for lesion segmentation
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
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(
                        keys[1], threshold=0.5,
                        invert=True, allow_missing_keys=True
                    ),
                    ToTensorD(keys, allow_missing_keys=True),
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
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(
                        keys[1], threshold=0.5,
                        invert=True, allow_missing_keys=True
                    ),  # remove for lesion segmentation
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

    def per_batch_callback(self, batch_num, image, label, prediction):
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
        dice_score = dice_metric(prediction.int(), label.int())[0].item()
        plot_2d_or_3d_image(
            data=image,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Volume:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data=label,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Mask:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data=prediction,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Prediction:dice_score:{dice_score}",
        )

    def per_epoch_callback(
        self, epoch, training_loss, valid_loss, training_metric, valid_metric
    ):
        """
        A function which is called every epoch
        to write the results in tensorboard

        Parameters
        ----------
        epoch : int
            The epoch number.
        training_loss : float
            The training loss for the epoch.
        valid_loss : float or None
            The validation loss for the epoch,
             or None if validation was not done.
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

        summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
        summary_writer.add_scalar("\nTraining Metric", training_metric, epoch)

        if valid_loss is not None:
            print(f"Validation Loss={valid_loss}")
            print(f"Validation Metric={valid_metric}")

            summary_writer.add_scalar("\nValidation Loss", valid_loss, epoch)
            summary_writer.add_scalar("\nValidation Metric",
                                      valid_metric, epoch)

    def predict_2dto3d(self, volume_path, temp_path="temp/"):
        """
        predicts the label of a 3D volume using a 2D network
        Parameters
        ----------
        volume_path: str
            path of the input file. expects a 3D nifti file.
        temp_path: str
            a temporary path to save 3d volume as 2d png slices.
             default is "temp/" and is automatically deleted
             before returning the prediction

        Returns
        -------
        tensor
            tensor of the predicted labels
            with shape (1,channel,length,width,depth)
        """
        keys = (config.transforms["img_key"], config.transforms["pred_key"])
        # read volume
        img_volume = SimpleITK.ReadImage(volume_path)
        img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        number_of_slices = img_volume_array.shape[0]
        # create temporary folder to store 2d png files
        if os.path.exists(temp_path) is False:
            os.mkdir(temp_path)
        # write volume slices as 2d png files
        for slice_number in range(number_of_slices):
            volume_silce = img_volume_array[slice_number, :, :]
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[
                -1
            ]  # delete extension from filename
            volume_png_path = (
                os.path.join(temp_path,
                             volume_file_name + "_" + str(slice_number))
                + ".png"
            )
            cv2.imwrite(volume_png_path, volume_silce)
        # predict slices individually then reconstruct 3D prediction
        batch = {keys[1]: self.predict(temp_path)}
        # transform shape from (batch,channel,length,width)
        # to (1,channel,length,width,batch)
        batch[keys[1]] = batch[keys[1]].permute(1, 2, 3, 0).unsqueeze(dim=0)
        # Apply post processing transforms on 3D prediction
        if config.transforms["mode"] == "3D":
            batch = self.post_process(batch, keys[1])
        # delete temporary folder
        shutil.rmtree(temp_path)
        return batch[keys[1]]

    def predict_sliding_window(self, data_dir):
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
                # predict by sliding window
                batch[keys[1]] = sliding_window_inference(
                    volume, (96, 96, 64), 4, self.network
                )
                # Apply post processing transforms on 3D prediction
                if config.transforms["mode"] == "3D":
                    batch = self.post_process(batch, keys[1])
                prediction_list.append(batch[keys[1]])
            prediction_list = torch.cat(prediction_list, dim=0)
        return prediction_list


def segment_liver(*args):
    """
    Segments the liver in 2D images using a 2D liver model.

    Returns
    -------
        tensor
            tensor of the liver segmentations.
    """

    set_seed()
    liver_model = AxialSegmentation2D()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction = liver_model.predict(config.dataset["prediction"])
    return liver_prediction


def segment_liver_3d(*args):
    """
    Segments the liver of a 3D volume using a 2D liver model.
    Parameters
    ----------
    *args: list
        Variable-length argument list.
    Returns
    -------
    tensor
        tensor of the liver segmentations.
    """

    set_seed()
    liver_model = AxialSegmentation2D()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction = liver_model.predict_2dto3d(volume_path=args[0])
    return liver_prediction


def train_liver(*args):
    """
    Trains a liver segmentation model.

    Returns:
        None
    """
    set_seed()
    model = AxialSegmentation2D()
    model.load_data()
    model.data_status()
    # model.load_checkpoint(config.save["potential_checkpoint"])
    print(
        "Initial test loss:", model.test(model.test_dataloader, callback=False)
    )  # FAlSE
    model.fit(
        evaluate_epochs=1,
        batch_callback_epochs=100,
        save_weight=True,
    )
    model.load_checkpoint(
        config.save["potential_checkpoint"]
    )  # evaluate on latest saved check point
    print("final test loss:",
          model.test(model.test_dataloader, callback=False))
