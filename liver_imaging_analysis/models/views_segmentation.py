"""
This module contains the implementation of ViewSegmentation, which combines the three views volumes
into one using a voting technique.

Contents:
- ViewSegmentation
  - slice
  - delete_temp_slices
  - reorder_dims
  - get_segmentation
"""
import os
import shutil
import torch
import numpy as np
from monai.transforms import Resize, EnsureChannelFirst, Compose, AddChannel
from liver_segmentation import LiverSegmentation
from engine.utils import nii2png_XYZ
from engine.config import config




class ViewSegmentation:
    """
    A class that provides various utility functions for combining
    three views volumes into one using a voting technique.
    """
    def __init__(self):
        self.slices_2d_path = config.dataset['slices_2d_path']
        self.orig_size = None

    def slice(self, volume_nii_path):
        """
        Slices a 3D volume into 2D slices and saves them to disk.

        Parameters
        ----------
        volume_nii_path : str
            The path to the 3D volume to slice.

        Returns
        ----------
        None
        """

        if os.path.exists(self.slices_2d_path) is False:
            os.mkdir(self.slices_2d_path)

        self.orig_size = nii2png_XYZ(
            "xy",
            volume_nii_path=volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_axial/",
        )
        self.orig_size = nii2png_XYZ(
            "xz",
            volume_nii_path=volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_coronal/",
        )

        self.orig_size = nii2png_XYZ(
            "yz",
            volume_nii_path=volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_sagittal/",
        )

    def delete_temp_slices(self):
        """
        Deletes the temporary 2D slices saved to disk.

        Returns
        ----------
        None
        """
        shutil.rmtree(self.slices_2d_path)

    def reorder_dims(self, plane, predicted_mask):
        """
        Reorders a predicted mask to return it to its original
         orientation based on the specified plane.

        Parameters
        ----------
        plane : str
            The plane of the predicted mask (valid values are "xy", "xz", or "yz").
        predicted_mask : np.ndarray
            The predicted mask to reorder.

        Returns
        ----------
        torch.Tensor
            The reordered predicted mask as a PyTorch tensor.
        """

        if plane == "xy":
            predicted_mask = np.transpose(
                predicted_mask, (1, 0, 2, 3)
            )  # axial reshaping
            predicted_mask = torch.from_numpy(predicted_mask)
            predicted_mask = predicted_mask[0, :, :, :]
            predicted_mask = torch.flip(predicted_mask, [1])
            predicted_mask = torch.rot90(predicted_mask, dims=[2, 1])

        elif plane == "xz":
            predicted_mask = np.transpose(
                predicted_mask, (1, 2, 0, 3)
            )  # coronal reshaping
            predicted_mask = torch.from_numpy(predicted_mask[0, :, :, :])
            predicted_mask = torch.rot90(
                predicted_mask, dims=[2, 0]
            )  # assuming zyx , correct
            predicted_mask = torch.flip(predicted_mask, [1])
            predicted_mask = torch.rot90(
                predicted_mask, dims=[1, 2], k=2
            )  # assuming zyx , correct

        elif plane == "yz":
            predicted_mask = np.transpose(
                predicted_mask, (1, 2, 3, 0)
            )  # saggital reshaping
            predicted_mask = torch.from_numpy(predicted_mask[0, :, :, :])
            predicted_mask = torch.rot90(
                predicted_mask, dims=[1, 0]
            )  # assuming zyx , correct
            predicted_mask = torch.flip(predicted_mask, [2])
            predicted_mask = torch.rot90(predicted_mask, dims=[2, 1], k=2)

        return predicted_mask

    def get_segmentation(self):
        """
        Uses three 2D segmentation models to generate a 3D segmentation of a volume.

        Returns
        ----------
        np.ndarray
            A 3D numpy array representing the segmented volume.
        """

        sagittal_plane = "yz"
        sagittal_model = LiverSegmentation(view="sagittal")
        sagittal_model.load_checkpoint(
            "/content/drive/MyDrive/voting_updated/engine/sagittal_checkpoint"
        )
        sagittal_prediction = sagittal_model.predict(
            f"{self.slices_2d_path}/volume_sagittal"
        )
        sagittal_prediction = self.reorder_dims(sagittal_plane,
                                                sagittal_prediction)

        coronal_plane = "xz"
        coronal_model = LiverSegmentation(mode="coronal")
        coronal_model.load_checkpoint(
            "/content/drive/MyDrive/voting_updated/engine/"
            "Copy of coronal_checkpoint"
        )
        coronal_prediction = coronal_model.predict(
            f"{self.slices_2d_path}/volume_coronal"
        )
        coronal_prediction = self.reorder_dims(coronal_plane,
                                               coronal_prediction)

        axial_plane = "xy"
        axial_model = LiverSegmentation(mode = "voting_axial")
        axial_model.load_checkpoint(
            "/content/drive/MyDrive/voting_updated/engine/axial_checkpoint"
        )
        axial_prediction =\
            axial_model.predict(f"{self.slices_2d_path}/volume_axial")
        axial_prediction = self.reorder_dims(axial_plane, axial_prediction)

        resize_to_original_shape = Compose(
            [
                AddChannel(),
                EnsureChannelFirst(channel_dim=0),
                Resize(self.orig_size, mode="nearest-exact"),
            ]
        )

        axial_prediction = resize_to_original_shape(axial_prediction)
        coronal_prediction = resize_to_original_shape(coronal_prediction)
        sagittal_prediction = resize_to_original_shape(sagittal_prediction)
        assert (
            coronal_prediction.shape
            == sagittal_prediction.shape
            == axial_prediction.shape
        )

        volume_3d = (
            coronal_prediction[0, :, :, :]
            + sagittal_prediction[0, :, :, :]
            + axial_prediction[0, :, :, :]
        )
        volume_3d[volume_3d < 2] = 0
        volume_3d[volume_3d >= 2] = 1

        return volume_3d
