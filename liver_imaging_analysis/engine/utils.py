"""
a module that contains supplementary methods used at the beginning/ending of the pipeline
"""
import os
from itertools import permutations
import cv2 as cv
import matplotlib.pyplot as plt
import natsort
import SimpleITK as sitk
from monai.transforms import KeepLargestConnectedComponent
import nibabel as nib
import numpy as np
from monai.transforms import ScaleIntensityRange
from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
from config import config

from monai.transforms import AsDiscrete

rc("animation", html="html5")


def progress_bar(progress, total):
    """
    A method to visualize the training progress by a progress bar
    Parameters
    ----------
    progress: float
        the current batch
    total: float
        the total number of batches
    """
    percent = 100 * (progress / float(total))
    bar = "#" * int(percent) + "_" * (100 - int(percent))
    print(f"\r|{bar}| {percent: .2f}%", end=f"  ---> {progress}/{total}")


def get_batch_names(batch, key):
    """
    A method to get the filenames of the current batch
    Parameters
    ----------
    batch: tensor
        the current batch dict
    key: str
        the key of the batch dict
    """
    return batch[f"{key}_meta_dict"]["filename_or_obj"]


class Overlay:
    """
    a class used to visualize the mask overlayed on the volume and saves output as GIF.
    """
    def gray_to_colored(VolumePath, MaskPath, alpha=0.2):
        """
        A method to generate the volume and the mask overlay
        Parameters
        ----------
        Volume Path: str
            the directory that includes the volume nii file
        Mask Path: str
            the directory that includes the segmented mask nii file
        alpha: float
            the opacity of the displayed mask. default=0.2
        Returns
        -------
        tensor
            The Stacked 4 channels array of the nifti input
        """

        def normalize(arr):
            return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        volume = nib.load(VolumePath).get_fdata()
        mask = nib.load(MaskPath).get_fdata()
        mask_label = []
        masksNo = np.unique(mask)[1:]
        dest = np.stack(
            (normalize(volume).astype(np.uint8),) * 3, axis=-1
        )  # stacked array of volume
        numbers = [0, 0.5, 1]
        perm = permutations(numbers)
        colors = [color for color in perm]
        for i, label in enumerate(
            masksNo
        ):  # a loop to iterate over each label in the mask and perform weighted add for each
            # label with a unique color for each one
            mask_label.append(mask == label)
            mask_label[i] = np.stack((mask_label[i],) * 3, axis=-1)
            mask_label[i] = np.multiply(
                (mask_label[i].astype(np.uint8) * 255), colors[i]
            ).astype(np.uint8)
            dest = cv.addWeighted(dest, alpha, mask_label[i], alpha, 0.0)
        return dest  # return an array of the volume with the mask overlayed on it with different label colors

    def animate(volume, output_name):
        """
        A method to save the animated gif from the overlay array
        Parameters
        ----------
        volume: tensor
            expects a 4d array of the volume/mask overlay
        output_name: str
            the name of the gif file to be saved
        """
        fig = plt.figure()
        ims = []
        for i in range(
            volume.shape[2]
        ):  # generate an animation over the slices of the array
            plt.axis("off")
            im = plt.imshow(volume[:, :, i], animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=100
        )
        ani.save(output_name, dpi=300, writer=PillowWriter(fps=5))

    def gray_to_colored_from_array(Volume, Mask, mask2=None, alpha=0.2):
        """
        A method to generate the volume and the mask overlay from arrays
        Parameters
        ----------
        Volume: tensor
            the volume array
        Mask: tensor
            the mask array
        mask2: tensor
            optional additional mask to be overlayed. default is None
        alpha: float
            the opacity of the displayed mask. default=0.2
        Returns
        ----------
        tensor
            The Stacked 4 channels array of the nifti input
        """

        def normalize(arr):
            return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        mask_label = []
        masks_number = np.unique(Mask)[1:]
        if mask2 is not None:
            mask_label2 = []
            masks_number2 = np.unique(mask2)[1:0]
        dest = np.stack(
            (normalize(Volume).astype(np.uint8),) * 3, axis=-1
        )  # stacked array of volume

        numbers = [0, 0.5, 1]
        perm = permutations(numbers)
        colors = [color for color in perm]

        for i, label in enumerate(
            masks_number
        ):  # a loop to iterate over each label in the mask and perform weighted add for each
            # label with a unique color for each one
            mask_label.append(Mask == label)
            mask_label[i] = np.stack((mask_label[i],) * 3, axis=-1)
            mask_label[i] = np.multiply(
                (mask_label[i].astype(np.uint8) * 255), colors[i]
            ).astype(np.uint8)
            dest = cv.addWeighted(dest, 1, mask_label[i], alpha, 0.0)
        if mask2 is not None:
            colors = np.flip(colors)
            for i, label in enumerate(
                masks_number2
            ):  # a loop to iterate over each label in the mask and perform weighted add for each
                # label with a unique color for each one
                mask_label2.append(mask2 == label)
                mask_label2[i] = np.stack((mask_label2[i],) * 3, axis=-1)
                mask_label2[i] = np.multiply(
                    (mask_label2[i].astype(np.uint8) * 255), colors[i]
                ).astype(np.uint8)
                dest = cv.addWeighted(dest, 1, mask_label2[i], alpha, 0.0)

        return dest  # return an array of the volume with the mask overlayed on it with different label colors


class VolumeSlicing:
    """
    a class used to call different functions to divide 3D Nfti files to 2D images, Nfti files .
    """
    def nii2png(volume_nii_path, mask_nii_path, volume_save_path, mask_save_path):
        """
        A method to generate 2d .png slices from 3d .nii volumes
        Parameters
        ----------
        volume_nii_path: str
            the directory of the 3d volumes
        mask_nii_path: str
            the directory of the 3d masks
        volume_save_path: str
            the save directory of the 2d volume slices
        mask_save_path: str
            the save directory of the 2d mask slices
        """
        volume_folders = natsort.natsorted(
            os.listdir(volume_nii_path)
        )  # sort the directory of files
        mask_folders = natsort.natsorted(os.listdir(mask_nii_path))

        for i in range(len(volume_folders)):

            volume_path = os.path.join(volume_nii_path, volume_folders[i])
            mask_path = os.path.join(mask_nii_path, mask_folders[i])

            img_volume = sitk.ReadImage(volume_path)
            img_mask = sitk.ReadImage(mask_path)

            img_volume_array = sitk.GetArrayFromImage(img_volume)
            img_mask_array = sitk.GetArrayFromImage(img_mask)

            number_of_slices = img_volume_array.shape[0]

            for slice_number in range(number_of_slices):

                volume_silce = img_volume_array[slice_number, :, :]
                mask_silce = img_mask_array[slice_number, :, :]

                volume_file_name = os.path.splitext(volume_folders[i])[
                    0
                ]  # delete extension from filename
                mask_file_name = os.path.splitext(mask_folders[i])[
                    0
                ]  # delete extension from filename

                # name =  "defaultNameWithoutExtention_sliceNum.png"
                volume_png_path = (
                    os.path.join(
                        volume_save_path, volume_file_name + "_" + str(slice_number)
                    )
                    + ".png"
                )
                mask_png_path = (
                    os.path.join(
                        mask_save_path, mask_file_name + "_" + str(slice_number)
                    )
                    + ".png"
                )

                cv.imwrite(volume_png_path, volume_silce)
                cv.imwrite(mask_png_path, mask_silce)

    def nii3d_To_nii2d(
        volume_nii_path, mask_nii_path, volume_save_path, mask_save_path
    ):
        """
        A method to generate 2d .nii slices from 3d .nii volumes
        Parameters
        ----------
        volume_nii_path: str
            the directory of the 3d volumes
        mask_nii_path: str
            the directory of the 3d masks
        volume_save_path: str
            the save directory of the 2d volume slices
        mask_save_path: str
            the save directory of the 2d mask slices
        """
        volume_folders = natsort.natsorted(
            os.listdir(volume_nii_path)
        )  # sort the directory of files
        mask_folders = natsort.natsorted(os.listdir(mask_nii_path))

        for i in range(len(volume_folders)):

            volume_path = os.path.join(volume_nii_path, volume_folders[i])
            mask_path = os.path.join(mask_nii_path, mask_folders[i])

            img_volume = sitk.ReadImage(volume_path)
            img_mask = sitk.ReadImage(mask_path)

            img_volume_array = sitk.GetArrayFromImage(img_volume)
            img_mask_array = sitk.GetArrayFromImage(img_mask)

            number_of_slices = img_volume_array.shape[0]

            for slice_number in range(number_of_slices):

                volume_silce = img_volume_array[slice_number, :, :]
                mask_silce = img_mask_array[slice_number, :, :]

                volume_file_name = os.path.splitext(volume_folders[i])[
                    0
                ]  # delete extension from filename
                mask_file_name = os.path.splitext(mask_folders[i])[
                    0
                ]  # delete extension from filename

                # nameConvention =  "defaultNameWithoutExtention_sliceNum.nii.gz"
                nii_volume_path = (
                    os.path.join(
                        volume_save_path, volume_file_name + "_" + str(slice_number)
                    )
                    + ".nii.gz"
                )
                nii_mask_path = (
                    os.path.join(
                        mask_save_path, mask_file_name + "_" + str(slice_number)
                    )
                    + ".nii.gz"
                )

                new_nii_volume = nib.Nifti1Image(
                    volume_silce, affine=np.eye(4)
                )  # ref : https://stackoverflow.com/questions/28330785/creating-a-nifti-file-from-a-numpy-array
                new_nii_mask = nib.Nifti1Image(mask_silce, affine=np.eye(4))

                nib.save(new_nii_volume, nii_volume_path)
                nib.save(new_nii_mask, nii_mask_path)


class Visualization:
    """
    a class used to call different visualization functions on tumors, volume and mask paths should be added
    to config['visualization], the mask should be labeled 0 for background, 1 for tumor.
    """
    
    def visualization_mood(self, mode='box',idx=None):
        """
        choose the visualization mood of tumor and load volume,mask and preprocess them. 
        ----------

        mode: str
            the visualization mood. available moods are:-
            'box': draw a bounding box arround tumor,
            'contour': draw a contour around tumor and draw the longest,shortest diameters.
            'zoom': draw bounding box, zoom, and draw longest,shortest diameters on tumors.
        idx: int
            if not None, it represents the index of a specific slice in the volume to execute
            code on. if None, the code will be executed for all slices and all tumors
        """
        # idx = self.calculate_largest_tumor(mask)
        # idx = None

        volume = nib.load(config.visualization["volume"]).get_fdata()
        volume = ScaleIntensityRange(
            a_min=-135,
            a_max=215,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )(volume)
        mask = nib.load(config.visualization["mask"]).get_fdata()
        mask = AsDiscrete(threshold=1.5)(mask)  # FIXED LATER 

        from visualization import visualize_tumor  # to avoid cyclic importing
        visualize_tumor(volume, mask, idx, mode)

    def calculate_largest_tumor(self, mask):
        """
        get the slice with largest tumor volume
        ----------

        mask: np array
            the tumors mask (0: background, 1: tumor)
        Returns
        -------
        idx: int
            index of the slice with largest volume
        """
        max_volume = -1
        idx = -1
        x, y, z = self.find_pix_dim(path=config.visualization["volume"])
        clone = mask.clone()
        largest_tumor = KeepLargestConnectedComponent()(clone)
        for i in range(largest_tumor.shape[-1]):
            slice = largest_tumor[:, :, i]
            if slice.any() == 1:
                count = np.unique(slice, return_counts=True)[1][1]
                if count > max_volume:
                    max_volume = count
                    idx = i
        max_volume = max_volume * x * y * z
        print("Largest Volume = ", max_volume, " In Slice ", idx)

        return idx

    def get_colors(self, numbers=[1, 0.5, 0]):
        """
        calculate a list with unique colors
        Parameters
        ----------
        colors: list
            the r,g,b values to be permuted
        Returns
        -------
        colors: list
            list with all possible permutations of rgb values
        """

        perm = permutations(numbers)
        colors = [color for color in perm]
        return colors

    def find_pix_dim(self, path=config.visualization['volume']):
        """
        calculate the pixel dimensions in mm in the 3 axes

        Parameters
        ----------
        path: str
            path of the input directory. expects nifti file.
        Returns
        -------
        list
            list if mm dimensions of 3 axes (L,W,N)
        """
        volume = nib.load(path)
        pix_dim = volume.header["pixdim"][1:4]
        pix_dimx = pix_dim[0]
        pix_dimy = pix_dim[1]
        pix_dimz = pix_dim[2]

        return [pix_dimx, pix_dimy, pix_dimz]


visualization = Visualization()
