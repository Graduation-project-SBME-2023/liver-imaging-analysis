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
from numpy.random import randint
import SimpleITK as sitk
from monai.transforms import ScaleIntensityRange
from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
import cv2
from liver_imaging_analysis.engine.config import config

from monai.transforms import AsDiscrete

import logging
logger = logging.getLogger(__name__)

rc("animation", html = "html5")

def concatenate_masks(mask1, mask2, volume):
    """
        a function to merge two masks and use them to supress 
        all other regions of the volume except the concatenated ROI

        Parameters
        ----------
        mask1: np array
            array contains first mask data
        mask2: np array
            array contains second mask data
        volume: np array
            array contains the original volume data
        Return
        ----------
        vol: array
            the volume with specified ROI values only
    """
    assert mask1.shape == mask2.shape == volume.shape, "Input shapes do not match"
    # Use logical OR to concatenate the masks and threshold the result to obtain a binary mask
    conc = np.logical_or(mask1, mask2).astype(np.uint8)
    # Multiply the binary mask with the original volume to obtain the segmented parts
    vol=np.where(
            conc == 1,
            volume,
            volume.min()
            ),#replace all background voxels with background intensity
    return vol

def mask_average(volume, mask):
    """
        a function to find the average value of a specific ROI in volume
        Parameters
        ----------
        volume: np array
            array contains the volume data
        mask: np array
            array contains the mask data ( ROI )
        Return
        ----------
        average: float
            the average value of the ROI
    """
    masked = np.multiply(volume, mask)
    masked = masked[masked != 0]
    average = masked.mean()
    return average

def transform_to_hu(path):
    """
        a function to transform the input nfti to its Hounsfield values
        Parameters
        ----------
        path: string
            the path of the input nfti file
        Return
        ----------
        HU_data: numpy array
            the array contains the data of input volume calibrated to Hounsfield 
    """
    nifti_img = nib.load(path)
    # Extract the image data as a numpy array
    img_data = nifti_img.get_fdata()
    # Get the slope and intercept values for the Hounsfield unit (HU) calculation
    slope = nifti_img.dataobj.slope
    intercept = nifti_img.dataobj.inter
    print(slope, intercept)
    # Calculate the HU values for each voxel in the image
    HU_data = (img_data * slope) + intercept
    # Print the minimum and maximum HU values in the image
    print(f"Minimum HU value: {HU_data.min()}")
    print(f"Maximum HU value: {HU_data.max()}")
    print(img_data.min(), img_data.max())
    logger.info(f"Transforming {path} to HU")
    logger.debug(f"Slope: {slope}, Intercept: {intercept}")
    logger.debug(f"Minimum voxel value: {img_data.min()}")
    logger.debug(f"Maximum voxel value: {img_data.max()}")
    logger.debug(f"Shape: {img_data.shape}")
    return HU_data

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
    logger.debug(f"\r|{bar}| {percent: .2f}%", end=f"  ---> {progress}/{total}")


def liver_isolate_crop(
        volumes_path,
        masks_path,
        new_volumes_path,
        new_masks_path
        ):
    """
        A method to crop liver volumes and masks in z direction 
        then isolate liver and lesions from abdomen
        Parameters
        ----------
        volumes_path: str
            the directory of the 3d volumes
        masks_path: str
            the directory of the 3d masks
        new_volumes_path: str
            the save directory of the cropped liver volume
        new_masks_path: str
            the save directory of the cropped lesions mask
    """
    volume_files = natsort.natsorted(
        os.listdir(volumes_path)
    )  # sort the directory of files
    mask_files = natsort.natsorted(os.listdir(masks_path))
    for i in range(len(volume_files)):
        volume_path = os.path.join(volumes_path, volume_files[i])
        mask_path = os.path.join(masks_path, mask_files[i])
        volume = nib.load(volume_path)
        mask = nib.load(mask_path)
        volume_array = volume.get_fdata()
        mask_array = mask.get_fdata() 
        min_slice = 0
        max_slice = 0
        for j in range (mask_array.shape[2]):
          if(len(np.unique(mask_array[:, :, j])) != 1):
            min_slice = j
            break
        for k in range(mask_array.shape[2]-1, -1, -1):
          if(len(np.unique(mask_array[:,:,k])) != 1):
            max_slice = k
            break
        volume_array = volume_array[:, :, min_slice : max_slice + 1]
        mask_array = mask_array[:, :, min_slice : max_slice + 1]
        new_volume = nib.Nifti1Image(
            np.where(
            (mask_array.astype(int) > 0.5).astype(int) == 1,
            volume_array.astype(int),
            volume_array.astype(int).min()), # replace nonliver voxels with background intensity
            affine = volume.affine,
            header = volume.header
            )
        new_mask = nib.Nifti1Image(
            (mask_array.astype(int) > 1.5).astype(int), # new mask contains lesions only
            affine = mask.affine,
            header = mask.header
            )
        nii_volume_path = (os.path.join(new_volumes_path, volume_files[i]))
        nii_mask_path = (os.path.join(new_masks_path, mask_files[i]))
        new_volume.to_filename(nii_volume_path)  # Save as NiBabel file
        new_mask.to_filename(nii_mask_path)  # Save as NiBabel file
        

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


def calculate_largest_tumor(mask):
    """
    Get the index of the slice with the largest tumor volume.

    Parameters:
    mask (np.array): The tumor mask where 0 represents the background and 1 represents the tumor.

    Returns:
    idx (int): The index of the slice with the largest tumor volume.
    """
    max_volume = -1
    idx = -1
    x, y, z = find_pix_dim(path=config.visualization["volume"])
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
    logger.debug(f"Largest Volume = {max_volume} In Slice {idx}") 

    return idx


def get_colors(numbers=[1, 0.5, 0]):
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


def find_pix_dim(path=config.visualization["volume"]):
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


class Overlay:
    """
    Used to visualize the mask overlayed on the volume and saves the output as GIF.
    """

    def __init__(self, volume_path, mask_path, output_name, mask2_path=None, alpha=0.2):
        """
        Initializes the variables needed in the class.

        Parameters
        ----------
        volume_path : str
            The path of the volume to be animated.
        mask_path : str
            The path of the mask to be overlaid on the volume and animated.
        output_name : str
            The name of the generated GIF overlay.
        alpha : float, optional
            The opacity of the mask. (Default: 0.2)
        """
        self.volume_path = volume_path
        self.mask_path = mask_path
        self.alpha = alpha
        self.output_name = output_name
        self.mask2_path = mask2_path

    def gray_to_colored(self):
        """
        Stacks the 1-channel gray volume to a 3-channel RGB volume and overlays the mask by assigning
        a different color to the mask with a reasonable opacity.
        Supports multi-class overlay and multi-label overlay.

        """

        def normalize(arr):
            return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        volume = nib.load(self.volume_path).get_fdata()
        mask = nib.load(self.mask_path).get_fdata()
        if self.mask2_path is not None:
            mask2 = nib.load(self.mask2_path).get_fdata()
        else:
            mask2 = None
        mask_label = []
        masks_number = np.unique(mask)[1:]
        if mask2 is not None:
            mask_label2 = []
            masks_number2 = np.unique(mask2)[1:0]
        self.dest = np.stack(
            (normalize(volume).astype(np.uint8),) * 3, axis=-1
        )  # stacked array of volume

        colors = get_colors()

        for i, label in enumerate(
            masks_number
        ):  # a loop to iterate over each label in the mask and perform weighted add for each
            # label with a unique color for each one
            mask_label.append(mask == label)
            mask_label[i] = np.stack((mask_label[i],) * 3, axis=-1)
            mask_label[i] = np.multiply(
                (mask_label[i].astype(np.uint8) * 255), colors[i]
            ).astype(np.uint8)
            self.dest = cv.addWeighted(self.dest, 1, mask_label[i], self.alpha, 0.0)
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
                self.dest = cv.addWeighted(
                    self.dest, 1, mask_label2[i], self.alpha, 0.0
                )

    def animate(self):
        """
        Animates the overlay and saves the output as GIF

        """
        fig = plt.figure()
        ims = []
        for i in range(
            self.dest.shape[2]
        ):  # generate an animation over the slices of the array
            plt.axis("off")
            im = plt.imshow(self.dest[:, :, i], animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=100
        )
        ani.save(self.output_name, dpi=300, writer=PillowWriter(fps=5))

    def generate_animation(self):
        """
        Used directly to generate and save the overlay animation.

        """
        self.gray_to_colored()
        self.animate()


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
    to config['visualization], the mask should be labeled 0 for background, 1 for liver, 2 for tumor.
    """

    def visualization_mode(self, mode="box", idx=None):
        """
        Choose the visualization mode for tumors and load volume, mask, and preprocess them.

        Parameters:
        mode (str): The visualization mode. Available modes are:
            - 'box': Draw a bounding box around the tumor.
            - 'contour': Draw a contour around the tumor and display the longest and shortest diameters.
            - 'zoom': Draw a bounding box, zoom, and display the longest and shortest diameters on tumors.


        idx (int): If not None, it represents the index of a specific slice in the volume to execute code on.
            If None, the code will be executed for all slices and all tumors.
        """
        from visualization import visualize_tumor           # to avoid cyclic importing

        
        volume = nib.load(config.visualization["volume"]).get_fdata()
        volume = ScaleIntensityRange(
            a_min=-135,
            a_max=215,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )(volume)
        mask = nib.load(config.visualization["mask"]).get_fdata()
        mask = AsDiscrete(threshold=1.5)(mask)  

        visualize_tumor(volume, mask, idx, mode)


visualization = Visualization()
