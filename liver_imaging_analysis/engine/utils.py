

### is project


"""
a module that contains supplementary methods used at the beginning/ending of the pipeline
"""
import os
import random
from itertools import permutations
import cv2 as cv
import matplotlib.pyplot as plt
import natsort
import SimpleITK as sitk
from monai.transforms import KeepLargestConnectedComponent
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from monai.transforms import ScaleIntensityRange
from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
from PIL import Image
from liver_imaging_analysis.engine.config import config
import json
import openai
import torch
from monai.transforms import AsDiscrete
plt.switch_backend('Agg') 
rc("animation", html="html5")

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
    # print("Largest Volume = ", max_volume, " In Slice ", idx)

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
    # random.seed(2)
    # perm = permutations(numbers, 3)
    # colors = [color for color in perm]
    # random.shuffle(colors)

    colors = [
    [1.0, 0.0, 0.0],       # Red
    [0.0, 0.0, 1.0],       # Blue
    [0.0, 1.0, 0.0],       # Green
    [1.0, 1.0, 0.0],       # Yellow
    [0.5, 0.0, 0.5],       # Purple
    [1.0, 0.647, 0.0],     # Orange
    [0.0, 0.502, 0.502],   # Teal
    [1.0, 0.753, 0.796],   # Pink
    [0.647, 0.165, 0.165], # Brown
    [0.0, 1.0, 1.0]        # Cyan
    ]
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


def create_image_grid(dir_path, output_filename):
    """
    Creates an image grid from all the JPG files in the specified directory
    and saves the resulting image to a file with the specified name.

    Arguments:
    - dir_path (str): the path to the directory containing the JPG files
    - output_filename (str): the dir and  name of the output file (should end in .png or .jpg)
    """
    # Get a list of all the JPG files in the directory
    jpg_files = os.listdir(dir_path)

    # Determine the number of columns in the figure
    n_files = len(jpg_files)
    if n_files % 2 == 0:
        n_cols = 2
    else:
        n_cols = 3

    # Create a new image with the appropriate dimensions
    img_height = 500
    img_width = int(n_files / n_cols) * img_height
    img_width += 100  # Add 100 pixels of white space to the right
    img = Image.new('RGB', (img_width, img_height * n_cols), color='white')

    # Iterate over the JPG files and paste them into the image
    for i, f in enumerate(jpg_files):
        # Calculate the coordinates of the upper-left corner of the current image
        x = int(i / n_cols) * img_height
        y = (i % n_cols) * img_height
        # Open the current image and paste it into the larger image
        if(f != ".DS_Store"):
            im = Image.open(os.path.join(dir_path, f))
            img.paste(im, (x, y))

    # Save the image as a PNG or JPG file
    img.save(output_filename)

class Overlay:
    """
    Used to visualize the mask overlayed on the volume and saves the output as GIF.
    """

    def __init__(self, volume, mask, mask2_path = None, alpha = 0.2):
        """
        Initializes the variables needed in the class.

        Parameters
        ----------
        volume_path : str
            The path of the volume to be animated.
        mask_path : str
            The path of the mask to be overlaid on the volume and animated.
        alpha : float, optional
            The opacity of the mask. (Default: 0.2)
        """
        self.volume = volume
        self.mask = mask
        self.alpha = alpha
        self.mask2_path = mask2_path

    def gray_to_colored(self):
        """
        Stacks the 1-channel gray volume to a 3-channel RGB volume and overlays the mask by assigning
        a different color to the mask with a reasonable opacity.
        Supports multi-class overlay and multi-label overlay.

        """

        def normalize(arr):
            return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

        
        if self.mask2_path is not None:
            mask2 = nib.load(self.mask2_path).get_fdata()
        else:
            mask2 = None
        mask_label = []
        masks_number = np.unique(self.mask)[1:]
        if mask2 is not None:
            mask_label2 = []
            masks_number2 = np.unique(mask2)[1:0]
        self.dest = np.stack(
            (normalize(self.volume).astype(np.uint8),) * 3, axis=-1
        )  # stacked array of volume

        colors = get_colors([1, .66 ,.33, 0])

        for i, label in enumerate(
            masks_number
        ):  # a loop to iterate over each label in the mask and perform weighted add for each
            # label with a unique color for each one
            mask_label.append(self.mask == label)
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

    def animate(self , output_filename, view ):
        """
        Animates the overlay and saves the output as GIF

        Parameters
        ----------
        view : int
            integer value to select the view of the slicing of the 3D volume 
            where (0 -> sagittal , 1 -> coronal , 2 -> axial)
        
        output_filename : str
            The name of the generated GIF.

        """
        fig = plt.figure(facecolor='black' , figsize=(1,1) )
        # fig = plt.figure(facecolor='black' , figsize=(3,3))
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ims = []

        if view == 0:
            for i in range( self.dest.shape[view]):  # generate an animation over the slices of the array
                im = plt.imshow(self.dest[i, :, :], animated=True)
                ims.append([im])

        elif view == 1:
            for i in range( self.dest.shape[view]):  # generate an animation over the slices of the array
                im = plt.imshow(self.dest[:, i, :], animated=True)
                ims.append([im])

        elif view == 2:
            for i in range( self.dest.shape[view]):  # generate an animation over the slices of the array
                im = plt.imshow(self.dest[:, :, i], animated=True)
                ims.append([im])

        ani = animation.ArtistAnimation(
            fig, ims, interval=200, blit=True, repeat_delay=100
        )
        ani.save(output_filename, dpi=300, writer=PillowWriter(fps=20))

    def generate_animation(self, output_filename, view):
        """
        Used directly to generate and save the overlay animation.

        """
        self.gray_to_colored()
        self.animate(output_filename,view)

    def create_biggest_slice(self, mask, save_path):
        '''
        Creates a PNG image for the slice which has the biggest
        overlayed mask 

        Parameters
        ----------
        mask : 3D tensor
        it's supposed to be an overlayed tensor (mask over volume)
        
        save_path: str
        the path at which the slice will be saved
        '''
        depth_sum = torch.sum(mask, dim=(1,2))
        slice_index = int(torch.argmax(depth_sum))
        print(f"dest shape:{self.dest.shape}")
        new_vol = self.dest.transpose((2, 0, 1,3))
        slice_data = new_vol[slice_index,:,:] # needs to rearrange dims
        # Create a PNG image of the slice data using matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(slice_data)
        plt.axis('off')

        # Save the PNG image to a specific location
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

        # Close the figure to free up memory
        plt.close(fig)


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

    def visualization_mode(self, mode="box", idx=None, plot=True):
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
        from liver_imaging_analysis.engine.visualization import visualize_tumor           # to avoid cyclic importing

        
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

        calculations=visualize_tumor(volume=volume, mask=mask, idx=idx, mode=mode,plot=plot)
        return calculations

visualization = Visualization()

def concatenate_masks(mask1,mask2,volume):
    """
        a function to merge two masks and use them to supress all other regions of the volume
        except the concatenated ROI

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
            conc==1,
            volume,
            volume.min()),#replace all nonliver voxels with background intensity    print(np.unique(vol))
    return vol

def mask_average(volume,mask):
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
    masked=np.multiply(volume,mask)
    masked=masked[masked!=0]
    average=masked.mean()
    return average

def transform_to_hu(path=config.visualization["volume"]):
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
    slope=nifti_img.dataobj.slope
    intercept=nifti_img.dataobj.inter
    print(slope,intercept)
    # Calculate the HU values for each voxel in the image
    HU_data = (img_data * slope) + intercept

    # Print the minimum and maximum HU values in the image
    print(f"Minimum HU value: {HU_data.min()}")
    print(f"Maximum HU value: {HU_data.max()}")
    print(img_data.min(),img_data.max())

    return HU_data


def stringify_dictionary(dictionary, indent=""):
    result = ""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            result += f"{indent}{key}:\n"
            result += stringify_dictionary(value, indent + "  ")
        else:
            result += f"{indent}{key}: {value}\n"
    return result


def generate_report(calculations_dict, max_retries=5):

    calculations=stringify_dictionary(calculations_dict)
    key = "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"
    openai.api_key = key

    delimiter = "####"

    instruction = f"""You are an expert liver radiologist. You will be provided a text with patient vital calculations,\
your role is to analyze the patient calculations, diagnose the patient, and generate a detailed report describing the health status \
of the patient and describing his current health conditions and diagnose him. The text will contain important calculations described below \
like liver volume volume in cm3 , spleen volume in cm3 , lesions volume in cm3 and dimensions mm, lobes volume in cm3 and attenuation ratio with spleen. and LSVR metric which is \
designed to measure the change of shape of liver. Also some diagnosis guidelines will be provided below to help .

Guidelines:-
-If Liver volume is larger than 1671 this is an indication for liver enlargement
-If Spleen volume is larger than 300 this is an indication for spleen enlargement and cirhosis
-If Liver/Spleen attenuation ratio is less than 1, this is an indication for fatty liver diseases like NAFLD or NASH.
-If the lesions diameter is largen than 10 or and volume are large, this is an indication for malignant tumor.
-If the Lobe/Spleen attenuation is less than 1, this is an indication for excess fat in the lobe.
-If the LSVR metric is higher than 0.24, this is an indication for Liver Cirrhosis.

You must follow the following instructions in the report
Instructions:-
-You shouldn't stick to these guidelines only, make more diagnosis beyond these guidelines to cover everything.
-Don't mention a lot of numbers in the report, focus more on giving an overview of the patient status and making diagnosis, \
and decisions about the patient health conditions.
-Analyze carefully the calculations and make an expert diagnosis and generate expert detailed report

Note that the calculations will be delimited by four hashtags, i.e {delimiter}.
"""
    
    message = f"""I will provide the patient calculation, analyze them carefully and diagnose the patient. 
Make sure you follow the instructions well and make an accurate diagnosis, don't mention a lot of numbers in the report

Calculations : {calculations}

Patient Report :
"""

    retries = 0
    tokens = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": message}
                ],
                temperature=0.15
            )

            tokens = response['usage']['total_tokens']
            result = response['choices'][0]['message']['content']
            return result, tokens
        except:
            if retries>=max_retries:
                return None, tokens
            else:
                retries+=1

class Report:
    
    def __init__(self,volume,mask=None,lobes_mask=None,spleen_mask=None):

        self.volume=volume
        self.volume_hu_transformed=transform_to_hu()
        self.liver_mask=np.where(mask == 1, 1, 0)
        self.lesions_mask=np.where(mask == 2, 1, 0)
        self.lobes_mask=lobes_mask
        self.spleen_mask=spleen_mask
        self.x, self.y, self.z = find_pix_dim()

    def liver_analysis(self):

        self.liver_volume=np.unique(self.liver_mask, return_counts=True)[1][1]*self.x*self.y*self.z/1000
        
        self.liver_attenuation=mask_average(self.volume_hu_transformed,self.liver_mask)
    
    def lesions_analysis(self):
        self.lesions_calculations=visualization.visualization_mode(mode="contour",plot=False)

    def lobes_analysis(self):
        values,total_pixels=np.unique(self.lobes_mask ,return_counts=True)
        values,total_pixels=values[1:],total_pixels[1:]
        self.lobes_volumes=total_pixels*self.x*self.y*self.z/1000
        
        self.lobes_average = [mask_average(volume=self.volume_hu_transformed, mask=np.where(self.lobes_mask==i, 1, 0)) 
                                for i in values]
        self.metric=np.sum(self.lobes_volumes[:3])/np.sum(self.lobes_volumes[3:])

    def spleen_analysis(self):
        self.spleen_volume=np.unique(self.spleen_mask, return_counts=True)[1][1]*self.x*self.y*self.z/1000
        self.spleen_attenuation=mask_average(self.volume_hu_transformed,self.spleen_mask)
    
    def build_report(self):
  
        report = {}

        if(self.spleen_mask is not None):
            self.spleen_analysis()
            report["Spleen Volume"]=self.spleen_volume
            report["Spleen Attenuation"]= self.spleen_attenuation  

        if(self.liver_mask is not None):
            self.liver_analysis()
            report["Liver Volume"]= self.liver_volume
            if(self.spleen_mask is not None):
                report["Liver/Spleen Attenuation Ratio"]= self.liver_attenuation/self.spleen_attenuation  
            else:
                report["Liver Attenuation"]= self.liver_attenuation


        if(self.lesions_mask is not None):
            self.lesions_analysis()
            lesions={}
            for i,calc in enumerate(self.lesions_calculations):
                if(calc[0]>calc[1]):

                    lesions[f"{i}"]={"Major Axis":calc[0], "Minor Axis":calc[1], "Volume":calc[2]}
                else:
                    lesions[f"{i}"]={"Major Axis":calc[1], "Minor Axis":calc[0], "Volume":calc[2]}
            report["Lesions Information"]=lesions

        if(self.lobes_mask is not None):
            self.lobes_analysis()
            lobes_volume={}
            lobes_attenuation={}

            for i,volume in enumerate(self.lobes_volumes):
                lobes_volume[f" Lobe {i+1} "]= volume
            if(self.spleen_mask is not None):
                for i,attenuation in enumerate(self.lobes_average):
                    lobes_attenuation[f" Lobe {i+1} "]= attenuation/self.spleen_attenuation
            else:
                for i,attenuation in enumerate(self.lobes_average):
                    lobes_attenuation[f" Lobe {i+1} "]= attenuation                  

            report["Each Lobe Volume"]= lobes_volume
            report["Each Lobe/Spleen Attenuation Ratio"]= lobes_attenuation
            report["LSVR Metric"]=self.metric

            msg,tokens=generate_report(report)
            report["msg"]=msg
        with open("report.json", "w") as json_file:
            json.dump(report, json_file, indent=4)
        return report

def round_dict(d,n = 2):
    """
    Rounds all the numbers inside a dictionary to the nearest 2 digits

    Args:
        d : dict 
        the input dictionary
        m : int 
        number of digits to round at , default to 2

    Returns:
        d : dict
        the rounded dictionary
    """
    for k, v in d.items():
        if isinstance(v, dict):
            round_dict(v, n)
        elif isinstance(v, float):
            d[k] = round(v, n)

    return d

