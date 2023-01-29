from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
from itertools import permutations
import os
import cv2 as cv
import matplotlib.pyplot as plt
import natsort
import nibabel as nib
import numpy as np
import SimpleITK as sitk


rc("animation", html="html5")


"""
    A method to display the original CT volume and the segmented mask and its labels
    with different color and opacity on it and saves the overlay into a gif file
    Methods:
        gray_to_colored: changes the input nfti from 3 channels gray scale ( L , W , N)
         to 4 channels RGB ( L , W , N , 3) by stacking the volume array
          and perform weighted add to put the segmented mask over the volume in one array
            Args:   Volume Path: the directory that includes the volume nii file
                    Mask Path: the directory that includes the segmented mask nii file
                    alpha: the opacity of the displayed mask
            Return: The Stacked 4 channels array of the nfti input
        normalize: normalize the input value to be in range 0:255
        animate: create the animated overlay and saves it as GIF
            Args: volume: the required array input to be animated
                  volumename: the name of the output gif file to be saved
"""


def gray_to_colored(VolumePath, MaskPath, alpha=0.2):
    def normalize(arr):
        return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    Volume = nib.load(VolumePath).get_fdata()
    Mask = nib.load(MaskPath).get_fdata()
    Masklabel = []
    masksNo = np.unique(Mask)[1:]
    dest = np.stack(
        (normalize(Volume).astype(np.uint8),) * 3, axis=-1
    )  # stacked array of volume
    numbers = [0, 0.5, 1]
    perm = permutations(numbers)
    colors = [color for color in perm]
    for i, label in enumerate(
        masksNo
    ):  # a loop to iterate over each label in the mask and perform weighted add for each
        # label with a unique color for each one
        Masklabel.append(Mask == label)
        Masklabel[i] = np.stack((Masklabel[i],) * 3, axis=-1)
        Masklabel[i] = np.multiply(
            (Masklabel[i].astype(np.uint8) * 255), colors[i]
        ).astype(np.uint8)
        dest = cv.addWeighted(dest, alpha, Masklabel[i], alpha, 0.0)
    return dest  # return an array of the volume with the mask overlayed on it with different label colors


def animate(volume, outputName):
    fig = plt.figure()
    ims = []
    for i in range(
        volume.shape[2]
    ):  # generate an animation over the slices of the array
        plt.axis("off")
        im = plt.imshow(volume[:, :, i], animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=100)
    ani.save(outputName, dpi=300, writer=PillowWriter(fps=5))


def gray_to_colored_from_array(Volume, Mask, mask2=None, alpha=0.2):
    def normalize(arr):
        return 255 * (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    Masklabel = []
    masksNo = np.unique(Mask)[1:]
    if mask2 != None:
        mask_label2 = []
        masks_number2 = np.unique(mask2)[1:0]
    dest = np.stack(
        (normalize(Volume).astype(np.uint8),) * 3, axis=-1
    )  # stacked array of volume

    numbers = [0, 0.5, 1]
    perm = permutations(numbers)
    colors = [color for color in perm]

    for i, label in enumerate(
        masksNo
    ):  # a loop to iterate over each label in the mask and perform weighted add for each
        # label with a unique color for each one
        Masklabel.append(Mask == label)
        Masklabel[i] = np.stack((Masklabel[i],) * 3, axis=-1)
        Masklabel[i] = np.multiply(
            (Masklabel[i].astype(np.uint8) * 255), colors[i]
        ).astype(np.uint8)
        dest = cv.addWeighted(dest, 1, Masklabel[i], alpha, 0.0)
    if mask2 != None:
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


def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = "#" * int(percent) + "_" * (100 - int(percent))
    print(f"\r|{bar}| {percent: .2f}%", end=f"  ---> {progress}/{total}")


def nii2png(volume_nii_path, mask_nii_path, volume_save_path, mask_save_path):

    """
    generates slices(with png extensions) from volumes (with nii extensions)
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
                os.path.join(mask_save_path, mask_file_name + "_" + str(slice_number))
                + ".png"
            )

            cv.imwrite(volume_png_path, volume_silce)
            cv.imwrite(mask_png_path, mask_silce)


def nii3d_To_nii2d(volume_nii_path, mask_nii_path, volume_save_path, mask_save_path):

    """
    generates slices (with nii extensions) from volumes (with nii extensions)
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
                os.path.join(mask_save_path, mask_file_name + "_" + str(slice_number))
                + ".nii.gz"
            )

            new_nii_volume = nib.Nifti1Image(
                volume_silce, affine=np.eye(4)
            )  # ref : https://stackoverflow.com/questions/28330785/creating-a-nifti-file-from-a-numpy-array
            new_nii_mask = nib.Nifti1Image(mask_silce, affine=np.eye(4))

            nib.save(new_nii_volume, nii_volume_path)
            nib.save(new_nii_mask, nii_mask_path)
