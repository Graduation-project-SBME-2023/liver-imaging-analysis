import liver_imaging_analysis.engine.visualization as vs
import torch
import nibabel as nib
import numpy as np
import pytest
from PIL import Image, ImageChops
import matplotlib
import os


# Prevents matplotlib from opening figure windows
matplotlib.use("Agg")


temp_folder_path = "tests/testdata/data/temp/"
volume_path = "tests/testdata/data/volume/resized_liver.nii"
mask_path = "tests/testdata/data/mask/resized_mask.nii"


def delete_temp_images():
    # Check if the folder exists
    if os.path.exists(temp_folder_path):
        # Remove all the files in the folder
        for item in os.listdir(temp_folder_path):
            item_path = os.path.join(temp_folder_path, item)
            os.remove(item_path)


def load_nifti_file(file_path):
    img = nib.load(file_path).get_fdata()
    return img


def read_image(image_path):
    return Image.open(image_path).convert("RGB")


def images_similarity(image_one, image_two):
    diff = ImageChops.difference(image_one, image_two)
    bbox = diff.getbbox()

    # Assert that the images are similar
    assert not bbox, "Images are different"


# Load the refrence volume and mask data
volume_data = load_nifti_file(volume_path)
mask_data = load_nifti_file(mask_path)

mask_data = np.where(mask_data == 2, 1, 0)
volume_data = torch.tensor(volume_data)
mask_data = torch.tensor(mask_data)



def test_contour():
    """
    Test the 'contour' visualization mode.

    This test function visualizes tumor data in 'contour' mode using the
    'visualize_tumor' function and compares the generated images with
    reference images from the 'contour_ref_path' folder. It ensures that
    the generated images are similar to the reference images.

    Raises:
        AssertionError: If any generated image is different from the
            reference image.

    """
    contour_ref_path = (
        "tests/testdata/testcases/visualization_test_photos/contour/volume/"
    )
    vs.visualize_tumor(
        volume=volume_data.numpy(),
        mask=mask_data,
        mode="contour",
        save_path="tests/testdata/data/temp",
    )
    for item in os.listdir(temp_folder_path):
        ref_image = read_image(os.path.join(contour_ref_path, item))
        test_image = read_image(os.path.join(temp_folder_path, item))
        images_similarity(test_image, ref_image)
    delete_temp_images()


def test_box():
    """
    Test the 'box' visualization mode.

    This test function visualizes tumor data in 'box' mode using the
    'visualize_tumor' function and compares the generated images with
    reference images from the 'contour_ref_path' folder. It ensures that
    the generated images are similar to the reference images.

    Raises:
        AssertionError: If any generated image is different from the
            reference image.

    """
    contour_ref_path = "tests/testdata/testcases/visualization_test_photos/box/volume/"
    vs.visualize_tumor(
        volume=volume_data.numpy(),
        mask=mask_data,
        mode="box",
        save_path="tests/testdata/data/temp",
    )
    for item in os.listdir(temp_folder_path):
        ref_image = read_image(os.path.join(contour_ref_path, item))
        test_image = read_image(os.path.join(temp_folder_path, item))
        images_similarity(test_image, ref_image)
    delete_temp_images()


def test_zoom():
    """
    Test the 'zoom' visualization mode.

    This test function visualizes tumor data in 'zoom' mode using the
    'visualize_tumor' function and compares the generated images with
    reference images from the 'contour_ref_path' folder. It ensures that
    the generated images are similar to the reference images.

    Raises:
        AssertionError: If any generated image is different from the
            reference image.

    """
    contour_ref_path = "tests/testdata/testcases/visualization_test_photos/zoom/volume/"
    vs.visualize_tumor(
        volume=volume_data.numpy(),
        mask=mask_data,
        mode="zoom",
        save_path="tests/testdata/data/temp",
    )
    for item in os.listdir(temp_folder_path):
        ref_image = read_image(os.path.join(contour_ref_path, item))
        test_image = read_image(os.path.join(temp_folder_path, item))
        images_similarity(test_image, ref_image)
    delete_temp_images()
