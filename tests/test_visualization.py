import torch
import nibabel as nib
import numpy as np
import pytest
from PIL import Image, ImageChops
import matplotlib
import os
from liver_imaging_analysis.engine.config import config
config.visualization['volume'] = 'tests/testdata/data/volume/resized_liver.nii'
config.visualization['mask'] = 'tests/testdata/data/mask/resized_mask.nii'
import liver_imaging_analysis.engine.visualization as vs



# Prevents matplotlib from opening figure windows
matplotlib.use("Agg")


@pytest.fixture
def volume_path():
    return config.visualization['volume']
# volume_path = config.visualization['volume']

@pytest.fixture
def mask_path():
    return config.visualization['mask']


@pytest.fixture
def volume_data(volume_path):
    volume_data = nib.load(volume_path).get_fdata()
    return  torch.tensor(volume_data)


@pytest.fixture
def mask_data(mask_path):
    mask_data = nib.load(mask_path).get_fdata()
    mask_data = np.where(mask_data == 2, 1, 0)
    return torch.tensor(mask_data)

@pytest.fixture
def temp_dir(tmpdir):
    """
    provides temporary directories for some test functions
    """
    tmpdir = tmpdir.mkdir("temp_folder_path")
    # temp_mask = tmpdir.mkdir("mask")
    return tmpdir

def read_image(image_path):
    return Image.open(image_path).convert("RGB")


def images_similarity(image_one, image_two):
    diff = ImageChops.difference(image_one, image_two)
    bbox = diff.getbbox()

    # Assert that the images are similar
    assert not bbox, "Images are different"


@pytest.mark.parametrize("mode", ["contour", "box", "zoom"])
def test_visualization_modes(volume_data, mask_data, temp_dir, mode):
    """
    Test different visualization modes.
    """
    ref_path = f"tests/testdata/testcases/visualization_test_photos/{mode}/volume/"
    vs.visualize_tumor(
        volume=volume_data.numpy(),
        mask=mask_data,
        mode=mode,
        save_path=temp_dir,
    )
    for item in os.listdir(temp_dir):
        ref_image = read_image(os.path.join(ref_path, item))
        test_image = read_image(os.path.join(temp_dir, item))
        images_similarity(test_image, ref_image)
    


