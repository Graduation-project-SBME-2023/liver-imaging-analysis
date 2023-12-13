import os
import pytest
from liver_imaging_analysis.engine.utils import (
    VolumeSlicing,
    Overlay,
    calculate_largest_tumor,
    liver_isolate_crop,
    find_pix_dim,
)
from liver_imaging_analysis.engine.config import config
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import natsort


@pytest.fixture
def volume_array():
    return nib.load(config.test["test_volume"]).get_fdata()


@pytest.fixture
def mask_array():
    # 0 background, 1 liver, 2 lesions
    return nib.load(config.test["test_label"]).get_fdata()


@pytest.fixture
def liver_isolated_volume(mask_array, volume_array):
    return np.where(
        (mask_array > 0.5), volume_array.astype(int), volume_array.astype(int).min()
    )


@pytest.fixture
def lesions_mask_array(mask_array):
    # 0 background, 1 lesions
    return np.where(mask_array == 2, 1, 0)


@pytest.fixture
def temp_dir(tmpdir):
    """
    provides temporary directories for some test functions
    """
    temp_vol = tmpdir.mkdir("vol")
    temp_mask = tmpdir.mkdir("mask")
    return temp_vol, temp_mask


def test_liver_isolate_crop(lesions_mask_array, liver_isolated_volume, temp_dir):
    """
    checks new volume and new mask are cropped in z direcrion,
    new volume contains only liver, while new mask contains only lesions.


    Parameters
    ----------
    lesions_mask_array: numpy.ndarray
        A NumPy array containing only lesions mask       
    liver_isolated_volume: numpy.ndarray
        A NumPy array containing volume of isolated liver
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving new_volume and new_mask
    """
    new_vol_dir = temp_dir[0]
    new_mask_dir = temp_dir[1]
    min_slice = 10
    max_slice = 32
    cropped_mask = lesions_mask_array[:, :, min_slice:max_slice]
    cropped_vol = liver_isolated_volume[:, :, min_slice:max_slice]

    liver_isolate_crop(
        os.path.join(config.test["test_folder"], "volume"),
        os.path.join(config.test["test_folder"], "mask"),
        new_vol_dir,
        new_mask_dir,
    )

    # testing on only one volume and one mask
    assert os.path.exists(new_vol_dir.join(os.listdir(new_vol_dir)[0]))
    assert os.path.exists(new_mask_dir.join(os.listdir(new_mask_dir)[0]))

    new_volume = nib.load(new_vol_dir.join(os.listdir(new_vol_dir)[0])).get_fdata()
    new_mask = nib.load(new_mask_dir.join(os.listdir(new_mask_dir)[0])).get_fdata()

    assert (
        new_volume.shape == new_mask.shape == (64, 64, max_slice - min_slice)
    )  # shape after cropping
    assert np.all(new_mask == cropped_mask)
    assert np.all(new_volume == cropped_vol)


@pytest.mark.parametrize(
    "input,expected",
    [
        (pytest.lazy_fixture("lesions_mask_array"), 16),
        (np.zeros((5, 5, 5), dtype=int), -1),
    ],
)
def test_largest_tumor(input, expected):
    """
    Tests index where largest tumor slice is,
    if no tumor ,slice_idx= -1

    Parameters
    ----------
    input: numpy.ndarray
        A NumPy array containing only lesions mask
    expected: int
        expected slice number
    """
    config.visualization["volume"] = config.test["test_volume"]
    slice_idx = calculate_largest_tumor(torch.from_numpy(input))
    assert slice_idx == expected


def test_find_pix_dim():
    """
    Tests nifti file pixel dimensions
    """
    x, y, z = find_pix_dim(config.test["test_volume"])
    assert np.allclose((x, y, z), (0.664062, 0.664062, 5.0))


def test_overlay(temp_dir):
    """
    Tests overlay gif is saved, and compares gif frames


    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            temporary directory for saving the gif
    """
    path = temp_dir[0].join("gif.gif")
    overlay = Overlay(config.test["test_volume"], config.test["test_label"], str(path))
    overlay.generate_animation()

    assert os.path.exists(path)
    gif = Image.open(str(path))

    # correct number of frames
    assert gif.n_frames == 33

    # loops on frames with a 3 step value
    for frame in range(0, gif.n_frames, 3):
        gif.seek(frame)
        assert np.all(
            np.asarray(gif)
            == np.load(
                os.path.join(config.test["reference_gif"], f"gif_frame_{frame}.npy")
            )
        )


def png2array(file):
    """
    Returns NumPy array from a PNG image.

    Parameters
    ----------
    file : str
        Path to the PNG image of a volume slice.

    Returns
    -------
    numpy.ndarray
        A NumPy array representing the image.

    """
    slice_image = Image.open(file)
    return np.asarray(slice_image)


@pytest.mark.parametrize("extension", [".nii.gz", ".png"])
def test_volumeslicing(volume_array, temp_dir, extension):
    """
    Tests all volume slices & mask slices are correctly saved

    Parameters
    ----------
    volume_array: numpy.ndarray
        A NumPy array containing original volume
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving volume slices & mask slices
    extension: str
            extension of the slice file
    """

    temp_vol_slices = temp_dir[0]
    temp_mask_slices = temp_dir[1]
    test_shape = (64, 64)
    test_num_of_slices = volume_array.shape[2]

    if extension == ".png":
        VolumeSlicing.nii2png(
            os.path.join(config.test["test_folder"], "volume"),
            os.path.join(config.test["test_folder"], "mask"),
            temp_vol_slices,
            temp_mask_slices,
        )
    else:
        VolumeSlicing.nii3d_To_nii2d(
            os.path.join(config.test["test_folder"], "volume"),
            os.path.join(config.test["test_folder"], "mask"),
            temp_vol_slices,
            temp_mask_slices,
        )

    # checks number of slices
    assert (
        len(os.listdir(temp_vol_slices))
        == len(os.listdir(temp_mask_slices))
        == test_num_of_slices
    )

    for i, (vol_name, mask_name) in enumerate(
        zip(
            natsort.natsorted(os.listdir(temp_vol_slices)),
            natsort.natsorted(os.listdir(temp_mask_slices)),
        )
    ):

        # checks file type
        assert vol_name.endswith(extension) is True
        assert mask_name.endswith(extension) is True

        if extension == ".png":
            vol_slice_array = png2array(os.path.join(temp_vol_slices, vol_name))
            mask_slice_array = png2array(os.path.join(temp_mask_slices, mask_name))
            vol_ref = png2array(
                os.path.join(config.test["liver_png_slices"], f"volume/{i}.png")
            )
            mask_ref = png2array(
                os.path.join(config.test["liver_png_slices"], f"mask/{i}.png")
            )

        else:
            vol_slice_array = nib.load(
                os.path.join(temp_vol_slices, vol_name)
            ).get_fdata()
            mask_slice_array = nib.load(
                os.path.join(temp_mask_slices, mask_name)
            ).get_fdata()
            vol_ref = nib.load(
                os.path.join(config.test["liver_slices"], f"volume/{i}.nii")
            ).get_fdata()
            mask_ref = nib.load(
                os.path.join(config.test["liver_slices"], f"mask/{i}.nii")
            ).get_fdata()

        # checks slice dimensions
        assert vol_slice_array.shape == test_shape
        assert mask_slice_array.shape == test_shape
        # checks pixel values per slice
        assert np.allclose(vol_slice_array, vol_ref)
        assert np.allclose(mask_slice_array, mask_ref)
