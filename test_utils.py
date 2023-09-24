import os
import pytest
from liver_imaging_analysis.engine.utils import (
    VolumeSlicing,
    Overlay,
    calculate_largest_tumor,
    liver_isolate_crop,
    find_pix_dim,
)
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import natsort


@pytest.fixture
def volume_dir():
    return "tests/testdata/data/volume"


@pytest.fixture
def mask_dir():
    return "tests/testdata/data/mask"


@pytest.fixture
def vol_path(volume_dir):
    return os.path.join(volume_dir, "resized_liver.nii")


@pytest.fixture
def mask_path(mask_dir):
    # 0 background, 1 liver, 2 lesion
    return os.path.join(mask_dir, "resized_mask.nii")


@pytest.fixture
def volume(vol_path):
    return nib.load(vol_path)


@pytest.fixture
def mask(mask_path):
    return nib.load(mask_path)


@pytest.fixture
def mask_array(mask):
    return np.where(mask.get_fdata() == 2, 1, 0)


@pytest.fixture
def temp_dir(tmpdir):
    """
    provides temporary directories for some test functions
    """
    temp_vol = tmpdir.mkdir("vol")
    temp_mask = tmpdir.mkdir("mask")
    return temp_vol, temp_mask


def test_liver_isolate_crop(mask, volume, mask_dir, volume_dir, temp_dir):
    """
    Tests volume and mask are cropped in z direcrion,
    volume contains only liver while mask contains only lesions.


    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving new_volume and new_mask
    """
    new_vol_dir = temp_dir[0]
    new_mask_dir = temp_dir[1]
    min_slice = 10
    max_slice = 32
    cropped_mask = mask.get_fdata()[:, :, min_slice:max_slice]
    cropped_vol = volume.get_fdata()[:, :, min_slice:max_slice]

    liver_isolate_crop(volume_dir, mask_dir, new_vol_dir, new_mask_dir)

    # testing on only one volume and one mask
    assert os.path.exists(new_vol_dir.join(os.listdir(new_vol_dir)[0]))
    assert os.path.exists(new_mask_dir.join(os.listdir(new_mask_dir)[0]))

    new_volume = nib.load(new_vol_dir.join(os.listdir(new_vol_dir)[0])).get_fdata()
    new_mask = nib.load(new_mask_dir.join(os.listdir(new_mask_dir)[0])).get_fdata()

    assert (
        new_volume.shape == new_mask.shape == (64, 64, max_slice - min_slice)
    )  # shape after cropping
    assert np.all(
        new_mask == np.where(cropped_mask == 2, 1, 0)
    )  # 0 background, 1 lesions
    assert np.all(
        new_volume
        == np.where(
            (cropped_mask > 0.5), cropped_vol.astype(int), cropped_vol.astype(int).min()
        )
    )  # liver isolated


@pytest.mark.parametrize(
    "input,expected",
    [
        (pytest.lazy_fixture("mask_array"), 16),
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
    slice_idx = calculate_largest_tumor(torch.from_numpy(input))
    assert slice_idx == expected


def test_find_pix_dim(vol_path):
    """
    Tests nifti file pixel dimensions
    """
    x, y, z = find_pix_dim(vol_path)
    assert np.allclose((x, y, z), (0.664062, 0.664062, 5.0))


def test_overlay(mask_path, vol_path, temp_dir):
    """
    Tests overlay gif is saved, and compares gif frames


    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            temporary directory for saving the gif
    """
    path = temp_dir[0].join("gif.gif")
    overlay = Overlay(vol_path, mask_path, str(path))
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
            == np.load(f"tests/testdata/testcases/gif_frames/gif_frame_{frame}.npy")
        )


def png2array(file):
    slice_image = Image.open(file)
    return np.asarray(slice_image)


@pytest.mark.parametrize("extension", [".nii.gz", ".png"])
def test_volumeslicing(volume, volume_dir, mask_dir, temp_dir, extension):
    """
    Tests all volume slices & mask slices are correctly saved

    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving volume slices & mask slices
    extension: str
            extension of the slice file
    """

    temp_vol_slices = temp_dir[0]
    temp_mask_slices = temp_dir[1]
    test_shape = (64, 64)
    test_num_of_slices = volume.get_fdata().shape[2]

    if extension == ".png":
        VolumeSlicing.nii2png(volume_dir, mask_dir, temp_vol_slices, temp_mask_slices)
    else:
        VolumeSlicing.nii3d_To_nii2d(
            volume_dir, mask_dir, temp_vol_slices, temp_mask_slices
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
                f"tests/testdata/testcases/liverSlices_png/volume/{i}.png"
            )
            mask_ref = png2array(
                f"tests/testdata/testcases/liverSlices_png/mask/{i}.png"
            )

        else:
            vol_slice_array = nib.load(
                os.path.join(temp_vol_slices, vol_name)
            ).get_fdata()
            mask_slice_array = nib.load(
                os.path.join(temp_mask_slices, mask_name)
            ).get_fdata()
            vol_ref = nib.load(
                f"tests/testdata/testcases/liverSlices/volume/{i}.nii"
            ).get_fdata()
            mask_ref = nib.load(
                f"tests/testdata/testcases/liverSlices/mask/{i}.nii"
            ).get_fdata()

        # checks slice dimensions
        assert vol_slice_array.shape == test_shape
        assert mask_slice_array.shape == test_shape
        # checks pixel values per slice
        assert np.allclose(vol_slice_array, vol_ref)
        assert np.allclose(mask_slice_array, mask_ref)


