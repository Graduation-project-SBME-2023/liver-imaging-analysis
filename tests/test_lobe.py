import pytest
from liver_imaging_analysis.models.lobe_segmentation import (
    LobeSegmentation,
    segment_lobe,
)
from liver_imaging_analysis.engine.transforms import Dilation, ConvexHull
from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.models.liver_segmentation import segment_liver
import torch
import nibabel as nib
from liver_imaging_analysis.engine.engine import set_seed
from liver_imaging_analysis.engine.dataloader import DataLoader
import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImageD,
    NormalizeIntensityD,
    OrientationD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    RandAdjustContrastd,
    RandZoomd,
    CropForegroundd,
    ActivationsD,
    AsDiscreteD,
    KeepLargestConnectedComponentD,
    FillHolesD,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    SpatialPadd,
    Spacingd,
    EnsureTyped,
    Activationsd,
    Invertd,
)


@pytest.fixture
def lobe_obj():
    set_seed()
    lobe_obj = LobeSegmentation()
    return lobe_obj


@pytest.mark.parametrize("transform_name", [("3d_transform"), ("2d_transform")])
def test_get_pretraining_transforms(transform_name, lobe_obj):
    """
    Tests get_pretraining_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform".
    lobe_obj (object): The lobe object that provides the get_pretraining_transforms method.
    ----------
    """

    output_transform = lobe_obj.get_pretraining_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            OrientationD,
            CropForegroundd,
            Spacingd,
            SpatialPadd,
            RandZoomd,
            RandFlipd,
            RandFlipd,
            RandRotated,
            RandAdjustContrastd,
            ScaleIntensityRanged,
            RandSpatialCropSamplesd,
            EnsureTyped,
            AsDiscreteD,
        ]

    elif transform_name == "2d_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            NormalizeIntensityD,
            AsDiscreteD,
            RandZoomd,
            RandFlipd,
            RandFlipd,
            RandRotated,
            RandAdjustContrastd,
            ToTensorD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)



@pytest.mark.parametrize("transform_name", [("3d_transform"), ("2d_transform")])
def test_get_pretesting_transforms(transform_name, lobe_obj):
    """
    Tests the get_pretesting_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform".
    lobe_obj (object): The lobe object that provides the get_pretesting_transforms method.
    ----------
    """
    output_transform = lobe_obj.get_pretesting_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            OrientationD,
            CropForegroundd,
            Spacingd,
            SpatialPadd,
            ScaleIntensityRanged,
            EnsureTyped,
        ]
    elif transform_name == "2d_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            NormalizeIntensityD,
            AsDiscreteD,
            ToTensorD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


@pytest.mark.parametrize("transform_name", [("3d_transform"), ("2d_transform")])
def test_get_postprocessing_transforms(transform_name, lobe_obj):
    """
    Tests get_postprocessing_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform".
    lobe_obj (object): The lobe object that provides the get_postprocessing_transforms function.
    ----------
    """

    output_transform = lobe_obj.get_postprocessing_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_transform":
        expected_transfom = [
            Invertd,
            Activationsd,
            AsDiscreteD,
            FillHolesD,
            KeepLargestConnectedComponentD,
        ]

    elif transform_name == "2d_transform":
        expected_transfom = [
            ActivationsD,
            AsDiscreteD,
            FillHolesD,
            KeepLargestConnectedComponentD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)



def test_segment_lobe_sw():
    """
    Tests segment_lobe function (lobe_inference = 'sliding_window').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.

    """
    config.transforms["transformation_size"] = [64, 64]
    lobe_prediction = segment_lobe(
        prediction_path=config.test["test_volume"],
        liver_inference= "3D",
        lobe_inference="sliding_window",
        liver_cp=config.test["liver_cp"],
        lobe_cp= config.test["lobe_cp_sw"],
    )
    # Assertion checks for liver_prediction
    assert isinstance(lobe_prediction, torch.Tensor)
    assert lobe_prediction.shape[0] == 1
    assert lobe_prediction.shape[1] == 1  # number of channels
    assert (
        lobe_prediction.shape[2:]
        == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    prediction = lobe_prediction.cpu()
    prediction = prediction.numpy()
    path = config.test["reference_lobe_array"]
    # np.save(path,prediction)
    reference = np.load(path)
    assert np.allclose(prediction, reference, 0.01)


@pytest.fixture
def lobe_obj():
    set_seed()
    config.transforms["transformation_size"] = [64, 64]
    lobe_obj = LobeSegmentation(inference="3D")
    return lobe_obj


def test_predict_2dto3(lobe_obj):
    """
    Tests predict_2dto3d function.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...)
    """

    liver_prediction = segment_liver(
        prediction_path=config.test["test_volume"],
        modality="CT",
        inference="sliding_window",
        cp_path=config.test["reference_sliding_window"],
    )

    extract = Compose([Dilation(), ConvexHull()])
    prediction = lobe_obj.predict_2dto3d(
        volume_path=config.test["test_volume"],
        liver_mask=extract(liver_prediction[0]).permute(3, 0, 1, 2)

    )
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension


    assert torch.min(prediction).item() == 0
    assert torch.max(prediction).item() == 9
    

@pytest.fixture
def lobe_object_sw():
    set_seed()
    lobe_object_sw = LobeSegmentation(inference="sliding_window")
    return lobe_object_sw


def test_predict_sliding_window(lobe_object_sw):
    """
    Tests predict_sliding_window function.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...)
    """
    liver_mask = segment_liver(
        prediction_path=config.test["test_volume"],
        modality="CT",
        inference="sliding_window",
        cp_path=config.test["reference_sliding_window"],
    )

    prediction = lobe_object_sw.predict_sliding_window(
        volume_path= config.test["test_volume"],
        liver_mask=liver_mask,
    )
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension


    assert torch.min(prediction).item() == 0
    assert torch.max(prediction).item() == 9

