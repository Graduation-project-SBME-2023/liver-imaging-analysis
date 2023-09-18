from liver_imaging_analysis.models.liver_segmentation import (
    LiverSegmentation,
    segment_liver,
    train_liver,
)
import pytest
from liver_imaging_analysis.engine.config import config
import torch
import nibabel as nib
import SimpleITK
from liver_imaging_analysis.engine.dataloader import DataLoader, Keys
import numpy as np
from liver_imaging_analysis.engine.utils import VolumeSlicing
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImageD,
    NormalizeIntensityD,
    RandFlipd,
    RandRotated,
    RandCropByPosNegLabeld,
    ResizeD,
    ToTensorD,
    RandAdjustContrastd,
    RandZoomd,
    ActivationsD,
    AsDiscreteD,
    KeepLargestConnectedComponentD,
    FillHolesD,
)


@pytest.mark.parametrize(
    "transform_name", [("3d_ct_transform"), ("2d_ct_transform"), ("2d_mri_transform")]
)
def test_get_pretraining_transforms(transform_name, liver_obj):
    """
    Tests get_pretraining_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform", "2d_mri_transform".
    liver_obj (object): The liver object that provides the get_pretraining_transforms method.
    ----------
    """
    output_transform = liver_obj.get_pretraining_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_ct_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            NormalizeIntensityD,
            ForegroundMaskD,
            RandCropByPosNegLabeld,
            ToTensorD,
        ]

    elif transform_name == "2d_ct_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            RandZoomd,
            RandFlipd,
            RandRotated,
            RandAdjustContrastd,
            NormalizeIntensityD,
            ForegroundMaskD,
            ToTensorD,
        ]

    elif transform_name == "2d_mri_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            RandZoomd,
            RandFlipd,
            RandRotated,
            RandAdjustContrastd,
            NormalizeIntensityD,
            ForegroundMaskD,
            ToTensorD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


@pytest.mark.parametrize(
    "transform_name", [("3d_ct_transform"), ("2d_ct_transform"), ("2d_mri_transform")]
)
def test_get_pretesting_transforms(transform_name, liver_obj):
    """
    Tests the get_pretesting_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform", "2d_mri_transform".
    liver_obj (object): The liver object that provides the get_pretesting_transforms method.
    ----------
    """
    output_transform = liver_obj.get_pretesting_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_ct_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            NormalizeIntensityD,
            ForegroundMaskD,
            ToTensorD,
        ]

    elif transform_name == "2d_ct_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            NormalizeIntensityD,
            ForegroundMaskD,
            ToTensorD,
        ]

    elif transform_name == "2d_mri_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ResizeD,
            NormalizeIntensityD,
            ForegroundMaskD,
            ToTensorD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


@pytest.mark.parametrize(
    "transform_name", [("3d_ct_transform"), ("2d_ct_transform"), ("2d_mri_transform")]
)
def test_get_postprocessing_transforms(transform_name, liver_obj):
    """
    Tests get_postprocessing_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "3d_ct_transform", "2d_ct_transform", "2d_mri_transform".
    liver_obj (object): The liver object that provides the get_postprocessing_transforms function.
    ----------
    """
    output_transform = liver_obj.get_postprocessing_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "3d_ct_transform":
        expected_transfom = [
            ActivationsD,
            AsDiscreteD,
            FillHolesD,
            KeepLargestConnectedComponentD,
        ]

    elif transform_name == "2d_ct_transform":
        expected_transfom = [
            ActivationsD,
            AsDiscreteD,
            FillHolesD,
            KeepLargestConnectedComponentD,
        ]

    elif transform_name == "2d_mri_transform":
        expected_transfom = [
            ActivationsD,
            AsDiscreteD,
            FillHolesD,
            KeepLargestConnectedComponentD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)



def test_segment_liver_2d(modality="CT", inference= "3D", path= "tests/testdata/predicted_3d.npy"):
    """ "
    Tests segment_liver function (liver_inference = '3D').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.

    Parameters:
    ----------
    modality (str): The modality of the input data. "CT","MRI".
    inference (str): The type of inference. "3D","Sliding_window.
    path (str): The path to the predicted 3D numpy array. In this test case, it is set to "predicted_3d.npy".
    ----------
    """
    config.transforms["transformation_size"] = [64, 64]
    liver_prediction = segment_liver(
        prediction_path=config.test["test_volume"],
        modality=modality,
        inference=inference,
        cp_path=config.test["liver_cp"],
    )
    # Assertion checks for liver_prediction
    assert isinstance(liver_prediction, torch.Tensor)
    assert liver_prediction.shape[0] == 1
    assert liver_prediction.shape[1] == 1  # number of channels
    assert (
        liver_prediction.shape[2:]
        == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    prediction = liver_prediction.cpu()
    prediction = prediction.numpy()
    final_path = path
    true = np.load(final_path)
    assert np.allclose(prediction, true, 0.01)


@pytest.fixture
def liver_obj():
    config.transforms["transformation_size"] = [64, 64]
    liver_obj = LiverSegmentation(modality="CT", inference="3D")
    return liver_obj


def test_predict_2dto3d(liver_obj):
    """
    Tests predict_2dto3d function.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...)
    """

    prediction = liver_obj.predict_2dto3d(config.test["test_volume"])
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    assert torch.min(prediction) >= 0
    assert torch.max(prediction) <= 1


@pytest.fixture
def liver_object_sw():
    liver_object_sw = LiverSegmentation(modality="CT", inference="sliding_window")
    return liver_object_sw


def test_predict_sliding_window(liver_object_sw):
    """
    Tests predict_sliding_window function.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...)
    """
    prediction = liver_object_sw.predict_sliding_window(
        volume_path=config.test["test_volume"]
    )
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    assert torch.min(prediction) >= 0
    assert torch.max(prediction) <= 1


def test_test_sliding_window(liver_object_sw):
    """
    Tests test_sliding_window function.
    verifies the functionality by comparing loss with true refrence loss on the same volume
    """
    test_dataloader = DataLoader(
        dataset_path=config.test["test_folder"],
        batch_size=config.training["batch_size"],
        train_transforms=liver_object_sw.train_transform,
        test_transforms=liver_object_sw.test_transform,
        num_workers=0,
        pin_memory=False,
        test_size=1,
        mode= "2D",
        shuffle=config.training["shuffle"],
    )
    test_dataload = test_dataloader.get_testing_data()
    test_loss, test_metric = liver_object_sw.test_sliding_window(
        dataloader=test_dataload
    )

    assert round(test_loss,1) == 0.9
    assert test_metric.item() <= 0.5




def test_segment_liver(modality = "CT" , inference = "sliding_window", path = "tests/testdata/predicted_sliding.npy"):
    """
    Tests segment_liver function (liver_inference = 'sliding_window').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.

    Parameters:
    ----------
    modality (str): The modality of the input data. "CT","MRI".
    inference (str): The type of inference. "Sliding_window , "3D".
    path (str): The path to the predicted 3D numpy array. In this test case, it is set to "predicted_sliding.npy".
    ----------
    """
    liver_prediction = segment_liver(
        prediction_path="tests/testdata/resized_liver/volume/resized_liver.nii",
        modality=modality,
        inference=inference,
        cp_path=config.test["reference_sliding_window"],
    )
    # Assertion checks for liver_prediction
    assert isinstance(liver_prediction, torch.Tensor)
    assert liver_prediction.shape[0] == 1
    assert liver_prediction.shape[1] == 1  # number of channels
    assert (
        liver_prediction.shape[2:]
        == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    prediction = liver_prediction.cpu()
    prediction = prediction.numpy()
    final_path = path
    true = np.load(final_path)
    assert np.allclose(prediction, true, 0.01)


def test_train():
    """ "
    Tests train function (epoch: 1 , inference = 'sliding_window', volume = '64,64,..' ) .
    Compares the weights of the resulted checkpoint with the reference checkpoint's weights.
    """
    model = LiverSegmentation(modality="CT", inference="sliding_window")

    # load refrence checkpoint of training 1 epoch on the same volume
    model.load_checkpoint(config.test["reference_sliding_window"])
    init_weights = model.network.state_dict()

    # Train a single epoch using the same volume and save the checkpoint to be compared with the reference
    train_liver(
        modality="CT",
        inference="sliding_window",
        pretrained=False,
        cp_path="tests/testdata/liver_sw",
        epochs=1,
        evaluate_epochs=1,
        batch_callback_epochs=100,
        save_weight=True,
        save_path="tests/testdata/liver_sw",
        test_batch_callback=False,
    )
    model.load_checkpoint("tests/testdata/liver_sw")
    loaded_weights = model.network.state_dict()

    # Check that weights match
    for i in init_weights.keys():
        assert torch.allclose(init_weights[i], loaded_weights[i])


