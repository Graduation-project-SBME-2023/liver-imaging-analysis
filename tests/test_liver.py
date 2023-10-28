from liver_imaging_analysis.models.liver_segmentation import (
    LiverSegmentation,
    segment_liver,
    train_liver,
)
from unittest.mock import MagicMock
from unittest.mock import patch
from liver_imaging_analysis.engine.engine import Engine, set_seed
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
    OrientationD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    RandSpatialCropd,
    RandAdjustContrastd,
    RandZoomd,
    CropForegroundd,
    ActivationsD,
    AsDiscreteD,
    KeepLargestConnectedComponentD,
    RemoveSmallObjectsD,
    FillHolesD,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
)


@pytest.fixture
def liver_obj():
    set_seed()
    liver_obj = LiverSegmentation(modality="CT", inference="3D")
    config.transforms["transformation_size"] = [64, 64]
    return liver_obj


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


@pytest.mark.parametrize("inference", [("3D"), ("sliding_window")])
def test_predict(inference):
    """
    Tests predict_2dto3d function and sliding window function.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...)
    """

    config.transforms["transformation_size"] = [64, 64]
    liver_object = LiverSegmentation(modality="CT", inference=inference)
    prediction = liver_object.predict(config.test["test_volume"])
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    assert torch.min(prediction) == 0
    assert torch.max(prediction) == 1


@pytest.fixture
def liver_object_sw():
    set_seed()
    liver_object_sw = LiverSegmentation(modality="CT", inference="sliding_window")
    config.transforms["transformation_size"] = [64, 64]
    return liver_object_sw


def test_test_sliding_window(liver_object_sw):
    """
    Tests test_sliding_window function.
    verifies the functionality by comparing loss with true refrence loss on the same volume
    """
    set_seed()
    test_dataloader = DataLoader(
        dataset_path=config.test["test_folder"],
        batch_size=config.training["batch_size"],
        train_transforms=liver_object_sw.train_transform,
        test_transforms=liver_object_sw.test_transform,
        num_workers=0,
        pin_memory=False,
        test_size=1,
        mode=config.dataset["mode"],
        shuffle=config.training["shuffle"],
    )
    test_dataload = test_dataloader.get_testing_data()
    test_loss, test_metric = liver_object_sw.test_sliding_window(
        dataloader=test_dataload
    )

    assert round(test_loss, 6) == 0.908422
    assert round(test_metric.item(), 4) == 0.0709


def set_configs(inference):
    """"
    Edits the new values for config parameters used in testing
    """
    if inference in ["2D", "3D"]:
        config.dataset["prediction"] = config.test["test_volume"]
        config.training["batch_size"] = 1
        config.dataset["training"], config.dataset["testing"] = (
            "tests\\testdata\\data\\resized_liver\\",
            "tests\\testdata\\data\\resized_liver\\",
        )
        config.network_parameters["dropout"] = 0
        config.network_parameters["out_channels"] = 1
        config.network_parameters["spatial_dims"] = 2
        config.network_parameters["strides"] = [2]
        config.network_parameters["num_res_units"] = 1
        config.network_parameters["channels"] = [32, 64]
        config.transforms["transformation_size"] = [64, 64]
    elif inference == "sliding_window":
        config.dataset["prediction"] = "test cases/volume/volume-64.nii"
        config.dataset["training"], config.dataset["testing"] = (
            "tests\\testdata\\data\\resized_liver\\",
            "tests\\testdata\\data\\resized_liver\\",
        )
        config.training["batch_size"] = 1
        config.training["scheduler_parameters"] = {
            "step_size": 20,
            "gamma": 0.5,
            "verbose": False,
        }
        config.network_parameters["dropout"] = 0
        config.network_parameters["out_channels"] = 1
        config.network_parameters["channels"] = [32, 64]
        config.network_parameters["strides"] = [2]
        config.network_parameters["num_res_units"] = 1
        config.network_parameters["spatial_dims"] = 3
        config.network_parameters["norm"] = "BATCH"
        config.network_parameters["bias"] = False
        config.save["liver_checkpoint"] = "liver_cp_sliding_window"
        config.transforms["sw_batch_size"] = 4
        config.transforms["roi_size"] = (32, 32, 32)
        config.transforms["overlap"] = 0.25
        config.transforms["train_transform"] = "3d_ct_transform"
        config.transforms["test_transform"] = "3d_ct_transform"
        config.transforms["post_transform"] = "3d_ct_transform"


@pytest.mark.parametrize(
    ("modality", "inference", "check_point", "refernece_path"),
    [
        (
            "CT",
            "sliding_window",
            config.test["reference_sliding_window"],
            config.test["reference_liver_sw_array"],
        )
    ],
)
def test_segment_liver(modality, inference, check_point, refernece_path):
    """
    Tests segment_liver function (liver_inference = 'sliding_window').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.

    Parameters:
    ----------
    modality (str): The modality of the input data. "CT".
    inference (str): The type of inference. "Sliding_window.
    check_point (str): The path of checkpoint depends on the model 3D or SLIDING WINDOW.
    refernece_path (str): The path to the reference 3D numpy array. 
    ----------
    """
    with patch.object(LiverSegmentation, "set_configs") as mock_set_configs:
        mock_set_configs.side_effect = lambda self, inference: print(
            "Mocked set_configs called"
        )
        # Call the set_configs function
        set_configs(inference="sliding_window")

        liver_prediction = segment_liver(
            prediction_path=config.test["test_volume"],
            modality=modality,
            inference=inference,
            cp_path=check_point,
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

        reference = np.load(refernece_path)
        assert np.allclose(prediction, reference, 0.01)


def test_train():
    """ 
    Tests train function (epoch: 1 , inference = 'sliding_window', volume = '64,64,..' ) .
    Compares the weights of the resulted checkpoint with the reference checkpoint's weights.
    """
    with patch.object(LiverSegmentation, "set_configs") as mock_set_configs:
        mock_set_configs.side_effect = lambda self, inference: print(
            "Mocked set_configs called"
        )

        # Call the set_configs function
        # set_configs(inference="sliding_window")

        model = LiverSegmentation(modality="CT", inference="sliding_window")

        # Train a single epoch using the same volume and save the checkpoint to be compared with the reference
        train_liver(
            modality="CT",
            inference="sliding_window",
            pretrained=False,
            cp_path=config.test["liver_cp_sw"],
            epochs=1,
            evaluate_epochs=1,
            batch_callback_epochs=100,
            save_weight=True,
            save_path=config.test["liver_cp_sw"],
            test_batch_callback=False,
        )

        # load the previous checkpoint of training 1 epoch on a volume
        model.load_checkpoint(config.test["liver_cp_sw"])
        trained_weights = model.network.state_dict()

        # load refrence checkpoint of training 1 epoch on the same volume
        model.load_checkpoint(config.test["reference_sliding_window"])
        reference_weights = model.network.state_dict()

        # Check that weights match
        for i in reference_weights.keys():
            assert torch.allclose(reference_weights[i], trained_weights[i])
