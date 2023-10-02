import pytest
from liver_imaging_analysis.models.lesion_segmentation import (
    LesionSegmentation,
    segment_lesion,
    train_lesion,
)
from unittest.mock import patch
from liver_imaging_analysis.engine.transforms import MorphologicalClosing
from liver_imaging_analysis.models.liver_segmentation import LiverSegmentation
from tests.test_liver import set_configs as liver_set_configs
from liver_imaging_analysis.models.liver_segmentation import segment_liver
from liver_imaging_analysis.engine.config import config
import torch
import nibabel as nib
from liver_imaging_analysis.engine.engine import set_seed
from liver_imaging_analysis.engine.dataloader import DataLoader, Keys
import numpy as np
from liver_imaging_analysis.engine.utils import VolumeSlicing
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImageD,
    RandFlipd,
    RandRotated,
    ToTensorD,
    RandAdjustContrastd,
    RandZoomd,
    ActivationsD,
    AsDiscreteD,
    FillHolesD,
    ScaleIntensityRanged,
    RemoveSmallObjectsD,
)


@pytest.fixture
def lesion_obj():
    set_seed()
    lesion_obj = LesionSegmentation()
    return lesion_obj


def test_get_pretraining_transforms(lesion_obj):
    """
    Tests get_pretraining_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    lesion_obj (object): The lesion object that provides the get_pretraining_transforms method.
    ----------
    """
    transform_name = "2d_ct_transform"
    output_transform = lesion_obj.get_pretraining_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    expected_transfom = [
        LoadImageD,
        EnsureChannelFirstD,
        ScaleIntensityRanged,
        RandZoomd,
        RandFlipd,
        RandFlipd,
        RandRotated,
        RandAdjustContrastd,
        ToTensorD,
    ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


def test_get_pretesting_transforms(lesion_obj):
    """
    Tests the get_pretesting_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    lesion_obj (object): The lesion object that provides the get_pretesting_transforms method.
    ----------
    """
    transform_name = "2d_ct_transform"
    output_transform = lesion_obj.get_pretesting_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    expected_transfom = [
        LoadImageD,
        EnsureChannelFirstD,
        ScaleIntensityRanged,
        ToTensorD,
    ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


def test_get_postprocessing_transforms(lesion_obj):
    """
    Tests get_postprocessing_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    lesion_obj (object): The lesion object that provides the get_postprocessing_transforms function.
    ----------
    """
    transform_name = "2d_ct_transform"
    output_transform = lesion_obj.get_postprocessing_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    expected_transfom = [ActivationsD, AsDiscreteD, FillHolesD, RemoveSmallObjectsD]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


def test_predict_2dto3d(lesion_obj):
    """
    Tests the prediction function for lesion object.
    verifies the functionality by performing predictions on a 3D volume (with size 64,64,...).

    Parameters:
    ----------
    lesion_obj (object): The lesion object that provides the predict_2dto3d function.
    ----------
    """
    with patch.object(LiverSegmentation, "set_configs") as mock_set_configs:
        mock_set_configs.side_effect = lambda self, inference: print(
            "Mocked set_configs called"
        )

        # Call the set_configs function
        liver_set_configs(inference="sliding_window")
        liver_mask = segment_liver(
            prediction_path=config.test["test_volume"],
            modality="CT",
            inference="sliding_window",
            cp_path=config.test["reference_sliding_window"],
        )

        volume_dir = config.test["test_volume"]
        close = MorphologicalClosing(iters=4)
        prediction = lesion_obj.predict_2dto3d(
            volume_path=volume_dir, liver_mask=close(liver_mask[0]).permute(3, 0, 1, 2)
        )

        assert isinstance(prediction, torch.Tensor)
        assert prediction.shape[0] == 1
        assert prediction.shape[1] == 1  # number of channels

        assert (
            prediction.shape[2:]
            == nib.load(config.test["test_volume"]).get_fdata().shape
        )
        assert torch.min(prediction).item() == 0
        assert torch.max(prediction).item() == 1


def set_configs():
    """
    Edits the new values for config parameters used in testing
    """

    config.dataset["prediction"] = "test cases/volume/volume-64.nii"
    config.dataset["training"], config.dataset["testing"] = (
        "tests/testdata/data/Temporary lessions/Train/",
        "tests/testdata/data/Temporary lessions/Test/",
    )
    config.training["batch_size"] = 8
    config.training["optimizer_parameters"] = {"lr": 0.01}
    config.training["scheduler_parameters"] = {
        "step_size": 20,
        "gamma": 0.5,
        "verbose": False,
    }
    config.network_parameters["dropout"] = 0
    config.network_parameters["out_channels"] = 1
    config.network_parameters["spatial_dims"] = 2
    # config.network_parameters["channels"] = [64, 128, 256, 512]
    config.network_parameters["channels"] = [32, 64]
    config.network_parameters["out_channels"] = 1
    # config.network_parameters["strides"] = [2, 2, 2]
    config.network_parameters["strides"] = [2]
    # config.network_parameters["num_res_units"] = 2
    config.network_parameters["num_res_units"] = 1
    config.network_parameters["norm"] = "BATCH"
    config.network_parameters["bias"] = False
    config.save["lesion_checkpoint"] = "lesion_cp"
    config.training["loss_parameters"] = {
        "sigmoid": True,
        "batch": True,
        "include_background": True,
    }
    config.training["metrics_parameters"] = {
        "ignore_empty": True,
        "include_background": False,
    }
    config.transforms["train_transform"] = "2d_ct_transform"
    config.transforms["test_transform"] = "2d_ct_transform"
    config.transforms["post_transform"] = "2d_ct_transform"


def test_segment_lesion():
    """
    Tests segment_lesion function (lesion_inference = '3D').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.
    """
    with patch.object(LiverSegmentation, "set_configs") as mock_set_configs:
        mock_set_configs.side_effect = lambda self, inference: print(
            "Mocked set_configs called"
        )
        liver_set_configs(inference="sliding_window")
        liver_model = LiverSegmentation(modality="CT", inference="sliding_window")

    with patch.object(LesionSegmentation, "set_configs"):
        set_configs()
        lesion_model = LesionSegmentation(inference="3D")
        liver_model.load_checkpoint(config.test["reference_sliding_window"])
        lesion_model.load_checkpoint(config.test["reference_lesion_cp"])
        liver_prediction = liver_model.predict(config.test["test_volume"])
        close = MorphologicalClosing(iters=4)
        lesion_prediction = lesion_model.predict(
            config.test["test_volume"],
            liver_mask=close(liver_prediction[0]).permute(3, 0, 1, 2),
        )
        liver_lesion_prediction = torch.tensor(
            np.where(lesion_prediction == 1, 2, liver_prediction)
        ).to(liver_prediction.device)

        # Assertion checks for lesion_prediction
        assert isinstance(liver_lesion_prediction, torch.Tensor)
        assert liver_lesion_prediction.shape[0] == 1
        assert liver_lesion_prediction.shape[1] == 1
        assert (
            liver_lesion_prediction.shape[2:]
            == nib.load(config.test["test_volume"]).get_fdata().shape
        )  # original volume dimension

        liver_lesion_prediction = liver_lesion_prediction.cpu()
        liver_lesion_prediction = liver_lesion_prediction.numpy()

        ref_path = config.test["refrence_lesions_array"]
        reference = np.load(ref_path)
        assert np.allclose(liver_lesion_prediction, reference, 0.01)


def test_train():
    """
    Tests train function (epoch: 1 , inference = '3D', volume = '64,64,..' ) .
    Compares the weights of the resulted checkpoint with the reference checkpoint's weights.
    """
    with patch.object(LesionSegmentation, "set_configs"):

        # Call the set_configs function
        set_configs()
        model = LesionSegmentation(inference="3D")
        # train a single epoch on the same volume
        train_lesion(
            pretrained=False,
            cp_path=config.test["lesion_cp"],
            epochs=1,
            evaluate_epochs=1,
            batch_callback_epochs=100,
            save_weight=True,
            save_path=config.test["lesion_cp"],
            test_batch_callback=False,
        )

        # load the previous checkpoint of training 1 epoch on a volume
        model.load_checkpoint(config.test["lesion_cp"])
        trained_weights = model.network.state_dict()

        # load reference checkpoint of training 1 epoch on the same volume
        model.load_checkpoint(config.test["reference_lesion_cp"])
        reference_weights = model.network.state_dict()

        # Check that weights match
        for i in reference_weights.keys():
            assert torch.allclose(reference_weights[i], trained_weights[i])
