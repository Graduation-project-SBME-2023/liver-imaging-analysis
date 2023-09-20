import pytest
from liver_imaging_analysis.models.lesion_segmentation import (
    LesionSegmentation,
    segment_lesion,
    train_lesion,
)
from liver_imaging_analysis.engine.transforms import MorphologicalClosing
from liver_imaging_analysis.models.liver_segmentation import segment_liver
from liver_imaging_analysis.engine.config import config
import torch
import nibabel as nib
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
    lesion_obj = LesionSegmentation()
    return lesion_obj


@pytest.mark.parametrize("transform_name", [("2d_ct_transform")])
def test_get_pretraining_transforms(transform_name, lesion_obj):
    """
    Tests get_pretraining_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "2d_ct_transform".
    lesion_obj (object): The lesion object that provides the get_pretraining_transforms method.
    ----------
    """

    output_transform = lesion_obj.get_pretraining_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "2d_ct_transform":
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


@pytest.mark.parametrize("transform_name", [("2d_ct_transform")])
def test_get_pretesting_transforms(transform_name, lesion_obj):
    """
    Tests the get_pretesting_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string): "2d_ct_transform".
    lesion_obj (object): The lesion object that provides the get_pretesting_transforms method.
    ----------
    """
    output_transform = lesion_obj.get_pretesting_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "2d_ct_transform":
        expected_transfom = [
            LoadImageD,
            EnsureChannelFirstD,
            ScaleIntensityRanged,
            ToTensorD,
        ]

    for t, e in zip(output_transform.transforms, expected_transfom):
        assert isinstance(t, e)


@pytest.mark.parametrize("transform_name", [("2d_ct_transform")])
def test_get_postprocessing_transforms(transform_name, lesion_obj):
    """
    Tests get_postprocessing_transforms function.
    verifies the functionality by comparing the output transforms with the expected transforms based on the provided transform_name.

    Parameters:
    ----------
    transform_name (string):"2d_ct_transform".
    lesion_obj (object): The lesion object that provides the get_postprocessing_transforms function.
    ----------
    """
    output_transform = lesion_obj.get_postprocessing_transforms(transform_name)
    assert isinstance(output_transform, Compose)

    if transform_name == "2d_ct_transform":
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

    liver_mask = segment_liver(
        prediction_path=config.test["test_volume"],
        modality="CT",
        inference="sliding_window",
        cp_path=config.test["reference_sliding_window"],
    )
    lesion_inference = "3D"
    volume_dir = config.test["test_volume"]
    close = MorphologicalClosing(iters=4)
    prediction = lesion_obj.predict_2dto3d(
        volume_path=volume_dir,
        liver_mask=close(liver_mask[0]).permute(3, 0, 1, 2)

    )

    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape[0] == 1
    assert prediction.shape[1] == 1  # number of channels

    assert (
        prediction.shape[2:] == nib.load(config.test["test_volume"]).get_fdata().shape
    )


def test_segment_lesion():
    """
    Tests segment_lesion function (lesion_inference = '3D').
    verifies the functionality by performing segmentation using a 3D volume (with size 64,64,...)
    Compares the results with the true reference segmentation.
    """
    lesion_prediction = segment_lesion(
        prediction_path=config.test["test_volume"],
        liver_inference="sliding_window",
        lesion_inference="3D",
        liver_cp=config.test["reference_sliding_window"],
        lesion_cp=config.test["reference_lesion_cp"],
    )

    # Assertion checks for lesion_prediction
    assert isinstance(lesion_prediction, torch.Tensor)
    assert lesion_prediction.shape[0] == 1
    assert lesion_prediction.shape[1] == 1
    assert (
        lesion_prediction.shape[2:]
        == nib.load(config.test["test_volume"]).get_fdata().shape
    )  # original volume dimension

    lesion_prediction = lesion_prediction.cpu()
    lesion_prediction = lesion_prediction.numpy()


    ref_path = "tests/testdata/testcases/predicted_array_lesion.npy"
    true = np.load(ref_path)
    assert np.allclose(lesion_prediction, true, 0.01)


def test_train():
    """
    Tests train function (epoch: 1 , inference = '3D', volume = '64,64,..' ) .
    Compares the weights of the resulted checkpoint with the reference checkpoint's weights.
    """
    model = LesionSegmentation(inference="3D")
    # load reference checkpoint of training 1 epoch on a volume
    model.load_checkpoint(config.test["reference_lesion_cp"])
    init_weights = model.network.state_dict()
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
    model.load_checkpoint(config.test["lesion_cp"])
    loaded_weights = model.network.state_dict()
    # Check that weights match
    for i in init_weights.keys():
        assert torch.allclose(init_weights[i], loaded_weights[i])
