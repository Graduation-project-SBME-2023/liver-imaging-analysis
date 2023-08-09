import os
import pytest
import torch
import monai
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from monai.data import DataLoader as MonaiLoader
from monai.losses import DiceLoss as monaiDiceLoss
from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.dataloader import Keys
from liver_imaging_analysis.engine.engine import Engine, set_seed
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImageD,
    NormalizeIntensityD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    RandAdjustContrastd,
    RandZoomd,
    ActivationsD,
    AsDiscreteD,
)


def set_configs():
    """
    Sets new values for config parameters.

    Parameters
    ----------
        None

    Returns
    ----------
        None
    """
    config.dataset["prediction"] = "test cases/sample"
    config.device = "cuda"
    config.dataset["training"] = "Temp2D/Train/"
    config.dataset["testing"] = "Temp2D/Test/"
    config.training["batch_size"] = 8
    config.training["scheduler_parameters"] = {
        "step_size": 20,
        "gamma": 0.5,
        "verbose": False,
    }
    config.network_parameters["dropout"] = 0
    config.network_parameters["spatial_dims"] = 3
    config.network_parameters["channels"] = [8, 16, 32, 64]
    config.network_parameters["strides"] = [2, 2, 2]
    config.network_parameters["num_res_units"] = 0
    config.network_parameters["norm"] = "INSTANCE"
    config.network_parameters["bias"] = True


@pytest.fixture
def engine():
    """
    Pytest fixture that initializes an instance of the Engine class with the configured parameters.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.

    Returns
    ----------
        None
    """
    set_configs()
    set_seed()
    engine = Engine()
    return engine


def test_set_configs(engine):
    """
    Tests the get_optimizer method of the engine with the Adam optimizer.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.

    Returns
    ----------
        None
    """

    assert isinstance(engine.optimizer, torch.optim.Adam)
    assert engine.optimizer.defaults["lr"] == 0.01

    assert isinstance(engine.scheduler, lr_scheduler.StepLR)
    assert engine.scheduler.step_size == 20
    assert engine.scheduler.gamma == 0.5

    assert isinstance(engine.network, monai.networks.nets.UNet)

    assert isinstance(engine.loss, monaiDiceLoss)


def get_transforms(engine):
    """
    Sets the pre-processing, and post-processing transforms for the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.

    Returns
    ----------
        None
    """
    engine.train_transform = Compose(
        [
            LoadImageD(Keys.all(), allow_missing_keys=True),
            EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
            ResizeD(
                Keys.all(),
                spatial_size=[256, 256, 128],
                mode=("trilinear", "nearest", "nearest"),
                allow_missing_keys=True,
            ),
            RandZoomd(
                Keys.all(),
                prob=0.5,
                min_zoom=0.8,
                max_zoom=1.2,
                allow_missing_keys=True,
            ),
            RandFlipd(Keys.all(), prob=0.5, spatial_axis=1, allow_missing_keys=True),
            RandRotated(
                Keys.all(),
                range_x=1.5,
                range_y=0,
                range_z=0,
                prob=0.5,
                allow_missing_keys=True,
            ),
            RandAdjustContrastd(Keys.IMAGE, prob=0.25),
            NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
            ForegroundMaskD(Keys.LABEL, threshold=0.5, invert=True),
            ToTensorD(Keys.all(), allow_missing_keys=True),
        ]
    )

    engine.test_transform = Compose(
        [
            LoadImageD(Keys.all(), allow_missing_keys=True),
            EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
            ResizeD(
                Keys.all(),
                spatial_size=[256, 256, 128],
                mode=("trilinear", "nearest", "nearest"),
                allow_missing_keys=True,
            ),
            NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
            ForegroundMaskD(
                Keys.LABEL, threshold=0.5, invert=True, allow_missing_keys=True
            ),
            ToTensorD(Keys.all(), allow_missing_keys=True),
        ]
    )

    engine.postprocessing_transforms = Compose(
        [ActivationsD(Keys.PRED, sigmoid=True), AsDiscreteD(Keys.PRED, threshold=0.5)]
    )


def test_load_data(engine):
    """
    Tests the load_data method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.

    Returns
    ----------
        None
    """

    get_transforms(engine)
    engine.load_data()

    assert len(engine.train_dataloader) > 0
    assert len(engine.test_dataloader) > 0
    assert isinstance(engine.train_dataloader, MonaiLoader)
    assert isinstance(engine.val_dataloader, MonaiLoader)
    assert isinstance(engine.test_dataloader, MonaiLoader)


def test_save_checkpoint(engine, ckpt_path="Checkpoint"):
    """
    Tests the save_checkpoint method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.
        ckpt_path: str
            Path of the input directory.

    Returns
    ----------
        None
    """

    assert os.path.exists(ckpt_path)

    loaded_checkpoint = torch.load(ckpt_path)

    engine.load_checkpoint(ckpt_path)

    # verify network state dict match
    assert set(engine.network.state_dict()) == set(loaded_checkpoint["state_dict"])
    # verify optimizer state dict match
    assert set(engine.optimizer.state_dict()) == set(loaded_checkpoint["optimizer"])
    # verify scheduler state dict match
    assert set(engine.scheduler.state_dict()) == set(loaded_checkpoint["scheduler"])


def test_load_checkpoint(
    engine, checkpoint_path="Checkpoint", ref_path="ref_Checkpoint"
):
    """
    Tests the load_checkpoint method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.
        checkpoint_path: str
            Path of the input directory.
        ref_path: str
            Path to check the model weights.

    Returns
    ----------
        None
    """
    get_transforms(engine)

    engine.load_checkpoint(ref_path)

    ref_weights = engine.network.state_dict()

    engine.load_checkpoint(checkpoint_path)

    # Get loaded weights
    loaded_weights = engine.network.state_dict()

    # Check that weights match
    for i in ref_weights.keys():
        assert torch.allclose(ref_weights[i], loaded_weights[i], atol=1e-3)


def test_fit(engine, save_path="Checkpoint", ref_path="ref_checkpoint"):
    """
    Tests the fit method of the engine.

    Parameters
    ----------

        engine: object
            An instance of the Engine class.

        save_path: str
            Path to save the model weights.

        ref_path: str
            Path to check the model weights.

    Returns
    ----------
        None
    """
    get_transforms(engine)
    engine.load_data()

    # Get initial weights
    init_weights = [p.clone() for p in engine.network.parameters()]

    # Get loaded weights
    engine.load_checkpoint(ref_path)
    ref_weights = engine.network.parameters()

    engine.fit(save_path=save_path)

    assert os.path.exists(save_path)

    # Verify weights were updated
    for p0, p1 in zip(init_weights, engine.network.parameters()):
        assert not torch.equal(p0, p1)

    # Check that weights match
    for p0, p1 in zip(ref_weights, engine.network.parameters()):
        assert torch.allclose(p0, p1, atol=1e-3)


def test_test(engine):
    """
    Tests the test method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.

    Returns
    ----------
        None
    """

    get_transforms(engine)
    engine.load_data()

    loss, metric = engine.test()

    assert np.allclose(loss, 0.954, atol=1e-3)
    assert np.allclose(metric, 0.0429, atol=1e-3)


def test_predict(engine, path="temp/", pred_path="test_temp\predicted.npy"):
    """
    Tests the predict method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.
        path: str
            Path of the input directory.
        pred_path: str
            The path to the predicted 3D numpy array. In this test case, it is set to "predicted.npy".

    Returns
    ----------
        None
    """

    get_transforms(engine)
    engine.load_data()

    predicted = engine.predict(path)
    prediction = predicted.cpu()
    prediction = prediction.numpy()
    final_path = pred_path
    np.save(final_path, prediction)
    true = np.load(final_path)

    assert predicted.shape == torch.Size([1, 1, 256, 256, 128])
    assert predicted.shape == true.shape
    assert np.allclose(prediction, true, 1e-2)
