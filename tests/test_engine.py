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

class TestEngine(Engine):
    """
    A class that inherits from the Engine class and overrides the get_pretraining_transforms,
    get_pretesting_transforms, and get_postprocessing_transforms methods.

    Parameters
    ----------
        Engine: object
            An instance of the Engine class.
    """

    def __init__(self):
        """
        Initializes the TestEngine class.
        """
        self.set_configs()
        super().__init__()

    def set_configs(self):
        """
        Sets new values for config parameters.

        """
        config.dataset["prediction"] = "tests\\testdata\\test\\volume"
        config.device = "cpu"
        config.dataset["training"] = "tests\\testdata\\train"
        config.dataset["testing"] = "tests\\testdata\\test"
        config.training["batch_size"] = 8
        config.training["optimizer"] = "Adam"
        config.training["optimizer_parameters"] = {"lr": 0.01}
        config.training["lr_scheduler"] = "StepLR"
        config.training["scheduler_parameters"] = {
            "step_size": 20,
            "gamma": 0.5,
            "verbose": False,
        }
        config.network_name = "monai_2DUNet"
        config.network_parameters["dropout"] = 0
        config.network_parameters["spatial_dims"] = 3
        config.network_parameters["channels"] = [8, 16, 32, 64]
        config.network_parameters["strides"] = [2, 2, 2]
        config.network_parameters["num_res_units"] = 0
        config.network_parameters["norm"] = "INSTANCE"
        config.network_parameters["bias"] = True
        config.transforms["train_transform"] = "3d_ct_transform"
        config.transforms["test_transform"] = "3d_ct_transform"
        config.transforms["post_transform"] = "3d_ct_transform"

    def get_pretraining_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the training data.

        Args:
             transform_name: str
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """
        transforms = {
            "3d_ct_transform": Compose(
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
                    RandFlipd(
                        Keys.all(), prob=0.5, spatial_axis=1, allow_missing_keys=True
                    ),
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
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the training data.

        Args:
             transform_name: str
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """
        transforms = {
            "3d_ct_transform": Compose(
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
        }
        return transforms[transform_name]

    def get_postprocessing_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the training data.

        Args:
             transform_name: str
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """
        transforms = {
            "3d_ct_transform": Compose(
                [
                    ActivationsD(Keys.PRED, sigmoid=True),
                    AsDiscreteD(Keys.PRED, threshold=0.5),
                ]
            )
        }
        return transforms[transform_name]


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
    set_seed()
    engine = TestEngine()
    engine.load_data()
    return engine


def test_set_configs(engine):
    """
    Tests the get_optimizer method of the engine with the Adam optimizer,
    get_scheduler with the StepLR scheduler, get_loss with the "monai_dice" loss,
    and also checks if the network attribute of the engine is an instance of the `monai.networks.nets.UNet`class.

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

    assert isinstance(engine.loss, monaiDiceLoss)

    assert isinstance(engine.network, monai.networks.nets.UNet)


def test_load_data(
    engine,
    image_path="tests\\testdata\\engine_image.npy",
    label_path="tests\\testdata\\engine_label.npy",
):
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
    image_path = image_path
    label_path = label_path

    for train_batch in engine.train_dataloader:
        expected_image = np.load(image_path)
        expected_label = np.load(label_path)

        assert train_batch[Keys.IMAGE].shape == expected_image.shape
        assert train_batch[Keys.LABEL].shape == expected_label.shape

        assert np.allclose(train_batch["image"].numpy(), expected_image, 1e-2)
        assert np.allclose(train_batch["label"].numpy(), expected_label, 1e-2)

    assert len(engine.train_dataloader) == 1
    assert len(engine.test_dataloader) == 1
    assert isinstance(engine.train_dataloader, MonaiLoader)
    assert isinstance(engine.val_dataloader, MonaiLoader)
    assert isinstance(engine.test_dataloader, MonaiLoader)


def test_save_checkpoint(engine, ckpt_path="tests\\testdata\\engine_cp"):
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

    init_weights = engine.network.state_dict()
    assert os.path.exists(ckpt_path)

    loaded_checkpoint = torch.load(ckpt_path)

    engine.load_checkpoint(ckpt_path)

    # Get loaded weights
    loaded_weights = engine.network.state_dict()

    # verify network state dict match
    assert set(engine.network.state_dict()) == set(loaded_checkpoint["state_dict"])
    # verify optimizer state dict match
    assert set(engine.optimizer.state_dict()) == set(loaded_checkpoint["optimizer"])
    # verify scheduler state dict match
    assert set(engine.scheduler.state_dict()) == set(loaded_checkpoint["scheduler"])

    # Check that weights match
    for i in init_weights.keys():
        assert torch.allclose(init_weights[i], loaded_weights[i])


def test_load_checkpoint(
    engine,
    checkpoint_path="tests\\testdata\\engine_cp",
    ref_path="tests\\testdata\\reference_engine_cp",
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

    engine.load_checkpoint(ref_path)

    ref_weights = engine.network.state_dict()

    engine.load_checkpoint(checkpoint_path)

    # Get loaded weights
    loaded_weights = engine.network.state_dict()

    # Check that weights match
    for i in ref_weights.keys():
        assert torch.allclose(ref_weights[i], loaded_weights[i], atol=1e-3)


def test_fit(
    engine,
    save_path="tests\\testdata\\engine_cp",
    ref_path="tests\\testdata\\reference_engine_cp",
):
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
    # Get initial weights
    init_weights = [p.clone() for p in engine.network.parameters()]

    # Get loaded weights
    engine.load_checkpoint(ref_path)
    ref_weights = engine.network.parameters()

    engine.fit(epochs=3, save_weight=True, save_path=save_path)

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
    loss, metric = engine.test()

    assert np.allclose(loss, 0.922, atol=1e-3)
    assert np.allclose(metric, 0.06, atol=1e-3)


def test_predict(
    engine,
    path="tests\\testdata\\test\\volume",
    pred_path="tests\\testdata\\engine_predicted.npy",
):
    """
    Tests the predict method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the Engine class.
        path: str
            Path of the input directory.
        pred_path: str
            The path to the predicted 3D numpy array. In this test case, it is set to "engine_predicted.npy".

    Returns
    ----------
        None
    """

    predicted = engine.predict(path)
    prediction = predicted.cpu().numpy()

    final_path = pred_path
    true = np.load(final_path)

    assert predicted.shape == torch.Size([1, 1, 256, 256, 128])
    assert predicted.shape == true.shape
    assert np.allclose(prediction, true, 1e-2)