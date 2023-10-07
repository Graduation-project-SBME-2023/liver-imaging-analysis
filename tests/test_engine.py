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
    ResizeD,
    ToTensorD,
    ActivationsD,
    AsDiscreteD,
)

class TestEngine(Engine):
    """
    A class that inherits from the Engine class and overrides the get_pretraining_transforms,
    get_pretesting_transforms, and get_postprocessing_transforms methods. Inherits from Engine.
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
        config.dataset["prediction"] = "tests/testdata/data/volume"
        config.device = "cuda" if torch.cuda.is_available() else "cpu"
        config.dataset["training"] = "tests/testdata/data"
        config.dataset["testing"] = "tests/testdata/data"
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
        config.save["engine_checkpoint"] = "tests/testdata/checkpoints/engine_cp.pt"
        config.save["reference_checkpoint"] = "tests/testdata/checkpoints/reference_cp.pt"

    def get_pretraining_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the training data.

        Parameters
        ----------
             transform_name: str
                Name of the desired set of transforms.

        Return
        ----------
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
                        spatial_size=[16,16,8],
                        mode=("trilinear", "nearest", "nearest"),
                        allow_missing_keys=True,
                    ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
                    ForegroundMaskD(Keys.LABEL, threshold=0.5, invert=True),
                    ToTensorD(Keys.all(), allow_missing_keys=True),
                ]
            )
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the testing data.

        Parameters
        ----------
             transform_name: str
                Name of the desired set of transforms.

        Return
        ----------
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
                        spatial_size=[16,16,8],
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
        Gets a stack of post processing transforms to be used on predictions.

        Parameters
        ----------
             transform_name: str
                Name of the desired set of transforms.

        Return
        ----------
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
    Pytest fixture that initializes an instance of the TestEngine class with the configured parameters.

    Returns
    ----------
        engine: object
            An instance of the TestEngine class.
    """
    # Set seed for reproducibility
    set_seed()
    engine = TestEngine()
    engine.load_data()

    return engine


def test_set_configs(engine):
    """
    Tests optimizer, scheduler, loss, and network initialization.

    Parameters
    ----------
        engine: object
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
    # Verify optimizer 
    assert isinstance(engine.optimizer, torch.optim.Adam)
    assert engine.optimizer.defaults["lr"] == 0.01

    # Verify scheduler 
    assert isinstance(engine.scheduler, lr_scheduler.StepLR)
    assert engine.scheduler.step_size == 20
    assert engine.scheduler.gamma == 0.5

    # Verify loss function 
    assert isinstance(engine.loss, monaiDiceLoss)

    # Verify network 
    assert isinstance(engine.network, monai.networks.nets.UNet)


def test_load_data(engine):
    """
    Tests the load_data method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
        
    image_path="tests/testdata/testcases/engine_image.npy"
    label_path="tests/testdata/testcases/engine_label.npy"

    for train_batch in engine.train_dataloader:

        expected_image = np.load(image_path)
        expected_label = np.load(label_path)

        assert train_batch[Keys.IMAGE].shape == expected_image.shape
        assert train_batch[Keys.LABEL].shape == expected_label.shape
        assert np.allclose(train_batch["image"].numpy(), expected_image, 1e-2)
        assert np.allclose(train_batch["label"].numpy(), expected_label, 1e-2)

    # Verify length of training data 
    assert len(engine.train_dataloader) == 1
    
    # Verify length of testing data
    assert len(engine.test_dataloader) == 1
   
    assert isinstance(engine.train_dataloader, MonaiLoader)
    assert isinstance(engine.val_dataloader, MonaiLoader)
    assert isinstance(engine.test_dataloader, MonaiLoader)


def test_save_load_checkpoint(engine):
    """
    Tests the save and load checkpoint method of the engine.

    Parameters
    ----------
        engine: TestEngine
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
    # Path of reference checkpoint
    ref_path =  config.save["reference_checkpoint"]

    # Path to save the checkpoint
    saved_cp ="tests/testdata/checkpoints/saved_cp.pt"

    # Assert reference checkpoint file exists
    assert os.path.exists(ref_path)

    # Test for saving checkpoint
    reference_checkpoint = {
        'state_dict': engine.network.state_dict(),
        'optimizer': engine.optimizer.state_dict(),
        'scheduler': engine.scheduler.state_dict(),
        }
    # Save checkpoint
    engine.save_checkpoint(saved_cp)

    # Assert checkpoint file exists
    assert os.path.exists(saved_cp)

    # load saved checkpoint
    loaded_checkpoint = torch.load(saved_cp)

    # verify network state dict match
    assert list(reference_checkpoint["state_dict"]) == list(loaded_checkpoint["state_dict"])
    # verify optimizer state dict match
    assert list(reference_checkpoint["optimizer"].values()) == list(loaded_checkpoint["optimizer"].values())
    # verify scheduler state dict match
    assert list(reference_checkpoint["scheduler"].values()) == list(loaded_checkpoint["scheduler"].values())
   
    # Delete saved checkpoint
    if os.path.exists(saved_cp):
        os.remove(saved_cp)

    # Test for loading checkpoint
    # load reference checkpoint
    loaded_checkpoint = torch.load(ref_path)
    loadd_weights = [value.to(engine.device) for value in loaded_checkpoint["state_dict"].values()]

    # Get loaded refrence weights
    engine.load_checkpoint(ref_path)
    ref_weights = [p.clone() for p in engine.network.parameters()]

    # Check that weights match
    for p0, p1 in zip(list(ref_weights) , list(loadd_weights)):
        assert torch.allclose(p0, p1, atol=1e-3)
    

def test_fit(engine):
    """
    Tests the fit method of the engine.

    Parameters
    ----------

        engine: object
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
    # Path to save the model weights
    save_path = config.save["engine_checkpoint"]

    # Path of reference checkpoint
    ref_path =  config.save["reference_checkpoint"]

    # Train model for one epoch
    engine.fit(epochs=1, save_weight=True, save_path=save_path)

    # Assert checkpoint file exists
    assert os.path.exists(save_path)

    checkpoint_weights = [p.clone() for p in engine.network.parameters()]

    # Get loaded refrence weights
    engine.load_checkpoint(ref_path)
    ref_weights = [p.clone() for p in engine.network.parameters()]

    
    # verify the wights match the reference checkpoint
    for p0, p1 in zip(checkpoint_weights, ref_weights):
        assert torch.allclose(p0, p1, atol=1e-3)
    
    # Delete saved checkpoint
    if os.path.exists(save_path):
        os.remove(save_path)
     

def test_test(engine):
    """
    Tests the test method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
    # Test model
    loss, metric = engine.test()

    # Verify loss 
    assert np.allclose(loss,0.913, atol=1e-3)

    # verify metric
    assert np.allclose(metric, 0.082, atol=1e-3)


def test_predict(engine):
    """
    Tests the predict method of the engine.

    Parameters
    ----------
        engine: object
            An instance of the TestEngine class.

    Returns
    ----------
        None
    """
    # Path of the input directory
    path=config.dataset["prediction"]

    # path to the predicted 3D numpy array
    pred_path="tests/testdata/testcases/engine_predicted.npy"

    # Get prediction
    prediction = engine.predict(path).cpu().numpy()

    # Load true prediction
    true = np.load(pred_path)

    # Verify prediction shape
    assert prediction.shape == true.shape == torch.Size([1, 1, 16, 16, 8]) 

    # Verify prediction values
    assert np.allclose(prediction, true, 1e-2)