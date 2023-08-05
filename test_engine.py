################################################### IMPORT ############################################################
import os
import pytest
import torch
import monai
from monai.metrics import DiceMetric, MeanIoU
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
#######################################################################################################################
def set_configs():
       
    config.dataset['prediction'] = "test cases/sample_image"
    config.device="cpu"
    config.dataset['training'] = "Temp2D/Train/"
    config.dataset['testing'] = "Temp2D/Test/"
    config.training['batch_size'] = 8
    config.training['scheduler_parameters'] = {
                                                "step_size" : 20,
                                                "gamma" : 0.5, 
                                                "verbose" : False
                                                }
    config.network_parameters['dropout'] = 0
    config.network_parameters['spatial_dims']= 3 
    config.network_parameters['channels'] = [8,16,32,64]
    config.network_parameters['strides'] =  [2, 2, 2]
    config.network_parameters['num_res_units'] =  0
    config.network_parameters['norm'] = "INSTANCE"
    config.network_parameters['bias'] = True

#######################################################################################################################
@pytest.fixture
def engine():
    set_configs()
    set_seed()
    engine= Engine()
    return engine

#######################################################################################################################
def test_get_optimizer_adam(engine):
    opt = engine.get_optimizer('Adam', lr=0.001)
    assert isinstance(opt, torch.optim.Adam)
    assert opt.defaults['lr'] == 0.001

def test_get_optimizer_sgd(engine):
    opt = engine.get_optimizer('SGD', lr=0.01, momentum=0.9)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults['lr'] == 0.01
    assert opt.defaults['momentum'] == 0.9
    
def test_get_scheduler_steplr(engine):
    scheduler = engine.get_scheduler('StepLR', step_size=10, gamma=0.1)
    assert isinstance(scheduler, lr_scheduler.StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1

def test_get_network_unet(engine):
    assert isinstance(engine.network ,monai.networks.nets.UNet ) 


def test_get_loss_monai_dice(engine):
    loss_fn = engine.get_loss('monai_dice')
    assert isinstance(loss_fn, monaiDiceLoss) 

def test_get_metrics_dice(engine):
    metric = engine.get_metrics('dice')
    assert isinstance(metric, DiceMetric)

def test_get_metrics_jaccard(engine):
    metric = engine.get_metrics('jaccard')
    assert isinstance(metric, MeanIoU)

#########################################################################################################################
def get_transforms(engine):
    engine.train_transform =Compose(
                    [
                        LoadImageD(Keys.all(), allow_missing_keys = True),
                        EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                        ResizeD(
                            Keys.all(), 
                            spatial_size=[256, 256,128], 
                            mode=("trilinear", "nearest", "nearest"), 
                            allow_missing_keys = True
                            ),
                        RandZoomd(
                            Keys.all(),
                            prob = 0.5, 
                            min_zoom = 0.8, 
                            max_zoom = 1.2, 
                            allow_missing_keys = True
                            ),
                        RandFlipd(
                            Keys.all(),
                            prob = 0.5, 
                            spatial_axis = 1, 
                            allow_missing_keys = True
                            ),
                        RandRotated(
                            Keys.all(),
                            range_x = 1.5, 
                            range_y = 0, 
                            range_z = 0, 
                            prob = 0.5, 
                            allow_missing_keys = True
                            ),
                        RandAdjustContrastd(Keys.IMAGE, prob = 0.25),
                        NormalizeIntensityD(Keys.IMAGE, channel_wise = True),
                        ForegroundMaskD(Keys.LABEL, threshold = 0.5, invert = True),
                        ToTensorD(Keys.all(), allow_missing_keys = True),
                    ]
                )
    
    engine.test_transform = Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(),
                        spatial_size=[256, 256,128], 
                        mode=("trilinear", "nearest", "nearest"), 
                        allow_missing_keys = True,
                    ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise = True),
                    ForegroundMaskD(
                        Keys.LABEL,
                        threshold = 0.5, 
                        invert = True, 
                        allow_missing_keys = True
                    ),
                    ToTensorD(Keys.all(), allow_missing_keys = True),
                ]
            )
    engine.postprocessing_transforms = Compose([
    ActivationsD(Keys.PRED,sigmoid = True),
    AsDiscreteD(Keys.PRED,threshold = 0.5)
    
])
#########################################################################################################################
def test_load_data(engine):

    get_transforms(engine)
    engine.load_data()

    assert len(engine.train_dataloader) > 0
    assert len(engine.test_dataloader) > 0
    assert isinstance(engine.train_dataloader, MonaiLoader)
    assert isinstance(engine.val_dataloader, MonaiLoader)
    assert isinstance(engine.test_dataloader, MonaiLoader)


#########################################################################################################################
def test_save_checkpoint(engine,ckpt_path='Checkpount'):

    engine.save_checkpoint(ckpt_path) 

    assert os.path.exists(ckpt_path)
    
#########################################################################################################################
def test_load_checkpoint(engine,checkpoint_path = 'Checkpount'):

    get_transforms(engine) 

    init_weights = engine.network.state_dict()

    engine.load_checkpoint(checkpoint_path)

    # Get loaded weights
    loaded_weights = engine.network.state_dict()

    # Check that weights match
    for i in init_weights.keys():
        assert torch.allclose(init_weights[i], loaded_weights[i])
#########################################################################################################################

def test_fit(engine,
        epochs = 2, 
        evaluate_epochs = 1,
        batch_callback_epochs = 100,
        save_weight =False,
        save_path ='Checkpount',

        ):
    get_transforms(engine)
    engine.load_data()

    engine.fit(
        epochs = epochs,
        evaluate_epochs = evaluate_epochs,
        batch_callback_epochs = batch_callback_epochs,
        save_weight = save_weight,
        save_path = save_path
    )

    assert os.path.exists(save_path)

#########################################################################################################################
def test_test(engine):

    get_transforms(engine)
    engine.load_data()

    loss, metric = engine.test()

    assert loss != 0
    assert metric != 0

#########################################################################################################################
def test_predict(engine,path="temp/"):
    
    get_transforms(engine)
    engine.load_data()

    predicted = engine.predict(path)

    assert predicted.shape == torch.Size([1,1,256,256,128])

#########################################################################################################################
