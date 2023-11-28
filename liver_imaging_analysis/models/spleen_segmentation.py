from logger import setup_logger
from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.engine import Engine, set_seed
from liver_imaging_analysis.engine.dataloader import Keys
from monai.inferers import sliding_window_inference
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
    EnsureTyped,
    Spacingd,
    Invertd,
    AsDiscreted,
    Activationsd
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
import os
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset
import torch
import numpy as np
import SimpleITK
import cv2
import shutil
import natsort
import nibabel as nib
import argparse
import logging
logger = logging.getLogger(__name__)

class SpleenSegmentation(Engine):
    """
    A class used for the spleen segmentation task using pretrained checkpoint. 
    Inherits from Engine.
    """
    logger.info("SpleenSegmentation")

    def __init__(self):
        setup_logger(self.__class__.__name__)

        logger.info("SpleenSegmentation")

        logger.info("Loading configuration")
        self.set_configs()
        super().__init__()
        self.predict = self.predict_sliding_window

    def set_configs(self):
        """
        Sets new values for config parameters.
        """
        
        config.dataset['prediction'] = "test cases/volume/volume-64.nii"
        config.training['batch_size'] = 1
        config.training['scheduler_parameters'] = {
                                                    "step_size":20,
                                                    "gamma":0.5, 
                                                    "verbose":False
                                                    }
        config.network_parameters['dropout'] = 0
        config.network_parameters["out_channels"] = 2
        config.network_parameters['channels'] = [16, 32, 64, 128, 256]
        config.network_parameters['spatial_dims'] = 3
        config.network_parameters['strides'] =  [2, 2, 2, 2]
        config.network_parameters['num_res_units'] =  2
        config.network_parameters['norm'] = "BATCH"
        config.network_parameters['bias'] = True
        config.save['spleen_checkpoint'] = 'spleen_cp'
        config.transforms['sw_batch_size'] = 4
        config.transforms['roi_size'] = (96, 96, 96)
        config.transforms['overlap'] = 0.25
        config.transforms['train_transform'] = "3d_transform"
        config.transforms['test_transform'] = "3d_transform"
        config.transforms['post_transform'] = "3d_transform"

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
            "3d_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "RAS", 
                        allow_missing_keys = True
                        ),
                    Spacingd(
                        Keys.all(), 
                        pixdim = [1.5, 1.5, 2.0], 
                        mode = ("bilinear"), 
                        allow_missing_keys = True
                        ),
                    ScaleIntensityRanged(
                        Keys.all(),
                        a_min = -57,
                        a_max = 164,
                        b_min = 0,
                        b_max = 1,
                        clip = True, 
                        allow_missing_keys = True
                    ),
                    EnsureTyped(Keys.all(), allow_missing_keys = True),
                ]
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the testing data.

        Args:
             transform_name: str
                Name of the desire set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        transforms = {
            "3d_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "RAS", 
                        allow_missing_keys = True
                        ),
                    Spacingd(
                        Keys.all(), 
                        pixdim = [1.5, 1.5, 2.0], 
                        mode = ("bilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
                        ),
                    ScaleIntensityRanged(
                        Keys.all(),
                        a_min = -57,
                        a_max = 164,
                        b_min = 0,
                        b_max = 1,
                        clip = True, 
                        allow_missing_keys = True
                    ),
                    EnsureTyped(Keys.all(), allow_missing_keys = True),
                ]
            ),
        }
        return transforms[transform_name]


    def get_postprocessing_transforms(self,transform_name):
        """
        Gets a stack of post processing transforms to be used on predictions.

        Args:
             transform_name: str
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """
        transforms= {
            "3d_transform": Compose(
                [
                    Invertd(
                        Keys.PRED,
                        transform = self.test_transform,
                        orig_keys = Keys.IMAGE,
                        meta_keys = Keys.PRED + "_meta_dict",
                        orig_meta_keys = Keys.IMAGE + "_meta_dict",
                        meta_key_postfix = "meta_dict",
                        nearest_interp = False,
                        to_tensor = True,
                        device = self.device,
                    ),
                    Activationsd(Keys.PRED, softmax = True),
                    AsDiscreted(Keys.PRED, argmax = True),
                ]
            ),
        } 
        return transforms[transform_name] 
    

    def predict_sliding_window(self, volume_path):
        """
        predict the label of the given input
        Parameters
        ----------
        volume_path: str
            path of the input directory. expects nifti or png files.
        Returns
        -------
        tensor
            tensor of the predicted labels
        """
        logger.info("predict_sliding_window")
        self.network.eval()
        with torch.no_grad():
            predict_files = [{Keys.IMAGE : volume_path}] 
            predict_set = Dataset(
                            data = predict_files, 
                            transform = self.test_transform
                            )
            predict_loader = MonaiLoader(
                                predict_set,
                                batch_size = self.batch_size,
                                num_workers = 0,
                                pin_memory = False,
                            )
            prediction_list = []
            for batch in predict_loader:
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                # Predict by sliding window
                batch[Keys.PRED] = sliding_window_inference(
                                        batch[Keys.IMAGE], 
                                        config.transforms['roi_size'], 
                                        config.transforms['sw_batch_size'],      
                                        self.network
                                        )
                # Apply post processing transforms
                batch = self.post_process(batch)    
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim = 0)
            logger.info(f"prediction_list: {prediction_list}")
            logger.info("Prediction complete")
        return prediction_list
    

def segment_spleen(prediction_path = None, cp_path = None):
    """
    a function used to segment the spleen of a 3d volume 
    using sliding window inference

    Parameters
    ----------
    prediciton_path: str
        expects a path of a 3D nii volume.
        if not defined, prediction dataset will be loaded from configs
    cp_path : str
        path of the model weights to be used for prediction. 
        if not defined, spleen_checkpoint will be loaded from configs.

    Returns
    ----------
        tensor: predicted 3D spleen mask
    """
    logger.info("segment_spleen")
    spleen_model = SpleenSegmentation()
    if prediction_path is None:
        prediction_path = config.dataset['prediction']
        logger.info(f"prediction_path: {prediction_path}")
    if cp_path is None:
        cp_path = config.save["spleen_checkpoint"]
        logger.info(f"cp_path: {cp_path}")
    spleen_model.load_checkpoint(cp_path)
    spleen_prediction = spleen_model.predict(prediction_path)
    logger.info(f"spleen_prediction: {spleen_prediction}")
    return spleen_prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Spleen Segmentation')
    parser.add_argument(
                '--predict_path', type = str, default = None,
                help = 'predicts the volume at the provided path (default: prediction config path)'
                )
    parser.add_argument(
                '--cp_path', type = bool, default = None,
                help = 'path of model weights (default: spleen_checkpoint config path)'
                )
    args = parser.parse_args()
    SpleenSegmentation() # to set configs
    if args.predict_path is None:
        args.predict_path = config.dataset['prediction']
    if args.cp_path is None:
        args.cp_path = config.save["spleen_checkpoint"]
    prediction = segment_spleen(
                    prediction_path = args.predict_path, 
                    cp_path = args.cp_path
                    )
    #save prediction as a nifti file
    original_header = nib.load(args.predict_path).header
    original_affine = nib.load(args.predict_path).affine
    spleen_volume = nib.Nifti1Image(
                        prediction[0,0].cpu(), 
                        affine = original_affine, 
                        header = original_header
                        )
    nib.save(spleen_volume, args.predict_path.split('.')[0] + '_spleen.nii')
    print('Prediction saved at', args.predict_path.split('.')[0] + '_spleen.nii')
    print("Run Complete")