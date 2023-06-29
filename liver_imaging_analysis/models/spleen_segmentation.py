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
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset
import torch



class SpleenSegmentation(Engine):
    """
    A class used for the spleen segmentation task using pretrained checkpoint. 
    Inherits from Engine.
    """

    def __init__(self):
        self.set_configs()
        super().__init__()
        self.predict = self.predict_sliding_window

    def set_configs(self):
        """
        Sets new values for config parameters.
        """
        config.dataset['prediction']="test cases/sample_volume"
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
        config.save['spleen_checkpoint'] = 'Liver-Segmentation-Website/models_checkpoints/spleen_cp'
        config.transforms['sw_batch_size'] = 4
        config.transforms['roi_size'] = (96, 96, 96)
        config.transforms['overlap'] = 0.25
        config.transforms['train_transform'] = "3DUnet_transform"
        config.transforms['test_transform'] = "3DUnet_transform"
        config.transforms['post_transform'] = "3DUnet_transform"

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
            "3DUnet_transform": Compose(
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
            "3DUnet_transform": Compose(
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
            "3DUnet_transform": Compose(
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
                batch = self.post_process(batch,Keys.PRED)    
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim = 0)
        return prediction_list
    

def segment_spleen(volume_path):
    """
    a function used to segment the spleen of a 3d volume 
    using sliding window inference

    Parameters
    ----------
        volume_path: str
            3D volume path, expects a nifti file.

    Returns
    ----------
        tensor: predicted 3D spleen mask
    """
    set_seed()
    spleen_model = SpleenSegmentation()
    spleen_model.load_checkpoint(config.save["spleen_checkpoint"])
    spleen_prediction = spleen_model.predict(volume_path = volume_path)
    return spleen_prediction