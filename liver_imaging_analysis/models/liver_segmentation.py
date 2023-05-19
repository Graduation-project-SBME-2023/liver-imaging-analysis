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
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
import os
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset, decollate_batch
import torch
import numpy as np
from monai.transforms import ToTensor
import SimpleITK
import cv2
import shutil
import natsort
from monai.handlers.utils import from_engine

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric=DiceMetric(ignore_empty=True,include_background=True)
    
class LiverSegmentation(Engine):
    """
    A class used for the liver segmentation task. Inherits from Engine.

    Args:
        mode: str
            determines the mode of inference. 
            Expects "2D" for slice inference, "3D" for volume inference,
            or "sliding_window" for sliding window inference.
            Default is "2D"
    """

    def __init__(self, mode = "2D"):
        self.set_configs(mode)
        super().__init__()
        if mode == '3D':
            self.predict = self.predict_2dto3d
        elif mode == 'sliding_window':
            self.predict = self.predict_sliding_window

    def set_configs(self, mode):
        """
        Sets new values for config parameters.

        Args:
            mode: str
                chooses the specified set of configs.
        """
        if mode in ['2D', '3D']:
            config.dataset['prediction'] = "test cases/sample_image"
            config.training['batch_size'] = 8
            config.training['scheduler_parameters'] = {
                                                        "step_size":20,
                                                        "gamma":0.5, 
                                                        "verbose":False
                                                        }
            config.network_parameters['dropout'] = 0
            config.network_parameters['channels'] = [64, 128, 256, 512]
            config.network_parameters['strides'] =  [2, 2, 2]
            config.network_parameters['num_res_units'] =  4
            config.network_parameters['norm'] = "INSTANCE"
            config.network_parameters['bias'] = True
            config.save['liver_checkpoint'] = 'liver_cp'
            config.transforms['test_transform'] = "2DUnet_transform"
            config.transforms['post_transform'] = "2DUnet_transform"
        elif mode == 'sliding_window':
            config.dataset['prediction']="test cases/sample_volume"
            config.training['batch_size'] = 1
            config.training['scheduler_parameters'] = {
                                                        "step_size":20,
                                                        "gamma":0.5, 
                                                        "verbose":False
                                                        }
            config.network_parameters['dropout'] = 0
            config.network_parameters['channels'] = [64, 128, 256, 512]
            config.network_parameters['spatial_dims'] = 3
            config.network_parameters['strides'] =  [2, 2, 2]
            config.network_parameters['num_res_units'] =  6
            config.network_parameters['norm'] = "BATCH"
            config.network_parameters['bias'] = False
            config.save['liver_checkpoint'] = 'liver_cp_sliding_window'
            config.transforms['test_transform'] = "3DUnet_transform"
            config.transforms['post_transform'] = "3DUnet_transform"
    
    def get_pretraining_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the training data.

        Args:
             transform_name: str
                Name of the required set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys=True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
                    # OrientationD(keys, axcodes="LAS", allow_missing_keys=True),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
                    ForegroundMaskD(
                        Keys.LABEL,
                        threshold=0.5,
                        invert=True,
                        allow_missing_keys=True
                        ),
                    ToTensorD(Keys.all(), allow_missing_keys=True),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys=True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode=("bilinear", "nearest", "nearest"), 
                        allow_missing_keys=True
                        ),
                    RandZoomd(
                        Keys.all(),
                        prob=0.5, 
                        min_zoom=0.8, 
                        max_zoom=1.2, 
                        allow_missing_keys=True
                        ),
                    RandFlipd(
                        Keys.all(),
                        prob=0.5, 
                        spatial_axis=1, 
                        allow_missing_keys=True
                        ),
                    RandRotated(
                        Keys.all(),
                        range_x=1.5, 
                        range_y=0, 
                        range_z=0, 
                        prob=0.5, 
                        allow_missing_keys=True
                        ),
                    RandAdjustContrastd(Keys.IMAGE, prob=0.25),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
                    ForegroundMaskD(Keys.LABEL, threshold=0.5, invert=True),
                    ToTensorD(Keys.all(), allow_missing_keys=True),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the testing data.

        Args:
             transform_name: str
                Name of the required set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys=True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
                    # OrientationD(keys, axcodes="LAS", allow_missing_keys=True),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
                    ForegroundMaskD(
                        Keys.LABEL,
                        threshold=0.5, 
                        invert=True, 
                        allow_missing_keys=True
                        ),
                    ToTensorD(Keys.all(), allow_missing_keys=True),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys=True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys=True),
                    ResizeD(
                        Keys.all(),
                        resize_size,
                        mode=("bilinear", "nearest", "nearest"),
                        allow_missing_keys=True,
                    ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise=True),
                    ForegroundMaskD(
                        Keys.LABEL,
                        threshold=0.5, 
                        invert=True, 
                        allow_missing_keys=True
                    ),
                    ToTensorD(Keys.all(), allow_missing_keys=True),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]


    def get_postprocessing_transforms(self,transform_name):
        """
        Gets a stack of post processing transforms to be used on predictions.

        Args:
             transform_name: str
                Name of the required set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        transforms= {

        '3DUnet_transform': Compose(
            [
                ActivationsD(Keys.PRED,sigmoid=True),
                AsDiscreteD(Keys.PRED,threshold=0.5),
                FillHolesD(Keys.PRED),
                KeepLargestConnectedComponentD(Keys.PRED),   
            ]
        ),

        '2DUnet_transform': Compose(
            [
                ActivationsD(Keys.PRED,sigmoid=True),
                AsDiscreteD(Keys.PRED,threshold=0.5),
                FillHolesD(Keys.PRED),
                KeepLargestConnectedComponentD(Keys.PRED),   
            ]
        )

        } 
        return transforms[transform_name] 


    def per_batch_callback(self, batch_num, image, label, prediction):
        """
        Plots image, label and prediction into tensorboard,
        and prints the prediction dice score.

        Args:
            batch_num: int
                The current batch index for identification.
            image: tensor
                original input image
            label: tensor
                target liver mask
            prediction:
                predicted liver mask
        """

        dice_score=dice_metric(prediction.int(),label.int())[0].item()
        plot_2d_or_3d_image(
            data=image,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Volume:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data=label,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Mask:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data=prediction,
            step=0,
            writer=summary_writer,
            frame_dim=-1,
            tag=f"Batch{batch_num}:Prediction:dice_score:{dice_score}",
        )

    def per_epoch_callback(
            self,
            epoch, 
            training_loss, 
            valid_loss, 
            training_metric, 
            valid_metric
            ):
        """
        Prints training and testing loss and metric,
        and plots them in tensorboard.

        Args:
            epoch: int
                Current epoch index for identification.
            training_loss: float
                Loss calculated over the training set.
            valid_loss: float
                Loss calculated over the testing set.
            training_metric: float
                Metric calculated over the training set.
            valid_metric: float
                Metric calculated over the testing set.
        """
        print("\nTraining Loss=", training_loss)
        print("Training Metric=", training_metric)
        summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
        summary_writer.add_scalar("\nTraining Metric", training_metric, epoch)
        if valid_loss is not None:
            print(f"Validation Loss={valid_loss}")
            print(f"Validation Metric={valid_metric}")
            summary_writer.add_scalar("\nValidation Loss", valid_loss, epoch)
            summary_writer.add_scalar("\nValidation Metric", valid_metric, epoch)


    def predict_2dto3d(self, volume_path, temp_path="temp/"):
        """
        Predicts the label of a 3D volume using a 2D network.

        Args:
            volume_path: str
                Path of the input file. expects a 3D nifti file.
            temp_path: str
                A temporary path to save 3d volume as 2d png slices. 
                Default is "temp/". 
                Automatically deleted before returning the prediction

        Returns:
            tensor
                Tensor of the predicted labels.
                Shape is (1,channel,length,width,depth).
        """
        # Read volume
        img_volume = SimpleITK.ReadImage(volume_path)
        img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        number_of_slices = img_volume_array.shape[0]
        # Create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
          os.mkdir(temp_path)
        # Write volume slices as 2d png files 
        for slice_number in range(number_of_slices):
            volume_slice = img_volume_array[slice_number, :, :]
            # Delete extension from filename
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
            volume_png_path = os.path.join(
                                    temp_path, 
                                    volume_file_name + "_" + str(slice_number)
                                    ) + ".png"
            cv2.imwrite(volume_png_path, volume_slice)
        # Predict slices individually then reconstruct 3D prediction
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(temp_path))
            volume_paths = [os.path.join(temp_path, file_name) 
                            for file_name in volume_names]
            predict_files = [{Keys.IMAGE: image_name} 
                             for image_name in volume_paths]
            predict_set = Dataset(
                data=predict_files, 
                transform=self.test_transform
                )
            predict_loader = MonaiLoader(
                predict_set,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,
            )
            prediction_list = []
            for batch in predict_loader:
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                batch[Keys.PRED] = self.network(batch[Keys.IMAGE])
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        batch = {Keys.PRED : prediction_list}
        # Transform shape from (batch,channel,length,width) 
        # to (1,channel,length,width,batch) 
        batch[Keys.PRED] = batch[Keys.PRED].permute(1,2,3,0).unsqueeze(dim=0) 
        # Apply post processing transforms
        batch = self.post_process(batch,Keys.PRED)
        # Delete temporary folder
        shutil.rmtree(temp_path)
        return batch[Keys.PRED]


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
                            data=predict_files, 
                            transform=self.test_transform
                            )
            predict_loader = MonaiLoader(
                                predict_set,
                                batch_size=self.batch_size,
                                num_workers=0,
                                pin_memory=False,
                            )
            prediction_list = []
            for batch in predict_loader:
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                # Predict by sliding window
                batch[Keys.PRED] = sliding_window_inference(
                                        batch[Keys.IMAGE], 
                                        (96,96,64), 
                                        4, 
                                        self.network
                                        )
                # Apply post processing transforms
                batch = self.post_process(batch,Keys.PRED)    
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        return prediction_list
    

def segment_liver(*args):
    """
    a function used to segment the liver of 2D images using a 2D liver model

    """

    set_seed()
    liver_model = LiverSegmentation(mode = '2D')
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict(config.dataset['prediction'])
    return liver_prediction


def segment_liver_3d(*args):
    """
    a function used to segment the liver of a 3d volume using a 2D liver model

    """
    set_seed()
    liver_model = LiverSegmentation(mode = '3D')
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict(volume_path=args[0])
    return liver_prediction


def segment_liver_sliding_window(*args):
    """
    a function used to segment the liver of a 3d volume using a 2D liver model

    """
    set_seed()
    liver_model = LiverSegmentation(mode = 'sliding_window')
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict(volume_path=args[0])
    return liver_prediction


def train_liver(*args):
    """
    a function used to start the training of liver segmentation

    """
    set_seed()
    model = LiverSegmentation(mode = '2D')
    model.load_data()
    model.data_status()
    model.load_checkpoint(config.save["potential_checkpoint"])
    print(
        "Initial test loss:", 
        model.test(model.test_dataloader, callback=False)
        )
    model.fit(
        evaluate_epochs=1,
        batch_callback_epochs=100,
        save_weight=True,
    )
    # Evaluate on latest saved check point
    model.load_checkpoint(config.save["potential_checkpoint"])
    print("final test loss:", model.test(model.test_dataloader, callback=False))