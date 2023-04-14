from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.engine import Engine, set_seed
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
from monai.data import Dataset,decollate_batch
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

def set_configs():
    config.dataset['prediction']="test cases/sample_image"
    config.training['batch_size']=8
    config.training['scheduler_parameters']={"step_size":20, "gamma":0.5, "verbose":False}
    config.network_parameters['dropout']= 0
    config.network_parameters['channels']= [64, 128, 256, 512]
    config.network_parameters['strides']=  [2, 2, 2]
    config.network_parameters['num_res_units']=  4
    config.network_parameters['norm']= "INSTANCE"
    config.network_parameters['bias']= 1
    config.save['liver_checkpoint']= 'liver_cp'
    config.transforms['mode']= "3D"


class LiverSegmentation(Engine):
    """

    a class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """
    def __init__(self):
        set_configs()
        super().__init__()
    
    def get_pretraining_transforms(self, transform_name, keys):
        """
        Function used to define the needed transforms for the training data

        Args:
             transform_name: string
             Name of the required set of transforms
             keys: list
             Keys of the corresponding items to be transformed.

        Return:
            transforms: compose
             Return the compose of transforms selected
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),  # preferred by radiologists
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(
                        keys,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        prob=0.5,
                        keep_size=True,
                    ),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest")),
                    RandZoomd(keys,prob=0.5, min_zoom=0.8, max_zoom=1.2),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(keys, range_x=1.5, range_y=0, range_z=0, prob=0.5),
                    RandAdjustContrastd(keys[0], prob=0.25),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True), #remove for lesion segmentation
                    ToTensorD(keys),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name, keys):
        """
        Function used to define the needed transforms for the training data

        Args:
             transform_name(string): name of the required set of transforms
             keys(list): keys of the corresponding items to be transformed.

        Return:
            transforms(compose): return the compose of transforms selected

        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),  # preferred by radiologists
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    ResizeD(
                        keys,
                        resize_size,
                        mode=("bilinear", "nearest"),
                        allow_missing_keys=True,
                    ),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(
                        keys[1], threshold=0.5, invert=True, allow_missing_keys=True
                    ),#remove for lesion segmentation
                    ToTensorD(keys, allow_missing_keys=True),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]


    def get_postprocessing_transforms(self,transform_name, keys):
        """
        Function used to define the needed post processing transforms for prediction correction

        Args:
             transform_name(string): name of the required set of transforms
        Return:
            transforms(compose): return the compose of transforms selected

        """
        transforms= {

        '2DUnet_transform': Compose(
            [
                ActivationsD(keys[1],sigmoid=True),
                AsDiscreteD(keys[1],threshold=0.5),
                FillHolesD(keys[1]),
                KeepLargestConnectedComponentD(keys[1]),   
            ]
        )
        } 
        return transforms[transform_name] 


    def per_batch_callback(self, batch_num, image, label, prediction):
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


    def per_epoch_callback(self, epoch, training_loss, valid_loss, training_metric, valid_metric):
        print("\nTraining Loss=", training_loss)
        print("Training Metric=", training_metric)

        summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
        summary_writer.add_scalar("\nTraining Metric", training_metric, epoch)

        if valid_loss is not None:
            print(f"Validation Loss={valid_loss}")
            print(f"Validation Metric={valid_metric}")

            summary_writer.add_scalar("\nValidation Loss", valid_loss, epoch)
            summary_writer.add_scalar("\nValidation Metric", valid_metric, epoch)


    def predict_2dto3d(self, volume_path,temp_path="temp/"):
        """
        predicts the label of a 3D volume using a 2D network
        Parameters
        ----------
        volume_path: str
            path of the input file. expects a 3D nifti file.
        temp_path: str
            a temporary path to save 3d volume as 2d png slices. default is "temp/"
            automatically deleted before returning the prediction

        Returns
        -------
        tensor
            tensor of the predicted labels with shape (1,channel,length,width,depth) 
        """
        keys = (config.transforms["img_key"], config.transforms["pred_key"])
        #read volume
        img_volume = SimpleITK.ReadImage(volume_path)
        img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        number_of_slices = img_volume_array.shape[0]
        #create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
          os.mkdir(temp_path)
        #write volume slices as 2d png files 
        for slice_number in range(number_of_slices):
            volume_silce = img_volume_array[slice_number, :, :]
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]  # delete extension from filename
            volume_png_path = (os.path.join(temp_path, volume_file_name + "_" + str(slice_number))+ ".png")
            cv2.imwrite(volume_png_path, volume_silce)
        #predict slices individually then reconstruct 3D prediction
        pred_dict={keys[1]:self.predict(temp_path)}
        #transform shape from (batch,channel,length,width) to (1,channel,length,width,batch) 
        pred_dict[keys[1]]=pred_dict[keys[1]].permute(1,2,3,0).unsqueeze(dim=0) 
        #Apply post processing transforms on 3D prediction
        if (config.transforms['mode']=="3D"):
            postpred_dict = [self.postprocessing_transforms(i) for i in decollate_batch(pred_dict)]
            pred_dict[keys[1]] = from_engine(keys[1])(postpred_dict)   
            pred_dict[keys[1]]= torch.stack(pred_dict[keys[1]],dim=0)
        #delete temporary folder
        shutil.rmtree(temp_path)
        return pred_dict[keys[1]]



def segment_liver(*args):
    """
    a function used to segment the liver of 2D images using a 2D liver model

    """

    set_seed()
    liver_model = LiverSegmentation()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict(config.dataset['prediction'])
    return liver_prediction


def segment_liver_3d(*args):
    """
    a function used to segment the liver of a 3d volume using a 2D liver model

    """
    set_seed()
    liver_model = LiverSegmentation()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict_2dto3d(volume_path=args[0])
    return liver_prediction


def train_liver(*args):
    """
    a function used to start the training of liver segmentation

    """
    set_seed()
    model = LiverSegmentation()
    model.data_status()
    # model.load_checkpoint(config.save["potential_checkpoint"])
    print("Initial test loss:", model.test(model.test_dataloader, callback=False))#FAlSE
    model.fit(
        evaluate_epochs=1,
        batch_callback_epochs=100,
        save_weight=True,
    )
    model.load_checkpoint(config.save["potential_checkpoint"]) # evaluate on latest saved check point
    print("final test loss:", model.test(model.test_dataloader, callback=False))