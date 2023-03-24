from engine.config import config
from engine.engine import Engine, set_seed
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
    CropForegroundd
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric
import os
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset
import torch
import numpy as np
from monai.transforms import ToTensor
from models.liver_segmentation import LiverSegmentation

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric=DiceMetric(ignore_empty=True,include_background=True)


class LesionSegmentation(Engine):
    """

    a class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """
    def __init__(self):
        config.dataset['prediction']="prediction_volume"
        config.training['batch_size']=1
        config.network_parameters['dropout']= 0
        config.network_parameters['channels']= [64, 128, 256, 512]
        config.network_parameters['strides']=  [2, 2, 2]
        config.network_parameters['num_res_units']=  0
        config.network_parameters['norm']= "INSTANCE"
        config.network_parameters['bias']= 1
        config.save['lesion_checkpoint']= 'lesion_cp'
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
                    # ForegroundMaskD(keys[1], threshold=0.5, invert=True), #remove for lesion segmentation
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
                    # ForegroundMaskD(
                    #     keys[1], threshold=0.5, invert=True, allow_missing_keys=True
                    # ),#remove for lesion segmentation
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

    def predict(self, data_dir, liver_mask):
        """
        predicts the liver & lesions mask given the liver mask
        Parameters
        ----------
        data_dir: str
            path of the input directory. expects nifti or png files.
        liver_mask: tensor
            the liver mask predicted by the liver model
        
        Returns
        -------
        tensor
            tensor of the predicted labels
        """
        self.network.eval()
        with torch.no_grad():
            volume_names = os.listdir(data_dir)
            volume_paths = [os.path.join(data_dir, file_name) for file_name in volume_names]
            predict_files = [{"image": image_name} for image_name in volume_paths]
            predict_set = Dataset(data=predict_files, transform=self.test_transform)
            predict_loader = MonaiLoader(
                predict_set,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,
            )
            prediction_list = []
            for batch in predict_loader:
                volume = batch["image"].to(self.device)
                suppressed_volume=np.where(liver_mask==1,volume,volume.min())
                suppressed_volume=ToTensor()(suppressed_volume).to(self.device)
                pred = self.network(volume)
                pred = (torch.sigmoid(pred) > 0.5).float()
                prediction_list.append(pred)
            prediction_list = torch.cat(prediction_list, dim=0)

        return prediction_list
    
    # def load_checkpoint(self, path=config.save["model_checkpoint"]):
    #     self.network.load_state_dict(
    #         torch.load(path, map_location=torch.device(self.device))
    #     )

def segment_lesion(*args):
    """
    a function used to segment the liver lesions using the liver and the lesion models

    """

    set_seed()
    liver_model = LiverSegmentation()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    lesion_model = LesionSegmentation()
    lesion_model.load_checkpoint(config.save["lesion_checkpoint"])
    liver_prediction=liver_model.predict(config.dataset['prediction'])
    lesion_prediction= lesion_model.predict(config.dataset['prediction'],liver_prediction)
    lesion_prediction=lesion_prediction*liver_prediction #no liver -> no lesion
    liver_lesion_prediction=lesion_prediction+liver_prediction #lesion label is 2
    return liver_lesion_prediction
