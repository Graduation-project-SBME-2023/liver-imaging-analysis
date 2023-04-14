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
from monai.inferers import sliding_window_inference
from monai.data import DataLoader as MonaiLoader
from monai.data import Dataset,decollate_batch
import torch
import natsort
import os

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric=DiceMetric(ignore_empty=True,include_background=True)

class LiverSegmentation(Engine):
    """

    a class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """
    def __init__(self):
        config.dataset['prediction']="test cases/sample_volume"
        config.training['batch_size']=1
        config.training['scheduler_parameters']={"step_size":20, "gamma":0.5, "verbose":False}
        config.network_parameters['dropout']= 0
        config.network_parameters['channels']= [64, 128, 256, 512]
        config.network_parameters['spatial_dims']= 3
        config.network_parameters['strides']=  [2, 2, 2]
        config.network_parameters['num_res_units']=  6
        config.network_parameters['norm']= "BATCH"
        config.network_parameters['bias']= False
        config.save['liver_checkpoint']= 'liver_cp_sliding_window'
        config.transforms['mode']= "3D"
        config.transforms['test_transform']= "3DUnet_transform"
        config.transforms['post_transform']= "3DUnet_transform"
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
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    # OrientationD(keys, axcodes="LAS", allow_missing_keys=True),  # preferred by radiologists
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True, allow_missing_keys=True),
                    ToTensorD(keys, allow_missing_keys=True),
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

        '3DUnet_transform': Compose(
            [
                ActivationsD(keys=keys[2], sigmoid=True),
                AsDiscreteD(keys=keys[2], threshold=0.5),
                FillHolesD(keys=keys[2]),
                KeepLargestConnectedComponentD(keys=keys[2]),   
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
            

    def load_checkpoint(self, path=config.save["model_checkpoint"]):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: str
            The path of the checkpoint. (Default is the model path in config)
        """
        # self.network.load_state_dict(
        #     torch.load(path, map_location=torch.device(self.device))
        # )

        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


    def predict(self, data_dir):
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
            volume_names = natsort.natsorted(os.listdir(data_dir))
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
                #predict by sliding window
                pred = sliding_window_inference(volume, (96,96,64), 4, self.network)
                #Apply post processing transforms on 3D prediction
                if (config.transforms['mode']=="3D"):
                    pred = [self.postprocessing_transforms(i) for i in decollate_batch(pred)]
                    pred=torch.stack(pred)        
                prediction_list.append(pred)
            prediction_list = torch.cat(prediction_list, dim=0)

        return prediction_list




def segment_liver(*args):
    """
    a function used to segment the liver of 3d images by sliding window network

    """
    set_seed()
    liver_model = LiverSegmentation()
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    liver_prediction=liver_model.predict(config.dataset['prediction'])
    return liver_prediction
