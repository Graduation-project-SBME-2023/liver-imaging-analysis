from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.engine import Engine, set_seed
from liver_imaging_analysis.engine.dataloader import Keys
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
from liver_imaging_analysis.models.liver_segmentation import LiverSegmentation
import SimpleITK
import cv2
import shutil
import natsort
import nibabel as nib
from monai.handlers.utils import from_engine

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric = DiceMetric(ignore_empty = True, include_background = False)

class LobeSegmentation(Engine):
    """
    A class used for the lobe segmentation task. Inherits from Engine.

    Args:
        mode: str
            determines the mode of inference. 
            Expects "2D" for slice inference or "3D" for volume inference.
            Default is "2D"
    """

    def __init__(self, mode = "2D"):
        self.set_configs()
        super().__init__()
        if mode == '3D':
            self.predict = self.predict_2dto3d

    def set_configs(self):
        config.dataset['prediction'] = "test cases/sample_image"
        config.dataset['training'] = "Temp2D/Train/"
        config.dataset['testing'] = "Temp2D/Test/"
        config.training['batch_size'] = 2
        config.training['optimizer_parameters'] = { "lr" : 0.01 }
        config.training['scheduler_parameters'] = { 
                                                    "step_size" : 200, 
                                                    "gamma" : 0.5, 
                                                    "verbose" : False 
                                                  }
        config.network_parameters['dropout'] = 0
        config.network_parameters['spatial_dims'] = 2
        config.network_parameters['channels'] = [64, 128, 256, 512, 1024]
        config.network_parameters["out_channels"] = 10
        config.network_parameters['strides'] =  [2 ,2, 2, 2]
        config.network_parameters['num_res_units'] =  6
        config.network_parameters['norm'] = "INSTANCE"
        config.network_parameters['bias'] = True
        config.save['lobe_checkpoint'] = 'lobe_cp'
        config.training['loss_parameters'] = { 
                                                "softmax" : True, 
                                                "batch" : True, 
                                                "include_background" : True, 
                                                "to_onehot_y" : False 
                                             }
        config.training['metrics_parameters'] = { 
                                                    "ignore_empty" : True, 
                                                    "include_background" : False, 
                                                    "reduction" : "mean" 
                                                }
        config.transforms['mode'] = "3D"
        config.transforms["train_transform"] = "2DUnet_transform"
        config.transforms["test_transform"] = "2DUnet_transform"
        config.transforms["post_transform"] = "2DUnet_transform"

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

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "LAS", 
                        allow_missing_keys = True
                        ),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode = ("trilinear", "nearest", "nearest"), 
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
                        range_x = 0.1,
                        range_y = 0.1,
                        range_z = 0.1,
                        prob = 0.5,
                        keep_size = True,
                        allow_missing_keys = True
                    ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise = True),
                    ForegroundMaskD(Keys.LABEL, threshold = 0.5, invert = True),
                    ToTensorD(Keys.all(), allow_missing_keys = True),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    # Transformations
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode = ("bilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
                        ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise = True),
                    AsDiscreteD(Keys.LABEL, to_onehot = 10),
                    # Augmentations
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
                    RandFlipd(
                        Keys.all(), 
                        prob = 0.5, 
                        spatial_axis = 0, 
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
                    RandAdjustContrastd(Keys.IMAGE, prob = 0.5),
                    # Array to Tensor
                    ToTensorD(Keys.all(), allow_missing_keys = True),
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
                Name of the desire set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "LAS", 
                        allow_missing_keys = True
                        ),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode = ("trilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
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
            ),
            "2DUnet_transform": Compose(
                [
                    #Transformations
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode = ("bilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
                        ),
                    NormalizeIntensityD(Keys.IMAGE, channel_wise = True),
                    AsDiscreteD(Keys.LABEL, to_onehot = 10, allow_missing_keys = True),
                    ToTensorD(Keys.all(), allow_missing_keys = True),
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
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        transforms= {
        '2DUnet_transform': Compose(
            [
                ActivationsD(Keys.PRED, softmax = True),
                AsDiscreteD(Keys.PRED, argmax = True),
                FillHolesD(Keys.PRED),
                KeepLargestConnectedComponentD(Keys.PRED),
                AsDiscreteD(Keys.PRED, to_onehot = 10)   # mandatory during training
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
                target lobe mask
            prediction: tensor
                predicted lobe mask
        """

        dice_metric(prediction.int(),label.int())
        dice_score = dice_metric.aggregate().item()
        if(label.shape[1] > 1):
            label = torch.argmax(label, dim = 1, keepdim = True)
        if(prediction.shape[1] > 1):
            prediction = torch.argmax(prediction, dim = 1, keepdim = True)
        plot_2d_or_3d_image(
            data = image,
            step = 0,
            writer = summary_writer,
            frame_dim = -1,
            tag = f"Batch{batch_num}:Volume:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data = label,
            step = 0,
            writer = summary_writer,
            frame_dim = -1,
            tag = f"Batch{batch_num}:Mask:dice_score:{dice_score}",
        )
        plot_2d_or_3d_image(
            data = prediction,
            step = 0,
            writer = summary_writer,
            frame_dim = -1,
            tag = f"Batch{batch_num}:Prediction:dice_score:{dice_score}",
        )
        dice_metric.reset()


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


    def predict(self, data_dir, liver_mask):
        """
        predicts the liver & lobes mask given the liver mask

        Args:
            data_dir: str
                Path of the input directory. expects nifti or png files.
            liver_mask: tensor
                Liver mask predicted by the liver model.
        
        Returns:
            tensor
                Predicted Labels. Values: background: 0, lobes: 1-9.
                Label 4 and 5 represents lobe 4A and 4B.
        """
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(data_dir))
            volume_paths = [os.path.join(data_dir, file_name) 
                            for file_name in volume_names]
            predict_files = [{Keys.IMAGE: image_name} 
                             for image_name in volume_paths]
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
            liver_set = Dataset(data = liver_mask)
            liver_loader = MonaiLoader(
                                        liver_set,
                                        batch_size = self.batch_size,
                                        num_workers = 0,
                                        pin_memory = False,
                                    )
            prediction_list = []
            for batch,liver_mask_batch in zip(predict_loader,liver_loader):
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                #isolate the liver and suppress other organs
                suppressed_volume = np.where(
                                        liver_mask_batch == 1,
                                        batch[Keys.IMAGE],
                                        batch[Keys.IMAGE].min()
                                        )
                suppressed_volume = ToTensor()(suppressed_volume).to(self.device)
                #predict lobes in isolated liver
                batch[Keys.PRED] = self.network(suppressed_volume)
                #Apply post processing transforms
                batch = self.post_process(batch,Keys.PRED)
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        return prediction_list
    
    
    def predict_2dto3d(self, volume_path, liver_mask, temp_path = "temp/"):
        """
        Predicts the lobes of a 3D volume using a 2D network given a liver mask.
        
        Args:
            volume_path: str
                path of the input directory. expects a 3D nifti file.
            liver_mask: tensor
                liver mask predicted by the liver model.
            temp_path: str
                A temporary path to save 3d volume as 2d png slices. 
                Default is "temp/".
                Automatically deleted before returning the prediction.

        Returns:
            tensor
                Predicted labels with shape (1, channel, length, width, depth).
                Values: background: 0, lobes: 1-9.
                Label 4 and 5 represents lobe 4A and 4B.
        """

        #read volume
        img_volume = SimpleITK.ReadImage(volume_path)
        img_volume_array = SimpleITK.GetArrayFromImage(img_volume)
        number_of_slices = img_volume_array.shape[0]
        #create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
          os.mkdir(temp_path)
        #write volume slices as 2d png files 
        for slice_number in range(number_of_slices):
            volume_slice = img_volume_array[slice_number, :, :]
            # Delete extension from filename
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
            volume_png_path = os.path.join(
                                temp_path, 
                                volume_file_name + "_" + str(slice_number)
                                ) + ".png"
            cv2.imwrite(volume_png_path, volume_slice)
        #predict slices individually then reconstruct 3D prediction
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(temp_path))
            volume_paths = [os.path.join(temp_path, file_name) 
                            for file_name in volume_names]
            predict_files = [{Keys.IMAGE: image_name} 
                             for image_name in volume_paths]
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
            liver_set = Dataset(data = liver_mask)
            liver_loader = MonaiLoader(
                                        liver_set,
                                        batch_size = self.batch_size,
                                        num_workers = 0,
                                        pin_memory = False,
                                    )
            prediction_list = []
            for batch, liver_mask_batch in zip(predict_loader, liver_loader):
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                #isolate the liver and suppress other organs
                suppressed_volume = np.where(
                                        liver_mask_batch == 1,
                                        batch[Keys.IMAGE],
                                        batch[Keys.IMAGE].min()
                                        )
                suppressed_volume = ToTensor()(suppressed_volume).to(self.device)
                #predict lobes in isolated liver
                batch[Keys.PRED] = self.network(suppressed_volume)
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        batch = {Keys.PRED : prediction_list}
        # Transform shape from (batch,channel,length,width) 
        # to (1,channel,length,width,batch) 
        batch[Keys.PRED] = batch[Keys.PRED].permute(1,2,3,0).unsqueeze(dim=0) 
        #Apply post processing transforms
        batch = self.post_process(batch,Keys.PRED)
        #delete temporary folder
        shutil.rmtree(temp_path)
        return batch[Keys.PRED]
    

def segment_lobe(*args):
    """
    A function used to segment the liver lobes using
    the liver and the lobes models.
    """

    set_seed()
    liver_model = LiverSegmentation(mode = '2D')
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    lobe_model = LobeSegmentation(mode = '2D')
    lobe_model.load_checkpoint(config.save["lobe_checkpoint"])
    liver_prediction=liver_model.predict(config.dataset['prediction'])
    lobe_prediction= lobe_model.predict(
                        config.dataset['prediction'], 
                        liver_mask = liver_prediction
                        )
    lobe_prediction = lobe_prediction * liver_prediction #no liver -> no lobe
    return lobe_prediction


def segment_lobe_3d(*args):
    """
    A function used to segment the liver lobes
    of a 3d volume using the liver and the lobes models.
    """
    set_seed()
    liver_model = LiverSegmentation(mode = '3D')
    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    lobe_model = LobeSegmentation(mode = '3D')
    lobe_model.load_checkpoint(config.save["lobe_checkpoint"])
    liver_prediction = liver_model.predict(volume_path = args[0])
    lobe_prediction = lobe_model.predict(
                            volume_path = args[0],
                            liver_mask = liver_prediction[0].permute(3,0,1,2)
                            )
    lobe_prediction = lobe_prediction * liver_prediction #no liver -> no lobe
    return lobe_prediction



def train_lobe(*args):
    """
    a function used to start the training of liver segmentation

    """
    set_seed()
    model = LobeSegmentation(mode = '2D')
    model.load_data()
    model.data_status()
    model.load_checkpoint(config.save["lobe_checkpoint"])
    print(
        "Initial test loss:", 
        model.test(model.test_dataloader, callback = False)
        )
    model.fit(
        evaluate_epochs = 1,
        batch_callback_epochs = 100,
        save_weight = True,
    )
    # evaluate on last saved checkpoint
    model.load_checkpoint(config.save["potential_checkpoint"]) 
    print("final test loss:", model.test(model.test_dataloader, callback = False))