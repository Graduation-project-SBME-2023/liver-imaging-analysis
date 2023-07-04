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
    RandSpatialCropSamplesd,
    SpatialPadd,
    Spacingd,
    EnsureTyped,
    AsDiscreted,
    Activationsd,
    Invertd
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
import argparse

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

    def __init__(self, inference = "sliding_window"):
        self.set_configs(inference)
        super().__init__()
        if inference == '3D':
            self.predict = self.predict_2dto3d
        elif inference == 'sliding_window':
            self.predict = self.predict_sliding_window
            self.test = self.test_sliding_window


    def set_configs(self, inference):
        """
        Sets new values for config parameters.

        Args:
            inference: str
                together with modality, determines the configurations to be loaded.
                Expects "2D" for slice inference, "3D" for volume inference,
                or "sliding_window" for sliding window inference.
        """
        
        if inference in ['2D', '3D']:
            config.dataset['prediction'] = "test cases/volume/volume-64.nii"
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
            config.network_parameters['strides'] = [2 ,2, 2, 2]
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
            config.transforms["train_transform"] = "2d_transform"
            config.transforms["test_transform"] = "2d_transform"
            config.transforms["post_transform"] = "2d_transform"
        elif inference == 'sliding_window':
            config.dataset['prediction'] = "test cases/volume/volume-64.nii"
            config.dataset['training'] = "MedSeg_Lobes/Train/"
            config.dataset['testing'] = "MedSeg_Lobes/Test/"
            config.training['batch_size'] = 1
            config.training['optimizer_parameters'] = { "lr" : 0.01 }
            config.training['scheduler_parameters'] = {
                                                        "step_size" : 250,
                                                        "gamma" : 0.5, 
                                                        "verbose" : False
                                                        }
            config.training['loss_parameters'] = { 
                                                    "softmax" : True, 
                                                    "batch" : True, 
                                                    "include_background" : True, 
                                                    "to_onehot_y" : False 
                                                }
            config.training['metrics_parameters'] = { 
                                                        "ignore_empty" : False, 
                                                        "include_background" : False, 
                                                        "reduction" : "mean_batch" 
                                                    }
            config.network_parameters['dropout'] = 0.5
            config.network_parameters["out_channels"] = 10
            config.network_parameters['channels'] = [16, 32, 64, 128, 256]
            config.network_parameters['spatial_dims'] = 3
            config.network_parameters['strides'] =  [2, 2, 2, 2]
            config.network_parameters['num_res_units'] =  4
            config.network_parameters['norm'] = "BATCH"
            config.network_parameters['bias'] = False
            config.save['lobe_checkpoint'] = 'Liver-Segmentation-Website/models_checkpoints/lobe_cp_sliding_window'
            config.transforms['sw_batch_size'] = 2
            config.transforms['roi_size'] = (192, 192, 32)
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
        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3d_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "RAS", 
                        allow_missing_keys = True
                        ),
                    CropForegroundd(
                        Keys.all(), 
                        source_key = Keys.IMAGE,
                        allow_missing_keys = True),
                    Spacingd(
                        Keys.all(), 
                        pixdim = [1, 1, 5], 
                        mode = ("bilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
                        ),
                    SpatialPadd(
                        Keys.all(), 
                        config.transforms['roi_size'], 
                        mode = 'minimum',
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
                    ScaleIntensityRanged(
                        Keys.IMAGE,
                        a_min = -135,
                        a_max = 215,
                        b_min = 0,
                        b_max = 1,
                        clip = True, 
                        allow_missing_keys = True
                    ),
                    RandSpatialCropSamplesd(
                        Keys.all(), 
                        config.transforms['roi_size'], 
                        num_samples = config.transforms['sw_batch_size'], 
                        random_center = True, 
                        random_size = False, 
                        allow_missing_keys = True
                        ),
                    EnsureTyped(Keys.all(), allow_missing_keys = True),
                    AsDiscreteD(Keys.LABEL, to_onehot = 10)   # mandatory during training

                ]
            ),
            "2d_transform" : Compose(
                [
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
                    ToTensorD(Keys.all(), allow_missing_keys = True),
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
            "3d_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    OrientationD(
                        Keys.all(), 
                        axcodes = "RAS", 
                        allow_missing_keys = True
                        ),
                    CropForegroundd(
                        Keys.all(), 
                        source_key = Keys.IMAGE,
                        allow_missing_keys = True),
                    Spacingd(
                        Keys.all(), 
                        pixdim = [1, 1, 5],
                        mode = ("bilinear", "nearest", "nearest"), 
                        allow_missing_keys = True
                        ),
                    SpatialPadd(
                        Keys.all(), 
                        config.transforms['roi_size'], 
                        mode ='minimum',
                        allow_missing_keys = True
                        ),
                    ScaleIntensityRanged(
                        Keys.IMAGE,
                        a_min = -135,
                        a_max = 215,
                        b_min = 0,
                        b_max = 1,
                        clip = True, 
                        allow_missing_keys = True
                    ),
                    EnsureTyped(Keys.all(), allow_missing_keys = True),
                    # AsDiscreteD(Keys.LABEL, to_onehot = 10)   # mandatory during training

                ]
            ),
            "2d_transform" : Compose(
                [
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
                        (Keys.LABEL, Keys.PRED),
                        transform = self.test_transform,
                        orig_keys = Keys.IMAGE,
                        meta_keys = (Keys.LABEL + "_meta_dict", Keys.PRED + "_meta_dict"),
                        orig_meta_keys = Keys.IMAGE + "_meta_dict",
                        meta_key_postfix = "meta_dict",
                        nearest_interp = False,
                        to_tensor = True,
                        device = self.device,
                        allow_missing_keys = True
                    ),
                    Activationsd(Keys.PRED, softmax = True),
                    AsDiscreted(Keys.PRED, argmax = True),
                    FillHolesD(Keys.PRED),
                    KeepLargestConnectedComponentD(Keys.PRED),
                    # AsDiscreteD(Keys.PRED, to_onehot = 10)   # mandatory during training
                ]
            ),
            '2d_transform': Compose(
                [
                    ActivationsD(Keys.PRED, softmax = True),
                    AsDiscreteD(Keys.PRED, argmax = True),
                    FillHolesD(Keys.PRED),
                    KeepLargestConnectedComponentD(Keys.PRED),
                    # AsDiscreteD(Keys.PRED, to_onehot = 10)   # mandatory during training
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
        print("Training Metric=", training_metric.mean().item(),':\n', training_metric.cpu().numpy())
        summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
        summary_writer.add_scalar("\nTraining Metric", training_metric.mean(), epoch)
        if valid_loss is not None:
            print("\nValidation Loss=", valid_loss)
            print("Validation Metric=", valid_metric.mean().item(),':\n', valid_metric.cpu().numpy())

            summary_writer.add_scalar("\nValidation Loss", valid_loss, epoch)
            summary_writer.add_scalar("\nValidation Metric", valid_metric.mean(), epoch)


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
                batch = self.post_process(batch)
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
        batch = self.post_process(batch)
        #delete temporary folder
        shutil.rmtree(temp_path)
        return batch[Keys.PRED]


    def predict_sliding_window(self, volume_path, liver_mask, temp_path = "temp/"):
        """
        predict the label of the given input
        Parameters
        ----------
        volume_path: str
            path of the input directory. expects nifti or png files.
        liver_mask: tensor
            liver mask predicted by the liver model.
        temp_path: str
            A temporary path to save 3d liver mask as nifti. 
            Default is "temp/".
            Automatically deleted before returning the prediction.
        Returns
        -------
        tensor
            tensor of the predicted labels
        """
        #create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
          os.mkdir(temp_path)
        original_header = nib.load(volume_path).header
        original_affine = nib.load(volume_path).affine
        liver_volume = liver_mask[0, 0].cpu().numpy()
        # Delete extension from filename
        volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
        liver_volume_path = os.path.join(temp_path, volume_file_name) + ".nii.gz"
        liver_volume = nib.Nifti1Image(
                            liver_volume, 
                            affine = original_affine, 
                            header = original_header
                            )
        nib.save(liver_volume, liver_volume_path)
        self.network.eval()
        with torch.no_grad():
            predict_files = [{
                                Keys.IMAGE : volume_path,
                                Keys.LABEL : liver_volume_path
                              }] 
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
                #isolate the liver and suppress other organs
                suppressed_volume = np.where(
                                        batch[Keys.LABEL] == 1,
                                        batch[Keys.IMAGE],
                                        batch[Keys.IMAGE].min()
                                        )
                suppressed_volume = ToTensor()(suppressed_volume).to(self.device)
                # Predict by sliding window
                batch[Keys.PRED] = sliding_window_inference(
                                        suppressed_volume, 
                                        config.transforms['roi_size'], 
                                        config.transforms['sw_batch_size'],      
                                        self.network
                                        )
                # Apply post processing transforms
                batch = self.post_process(batch)    
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim = 0)
        shutil.rmtree(temp_path)
        return prediction_list


    def test_sliding_window(self, dataloader = None, callback = False):
        """
        calculates loss on input dataset

        Parameters
        ----------
        dataloader: dict
                Iterator of the dataset to evaluate on.
                If not specified, the test_dataloader will be used.
        callback: bool
                Flag to call per_batch_callback or not. Default is False.

        Returns
        -------
        float
            the averaged loss calculated during testing
        float
            the averaged metric calculated during testing
        """
        if dataloader is None: #test on test set by default
            dataloader = self.test_dataloader
        num_batches = len(dataloader)
        test_loss = 0
        test_metric = 0
        self.network.eval()
        with torch.no_grad():
            for batch_num,batch in enumerate(dataloader):
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                batch[Keys.LABEL] = batch[Keys.LABEL].to(self.device)
                batch[Keys.PRED] = sliding_window_inference(
                                        batch[Keys.IMAGE], 
                                        config.transforms['roi_size'], 
                                        config.transforms['sw_batch_size'], 
                                        self.network,
                                        config.transforms['overlap']
                                        )
                test_loss += self.loss(
                    batch[Keys.PRED],
                    batch[Keys.LABEL]
                    ).item()
                #Apply post processing transforms on prediction
                batch = self.post_process(batch)
                self.metrics(batch[Keys.PRED].int(), batch[Keys.LABEL].int())
                if callback:
                  self.per_batch_callback(
                      batch_num,
                      batch[Keys.IMAGE],
                      batch[Keys.LABEL],
                      batch[Keys.PRED]
                      )
            test_loss /= num_batches
            # aggregate the final metric result
            test_metric = self.metrics.aggregate()
            # reset the status for next computation round
            self.metrics.reset()
        return test_loss, test_metric
    
    
def segment_lobe(
        prediction_path = None,
        liver_inference = '3D',
        lobe_inference = 'sliding_window', 
        liver_cp = None, 
        lobe_cp = None
        ):
    """
    Segments the Lobes from a liver scan.

    Parameters
    ----------
    prediciton_path : str
        if inferences are 2D, expects a directory containing a set of png images.
        if inferences are 3D or sliding_window, expects a path of a 3D nii volume.
        if not defined, prediction dataset will be loaded from configs
    liver_inference : str
        the type of inference to be used for liver model.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
        Default is 3D
    lobe_inference : str
        the type of inference to be used for lobe model.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
        Default is sliding_window
    liver_cp : str
        path of the liver model weights to be used for prediction. 
        if not defined, liver_checkpoint will be loaded from configs.
    lobe_cp : str
        path of the lobe model weights to be used for prediction. 
        if not defined, lobe_checkpoint will be loaded from configs.
    Returns
    ----------
        tensor : predicted lobes segmentation
    """
    liver_model = LiverSegmentation(modality = 'CT', inference = liver_inference)
    lobe_model = LobeSegmentation(inference = lobe_inference)
    if prediction_path is None:
        prediction_path = config.dataset['prediction']
    if liver_cp is None:
        liver_cp = config.save["liver_checkpoint"]
    if lobe_cp is None:
        lobe_cp = config.save["lobe_checkpoint"]
    liver_model.load_checkpoint(liver_cp)
    lobe_model.load_checkpoint(lobe_cp)
    liver_prediction = liver_model.predict(prediction_path)
    lobe_prediction = lobe_model.predict(
                        prediction_path, 
                        liver_mask = liver_prediction[0].permute(3,0,1,2) 
                                     if lobe_inference == '3D' 
                                     else liver_prediction
                        )
    lobe_prediction = lobe_prediction * liver_prediction #no liver -> no lobe
    return lobe_prediction


def train_lobe(
        inference = 'sliding_window', 
        pretrained = True, 
        cp_path = None,
        epochs = None, 
        evaluate_epochs = 1,
        batch_callback_epochs = 100,
        save_weight = True,
        save_path = None,
        test_batch_callback = False,
        ):
    """
    Starts training of lobe segmentation model.
    inference : str
        the type of inference to be used.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
        Default is 3D
    pretrained : bool
        if true, loads pretrained checkpoint. Default is True.
    cp_path : str
        determines the path of the checkpoint to be loaded 
        if pretrained is true. If not defined, the potential 
        cp path will be loaded from config.
    epochs : int
        number of training epochs.
        If not defined, epochs will be loaded from config.
    evaluate_epochs : int
        The number of epochs to evaluate model after. Default is 1.
    batch_callback_epochs : int
        The frequency at which per_batch_callback will be called. 
        Expects a number of epochs. Default is 100.
    save_weight : bool
        whether to save weights or not. Default is True.
    save_path : str
        the path to save weights at if save_weights is True.
        If not defined, the potential cp path will be loaded 
        from config.
    test_batch_callback : bool
        whether to call per_batch_callback during testing or not.
        Default is False
    """
    if cp_path is None:
        cp_path = config.save["potential_checkpoint"]
    if epochs is None:
        epochs = config.training["epochs"]
    if save_path is None:
        save_path = config.save["potential_checkpoint"]
    set_seed()
    model = LobeSegmentation(inference)
    model.load_data()
    model.data_status()
    if pretrained:
        model.load_checkpoint(cp_path)
    model.compile_status()
    init_loss, init_metric = model.test(
                                model.test_dataloader, 
                                callback = test_batch_callback
                                )
    print(
        "Initial test loss:", 
        init_loss,
        )
    print(
        "\nInitial average test metric:", 
        init_metric.mean().item(),
        ':\nInitial average test metric per lobe',
        init_metric.cpu().numpy()
        )
    model.fit(
        epochs = epochs,
        evaluate_epochs = evaluate_epochs,
        batch_callback_epochs = batch_callback_epochs,
        save_weight = save_weight,
        save_path = save_path
    )
    # Evaluate on latest saved check point
    model.load_checkpoint(save_path)
    final_loss, final_metric = model.test(
                                model.test_dataloader, 
                                callback = test_batch_callback
                                )
    print(
        "Final test loss:", 
        final_loss,
        )
    print(
        "\nFinal average test metric:", 
        final_metric.mean().item(),
        ':\nFinal average test metric per lobe',
        final_metric.cpu().numpy()
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Lobe Segmentation')
    parser.add_argument(
                '--liver_inference', type = str, default = '3D',
                help = 'choose liver inference mode: 2D, 3D, or sliding_window (default: 3D)'
                )    
    parser.add_argument(
                '--lobe_inference', type = str, default = 'sliding_window',
                help = 'choose lobe inference mode: 2D, 3D, or sliding_window (default: sliding_window)'
                )
    parser.add_argument(
                '--cp', type = bool, default = True,
                help = 'if True loads pretrained checkpoint (default: True)'
                )
    parser.add_argument(
                '--liver_cp_path', type = bool, default = None,
                help = 'path of pretrained liver checkpoint (default: liver cp config path)'
                )
    parser.add_argument(
                '--lobe_cp_path', type = bool, default = None,
                help = 'path of pretrained lobe checkpoint (default: lobe cp config path)'
                )
    parser.add_argument(
                '--train', type = bool, default = False,
                help = 'if True runs training loop (default: False)'
                )
    parser.add_argument(
                '--epochs', type = int, default = 1,
                help = 'number of epochs to train (default: 1)'
                )
    parser.add_argument(
                '--eval_epochs', type = int, default = 1,
                help = 'number of epochs to evaluate after (default: 1)'
                )
    parser.add_argument(
                '--batch_callback', type = int, default = 100,
                help = 'number of epochs to run batch callback after (default: 100)'
                )
    parser.add_argument(
                '--save', type = bool, default = False,
                help = 'if True save weights after training (default: False)'
                )
    parser.add_argument(
                '--save_path', type = str, default = None,
                help = 'path to save weights at if save is True (default: potential cp config path)'
                )
    parser.add_argument(
                '--test_callback', type = bool, default = False,
                help = 'if True call batch callback during testing (default: False)'        
                )
    parser.add_argument(
                '--test', type = bool, default = False,
                help = 'if True runs separate testing loop (default: False)'
                )
    parser.add_argument(
                '--predict', type = bool, default = False,
                help = 'if True, predicts the volume at predict_path (default: False)'
                )
    parser.add_argument(
                '--predict_path', type = str, default = None,
                help = 'predicts the volume at the provided path (default: prediction config path)'
                )
    args = parser.parse_args()
    LiverSegmentation(inference = args.liver_inference) # to set configs
    LobeSegmentation(inference = args.lobe_inference) # to set configs
    if args.predict_path is None:
        args.predict_path = config.dataset['prediction']
    if args.liver_cp_path is None:
        args.liver_cp_path = config.save["liver_checkpoint"]
    if args.lobe_cp_path is None:
        args.lobe_cp_path = config.save["lobe_checkpoint"]
    if args.save_path is None:
        args.save_path = config.save["potential_checkpoint"]
    if args.train: 
        train_lobe(
            inference = args.lobe_inference, 
            pretrained = args.cp, 
            cp_path = args.lobe_cp_path,
            epochs = args.epochs, 
            evaluate_epochs = args.eval_epochs,
            batch_callback_epochs = args.batch_callback,
            save_weight = args.save,
            save_path = args.save_path,
            test_batch_callback = args.test_callback,
            )
    if args.test:
        model = LobeSegmentation(args.lobe_inference)
        model.load_data() #dataset should be located at the config path
        loss, metric = model.test(
                                    model.test_dataloader, 
                                    callback = args.test_callback
                                    )
        print(
            "Test loss:", 
            loss,
            )
        print(
            "\nAverage test metric:", 
            metric.mean().item(),
            ':\nAverage test metric per lobe',
            metric.cpu().numpy()
            )
    if args.predict:
        prediction = segment_lobe(
                        args.predict_path, 
                        liver_inference = args.liver_inference,
                        lobe_inference = args.lobe_inference,
                        liver_cp = args.liver_cp_path,
                        lobe_cp = args.lobe_cp_path
                        )
        #save prediction as a nifti file
        original_header = nib.load(args.predict_path).header
        original_affine = nib.load(args.predict_path).affine
        liver_volume = nib.Nifti1Image(
                            prediction[0,0].cpu(), 
                            affine = original_affine, 
                            header = original_header
                            )
        nib.save(liver_volume, args.predict_path.split('.')[0] + '_lobes.nii')
        print('Prediction saved at', args.predict_path.split('.')[0] + '_lobes.nii')
    print("Run Complete")