from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.engine import Engine, set_seed
from liver_imaging_analysis.engine.dataloader import Keys
from liver_imaging_analysis.engine.transforms import MorphologicalClosingd
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
import argparse
import nibabel as nib
import natsort 
from liver_imaging_analysis.engine.utils import progress_bar


summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric=DiceMetric(ignore_empty=True,include_background=True)
    
class LiverSegmentation(Engine):
    """
    A class used for the liver segmentation task. Inherits from Engine.

    Args:
        modality: str
            determines the configurations to be loaded.
            Expects 'CT' for CT configurations, or 'MRI' for MRI configurations.
            Default is 'CT'
        inference: str
            determines the mode of inference. 
            Expects "2D" for slice inference, "3D" for volume inference,
            or "sliding_window" for sliding window inference.
            Default is "3D"
    """

    def __init__(self, modality = 'CT', inference = '3D'):
        self.set_configs(modality, inference)
        super().__init__()
        if inference == '3D':
            self.predict = self.predict_2dto3d
        elif inference == 'sliding_window':
            self.predict = self.predict_sliding_window
            self.test = self.test_sliding_window

    def set_configs(self, modality, inference):
        """
        Sets new values for config parameters.

        Args:
            modality: str
                together with inference, determines the configurations to be loaded.
                Expects 'CT' for CT configurations, or 'MRI' for MRI configurations.
            inference: str
                together with modality, determines the configurations to be loaded.
                Expects "2D" for slice inference, "3D" for volume inference,
                or "sliding_window" for sliding window inference.
        """
        
        if modality == 'CT':
            if inference in ['2D', '3D']:
                config.dataset['prediction'] = "test cases/volume/volume-64.nii"
                config.training['batch_size'] = 8
                config.training['scheduler_parameters'] = {
                                                            "step_size" : 20,
                                                            "gamma" : 0.5, 
                                                            "verbose" : False
                                                            }
                config.network_parameters['dropout'] = 0
                config.network_parameters["out_channels"] = 1
                config.network_parameters['spatial_dims'] = 2
                config.network_parameters['channels'] = [64, 128, 256, 512]
                config.network_parameters['strides'] =  [2, 2, 2]
                config.network_parameters['num_res_units'] =  4
                config.network_parameters['norm'] = "INSTANCE"
                config.network_parameters['bias'] = True
                config.save['liver_checkpoint'] = 'liver_cp'
                config.transforms['train_transform'] = "2d_ct_transform"
                config.transforms['test_transform'] = "2d_ct_transform"
                config.transforms['post_transform'] = "2d_ct_transform"
            elif inference == 'sliding_window':
                config.dataset['prediction'] = "test cases/volume/volume-64.nii"
                config.training['batch_size'] = 1
                config.training['scheduler_parameters'] = {
                                                            "step_size" : 20,
                                                            "gamma" : 0.5, 
                                                            "verbose" : False
                                                            }
                config.network_parameters['dropout'] = 0
                config.network_parameters["out_channels"] = 1
                config.network_parameters['channels'] = [64, 128, 256, 512]
                config.network_parameters['spatial_dims'] = 3
                config.network_parameters['strides'] =  [2, 2, 2]
                config.network_parameters['num_res_units'] =  6
                config.network_parameters['norm'] = "BATCH"
                config.network_parameters['bias'] = False
                config.save['liver_checkpoint'] = 'liver_cp_sliding_window'
                config.transforms['sw_batch_size'] = 4
                config.transforms['roi_size'] = (96,96,64)
                config.transforms['overlap'] = 0.25
                config.transforms['train_transform'] = "3d_ct_transform"
                config.transforms['test_transform'] = "3d_ct_transform"
                config.transforms['post_transform'] = "3d_ct_transform"
        elif modality == 'MRI':
            if inference in ['2D', '3D']:
                config.dataset['prediction'] = "test cases/volume/volume-64.nii"
                config.training['batch_size'] = 12
                config.training['scheduler_parameters'] = {
                                                            "step_size" : 20,
                                                            "gamma" : 0.5, 
                                                            "verbose" : True
                                                            }
                config.network_parameters['dropout'] = 0.35
                config.network_parameters["out_channels"] = 1
                config.network_parameters['spatial_dims'] = 2
                config.network_parameters['channels'] = [64, 128, 256, 512]
                config.network_parameters['strides'] =  [2, 2, 2]
                config.network_parameters['num_res_units'] =  4
                config.network_parameters['norm'] = "INSTANCE"
                config.network_parameters['bias'] = True
                config.save['liver_checkpoint'] = 'mri_cp'
                config.transforms['train_transform'] = "2d_mri_transform"
                config.transforms['test_transform'] = "2d_mri_transform"
                config.transforms['post_transform'] = "2d_mri_transform"

    
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
            "3d_ct_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    # OrientationD(keys, axcodes="LAS", allow_missing_keys = True),
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
            "2d_ct_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode=("bilinear", "nearest", "nearest"), 
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
            ),
            "2d_mri_transform": Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(), 
                        resize_size, 
                        mode = ("bilinear", "nearest", "nearest"), 
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
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name):
        """
        Gets a stack of preprocessing transforms to be used on the testing data.

        Args:
             transform_name: str
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3d_ct_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
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
            "2d_ct_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ResizeD(
                        Keys.all(),
                        resize_size,
                        mode = ("bilinear", "nearest", "nearest"),
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
            ),
            "2d_mri_transform": Compose(
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
                    ForegroundMaskD(
                        Keys.LABEL, 
                        threshold = 0.5, 
                        invert = True, 
                        allow_missing_keys=True
                        ),
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
            '3d_ct_transform' : Compose(
                [
                    ActivationsD(Keys.PRED,sigmoid = True),
                    AsDiscreteD(Keys.PRED,threshold = 0.5),
                    FillHolesD(Keys.PRED),
                    KeepLargestConnectedComponentD(Keys.PRED),   
                ]
            ),
            '2d_ct_transform' : Compose(
                [
                    ActivationsD(Keys.PRED,sigmoid = True),
                    AsDiscreteD(Keys.PRED,threshold = 0.5),
                    FillHolesD(Keys.PRED),
                    KeepLargestConnectedComponentD(Keys.PRED),
                ]
            ),
            '2d_mri_transform': Compose(
                [
                    ActivationsD(Keys.PRED,sigmoid = True),
                    AsDiscreteD(Keys.PRED,threshold = 0.5),
                    FillHolesD(Keys.PRED),
                    KeepLargestConnectedComponentD(Keys.PRED),   
                ]
            ),
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
                batch_size = self.batch_size,
                num_workers = 0,
                pin_memory = False,
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
        batch[Keys.PRED] = batch[Keys.PRED].permute(1,2,3,0).unsqueeze(dim = 0) 
        # Apply post processing transforms
        batch = self.post_process(batch)
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
                                        self.network,
                                        config.transforms['overlap']
                                        )
                # Apply post processing transforms
                batch = self.post_process(batch)    
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim = 0)
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
        
        
    def test3d(self, dir_3d = None):
        """
        Calculate 3d dice in 2d models
        
        Parameters
        ----------
        dir_3d : path
                 path for test dataset and its true labels.
        
        Returns
        -------
        float
            the averaged 3d metric calculated during testing
        """

        test_metric = 0   
        
        test_dir = os.path.join(dir_3d, "volume/")
        labels_dir = os.path.join(dir_3d, "mask/")
        
        tests=natsort.natsorted(os.listdir(test_dir))
        labels=natsort.natsorted(os.listdir(labels_dir))

        print('\n3D TESTING:')
        for test_num, (test_name, label_name) in enumerate(zip(tests, labels)):
            progress_bar(test_num + 1, len(tests))
            
            test_path=os.path.join(test_dir, test_name)
            prediction = self.predict_2dto3d(test_path).to(config.device)

            label_path=os.path.join(labels_dir,label_name)
            nifti_label = nib.load(label_path).get_fdata()
            nifti_label = np.where(nifti_label > 0.5, 1, nifti_label)
            label = torch.from_numpy(nifti_label)
            label = label.to(config.device)
            label=label.view(1, 1, *label.shape)

            dice= self.metrics(prediction,label)
            
            print(f'{test_name}')
            print(f'prediced : {label_name}')
            print(f'Dice : {dice}')
            
            # Free GPU memory if CUDA is used
            if torch.cuda.is_available():
                del prediction, label
                torch.cuda.empty_cache()
        
        # aggregate the final metric result
        test_metric = self.metrics.aggregate().item()
        # reset the status for next computation round
        self.metrics.reset()

        return test_metric


def segment_liver(
        prediction_path = None, 
        modality = 'CT', 
        inference = '3D', 
        cp_path = None
        ):
    """
    Segments the liver from an abdominal scan.

    Parameters
    ----------
    prediciton_path : str
        if inference is 2D, expects a directory containing a set of png images.
        if inference is 3D or sliding_window, expects a path of a 3D nii volume.
        if not defined, prediction dataset will be loaded from configs
    modality : str
        the type of imaging modality to be segmented.
        expects 'CT' for CT images, or 'MRI' for MRI images.
        Default is CT
    inference : str
        the type of inference to be used.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
    cp_path : str
        path of the model weights to be used for prediction. 
        if not defined, liver_checkpoint will be loaded from configs.
    Returns
    ----------
        tensor : predicted liver segmentation
    """
    liver_model = LiverSegmentation(modality, inference)
    if prediction_path is None:
        prediction_path = config.dataset['prediction']
    if cp_path is None:
        cp_path = config.save["liver_checkpoint"]
    liver_model.load_checkpoint(cp_path)
    liver_prediction = liver_model.predict(prediction_path)
    return liver_prediction


def train_liver(
        modality = 'CT', 
        inference = '3D', 
        test_inference = '2D',
        dir_3d = None,
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
    Starts training of liver segmentation model.
    modality : str
        the type of imaging modality to be segmented.
        expects 'CT' for CT images, or 'MRI' for MRI images.
        Default is CT
    inference : str
        the type of inference to be used.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
        Default is 3D
    test_inference : str
        The type of inference to be used during testing.
        Expects "2D" for 2d dice test, "3D" for 3d dice test.
        Default is '2D'.
    dir_3d : str
        The path for the 3d volumes and true labels of test dataset, 
        This is used if `test_inference` is set to '3D'.
        Default is None.    
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
    model = LiverSegmentation(modality, inference)
    model.load_data()
    model.data_status()
    if pretrained:
        model.load_checkpoint(cp_path)
    model.compile_status()
    
    if test_inference == '3D' :
        init_metric = model.test3d(
                        dir_3d = dir_3d
                    )
    else :
        init_loss, init_metric = model.test(
                                    model.test_dataloader, 
                                    callback = test_batch_callback
                                )
        print(
            "Initial test loss:", 
            init_loss,
            )
    print(
        "\nInitial test metric:", 
        init_metric.mean().item(),
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
    
    if test_inference == '3D' :
        final_metric = model.test3d(
                        dir_3d = dir_3d
                    )
    else :
        final_loss, final_metric = model.test(
                                    model.test_dataloader, 
                                    callback = test_batch_callback
                                    )
        print(
            "Final test loss:", 
            final_loss,
            )
    print(
        "\nFinal test metric:", 
        final_metric.mean().item(),
        )



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Liver Segmentation')
    parser.add_argument(
                '--modality', type = str, default = 'CT',
                help = 'choose imaging modality from CT or MRI (default: CT)'
                )
    parser.add_argument(
                '--inference', type = str, default = '3D',
                help = 'choose inference mode: 2D, 3D, or sliding_window (default: 3D)'
                )
    parser.add_argument(
                '--cp', type = bool, default = True,
                help = 'if True loads pretrained checkpoint (default: True)'
                )
    parser.add_argument(
                '--cp_path', type = bool, default = None,
                help = 'path of pretrained checkpoint (default: liver cp config path)'
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
    LiverSegmentation(args.modality, args.inference) # to set configs
    if args.predict_path is None:
        args.predict_path = config.dataset['prediction']
    if args.cp_path is None:
        args.cp_path = config.save["liver_checkpoint"]
    if args.save_path is None:
        args.save_path = config.save["potential_checkpoint"]
    if args.train: 
        train_liver(
            modality = args.modality, 
            inference = args.inference, 
            pretrained = args.cp, 
            cp_path = args.cp_path,
            epochs = args.epochs, 
            evaluate_epochs = args.eval_epochs,
            batch_callback_epochs = args.batch_callback,
            save_weight = args.save,
            save_path = args.save_path,
            test_batch_callback = args.test_callback,
            )
    if args.test:
        model = LiverSegmentation(args.modality, args.inference)
        model.load_data() #dataset should be located at the config path
        loss, metric = model.test(
                                    model.test_dataloader, 
                                    callback = args.test_callback
                                    )
        print(
            "test loss:", 
            loss,
            )
        print(
            "\ntest metric:", 
            metric.mean().item(),
            )
    if args.predict:
        prediction = segment_liver(
                        args.predict_path, 
                        modality = args.modality, 
                        inference = args.inference,
                        cp_path = args.cp_path
                        )
        #save prediction as a nifti file
        original_header = nib.load(args.predict_path).header
        original_affine = nib.load(args.predict_path).affine
        liver_volume = nib.Nifti1Image(
                            prediction[0,0].cpu(), 
                            affine = original_affine, 
                            header = original_header
                            )
        nib.save(liver_volume, args.predict_path.split('.')[0] + '_liver.nii')
        print('Prediction saved at', args.predict_path.split('.')[0] + '_liver.nii')
    print("Run Complete")
