from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.engine import Engine, set_seed
from liver_imaging_analysis.engine.dataloader import Keys
from liver_imaging_analysis.engine.transforms import MorphologicalClosing
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
from liver_imaging_analysis.models.liver_segmentation import LiverSegmentation
import SimpleITK
import cv2
import shutil
import natsort
import nibabel as nib
from monai.handlers.utils import from_engine
import argparse
import logging
logger = logging.getLogger(__name__)

summary_writer = SummaryWriter(config.save["tensorboard"])
dice_metric = DiceMetric(ignore_empty = True, include_background = False)

class LesionSegmentation(Engine):
    """
    A class used for the lesion segmentation task. Inherits from Engine.

    Args:
        inference: str
            determines the mode of inference. 
            Expects "2D" for slice inference or "3D" for volume inference.
            Default is "3D"
    """
    logger.info('LesionSegmentation')

    def __init__(self, inference = "3D"):
        self.set_configs()
        super().__init__()
        if inference == '3D':
            self.predict = self.predict_2dto3d

    def set_configs(self):
        """
        Sets new values for config parameters.
        """
       
        config.dataset['prediction'] = "test cases/volume/volume-64.nii"
        config.dataset['training'] = "Temp2D/Train/"
        config.dataset['testing'] = "Temp2D/Test/"
        config.training['batch_size'] = 8
        config.training['optimizer_parameters'] = {"lr" : 0.01}
        config.training['scheduler_parameters'] = {
                                                    "step_size" : 20,
                                                    "gamma" : 0.5, 
                                                    "verbose" : False
                                                  }
        config.network_parameters['dropout'] = 0
        config.network_parameters["out_channels"] = 1
        config.network_parameters['spatial_dims'] = 2
        config.network_parameters['channels'] = [64, 128, 256, 512]
        config.network_parameters["out_channels"] = 1
        config.network_parameters['strides'] =  [2, 2, 2]
        config.network_parameters['num_res_units'] =  2
        config.network_parameters['norm'] = "BATCH"
        config.network_parameters['bias'] = False
        config.save['lesion_checkpoint'] = 'lesion_cp'
        config.training['loss_parameters'] = {
                                                "sigmoid" : True,
                                                "batch" : True,
                                                "include_background" : True
                                             }
        config.training['metrics_parameters'] = {
                                                    "ignore_empty" : True,
                                                    "include_background" : False
                                                }
        config.transforms['train_transform'] = "2d_transform"
        config.transforms['test_transform'] = "2d_transform"
        config.transforms['post_transform'] = "2d_transform"

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
            "2d_transform" : Compose(
                [
                    #Transformations
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ScaleIntensityRanged(
                        Keys.IMAGE,
                        a_min = -135,
                        a_max = 215,
                        b_min = 0.0,
                        b_max = 1.0,
                        clip = True,
                    ),
                    #Augmentations
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
                Name of the desired set of transforms.

        Return:
            Compose
                Stack of selected transforms.
        """

        transforms = {
            "2d_transform" : Compose(
                [
                    LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ScaleIntensityRanged(
                        Keys.IMAGE,
                        a_min = -135,
                        a_max = 215,
                        b_min = 0.0,
                        b_max = 1.0,
                        clip = True,
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
            '2d_transform': Compose(
                [
                    ActivationsD(Keys.PRED, sigmoid = True),
                    AsDiscreteD(Keys.PRED, threshold = 0.5),
                    FillHolesD(Keys.PRED),
                    RemoveSmallObjectsD(Keys.PRED, min_size = 5),
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
                target lesion mask
            prediction: tensor
                predicted lesion mask
        """

        dice_score = dice_metric(prediction.int(),label.int())[0].item()
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
        logger.debug(f"Training Loss={training_loss}")
        logger.debug(f"Training Metric={training_metric}")
        if valid_loss is not None:
            print(f"\nValidation Loss={valid_loss}")
            print(f"\nValidation Metric={valid_metric}")
            summary_writer.add_scalar("\nValidation Loss", valid_loss, epoch)
            summary_writer.add_scalar("\nValidation Metric", valid_metric, epoch)
            logger.debug(f"\nValidation Loss={valid_loss}")
            logger.debug(f"\nValidation Metric={valid_metric}")

    def predict(self, data_dir, liver_mask):
        """
        Predicts the lesion mask given the liver mask.

        Args:
            data_dir: str
                Path of the input directory. expects nifti or png files.
            liver_mask: tensor
                 Liver mask predicted by the liver model.
        
        Returns:
            tensor
                Predicted Labels. Values: background: 0, liver: 1, lesion: 2.
        """
        logger.info('predict lesion')

        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(data_dir))
            volume_paths = [os.path.join(data_dir, file_name) 
                            for file_name in volume_names]
            logger.info(f"volume_paths={volume_paths}")
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
                #predict lesions in isolated liver
                batch[Keys.PRED] = self.network(suppressed_volume)
                #Apply post processing transforms
                batch = self.post_process(batch)
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
            logger.info(f"prediction_list={prediction_list}")
        return prediction_list
    
    def predict_2dto3d(self, volume_path, liver_mask, temp_path="temp/"):
        """
        Predicts the lesions of a 3D volume using a 2D network given a liver mask.
        
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
                Values: background: 0, liver: 1, lesion: 2.
        """
        logger.info('predict_2dto3d')

        # Read volume
        img_volume_array=nib.load(volume_path).get_fdata()
        number_of_slices = img_volume_array.shape[2]
        # Create temporary folder to store 2d png files 
        if os.path.exists(temp_path) == False:
          os.mkdir(temp_path)
        # Write volume slices as 2d png files 
        for slice_number in range(number_of_slices):
            volume_slice = img_volume_array[:, :,slice_number]
            # Delete extension from filename
            volume_file_name = os.path.splitext(volume_path)[0].split("/")[-1]
            nii_volume_path = os.path.join(
                                temp_path, 
                                volume_file_name + "_" + str(slice_number)
                                ) + ".nii.gz"
            new_nii_volume = nib.Nifti1Image(volume_slice, affine = np.eye(4))
            nib.save(new_nii_volume, nii_volume_path)
        # Predict slices individually then reconstruct 3D prediction
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(temp_path))
            volume_paths = [os.path.join(temp_path, file_name) 
                            for file_name in volume_names]
            logger.info(f"volume_paths={volume_paths}")
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
            liver_set = Dataset(data = liver_mask)
            liver_loader = MonaiLoader(
                liver_set,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,
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
                suppressed_volume=ToTensor()(suppressed_volume).to(self.device)
                #predict lesions in isolated liver
                batch[Keys.PRED] = self.network(suppressed_volume)
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        batch = {Keys.PRED : prediction_list}
        # Transform shape from (batch,channel,length,width) 
        # to (1,channel,length,width,batch) 
        batch[Keys.PRED] = batch[Keys.PRED].permute(1,2,3,0).unsqueeze(dim=0) 
        # Apply post processing transforms
        batch = self.post_process(batch)
        # Delete temporary folder
        shutil.rmtree(temp_path)
        logger.info(f"batch[Keys.PRED]={batch[Keys.PRED]}")
        return batch[Keys.PRED]
    

def segment_lesion(
        prediction_path = None,
        liver_inference = '3D',
        lesion_inference = '3D', 
        liver_cp = None, 
        lesion_cp = None
        ):
    """
    Segments the lesions from a liver scan.

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
    lesion_inference : str
        the type of inference to be used for lesion model.
        Expects "2D" for slice inference, "3D" for volume inference,
        or "sliding_window" for sliding window inference.
        Default is 3D
    liver_cp : str
        path of the liver model weights to be used for prediction. 
        if not defined, liver_checkpoint will be loaded from configs.
    lesion_cp : str
        path of the lesion model weights to be used for prediction. 
        if not defined, lesion_checkpoint will be loaded from configs.
    Returns
    ----------
        tensor : predicted lesions segmentation
    """
    logger.info('segment_lesion')
    liver_model = LiverSegmentation(modality = 'CT', inference = liver_inference)
    lesion_model = LesionSegmentation(inference = lesion_inference)
    if prediction_path is None:
        prediction_path = config.dataset['prediction']
        logger.info(f"prediction_path={prediction_path}")
    if liver_cp is None:
        liver_cp = config.save["liver_checkpoint"]
        logger.info(f"liver_cp={liver_cp}")
    if lesion_cp is None:
        lesion_cp = config.save["lesion_checkpoint"]
        logger.info(f"lesion_cp={lesion_cp}")
    liver_model.load_checkpoint(liver_cp)
    lesion_model.load_checkpoint(lesion_cp)
    liver_prediction = liver_model.predict(prediction_path)
    close = MorphologicalClosing(iters = 4)
    lesion_prediction = lesion_model.predict(
                            prediction_path,
                            liver_mask = close(liver_prediction[0]).permute(3,0,1,2) 
                                         if lesion_inference == '3D' 
                                         else liver_prediction
                            )
    liver_lesion_prediction = torch.tensor(np.where(
                                                lesion_prediction == 1,
                                                2, 
                                                liver_prediction
                                                )).to(liver_prediction.device)
    logger.info(f"liver_lesion_prediction={liver_lesion_prediction}")
    return liver_lesion_prediction



def train_lesion(
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
    Starts training of lesion segmentation model.
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
    logger.info('train_lesion')
    if cp_path is None:
        cp_path = config.save["potential_checkpoint"]
        logger.info(f"cp_path={cp_path}")
    if epochs is None:
        epochs = config.training["epochs"]
        logger.info(f"epochs={epochs}")
    if save_path is None:
        save_path = config.save["potential_checkpoint"]
        logger.info(f"save_path={save_path}")
    set_seed()
    model = LesionSegmentation()
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
    logger.debug(f"Initial test loss={init_loss}")
    print(
        "\nInitial test metric:", 
        init_metric.mean().item(),
        )
    logger.debug(f"Initial test metric={init_metric.mean().item()}")
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
    logger.debug(f"Final test loss={final_loss}")
    print(
        "\nFinal test metric:", 
        final_metric.mean().item(),
        )
    logger.debug(f"Final test metric={final_metric.mean().item()}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Lobe Segmentation')
    parser.add_argument(
                '--liver_inference', type = str, default = '3D',
                help = 'choose liver inference mode: 2D, 3D, or sliding_window (default: 3D)'
                )    
    parser.add_argument(
                '--lesion_inference', type = str, default = '3D',
                help = 'choose lesion inference mode: 2D, 3D, or sliding_window (default: 3D)'
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
                '--lesion_cp_path', type = bool, default = None,
                help = 'path of pretrained lesion checkpoint (default: lesion cp config path)'
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
    LesionSegmentation(inference = args.lesion_inference) # to set configs
    if args.predict_path is None:
        args.predict_path = config.dataset['prediction']
    if args.liver_cp_path is None:
        args.liver_cp_path = config.save["liver_checkpoint"]
    if args.lesion_cp_path is None:
        args.lesion_cp_path = config.save["lesion_checkpoint"]
    if args.save_path is None:
        args.save_path = config.save["potential_checkpoint"]
    if args.train: 
        train_lesion(
            inference = args.lesion_inference, 
            pretrained = args.cp, 
            cp_path = args.lesion_cp_path,
            epochs = args.epochs, 
            evaluate_epochs = args.eval_epochs,
            batch_callback_epochs = args.batch_callback,
            save_weight = args.save,
            save_path = args.save_path,
            test_batch_callback = args.test_callback,
            )
    if args.test:
        model = LesionSegmentation()
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
        prediction = segment_lesion(
                        args.predict_path, 
                        liver_inference = args.liver_inference,
                        lesion_inference = args.lesion_inference,
                        liver_cp = args.liver_cp_path,
                        lesion_cp = args.lesion_cp_path
                        )
        #save prediction as a nifti file
        original_header = nib.load(args.predict_path).header
        original_affine = nib.load(args.predict_path).affine
        liver_volume = nib.Nifti1Image(
                            prediction[0,0].cpu(), 
                            affine = original_affine, 
                            header = original_header
                            )
        nib.save(liver_volume, args.predict_path.split('.')[0] + '_lesions.nii')
        print('Prediction saved at', args.predict_path.split('.')[0] + '_lesions.nii')
    print("Run Complete")