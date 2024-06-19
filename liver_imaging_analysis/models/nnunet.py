from liver_imaging_analysis.engine.config import config
from monai.metrics import DiceMetric
import os
import torch

class NnUnet():
    """
    The `NnUnet` class provides an interface for training and segmenting medical imaging data using the nnU-Net framework.
    """

    def __init__(self, dataset_id, dataset_name, preprocessed_folder=None, results_folder=None, raw_folder=None):
        """
        Initializes the `NnUnet` class with the specified dataset details and sets up the environment.

        Args:
            dataset_id (str): The ID of the dataset.
            dataset_name (str): The name of the dataset.
            preprocessed_folder (str, optional): Path to the preprocessed data folder.
            results_folder (str, optional): Path to the results folder.
            raw_folder (str, optional): Path to the raw data folder.
        """
        self.device = config.device
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.setup_environment(preprocessed_folder, results_folder, raw_folder)


    def setup_environment(self, preprocessed_folder, results_folder, raw_folder):
        """
        Sets up the environment variables required by nnU-Net.

        Args:
            preprocessed_folder (str): Path to the preprocessed data folder.
            results_folder (str): Path to the results folder.
            raw_folder (str): Path to the raw data folder.
        """
        os.environ['nnUNet_preprocessed'] = preprocessed_folder
        os.environ['nnUNet_results'] = results_folder
        os.environ['nnUNet_raw'] = raw_folder

    def segment(self, file=None, fold=0, configuration="2d"):
        """
        Performs segmentation on a specified file using the trained nnU-Net model.

        Args:
            file (str, optional): Path to the file to be segmented.
            fold (int, optional): Fold number to be used for prediction. Default is 0.
            configuration (str, optional): Configuration type for nnU-Net. Default is "2d".
        """

        from batchgenerators.utilities.file_and_folder_operations import join
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device(self.device),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        # Initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(self.results_folder, f'Dataset{self.dataset_id}_{self.dataset_name}/nnUNetTrainer__nnUNetPlans__{configuration}'),
            use_folds=(fold,),
            checkpoint_name='checkpoint_best.pth',
        )

        predictor.predict_from_files([[file]],
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    def train(self, pretrained=False, fold=0, configuration="2d"):
        """
        Trains the nnU-Net model with the specified configuration and fold.

        Args:
            pretrained (bool, optional): Whether to use a pretrained model. Default is False.
            fold (int, optional): Fold number to be used for training. Default is 0.
            configuration (str, optional): Configuration type for nnU-Net. Default is "2d".
        """
        from nnunetv2.run.run_training import run_training
        run_training(dataset_name_or_id=self.dataset_id, configuration=configuration, fold=fold, continue_training= pretrained, device=torch.device(self.device))
