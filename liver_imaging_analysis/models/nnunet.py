from liver_imaging_analysis.engine.config import config
import os
import torch

class NnUnet():
    """
    The `NnUnet` class provides an interface for training and segmenting medical imaging data using the nnU-Net framework.
    """

    def __init__(self, dataset_id, dataset_name, preprocessed_folder=None, results_folder=None, raw_folder=None):
        """
        Initializes the `NnUnet` class with the specified dataset details and sets up the environment.
        The `preprocessed_folder`, `results_folder`, and `raw_folder` should be created beforehand. 
        The nnU-Net framework expects datasets to be organized in a specific structure. 
        For detailed information on setting up datasets compatible with nnU-Net, please refer to (https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

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

    def segment(self, file, fold=0, configuration="2d",output_file=None):
        """
        Performs segmentation on a specified file using the trained nnU-Net model.

        Args:
            file (str): Path to the file to be segmented.
            fold (int, optional): Fold number to be used for prediction. Default is 0.
            configuration (str, optional): Configuration type for nnU-Net. Default is "2d".
            output_file (str, optional): Path to the output file for  segmentation to be saved.

        Returns:
            np.ndarray: The segmentation result produced by the nnU-Net model.    
        """

        from batchgenerators.utilities.file_and_folder_operations import join
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.paths import nnUNet_results
        
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            device=torch.device(self.device),
            verbose=True,
            verbose_preprocessing=True,
            allow_tqdm=True
        )
        
        # Initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(nnUNet_results, f'Dataset{self.dataset_id}_{self.dataset_name}/nnUNetTrainer__nnUNetPlans__{configuration}'),
            use_folds=(fold,),
            checkpoint_name='checkpoint_best.pth',
        )

        seg=predictor.predict_from_files([[file]],
                                     output_folder_or_list_of_truncated_output_files=output_file,    
                                     save_probabilities=False, overwrite=True,
                                     num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                     folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        
        return seg
    

    def plan_and_preprocess(self):
        """
        Plans and preprocesses the dataset for nnU-Net.

        This method performs the following steps:
        
        1. Extracts fingerprints: Systematically analyzes the provided training cases and creates a 'dataset fingerprint'. 
           This fingerprint is used to understand the dataset characteristics.
        2. Plans experiments: Uses the dataset fingerprint to design three U-Net configurations tailored to the dataset.
        3. Preprocesses data: Each U-Net configuration operates on its own preprocessed version of the dataset.

        """
        from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess

        extract_fingerprints(dataset_ids=[self.dataset_id], num_processes = 1, check_dataset_integrity = True)
        plan_experiments(dataset_ids=[self.dataset_id])
        preprocess(dataset_ids=[self.dataset_id],num_processes=1)

    

    def train(self, pretrained=False, fold=0, configuration="2d"):
        """
        Trains the nnU-Net model with the specified configuration and fold.

        Args:
            pretrained (bool, optional): Whether to use a pretrained model. Default is False.
            fold (int, optional): Fold number to be used for training (0 to 4). Default is 0.
            configuration (str, optional): Configuration type for nnU-Net. Default is "2d".
        """
        from nnunetv2.run.run_training import run_training
        run_training(dataset_name_or_id=self.dataset_id, configuration=configuration, fold=fold, continue_training= pretrained, device=torch.device(self.device))
