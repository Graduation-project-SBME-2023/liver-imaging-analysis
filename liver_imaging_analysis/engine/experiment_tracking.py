import os
import sys
import shutil
from torch.utils.tensorboard import SummaryWriter
from clearml import Task, Dataset
from liver_imaging_analysis.engine.config import config


class ExperimentTracking:
    """
    Class for experiment tracking using ClearML, Tensorboard and Google Drive.

    """

    def __init__(self):
        """
        Initializes an ExperimentTracking instance for the chosen model and hyperparameters.

        """
        # CLearML credentials for experiment tracking
        Task.set_credentials(
            api_host="https://api.clear.ml",
            web_host="https://app.clear.ml",
            files_host="https://files.clear.ml",
            key=config.clearml_credentials["key"],
            secret=config.clearml_credentials["secret"],
        )
        self.hyperparameters = self.get_hyperparameters()
        self.experiment_name, self.run_name = self.exp_naming()
        self.task = None  # Task object that represents the current running experiment in CLearML
        self.writer = None  # Writer object for Tensorboard logging

    def hash(self, text):
        """
        Calculates a hash value for the input text.

        Parameters
        ----------
        text: str
            The input text.
        Returns
        -------
        int
            The calculated hash value as an integer.
        """
        hash = 0
        for ch in text:
            hash = (hash * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
        return hash

    def exp_naming(self):
        """
        Names the experiment and run based on network name and hyperparameters.

        Returns
        -------
        Tuple
            A tuple containing the experiment name and the run name.
        """
        config.name["experiment_name"] = config.network_name
        run_name = ""
        for key, value in self.hyperparameters.items():
            run_name += f"{key}_{value}_"
        print(f"Experimemnt name: {config.name['experiment_name']}")
        config.name["run_name"] = str(self.hash(run_name))
        print(f"Run ID: {config.name['run_name']}")
        return config.name["experiment_name"], config.name["run_name"]

    def get_hyperparameters(self):
        """
        Get all hyperparameters specified in configs.

        Returns:
            dict: A dictionary containing the hyperparameters.
        """

        hyperparameters = {
            **config.network_parameters
        }  # Copy config.network_parameters dict
        hyperparameters.update(
            {key: value for key, value in config.training.items() if key != "epochs"}
        )  # Merge config.network_parameters and config.training dictionaries
        hyperparameters = {
            key: str(value) if isinstance(value, (list, dict)) else value
            for key, value in hyperparameters.items()
        }
        return hyperparameters

    def new_clearml_logger(self):
        """
        Creates a new ClearML task for experiment tracking.

        Returns
        ----------
            clearml.Task: A ClearML task for the experiment.
        """
        # new task
        self.task = Task.init(
            project_name=self.experiment_name, task_name=self.run_name
        )
        self.task.add_requirements
        # log all configs to ClearMl
        self.task.connect_configuration(config.__dict__, name="configs")
        return self.task

    def update_clearml_logger(self):
        """
        Updates the ClearML logger for continuing from previous checkpoints.

        Returns
        ----------
            clearml.Task: A ClearML task for the experiment.
        """
        # continuing from previous checkpoints
        tasks = Task.get_tasks(
            project_name=self.experiment_name, task_name=self.run_name
        )
        self.task = Task.init(
            continue_last_task=True, reuse_last_task_id=tasks[-1].task_id
        )
        # log all configs to ClearMl
        self.task.connect_configuration(config.__dict__, name="configs")
        return self.task

    def new_tb_logger(self):
        """
        Initializes a Tensorboard logger.

        Returns
        ----------
            torch.utils.tensorboard.SummaryWriter: Tensorboard writer.
        """
        # Construct the log directory path
        log_dir = os.path.join(
            config.save["output_folder"], self.experiment_name, self.run_name
        )

        # Check if the directory exists and create it if not
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize the tensorboard writer
        self.writer = SummaryWriter(log_dir)
        # Log hyperparameters to Tensorboard; they will be automatically reported by ClearML.
        self.writer.add_hparams(self.hyperparameters, metric_dict={})

        return self.writer

    @staticmethod
    def __upload_folder_to_drive_from_colab__(runs_dir):
        """
        Uploads experiment data to Google Drive from a Google Colab environment.

        Parameters
        ----------
        runs_dir : str
            Local path to the model's output files to upload.
        Note:
            This method is intended to be used in a Google Colab environment. It mounts Google Drive
            to '/content/drive/' and copies the experiment data to a user-specified location within
            Google Drive.
        """
        from google.colab import drive

        drive.mount("/content/drive/")

        while True:
            try:
                upload_path = input(
                    "Please enter the path to the 'runs' folder in the shared Google Drive: "
                )
                # Check if the provided path exists
                if os.path.exists(upload_path):
                    break  # Exit the loop when the path is valid
                else:
                    print("Error: The provided path does not exist. Please try again.")
            except Exception as e:
                print(f"An error occurred: {e}")

        experiment_name = os.path.basename(os.path.dirname(runs_dir))
        upload_path = os.path.join(upload_path, experiment_name)

        if not os.path.exists(upload_path):
            os.makedirs(upload_path)

        shutil.move(runs_dir, upload_path)

    @staticmethod
    def upload_to_drive():
        """
        Uploads all run data to Google Drive.

        """
        runs_dir = os.path.join(
            config.save["output_folder"],
            config.name["experiment_name"],
            config.name["run_name"],
        )
        if ExperimentTracking.is_google_colab():
            print("This code is running on a Google Colab machine.")
            ExperimentTracking.__upload_folder_to_drive_from_colab__(runs_dir)

    @staticmethod
    def is_google_colab():
        """
        Checks if the code is running on Google Colab.

        Returns
        ----------
            bool: True if running on Google Colab, False otherwise.
        """
        return "google.colab" in sys.modules
