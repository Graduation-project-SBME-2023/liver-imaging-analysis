import os
import sys
import shutil
from liver_imaging_analysis.engine.config import config
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from torch.utils.tensorboard import SummaryWriter
from clearml import Task, Dataset
from liver_imaging_analysis.engine.config import config
from google.auth.transport.requests import Request

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class ExperimentTracking:
    """
    Class for experiment tracking using ClearML, Tensorboard and Google Drive.

    Args:
        experiment_name (str): Name of the experiment.
        run_name (str): Name of the run within the experiment.
    """

    def __init__(self, experiment_name, run_name):
        """
        Initializes an ExperimentTracking instance with the given experiment and run names.

        Args:
            experiment_name (str): Name of the experiment.
            run_name (str): Name of the run within the experiment.
        """
        # clearml credentials for experiment tracking
        Task.set_credentials(
            api_host="https://api.clear.ml",
            web_host="https://app.clear.ml",
            files_host="https://files.clear.ml",
            key="8YCA72388IV3AT36EWGJ",
            secret="C6LCwXiOxAX8xgKBffW3qUKybGEtGWXMtzfYr0xMgkBDVlMscl",
        )
        self.experiment_name, self.run_name = experiment_name, run_name
        self.task=None
        self.writer=None

    def new_clearml_logger(self):
        """
        Creates a new ClearML task for experiment tracking.

        Returns:
            clearml.Task: A ClearML task for the experiment.
        """
        # new task
        self.task = Task.init(
            project_name=self.experiment_name, task_name=self.run_name
        )
        Task.add_requirements
        return self.task

    def update_clearml_logger(self):
        """
        Updates the ClearML logger for continuing from previous checkpoints.

        Returns:
            clearml.Task: A ClearML task for the experiment.
        """
        # continuing from previous checkpoints
        tasks = Task.get_tasks(
            project_name=self.experiment_name, task_name=self.run_name
        )
        self.task = Task.init(
            continue_last_task=True, reuse_last_task_id=tasks[-1].task_id
        )
        return self.task

    def tb_logger(self):
        """
        Initializes a Tensorboard logger.

        Returns:
            torch.utils.tensorboard.SummaryWriter: Tensorboard writer.
        """
        # Construct the log directory path
        log_dir = os.path.join(
            config.save["tensorboard"], self.experiment_name, self.run_name
        )

        # Check if the directory exists and create it if not
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Initialize the tensorboard writer
        self.writer = SummaryWriter(log_dir)

        return self.writer
    
   