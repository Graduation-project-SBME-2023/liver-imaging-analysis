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
    
    @staticmethod
    def __upload_folder__(service, folder_path, parent_folder_id):
        """
        Recursively uploads files and subdirectories in the given folder to Google Drive.

        Args:
            service (googleapiclient.discovery.Resource): Google Drive API service.
            folder_path (str): Local path to the folder to upload.
            parent_folder_id (str): ID of the parent folder in Google Drive.
        """
        # List all files and subdirectories in the folder
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            if os.path.isfile(item_path):
                # Upload file
                file_metadata = {"name": item_name, "parents": [parent_folder_id]}
                media = MediaFileUpload(item_path, resumable=True)
                request = service.files().create(media_body=media, body=file_metadata)
                response = request.execute()
                print(f'Successfully uploaded file: {item_name} (ID: {response["id"]}')

            elif os.path.isdir(item_path):
                # Recursively upload subfolder
                ExperimentTracking.__upload_folder__(service, item_path, parent_folder_id)
    
    @staticmethod
    def __upload_folder_to_drive_from_local__(
        runs_dir, cp_path, parent_folder_id
    ):
        """
        Uploads a local folder to Google Drive.

        Args:
            runs_dir (str): Local path to the tensorboard files to upload.
            cp_path (str): Local path to the checkpoint to upload.
            parent_folder_id (str): ID of the parent folder in Google Drive.
        """
        creds = None

        # The file token.json stores the user's access and refresh tokens and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        try:
            service = build("drive", "v3", credentials=creds)

            # Check if the experiment folder exists, if not, create it
            experiment_folder_id = ExperimentTracking.__get_or_create_folder__(
                service, parent_folder_id, config.name['experiment_name']
            )

            # Check if the run folder exists, if not, create it inside the experiment folder
            run_folder_id = ExperimentTracking.__get_or_create_folder__(
                service, experiment_folder_id, config.name['run_name']
            )

            # Upload the runs folder and its contents to the drive
            ExperimentTracking.__upload_folder__(service, runs_dir, run_folder_id)

            # Upload the checkpoint to the run folder
            parts = cp_path.split("/")
            cp_path = "/".join(parts[:-1]) + "/"
            ExperimentTracking.__upload_folder__(service, cp_path, run_folder_id)

        except HttpError as error:
            print(f"An error occurred: {error}")
        finally:
            # After uploading, delete the token file
            if os.path.exists("token.json"):
                os.remove("token.json")
                print("Deleted token.json after upload")
    
    @staticmethod
    def __get_or_create_folder__(service, parent_folder_id, folder_name):
        """
        Checks if a folder exists in Google Drive and creates it if it doesn't.

        Args:
            service (googleapiclient.discovery.Resource): Google Drive API service.
            parent_folder_id (str): ID of the parent folder in Google Drive.
            folder_name (str): Name of the folder to check/create.

        Returns:
            str: ID of the folder in Google Drive.
        """
        folder_id = None
        # Check if the folder already exists
        query = f"'{parent_folder_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
        results = service.files().list(q=query).execute()
        files = results.get("files", [])
        if not files:
            # If the folder doesn't exist, create it
            folder_metadata = {
                "name": folder_name,
                "parents": [parent_folder_id],
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = service.files().create(body=folder_metadata, fields="id").execute()
            folder_id = folder.get("id")
        else:
            folder_id = files[0]["id"]
        return folder_id
    
    @staticmethod
    def __upload_folder_to_drive_from_colab__(runs_dir,cp_path):
        """
        Uploads experiment data to Google Drive from a Google Colab environment.

        Args:
            runs_dir (str): Local path to the tensorboard files to upload.
            cp_path (str): Local path to the checkpoint to upload.

        Note:
            This method is intended to be used in a Google Colab environment. It mounts Google Drive
            to '/content/drive/' and copies the experiment data to a user-specified location within
            Google Drive.
        """
        from google.colab import drive

        drive.mount("/content/drive/")
        upload_path = input(
            "Please enter the path to the 'runs' folder in the shared Google Drive: "
        )

        experiment_name = os.path.basename(os.path.dirname(runs_dir))
        run_name = os.path.basename(runs_dir)
        upload_path=os.path.join(upload_path,experiment_name,run_name)

        if not os.path.exists(upload_path):
          os.makedirs(upload_path)
        
        for filename in os.listdir(runs_dir):
          source_file = os.path.join(runs_dir, filename)
          target_file = os.path.join(upload_path, filename)
          shutil.copy(source_file, target_file)
        # shutil.copy(cp_path, upload_path)
        # shutil.copytree(runs_dir, upload_path)


        # Upload the checkpoint to the run folder
        # parts = cp_path.split("/")
        # cp_path = "/".join(parts[:-1]) + "/"
        # shutil.copy(cp_path, upload_path)
    
    @staticmethod
    def upload_to_drive():
        """
        Uploads all run data to Google Drive.

        """
        cp_dir=os.path.join(config.save['potential_checkpoint'], config.name['experiment_name'], config.name['run_name'])
        cp_path = f"{cp_dir}/{config.save['potential_checkpoint']}" 
        runs_dir=os.path.join(config.save["tensorboard"], config.name['experiment_name'], config.name['run_name'])
        if ExperimentTracking.is_google_colab():
            print("This code is running on a Google Colab machine.")
            # ExperimentTracking.__upload_folder_to_drive_from_colab__(runs_dir,cp_path)
        else:
            print("This code is running on a local machine.")
            parent_folder_id = "1IHOuM7JyptK20PWJpWKJfIxJCI2QKKfm"
            ExperimentTracking.__upload_folder_to_drive_from_local__(
                runs_dir, cp_path, parent_folder_id
            )

    @staticmethod
    def is_google_colab():
        """
        Checks if the code is running on Google Colab.

        Returns:
            bool: True if running on Google Colab, False otherwise.
        """
        return "google.colab" in sys.modules