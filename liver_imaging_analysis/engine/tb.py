import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
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
    def __init__(self, experiment_name, run_name):
        # clearml credentials for experiment tracking
        Task.set_credentials(
            api_host="https://api.clear.ml",
            web_host="https://app.clear.ml",
            files_host="https://files.clear.ml",
            key="OWR4YS0HSZHYPPIZ85AE",
            secret="tuOTEuVPv5UnaR4sE0VUJw8C77IugEWb7bbNYv9rN1PSOPykku",
        )
        self.experiment_name, self.run_name = experiment_name, run_name

    def new_clearml_logger(self):
        # new task
        self.task = Task.init(
            project_name=self.experiment_name, task_name=self.run_name
        )
        Task.add_requirements
        return self.task

    def update_clearml_logger(self):
        # continuing from previous checkpoints
        tasks = Task.get_tasks(
            project_name=self.experiment_name, task_name=self.run_name
        )
        self.task = Task.init(
            continue_last_task=True, reuse_last_task_id=tasks[-1].task_id
        )
        return self.task

    def tb_logger(self):
            # Construct the log directory path
            log_dir = os.path.join(config.save['tensorboard'], self.experiment_name, self.run_name)

            # Check if the directory exists and create it if not
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Initialize the tensorboard writer
            self.writer = SummaryWriter(log_dir)

            return self.writer

    def upload_folder(self, service, folder_path, parent_folder_id):
        # Create a folder metadata
        folder_name = os.path.basename(folder_path)
        folder_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_folder_id],
        }

        # Create the folder in Google Drive
        folder = service.files().create(body=folder_metadata, fields="id").execute()

        print(f'Created folder: {folder_name} (ID: {folder["id"]})')

        # List all files and subdirectories in the folder
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            if os.path.isfile(item_path):
                # Upload file
                file_metadata = {"name": item_name, "parents": [folder["id"]]}
                media = MediaFileUpload(item_path, resumable=True)
                request = service.files().create(media_body=media, body=file_metadata)
                response = request.execute()
                print(f'Successfully uploaded file: {item_name} (ID: {response["id"]})')
            elif os.path.isdir(item_path):
                # Recursively upload subfolder
                self.upload_folder(service, item_path, folder["id"])

    def upload_folder_to_drive_local(self, folder_path, parent_folder_id):
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
            # Upload the folder and its contents
            self.upload_folder(service, folder_path, parent_folder_id)
        except HttpError as error:
            print(f"An error occurred: {error}")

    def upload_folder_to_drive(self):
        if self.is_google_colab():
            print("This code is running on a Google Colab machine.")
        else:
            print("This code is running on a local machine.")
            runs_dir = f"{self.logdir}\{self.experiment_name}"
            parent_folder_id = "1IHOuM7JyptK20PWJpWKJfIxJCI2QKKfm"
            self.upload_folder_to_drive_local(runs_dir, parent_folder_id)

    def is_google_colab(self):
        return "google.colab" in sys.modules
