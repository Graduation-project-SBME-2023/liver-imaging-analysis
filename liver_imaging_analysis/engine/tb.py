import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import shutil
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
SCOPES = ['https://www.googleapis.com/auth/drive.file']

class Benchmarking:

    def __init__(self, logdir):
        self.logdir = logdir
        self.drive_dir = None  # Set your Google Drive folder path here
        Task.set_credentials(
            api_host="https://api.clear.ml",
            web_host="https://app.clear.ml",
            files_host="https://files.clear.ml",
            key="Y08KY2Y42A1CYK5LLMI5",
            secret="5LxqZXjbwlYCMvm0FlaqE0fA5EbXHNExUlcIL2EcHlye3kai8h",
        )


    def exp_naming(self):
        
        self.run_name = (
            f"dropout_{config.network_parameters['dropout']}, "
            f"epochs_{config.training['epochs']}, "
            f"batch_size_{config.training['batch_size']}, "
            f"optimizer_{config.training['optimizer']}, "
            f"lr_scheduler_{config.training['lr_scheduler']}, "
            f"loss_name_{config.training['loss_name']}"
        )
        all_config_variables = str(vars(config))
        self.experiment_name = (f"{config.network_name}" + "_" + str(hash(all_config_variables)))
        print(self.experiment_name)

    def clearml_logger(self):
        task = Task.init(project_name=self.experiment_name, task_name=self.run_name)
        return task

  #  import os

    def tb_logger(self):
        logdir = os.path.join(self.logdir, self.experiment_name, self.run_name)
        writer = SummaryWriter(logdir)
        print(logdir)
        return writer


    def move_folders(self, source_dir):
        # Moves local folders to drive
        if os.listdir(source_dir):
            for folder in os.listdir(source_dir):
                destination_dir = os.path.join(self.drive_dir, folder)
                os.makedirs(destination_dir, exist_ok=True)
                folder_dir = os.path.join(source_dir, folder)
                if os.listdir(folder_dir):
                    for file in os.listdir(folder_dir):
                        file_path = os.path.join(folder_dir, file)
                        destination_path = os.path.join(destination_dir, file)
                        if not os.path.isdir(destination_path):
                            shutil.copytree(file_path, destination_path)

    def log_dataset(self, path, dataset_name, dataset_project):
        ds = Dataset.create(dataset_name=dataset_name, dataset_project=dataset_project)
        ds.add_files(path=path)
        ds.finalize(auto_upload=True)

    def upload_folder(self, service, folder_path, parent_folder_id):
            # Create a folder metadata
            folder_name = os.path.basename(folder_path)
            folder_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }

            # Create the folder in Google Drive
            folder = service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()

            print(f'Created folder: {folder_name} (ID: {folder["id"]})')

            # List all files and subdirectories in the folder
            for item_name in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item_name)

                if os.path.isfile(item_path):
                    # Upload file
                    file_metadata = {
                        'name': item_name,
                        'parents': [folder["id"]]
                    }
                    media = MediaFileUpload(item_path, resumable=True)
                    request = service.files().create(
                        media_body=media,
                        body=file_metadata
                    )
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
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        try:
            service = build('drive', 'v3', credentials=creds)

            # Upload the folder and its contents
            self.upload_folder(service, folder_path, parent_folder_id)
        except HttpError as error:
            print(f'An error occurred: {error}')

    def upload_folder_to_drive(self):
        if self.is_google_colab():
            print("This code is running on a Google Colab machine.")
        else:
            print("This code is running on a local machine.")
            runs_dir = (f"{self.logdir}\{self.experiment_name}")
            parent_folder_id = input("Enter the ID of the parent folder: ")
            self.upload_folder_to_drive_local(runs_dir, parent_folder_id)

    def is_google_colab(self):
        return 'google.colab' in sys.modules

# if __name__ == "__main__":
#     log_directory = "/path/to/your/log/directory"
#     benchmark = Benchmarking(log_directory)
#     benchmark.exp_naming()
#     benchmark.clearml_logger()
#     benchmark.tb_logger()
#     benchmark.upload_folder_to_drive()
