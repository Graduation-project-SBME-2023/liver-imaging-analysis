"""
a module contains the fixed structure of the core of our code

"""
import os
import random
from liver_imaging_analysis.engine.dataloader import DataLoader, Keys
import numpy as np
import torch
import torch.optim.lr_scheduler
from liver_imaging_analysis.engine.config import config
import monai
from monai.data import Dataset, decollate_batch,  DataLoader as MonaiLoader
from monai.losses import DiceLoss as monaiDiceLoss
from torchmetrics import Accuracy
from liver_imaging_analysis.engine.utils import progress_bar
from monai.metrics import DiceMetric, MeanIoU
import natsort
from monai.transforms import Compose
import time

from monai.handlers.utils import from_engine

class Engine:
    """
    Base class for all segmentation tasks. Tasks should inherit from this class.
    """

    def __init__(self):
        self.device = config.device
        self.batch_size = config.training["batch_size"]
        self.loss = self.get_loss(
            loss_name=config.training["loss_name"],
            **config.training["loss_parameters"],
        )
        self.network = self.get_network(
            network_name=config.network_name,
            **config.network_parameters
        ).to(self.device)
        self.optimizer = self.get_optimizer(
            optimizer_name=config.training["optimizer"],
            **config.training["optimizer_parameters"],
        )
        self.scheduler = self.get_scheduler(
            scheduler_name=config.training["lr_scheduler"],
            **config.training["scheduler_parameters"],
        )
        self.metrics = self.get_metrics(
            metrics_name=config.training["metrics"],
            **config.training["metrics_parameters"],
        )
        self.train_transform = self.get_pretraining_transforms(
            config.transforms["train_transform"]
        )
        self.test_transform = self.get_pretesting_transforms(
            config.transforms["test_transform"]
        )
        self.postprocessing_transforms = self.get_postprocessing_transforms(
             config.transforms["post_transform"]
        )

    def get_optimizer(self, optimizer_name, **kwargs):
        """
        internally used to load optimizer specified in configs.
        Parameters
        ----------
            optimizer_name: str
                name of optimizer to fetch from dictionary
                should be chosen from: 'Adam','SGD'
            **kwargs: dict
                parameters of optimizer, if exist.
        """
        optimizers = {
            "Adam" : torch.optim.Adam,
            "SGD" : torch.optim.SGD,
        }
        return optimizers[optimizer_name](self.network.parameters(), **kwargs)

    def get_scheduler(self, scheduler_name, **kwargs):
        """
        internally used to load lr scheduler specified in configs.
        Parameters
        ----------
            scheduler_name: str
                name of scheduler to fetch from dictionary
                should be chosen from: 'StepLR','CyclicLR','...'
            kwargs: dict
                parameters of optimizer, if exist.
        """
        schedulers = {
            "StepLR" : torch.optim.lr_scheduler.StepLR,
            "CyclicLR" : torch.optim.lr_scheduler.CyclicLR,
        }
        return schedulers[scheduler_name](self.optimizer, **kwargs)

    def get_network(self, network_name, **kwargs):
        """
        internally used to load network specified in configs.
        Parameters
        ----------
            network_name: str
                name of network to fetch from dictionary
                should be chosen from: '3DUNet','3DResNet','2DUNet'
            **kwargs: dict
                parameters of network, if exist.
        """
        networks = {
            "monai_2DUNet" : monai.networks.nets.UNet,
        }
        return networks[network_name](**kwargs)

    def get_loss(self, loss_name, **kwargs):
        """
        internally used to load loss function specified in configs.
        Parameters
        ----------
            loss_name: str
                name of loss function to fetch from dictionary
                should be chosen from: 'dice_loss','monai_dice'
            **kwargs: dict
                parameters of loss function, if exist.
        """
        loss_functions = {
            "monai_dice" : monaiDiceLoss,
        }
        return loss_functions[loss_name](**kwargs)

    def get_metrics(self, metrics_name, **kwargs):
        """
        internally used to load metrics specified in configs.
        Parameters
        ----------
            metrics_name: str
                name of metrics to be fetched from dictionary
                should be chosen from: 'accuracy','dice',
            **kwargs: dict
                parameters of metrics, if exist.
        """
        metrics = {
            "accuracy" : Accuracy,
            "dice" : DiceMetric,
            "jaccard" : MeanIoU,
        }
        return metrics[metrics_name](**kwargs)
    
    def get_hparams(self, network_name):
        """
        used to return a hyperparameters dictionary specific to each network 
        ----------
            network_name: str
                name of network to fetch from dictionary
                should be chosen from: '3DUNet','3DResNet','2DUNet'
        """
        hparams = {
            "monai_2DUNet" :  {
            "spatial_dims": config.network_parameters["spatial_dims"],
            "num_res_units": config.network_parameters["num_res_units"],
            "bias": config.network_parameters["bias"],
            "norm": config.network_parameters["norm"],
            "dropout": config.network_parameters["dropout"],
            "batch_size": config.training["batch_size"],
            "optimizer": config.training["optimizer"],
            "lr_scheduler": config.training["lr_scheduler"],
            "loss_name": config.training["loss_name"],
            "metrics": config.training["metrics"],
            "shuffle": config.training["shuffle"]
        }
        }
        return hparams[network_name]
    
    def Hash(self, text:str):
        """
        Calculate a hash value for the input text.

        Parameters:
        text (str): The input text.
        Returns:
        int: The calculated hash value as an integer.
        """
        hash=0
        for ch in text:
            hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
        return hash

    def exp_naming(self):
        """
        Names the experiment after network name,
        and names the run based on hyperparameters used.

        """

        config.name['experiment_name']=config.network_name
        run_name = ""
        hparams=self.get_hparams(config.network_name)
        for key, value in hparams.items():
            run_name += f'{key}_{value}_'
        print( f"Experimemnt name: {config.name['experiment_name']}")
        config.name['run_name']=str(self.Hash(run_name))
        print( f"Run ID: {config.name['run_name']}") 

    def get_pretraining_transforms(self, *args, **kwargs):
        """
        Should be Implemented by user in task module.
        Transforms to be applied on training set before training.
        Expected to return a monai Compose object with desired transforms.

        Raises:
            NotImplementedError: When the function is not implemented.
        """
        raise NotImplementedError()

    def get_pretesting_transforms(self, *args, **kwargs):
        """
        Should be Implemented by user in task module.
        Transforms to be applied on testing set before evaluation.
        Expected to return a monai Compose object with desired transforms.

        Raises:
            NotImplementedError: When the function is not implemented.
        """
        raise NotImplementedError()

    def get_postprocessing_transforms(self, *args, **kwargs):
        """
        Should be Implemented by user in task module.
        Transforms to be applied on predicted data to correct prediction.
        Expected to return a monai Compose object with desired transforms.
        """
        return Compose([])

    def post_process(self, batch):
            """
            Applies the transformations specified in get_postprocessing_transforms
            to the network output.

            Parameters
            ----------
            batch: dict
                a dictionary containing the model's output to be post-processed
            """
            post_batch = [self.postprocessing_transforms(i) 
                        for i in decollate_batch(batch)]
            for key in batch.keys():
                if key in Keys.all():
                    batch[key] = from_engine(key)(post_batch)
                    batch[key] = torch.stack(batch[key], dim = 0)
            return batch 

    def load_data(self):
        """
        Internally used to load and save the data to the data attributes.
        Uses the parameter values in config.
        """
        self.train_dataloader = []
        self.val_dataloader = []
        self.test_dataloader = []
        trainloader = DataLoader(
            dataset_path = config.dataset["training"],
            batch_size = config.training["batch_size"],
            train_transforms = self.train_transform,
            test_transforms = self.test_transform,
            num_workers = 0,
            pin_memory = False,
            test_size = config.training["train_valid_split"],
            mode = config.dataset["mode"],
            shuffle = config.training["shuffle"]
        )
        testloader = DataLoader(
            dataset_path = config.dataset["testing"],
            batch_size = config.training["batch_size"],
            train_transforms = self.train_transform,
            test_transforms = self.test_transform,
            num_workers = 0,
            pin_memory = False,
            test_size = 1,  # test set should all be used for evaluation
            mode = config.dataset["mode"],
            shuffle = config.training["shuffle"]
        )
        self.train_dataloader = trainloader.get_training_data()
        self.val_dataloader = trainloader.get_testing_data()
        self.test_dataloader = testloader.get_testing_data()

    def data_status(self):
        """
        Prints the shape and data type of a training batch, a validation batch,
        and a testing batch, if exists.
        """
        dataloader_iterator = iter(self.train_dataloader)
        try:
            print("Number of Training Batches:", len(dataloader_iterator))
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Training Features:"
                f" {batch[Keys.IMAGE].shape} {batch[Keys.IMAGE].dtype}"
            )
            print(
                f"Batch Shape of Training Labels:"
                f" {batch[Keys.LABEL].shape} {batch[Keys.LABEL].dtype}"
            )
        except StopIteration:
            print("No Training Set")
        dataloader_iterator = iter(self.val_dataloader)
        try:
            print("Number of Validation Batches:", len(dataloader_iterator))
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Validation Features:"
                f" {batch[Keys.IMAGE].shape} {batch[Keys.IMAGE].dtype}"
            )
            print(
                f"Batch Shape of Validation Labels:"
                f" {batch[Keys.LABEL].shape} {batch[Keys.LABEL].dtype}"
            )
        except StopIteration:
            print("No Validation Set")
        dataloader_iterator = iter(self.test_dataloader)
        try:
            print("Number of Testing Batches:", len(dataloader_iterator))
            batch = next(dataloader_iterator)
            print(
                f"Batch Shape of Testing Features:"
                f" {batch[Keys.IMAGE].shape} {batch[Keys.IMAGE].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch[Keys.LABEL].shape} {batch[Keys.LABEL].dtype}"
            )
        except StopIteration:
            print("No Testing Set")

    def save_checkpoint(self, path = config.save["model_checkpoint"]):
        """
        Saves the current network, optimizer, and scheduler states.

        Parameters
        ----------
        path: str
            The path at which the checkpoint is to be saved. 
            Default path is the one specified in config.
        """
        checkpoint = {
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path = config.save["model_checkpoint"]):
        """
        Loads network, optimizer, and scheduler states, if exist.

        Parameters
        ----------
        path: str
            The path of the checkpoint to be loaded.
            Default path is the one specified in config.
        """
        checkpoint = torch.load(path)
        if ('state_dict' in checkpoint.keys()): #dict checkpoint
            self.network.load_state_dict(checkpoint['state_dict'])
            if ('optimizer' in checkpoint.keys()):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if ('scheduler' in checkpoint.keys()):
                self.scheduler.load_state_dict(checkpoint['scheduler'])
        else: #weights only
            self.network.load_state_dict(checkpoint)

    def compile_status(self):
        """
        Prints the loss function and the optimizer status.
        """
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")

    def per_batch_callback(self, *args, **kwargs):
          """
          A generic callback function to be executed every batch.
          Supposed to output information desired by user.
          Should be Implemented in task module.
          """
          pass

    def per_epoch_callback(self, *args, **kwargs):
        """
        A generic callback function to be executed every epoch.
        Supposed to output information desired by user.
        Should be Implemented in task module.
        """
        pass

    def fit(
        self,
        summary_writer,
        offset,
        epochs = config.training["epochs"],
        evaluate_epochs = 1,
        batch_callback_epochs = None,
        save_weight = False,
        save_path = config.save["potential_checkpoint"],
    ):
        """
        train the model using the stored training set

        Parameters
        ----------
        epochs: int
            The number of training iterations over data.
            Default is the value specified in config.
        evaluate_epochs: int
            The number of epochs to evaluate model after. Default is 1.
        batch_callback_epochs: int
            The frequency at which per_batch_callback will be called, if exists.
            Expects a number of epochs. Default is None.
        save_weight: bool
            Flag to save best weights. Default is False.
        save_path: str
            Directory to save best weights at. 
            Default is the potential path in config.
        """
 
        for epoch in range(epochs):
            pre_train = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
            training_loss = 0
            training_metric = 0
            self.network.train()
            progress_bar(0, len(self.train_dataloader))  # epoch progress bar
            epoch_start_timestamps=time.time()
            for batch_num, batch in enumerate(self.train_dataloader):
                progress_bar(batch_num + 1, len(self.train_dataloader))
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                batch[Keys.LABEL] = batch[Keys.LABEL].to(self.device)
                batch[Keys.PRED] = self.network(batch[Keys.IMAGE])
                loss = self.loss(batch[Keys.PRED], batch[Keys.LABEL])
                # Apply post processing transforms and calculate metrics
                pre_post_process_time = time.time()
                batch = self.post_process(batch)
                post_post_process_time = time.time() -pre_post_process_time
                post_process_time_per_epoch += post_post_process_time
                self.metrics(batch[Keys.PRED].int(), batch[Keys.LABEL].int())
                # Backpropagation
                pre_back_prop_time = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                post_back_prop_time = time.time()-pre_back_prop_time
                back_prop_time_per_epoch+=  post_back_prop_time
                training_loss += loss.item()
                if batch_callback_epochs is not None:
                    if (epoch + 1) % batch_callback_epochs == 0:
                        self.per_batch_callback(
                                summary_writer,
                                batch_num,
                                batch[Keys.IMAGE],
                                batch[Keys.LABEL],
                                batch[Keys.PRED], # thresholded prediction
                            )
            print(f"time of post processing per epoch in training:{post_process_time_per_epoch}")
            print(f"time of backpropagation time per epoch in training:{back_prop_time_per_epoch}")
            self.scheduler.step()
            # normalize loss over batch size
            training_loss = training_loss / len(self.train_dataloader)  
            # aggregate batches metrics of current epoch
            training_metric = self.metrics.aggregate().item()
            # reset the status for next computation round
            self.metrics.reset()
            # every evaluate_epochs, test model on test set
            if (epoch + 1) % evaluate_epochs == 0:  
                valid_loss, valid_metric = self.test(self.test_dataloader)
            if save_weight:
                self.save_checkpoint(save_path)
            else:
                valid_loss = None
                valid_metric = None
            self.per_epoch_callback(
                    summary_writer,
                    epoch+offset,
                    training_loss,
                    valid_loss,
                    training_metric,
                    valid_metric,
                    epoch_start_timestamps,
                    self.optimizer.param_groups[0]['lr']
                )
            post_train = time.time() - pre_train
            print(f"time of training per epoch {post_train}")

    def test(self, dataloader = None, callback = False):
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
        print('\nTESTING:')
        with torch.no_grad():
            for batch_num,batch in enumerate(dataloader):
                progress_bar(batch_num + 1, len(dataloader))
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                batch[Keys.LABEL] = batch[Keys.LABEL].to(self.device)
                batch[Keys.PRED] = self.network(batch[Keys.IMAGE])
                test_loss += self.loss(
                    batch[Keys.PRED],
                    batch[Keys.LABEL]
                    ).item()
                #Apply post processing transforms on prediction
                pre_post_processing = time.time()
                batch = self.post_process(batch)
                post_post_processing = time.time()-pre_post_processing
                post_processing_time += post_post_processing
                self.metrics(batch[Keys.PRED].int(), batch[Keys.LABEL].int())
                if callback:
                  self.per_batch_callback(
                      batch_num,
                      batch[Keys.IMAGE],
                      batch[Keys.LABEL],
                      batch[Keys.PRED]
                      )
            print(f"post procesing time in validation {post_processing_time}")

            test_loss /= num_batches
            # aggregate the final metric result
            test_metric = self.metrics.aggregate().item()
            # reset the status for next computation round
            self.metrics.reset()
        return test_loss, test_metric

    def predict(self, data_dir):
        """
        predict the label of the given input
        Parameters
        ----------
        data_dir: str
            path of the input directory. expects to contain nifti or png files.
        
        Returns
        -------
        tensor
            tensor of the predicted labels
        """
        self.network.eval()
        with torch.no_grad():
            volume_names = natsort.natsorted(os.listdir(data_dir))
            volume_paths = [os.path.join(data_dir, file_name) 
                            for file_name in volume_names]
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
            prediction_list = []
            for batch in predict_loader:
                batch[Keys.IMAGE] = batch[Keys.IMAGE].to(self.device)
                batch[Keys.PRED] = self.network(batch[Keys.IMAGE])
                #Apply post processing transforms
                batch = self.post_process(batch)
                prediction_list.append(batch[Keys.PRED])
            prediction_list = torch.cat(prediction_list, dim=0)
        return prediction_list


def set_seed():
    """
    Sets seed for all randomized attributes of the packages and modules.
    Usually called before engine initialization. 
    """
    seed = config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)
