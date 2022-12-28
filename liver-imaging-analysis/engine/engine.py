import sys
from typing import Dict
import torch
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss as monaiDiceLoss
from monai.visualize import plot_2d_or_3d_image
from engine.config import config
from engine import dataloader
from engine import models
from engine import losses 
from engine.utils import progress_bar

class Engine():
    """
    Class that implements the basic PyTorch methods for deep learning tasks
    tasks should inherit this class
    """

    def __init__(
        self 
        ):

        self.device = config.device
        print("Used Device: ",self.device)
        self.loss = self.get_loss(
            loss_name=config.loss_function
       
        )

        self.network = self.get_network(
            network_name=config.network_name,
            **config.network_parameters
            ).to(self.device)


        self.optimizer = self.get_optimizer(
            optimizer_name = config.optimizer,
            **config.optimizer_parameters
            )

        self.load_data()
    

    def get_optimizer(self,optimizer_name,**kwargs):        
        """
        internally used to load optimizer.
        Parameters
        ----------
            optimizer_name: str
                name of optimizer to fetch from dictionary 
                should be chosen from: 'Adam','SGD'
            **kwargs: dict
                parameters of optimizer, if exist.
        """
        optimizers={
            'Adam': torch.optim.Adam,
            'SGD': torch.optim.SGD,
        }
        return optimizers[optimizer_name](self.network.parameters(), **kwargs)

    def get_network(self,network_name,**kwargs):
        """
        internally used to load network.
        Parameters
        ----------
            network_name: str
                name of network to fetch from dictionary 
                should be chosen from: '3DUNet','3DResNet','2DUNet'
            **kwargs: dict
                parameters of network, if exist.
        """
        networks = {
            '3DUNet': models.UNet3D,
            '3DResNet': models.ResidualUNet3D,
            '2DUNet' : models.UNet2D
        }
        return networks[network_name](**kwargs)

    def get_loss(self,loss_name):
        """
        internally used to load loss function.
        Parameters
        ----------
            loss_name: str
                name of loss function to fetch from dictionary 
                should be chosen from: 'dice_loss','monai_dice'
            **kwargs: dict
                parameters of loss function, if exist.
        """
        loss_functions = {
            'dice_loss': losses.DiceLoss,
            'monai_dice' : monaiDiceLoss,
            'bce_dice': losses.BCEDiceLoss
            

        }        
        return loss_functions[loss_name]()


    def get_pretraining_transforms(self):
        """
        Should be Implemented by user.
        Transforms to be applied on training set before training.
        Expected to return a monai Compose object with desired transforms.
        """
        raise NotImplementedError()

    def get_pretesting_transforms(self):
        """
        Should be Implemented by user.
        Transforms to be applied on testing set before evaluation.
        Expected to return a monai Compose object with desired transforms.        
        """
        raise NotImplementedError()


    def load_data(
        self,
    ):
        """
        Internally used to load and save the data to the data attributes.
        Uses the parameter values in config.
        """

        self.train_dataloader = []
        self.val_dataloader = []
        self.test_dataloader = []
        self.train_transform = self.get_pretraining_transforms(
            config.tranform_name,
            keys = (config.img_key,config.label_key)
            )
        self.test_transform = self.get_pretesting_transforms(
            config.tranform_name,
            keys = (config.img_key,config.label_key)
            )

        trainloader = dataloader.DataLoader(
            dataset_path = config.train_data_path,
            batch_size = config.batch_size,
            transforms = self.train_transform,
            num_workers = 0,
            pin_memory = False,
            test_size = config.train_valid_split,
            keys = (config.img_key,config.label_key),
        )
        testloader = dataloader.DataLoader(
            dataset_path = config.test_data_path,
            batch_size = config.batch_size,
            transforms = self.test_transform,
            num_workers = 0,
            pin_memory = False,
            test_size = 0, #testing set should all be set as evaluation (no training)
            keys = (config.img_key,config.label_key),
        )
        self.train_dataloader = trainloader.get_training_data()
        self.val_dataloader = trainloader.get_testing_data()
        self.test_dataloader = testloader.get_training_data()

    def data_status(self):
        """
        Prints the shape and data type of a training batch
        and a testing batch, if exists.
        """
        for batch in self.train_dataloader:
            print(
                f"Batch Shape of Training Features:"
                f" {batch[config.img_key].shape} {batch[config.img_key].dtype}"
            )
            print(
                f"Batch Shape of Training Labels:"
                f" {batch[config.label_key].shape} {batch[config.label_key].dtype}"
            )
            break
        for batch in self.val_dataloader:
            print(
                f"Batch Shape of Testing Features:"
                f" {batch[config.img_key].shape} {batch[config.img_key].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch[config.label_key].shape} {batch[config.label_key].dtype}"
            )
            break

    def save_checkpoint(self, path = config.model_checkpoint):
        """
        Saves current checkpoint to a specific path. (Default is the model path in config)

        Parameters
        ----------
        path: str
            The path where the checkpoint will be saved at
        """
        torch.save(self.network.state_dict(), path)

    def load_checkpoint(self, path = config.model_checkpoint):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: str
            The path of the checkpoint. (Default is the model path in config)
        """
        self.network.load_state_dict(
            torch.load(path, map_location = torch.device(self.device))
        ) 


    def compile_status(self):
        """
        Prints the loss function and optimizer to be used.
        """
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")
        

    def fit(
        self,
        epochs = config.epochs,
        do_evaluation = False,
        evaluate_epochs = 1,
        visualize_epochs = None,
        save_weight = False,
        save_path = config.potential_checkpoint
        ):
        """
        train the model using the stored training set

        Parameters
        ----------
        epochs: int
            the number of iterations for fitting. (Default is the value in config)
        do_evaluation: bool
            if true, evaluate on the test set, occurs every evaluate_epochs. (Default is False)
        evaluate_epochs: int
            the number of epochs to evaluate model after. (Default is 1)
        visualize_epochs: int
            the number of epochs to visualize gifs after, if exists. (Default is None)
        save_weight: bool
            flag to save best weights. (Default is False)
        save_path: str
            directory to save best weights at. (Default is the potential path in config)
        """
        # summary_writer = SummaryWriter(config["save"]["Tensor Board"])    
        best_training_loss=float('inf') #initialization with largest possible number
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}\n-------------------------------")
            training_loss = 0
            self.network.train()  
            progress_bar(0, len(self.train_dataloader))  ## batch progress bar
            for batch_num, batch in enumerate(self.train_dataloader):
                progress_bar(batch_num, len(self.train_dataloader))
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                pred = self.network(volume)
                loss = self.loss(pred, mask)                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                training_loss += loss.item()

                # Print Progress
                if visualize_epochs != None:
                    if ((epoch+1)%visualize_epochs==0): #every visualize_epochs create gifs
                        plot_2d_or_3d_image(data=batch['image'],step=0,writer=summary_writer,
                                            frame_dim=-1,tag=f"Batch{batch_num}:Volume")
                        plot_2d_or_3d_image(data=batch['label'],step=0,writer=summary_writer,
                                            frame_dim=-1,tag=f"Batch{batch_num}:Mask")
                        plot_2d_or_3d_image(data=(torch.sigmoid(pred)>0.5).float(),step=0,writer=summary_writer,
                                            frame_dim=-1,tag=f"Batch{batch_num}:Prediction") 

            training_loss = training_loss / config.batch_size  ## normalize loss over batch size
            print("\nTraining Loss=",training_loss)
            # summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
            
            if save_weight: #save model if performance improved on validation set
                if training_loss <= best_training_loss:
                    best_training_loss = training_loss
                    self.save_checkpoint(path = save_path)

            if ((epoch+1)%evaluate_epochs==0): #every evaluate_epochs, test model on test set
                if do_evaluation == True:
                    valid_loss=self.test(self.test_dataloader)
                    print(f"Validation Loss={valid_loss}")
                    # summary_writer.add_scalar("Validation Loss", valid_loss, epoch)


    def test(self, dataloader):
        """
        calculates loss on input dataset

        Parameters
        ----------
        dataloader: dict
                the dataset to evaluate on
        """
        num_batches = len(dataloader)
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                pred = self.network(volume)
                test_loss += self.loss(pred, mask).item()
            test_loss /= num_batches
        return test_loss


    # def predict(self, volume_path):
    #     """
    #     predict the label of the given input using the current weights

    #     Parameters
    #     ----------
    #     volume_path: str
    #               path of the input feature. expects a nifti file.
    #     Returns
    #     -------
    #     tensor
    #         tensor of the predicted label
    #     """
    #     dict_loader = dataloader.LoadImageD(keys=("image", "label"))
    #     data_dict = dict_loader({"image": volume_path, "label": volume_path})
    #     preprocess = dataloader.Preprocessing(("image", "label"),
    #                                           self.data_size)
    #     data_dict_processed = preprocess(data_dict)
    #     volume = data_dict_processed["image"]
    #     volume = volume.expand(
    #         1, volume.shape[0], volume.shape[1],
    #         volume.shape[2], volume.shape[3]
    #     )       
    #     with torch.no_grad():
    #         pred = self.network(volume.to(self.device))
    #     return pred
