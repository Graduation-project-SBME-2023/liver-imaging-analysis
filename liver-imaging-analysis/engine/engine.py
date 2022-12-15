import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import dataloader
import matplotlib.pyplot as plt
import utils
import nibabel as nib
from monai.visualize import plot_2d_or_3d_image



class Engine(nn.Module):
    """Class that implements the basic PyTorch methods for neural network
    Neural Networks should inherit from this class
    """

    def __init__(self,device):
        self.device = device
        super(Engine,self).__init__()

    def load_data(
        self,
        training_data_path,
        testing_data_path,
        transformation_flag,
        transformation,
        batchsize=1,
        test_valid_split=0,
    ):
        """
        Loads and saves the data to the data attribute

        Parameters
        ----------
        training_data_path: str
              String containing paths of the training volumes at the folder
              Path and masks at the folder Path2.
        testing_data_path: str
              String containing paths of the testing volumes at the folder
              Path and masks at the folder Path2.
        transformation_flag: bool
              indicates if data preprocessing should be performed or not
              False will ignore "transformation" argument
        transformation: array_like
              an array of the shape data will be transformed into.
              eg; [64,512,512]
        batchsize: int
            the number of features to be loaded in each batch.(Default: 1)
        test_valid_split: float
            a fraction between 0-1 that indicate the portion of dataset to be
            loaded to the validation set.( Default: 0)
        """

        self.transformation = transformation
        self.expand_flag = not transformation_flag
        self.train_dataloader = []
        self.val_dataloader = []
        self.test_dataloader = []

        trainloader = dataloader.DataLoader(
            dataset_path=training_data_path,
            batch_size=batchsize,
            num_workers=0,
            pin_memory=False,
            test_size=test_valid_split,
            transform=transformation_flag,
            # keys=dataloader.keys,
            size=transformation,
        )
        testloader = dataloader.DataLoader(
            dataset_path=testing_data_path,
            batch_size=batchsize,
            num_workers=0,
            pin_memory=False,
            test_size=test_valid_split,
            transform=transformation_flag,
            # keys=dataloader.keys,
            size=transformation,
        )
        self.train_dataloader = trainloader.get_training_data()
        self.val_dataloader = trainloader.get_testing_data()
        self.test_dataloader = testloader.get_training_data()

    def data_status(self):
        """
        Prints the shape and data type of the first batch of the training set
        and testing set, if exists
        """
        for batch in self.train_dataloader:
            print(
                f"Batch Shape of Training Features:"
                f" {batch['image'].shape} {batch['image'].dtype}"
            )
            print(
                f"Batch Shape of Training Labels:"
                f" {batch['label'].shape} {batch['label'].dtype}"
            )
            break
        for batch in self.val_dataloader:
            print(
                f"Batch Shape of Testing Features:"
                f" {batch['image'].shape} {batch['image'].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch['label'].shape} {batch['label'].dtype}"
            )
            break

    def save_checkpoint(self, path):
        """
        Saves current checkpoint to a specific path

        Parameters
        ----------
        path: int
            The path where the checkpoint will be saved at
        """
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        """
        Loads checkpoint from a specific path

        Parameters
        ----------
        path: int
            The path of the checkpoint
        """
        self.load_state_dict(
            torch.load(path, map_location=torch.device("cpu"))
        )  # if working with CUDA remove torch.device('cpu')

    def compile(self, loss, optimizer, metrics=["loss"]):
        """
        Stores the loss function, the optimizer,
        and the metrics to be used during fitting and evaluating

        Parameters
        ----------
        loss: str
            the loss function to be used,
            should be imported from loss_functions class
        optimizer: str
            the optimizer to be used, should be imported from torch.optim
        metrics: array_like
            the metrics calculated for each batch per epoch during training,
            and for the whole data during evaluating expects an array of string
            of one or more of: 'loss', 'dice_score'. Default: ['loss']
        """

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def compile_status(self):
        """
        Prints the stored loss function, optimizer, and metrics.
        """
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")
        print(f"Metrics= {self.metrics} \n")

    def fit(self, epochs=1):
        """
        train the model using the stored training set

        Parameters
        ----------
        epochs: int
             the number of iterations for fitting. (Default = 1)
        """
        self.epochs = epochs
        self.total_epochs_loss = []
        tb = SummaryWriter()
        best_epoch_loss=1
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            epoch_loss = 0
            size = self.train_dataloader.__len__()
            self.train()  # from pytorch
            for batch_num, batch in enumerate(self.train_dataloader):
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                if self.expand_flag:
                    volume = volume.expand(
                        1,
                        volume.shape[0],
                        volume.shape[1],
                        volume.shape[2],
                        volume.shape[3],
                    )
                    mask = mask.expand(
                        1, mask.shape[0], mask.shape[1],
                        mask.shape[2], mask.shape[3]
                    )
                pred = self(volume)
                loss = self.loss(pred, mask)                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Print Progress
                current = batch_num * len(volume) + 1
                if "loss" in self.metrics:
                    print(f"loss: {loss.item():>7f}"
                          f"      [{current:>5d}/{size:>5d}]")
                    epoch_loss = epoch_loss + loss.item()

                if "dice_score" in self.metrics:
                    print(
                        f"Dice Score: "
                        f"{(1-loss.item()):>7f}  [{current:>5d}/{size:>5d}]"
                    )
                
                plot_2d_or_3d_image(data=batch['image'],step=0,writer=tb,frame_dim=-1,tag=f"volume{batch_num}")
                plot_2d_or_3d_image(data=batch['label'],step=0,writer=tb,frame_dim=-1,tag=f"mask{batch_num}")
            epoch_loss = epoch_loss / len(self.train_dataloader)
            self.total_epochs_loss.append(epoch_loss)
            # print(" TOTAL LOSS = ",self.totalloss)
            tb.add_scalar("Epoch average loss", epoch_loss, epoch)
            if epoch == 0:
                best_epoch_loss=epoch_loss
                self.save_checkpoint("first_epoch")
            elif epoch_loss < best_epoch_loss:
                best_epoch_loss=epoch_loss
                self.save_checkpoint("best_epoch")


    def test(self, dataloader):
        """
        function that calculates metrics without updating weights

        Parameters
        ----------
        dataloader: dict
                the dataset to evaluate on
        """
        num_batches = len(dataloader)
        # self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = batch["image"].to(self.device),\
                               batch["label"].to(self.device)
                if self.expand_flag:
                    volume = volume.expand(
                        1,
                        volume.shape[0],
                        volume.shape[1],
                        volume.shape[2],
                        volume.shape[3],
                    )
                    mask = mask.expand(
                        1, mask.shape[0], mask.shape[1],
                        mask.shape[2], mask.shape[3]
                    )
                pred = self(volume)
                if "loss" or "dice_score" in self.metrics:
                    test_loss += self.loss(pred, mask).item()
        test_loss /= num_batches
        if "loss" in self.metrics:
            print(f"loss: {test_loss:>7f}")
        if "dice_score" in self.metrics:
            print(f"Dice Score: {(1-test_loss):>7f}")

    def evaluate_train(self):
        """
        function that evaluates the model on the stored
        training dataset by calling "Test"
        """
        self.test(self.train_dataloader)
        epochs = np.arange(0, self.Epochs)
        plt.plot(epochs, self.total_epochs_loss)
        plt.show()

    def evaluate_test(self):
        """
        function that evaluates the model on the stored
        testing dataset by calling "Test"
        """
        self.test(self.test_dataloader)

    def predict(self, volume_path):
        """
        predict the label of the given input using the current weights

        Parameters
        ----------
        volume_path: str
                  path of the input feature. expects a nifti file.
        Returns
        -------
        tensor
            tensor of the predicted label
        """
        dict_loader = dataloader.LoadImageD(keys=("image", "label"))
        data_dict = dict_loader({"image": volume_path, "label": volume_path})
        preprocess = dataloader.Preprocessing(("image", "label"),
                                              self.transformation)
        inverse_transformation= data_dict['label'].shape
        plt.imshow(data_dict['label'][:,:,105])
        print(inverse_transformation)
        data_dict_processed = preprocess(data_dict)
        volume = data_dict_processed["image"]
        volume = volume.expand(
            1, volume.shape[0], volume.shape[1],
            volume.shape[2], volume.shape[3]
        )
        # self.eval()       
        with torch.no_grad():
            pred = self(volume.to(self.device))
        return pred

    
    def fit2d(self,epochs=1,callback_flag=False,callback_epochs=1):
        self.train()
        tb = SummaryWriter()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}\n-------------------------------")
            epoch_loss=0
            self.train()  
            for batch_num, batch in enumerate(self.train_dataloader):
                print(f"Batch: {batch_num+1}/{len(self.train_dataloader)}")
                volume,mask= batch['image'].to(self.device),batch['label'].to(self.device)
                volume=volume.permute(3, 0, 1, 2)
                mask=mask.permute(3, 0, 1, 2)
                pred2d = self(volume)
                loss = self.loss(pred2d, mask)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()
                plot_2d_or_3d_image(data=[batch['image']],step=0,writer=tb,frame_dim=-1,tag=f"Batch{batch_num}: Volume")
                plot_2d_or_3d_image(data=[batch['label']],step=0,writer=tb,frame_dim=-1,tag=f"Batch{batch_num}: Label")
                
                pred2d = self(volume)
                pred2dnew=torch.sigmoid(pred2d.detach())
                pred2dnew=(pred2dnew>0.5).float().permute( 1,2,3,0)
                plot_2d_or_3d_image(data=[pred2dnew],step=0,writer=tb,frame_dim=-1,tag=f"Batch{batch_num}: Prediction")

            #every callback_epochs plot prediction and save weights
            epoch_loss/=len(self.train_dataloader)
            tb.add_scalar("Epoch average loss", epoch_loss, epoch)
            print(f"Epoch Loss: {epoch_loss}")
            
            if callback_flag:  
              if ((epoch+1)%callback_epochs) == 0:
                self.save_checkpoint(
                  "/content/drive/MyDrive/liver-imaging-analysis/engine/potential_checkpoint")
                # for batch_num, batch in enumerate(self.train_dataloader):
                #       volume,mask= batch['image'].to(self.device),batch['label'].to(self.device)
                #       volume=volume.permute(3, 0, 1, 2)
                #       mask=mask.permute(3, 0, 1, 2)
                #       plt.subplot(1, 3, 1)
                #       plt.imshow(volume[16][0])
                #       plt.title("Volume")
                #       plt.subplot(1, 3, 2)
                #       plt.imshow(mask[16][0])
                #       plt.title("True Mask")
                #       pred2d = self(volume)
                #       pred2dnew=torch.sigmoid(pred2d.detach())
                #       pred2dnew=(pred2dnew>0.5).float()
                #       plt.subplot(1, 3, 3)
                #       plt.imshow(pred2dnew[16][0])
                #       plt.title("Predicted Mask")
                #       plt.show()
                #       if batch_num>5: #plot only 5 pictures
                #           break

    def test2d(self,dataloader):
        self.train()
        with torch.no_grad():
            total_loss=0
            for batch_num,batch in enumerate(dataloader):
                self.train()
                print(f"Batch: {batch_num+1}/{len(dataloader)}")
                volume,mask= batch['image'].to(self.device),batch['label'].to(self.device)
                volume=volume.permute(3, 0, 1, 2)
                mask=mask.permute(3, 0, 1, 2)
                pred = self(volume)
                loss = self.loss(pred, mask)
                total_loss+=loss.item()
                print(f"Batch Loss: {loss.item()}")
            total_loss/=len(dataloader)
            print(f"Total Loss: {total_loss}")

    def evaluate_test2d(self):
        """
        function that evaluates the model on the stored
        testing dataset by calling "Test"
        """
        self.test2d(self.test_dataloader)

    def pred2d(self,batch_path):
        self.train()
        DataLoader= dataloader.DataLoader(batch_path,1,0,False,0,self.transformation_flag,dataloader.keys,self.transformation)
        predict_data= DataLoader.get_training_data()
        pred_list=[]
        mask_list=[]
        vol_list=[]
        for batch_num,predict_dict in enumerate(predict_data):
            print(f"batch: {batch_num+1}/{len(predict_data)}")
            predict_volume,useless_var=predict_dict['image'].to(self.device),predict_dict['label'].to(self.device)
            predict_volume=predict_volume.permute(3, 0, 1, 2)
            self.train()
            with torch.no_grad():
                pred = self(predict_volume)
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float().permute(1, 2, 3, 0)
                pred_list.append(pred)
                mask_list.append(useless_var)#useless_var is either vol or mask after processing
                vol_list.append(predict_volume.permute(1, 2, 3, 0))
        return torch.stack(pred_list),torch.stack(mask_list),torch.stack(vol_list)

