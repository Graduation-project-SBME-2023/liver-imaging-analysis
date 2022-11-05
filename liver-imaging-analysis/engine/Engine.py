import torch
from torch import nn

import dataloader


class Engine (nn.Module):
    """"
        Class that implements the basic PyTorch methods for neural network
        Neural Networks should inherit from this class
        Methods:
            load_data: load and save the data to the data attribute
                Args:   Paths: dictionary that includes dataset names and their respective directories
                               directories should include two folders: "Path1" and "Path2" containing features and labels, respectively.
                        dataset_name: string of the dataset name to be loaded from Paths
                        transformation_flag: boolean that indicates if data preprocessing will occur or not. 
                                             False will ignore "transformation" arguement 
                        transformation: an array of the shape data will be transformed into. eg; [64,512,512]
                        batchsize: the number of features to be loaded in each batch. Default: 1
                        test_valid_split: a number between 0-1 that indicate the portion of dataset to be loaded to the validation set. Default: 0
            data_status: Prints the shape and dtype of the first batch of the training set and testing set, if exists.
            compile: Stores the loss function, the optimizer, and the metrics to be used during fitting and evaluating
                Args: loss: the loss function to be used, should be imported from loss_functions
                      optimizer: the optimizer to be used, should be imported from torch.optim
                      metrics: the metrics calculated for each batch per epoch during training, and for the whole data during evaluating
                               expects and array of string of one or more of: 'loss', 'dice_score'. Default: ['loss']
            compile_status: Prints the stored loss function, optimizer, and metrics.
            fit: train the model using the stored training set
                Args: epochs: the number of iterations for fitting. Default: 1
            test: function the calculate metrics without updating weights
                Args: dataloader: the dataset to evaluate on
            evaluate_train: function that evaluates the model on the stored training dataset by calling "Test" 
            evaluate_test: function that evaluates the model on the stored testing dataset by calling "Test" 
            predict: predict the label of the given input using the current weights
                Args: XPath: path of the input feature. expects a nifti file.
                returns: tensor of the predicted label
    """


    def __init__(self):
        #self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device= "cpu"
        super(Engine,self).__init__()


    def load_data(self,dataset_path,transformation_flag,transformation,batchsize=1,test_valid_split=0):
        '''
        description: loads and saves the data to the data attribute
        
        Paths: dictionary that includes dataset names and their respective directories that should include two folders:"Path1" and "Path2" containing features and labels, respectively.

        dataset_name: string of the dataset name to be loaded from Paths

        transformation_flag: boolean that indicates if data preprocessing should be performed or not ,False will ignore "transformation" arguement 

        transformation: an array of the shape data will be transformed into. eg; [64,512,512]

        batchsize: the number of features to be loaded in each batch.(Default: 1) 
            
        test_valid_split: a fraction between 0-1 that indicate the portion of dataset to be loaded to the validation set.( Default: 0)
        '''
        self.transformation=transformation
        self.expand_flag= not transformation_flag
        self.train_dataloader=[]
        self.test_dataloader=[]
        DataLoader= dataloader.DataLoader(dataset_path,batchsize,0,False,test_valid_split,transformation_flag,dataloader.keys,transformation)
        self.train_dataloader= DataLoader.get_training_data()
        self.test_dataloader= DataLoader.get_testing_data()
             

    def data_status(self):
        '''
        data_status: Prints the shape and dtype of the first batch of the training set and testing set, if exists
        '''
        for batch in self.train_dataloader:
            print(f"Batch Shape of Training Features: {batch['image'].shape} {batch['image'].dtype}")
            print(f"Batch Shape of Training Labels: {batch['label'].shape} {batch['label'].dtype}")
            break
        for batch in self.test_dataloader:
            print(f"Batch Shape of Testing Features: {batch['image'].shape} {batch['image'].dtype}")
            print(f"Batch Shape of Testing Labels: {batch['label'].shape} {batch['label'].dtype}")
            break


        
    def compile (self,loss,optimizer,metrics=['loss']):
        '''
        Description: Stores the loss function, the optimizer, and the metrics to be used during fitting and evaluating

        loss: the loss function to be used, should be imported from loss_functions class

        optimizer: the optimizer to be used, should be imported from torch.optim

        metrics: the metrics calculated for each batch per epoch during training, and for the whole data during evaluating
        expects and array of string of one or more of: 'loss', 'dice_score'. Default: ['loss']

        '''

        self.loss=loss
        self.optimizer=optimizer
        self.metrics=metrics


    def compile_status (self):
        '''
        description: Prints the stored loss function, optimizer, and metrics.
        '''
        print(f"Loss= {self.loss} \n")
        print(f"Optimizer= {self.optimizer} \n")
        print(f"Metrics= {self.metrics} \n")
        

    def fit(self,epochs=1):
        '''
        description : train the model using the stored training set

        epochs: the number of iterations for fitting. (Default = 1) 
        '''
        self.Epochs=epochs
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            size = self.train_dataloader.__len__()
            self.train()  # from pytorch
            for batch_num, batch in enumerate(self.train_dataloader):
                volume,mask= batch['image'].to(self.device),batch['label'].to(self.device)
                if (self.expand_flag):
                    volume=volume.expand(1,volume.shape[0],volume.shape[1],volume.shape[2],volume.shape[3])
                    mask=mask.expand(1,mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3])
                pred = self(volume)
                loss = self.loss(pred, mask)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #Print Progress
                current= batch_num * len(volume) +1
                if 'loss' in self.metrics:
                    print(f"loss: {loss.item():>7f}        [{current:>5d}/{size:>5d}]")
                if 'dice_score' in self.metrics:
                    print(f"Dice Score: {(1-loss.item()):>7f}  [{current:>5d}/{size:>5d}]")
                # if 'accuracy' in self.Metrics:
                #     self.eval()
                #     with torch.no_grad():
                #             pred = self(X)
                #     correct = int((pred.round()==y).sum())
                #     correct /= math.prod(pred.shape)
                #     print(f"Accuracy: {(100*correct):>0.1f}%       [{current:>5d}/{size:>5d}]")

                    
    def test(self, dataloader):
        '''
        description: function the calculate metrics without updating weights

        dataloader: the dataset to evaluate on
            
        '''
        num_batches = len(dataloader)
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = batch['image'].to(self.device),batch['label'].to(self.device)
                if (self.expand_flag):
                    volume=volume.expand(1,volume.shape[0],volume.shape[1],volume.shape[2],volume.shape[3])
                    mask=mask.expand(1,mask.shape[0],mask.shape[1],mask.shape[2],mask.shape[3])
                pred = self(volume)
                if 'loss' or 'dice_score' in self.metrics:
                    test_loss += self.loss(pred, mask).item()
        test_loss /= num_batches
        if 'loss' in self.metrics:
            print(f"loss: {test_loss:>7f}")
        if 'dice_score' in self.metrics:
            print(f"Dice Score: {(1-test_loss):>7f}")


    def evaluate_train(self):
        '''
        description: function that evaluates the model on the stored training dataset by calling "Test" 
        '''
        self.test(self.train_dataloader)

    def evaluate_test(self):
        '''
        description: function that evaluates the model on the stored testing dataset by calling "Test"
        '''
        self.test(self.test_dataloader)
        

    def predict(self,volume_path):
        '''
        description: predict the label of the given input using the current weights

        XPath: path of the input feature. expects a nifti file.

        returns: tensor of the predicted label
        '''
        dict_loader = dataloader.LoadImageD(keys=("image", "label"))
        data_dict = dict_loader({"image": volume_path ,"label": volume_path})
        preprocess = dataloader.Preprocessing(("image", "label"), self.transformation)
        data_dict_processed = preprocess(data_dict)
        volume=data_dict_processed["image"]
        volume=volume.expand(1,volume.shape[0],volume.shape[1],volume.shape[2],volume.shape[3])
        self.eval()
        with torch.no_grad():
            pred = self(volume.to(self.device))
        return pred
