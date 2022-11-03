import DataLoader as dp
import torch
from torch import nn


class Engine (nn.Module):
    """"
        Class that implements the basic PyTorch methods for neural network
        Neural Networks should inherit from this class
        Methods:
            LoadData: load and save the data to the data attribute
                Args:   Paths: dictionary that includes dataset names and their respective directories
                               directories should include two folders: "Path1" and "Path2" containing features and labels, respectively.
                        dataset_name: string of the dataset name to be loaded from Paths
                        transformation_flag: boolean that indicates if data preprocessing will occur or not. 
                                             False will ignore "transformation" arguement 
                        transformation: an array of the shape data will be transformed into. eg; [64,512,512]
                        batchsize: the number of features to be loaded in each batch. Default: 1
                        test_valid_split: a number between 0-1 that indicate the portion of dataset to be loaded to the validation set. Default: 0
            DataStatus: Prints the shape and dtype of the first batch of the training set and testing set, if exists.
            Compile: Stores the loss function, the optimizer, and the metrics to be used during fitting and evaluating
                Args: loss: the loss function to be used, should be imported from loss_functions
                      optimizer: the optimizer to be used, should be imported from torch.optim
                      metrics: the metrics calculated for each batch per epoch during training, and for the whole data during evaluating
                               expects and array of string of one or more of: 'loss', 'dice_score'. Default: ['loss']
            CompileStatus: Prints the stored loss function, optimizer, and metrics.
            Fit: train the model using the stored training set
                Args: epochs: the number of iterations for fitting. Default: 1
            Test: function the calculate metrics without updating weights
                Args: dataloader: the dataset to evaluate on
            Evaluate_train: function that evaluates the model on the stored training dataset by calling "Test" 
            Evaluate_test: function that evaluates the model on the stored testing dataset by calling "Test" 
            Predict: predict the label of the given input using the current weights
                Args: XPath: path of the input feature. expects a nifti file.
                returns: tensor of the predicted label
    """


    def __init__(self):
        #self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.Device= "cpu"
        super(Engine,self).__init__()


    def LoadData(self,Paths,dataset_name,transformation_flag,transformation,batchsize=1,test_valid_split=0):
        self.transformation=transformation
        self.expand_flag= not transformation_flag
        self.train_dataloader=[]
        self.test_dataloader=[]
        DataLoader= dp.DataLoader(Paths,dataset_name,batchsize,0,False,test_valid_split,transformation_flag,dp.KEYS,transformation)
        self.train_dataloader= DataLoader.get_training_data()
        self.test_dataloader= DataLoader.get_testing_data()
             

    def DataStatus(self):
        for Batch in self.train_dataloader:
            print(f"Batch Shape of Training Features: {Batch['image'].shape} {Batch['image'].dtype}")
            print(f"Batch Shape of Training Labels: {Batch['label'].shape} {Batch['label'].dtype}")
            break
        for Batch in self.test_dataloader:
            print(f"Batch Shape of Testing Features: {Batch['image'].shape} {Batch['image'].dtype}")
            print(f"Batch Shape of Testing Labels: {Batch['label'].shape} {Batch['label'].dtype}")
            break


    def Compile (self,loss,optimizer,metrics=['loss']):
        self.Loss=loss
        self.Optimizer=optimizer
        self.Metrics=metrics


    def CompileStatus (self):
        print(f"Loss= {self.Loss} \n")
        print(f"Optimizer= {self.Optimizer} \n")
        print(f"Metrics= {self.Metrics} \n")
        

    def Fit(self,epochs=1):
        self.Epochs=epochs
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            size = self.train_dataloader.__len__()
            self.train()
            for batch_num, batch in enumerate(self.train_dataloader):
                X,y= batch['image'].to(self.Device),batch['label'].to(self.Device)
                if (self.expand_flag):
                    X=X.expand(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
                    y=y.expand(1,y.shape[0],y.shape[1],y.shape[2],y.shape[3])
                pred = self(X)
                loss = self.Loss(pred, y)
                # Backpropagation
                self.Optimizer.zero_grad()
                loss.backward()
                self.Optimizer.step()
                #Print Progress
                current= batch_num * len(X) +1
                if 'loss' in self.Metrics:
                    print(f"loss: {loss.item():>7f}        [{current:>5d}/{size:>5d}]")
                if 'dice_score' in self.Metrics:
                    print(f"Dice Score: {(1-loss.item()):>7f}  [{current:>5d}/{size:>5d}]")
                # if 'accuracy' in self.Metrics:
                #     self.eval()
                #     with torch.no_grad():
                #             pred = self(X)
                #     correct = int((pred.round()==y).sum())
                #     correct /= math.prod(pred.shape)
                #     print(f"Accuracy: {(100*correct):>0.1f}%       [{current:>5d}/{size:>5d}]")

                    
    def Test(self, dataloader):
        num_batches = len(dataloader)
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                X,y= batch['image'].to(self.Device),batch['label'].to(self.Device)
                if (self.expand_flag):
                    X=X.expand(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
                    y=y.expand(1,y.shape[0],y.shape[1],y.shape[2],y.shape[3])
                pred = self(X)
                if 'loss' or 'dice_score' in self.Metrics:
                    test_loss += self.Loss(pred, y).item()
        test_loss /= num_batches
        if 'loss' in self.Metrics:
            print(f"loss: {test_loss:>7f}")
        if 'dice_score' in self.Metrics:
            print(f"Dice Score: {(1-test_loss):>7f}")


    def Evaluate_train(self):
        self.Test(self.train_dataloader)

    def Evaluate_test(self):
        self.Test(self.test_dataloader)
        

    def Predict(self,XPath):
        dict_loader = dp.LoadImageD(keys=("image", "label"))
        data_dict = dict_loader({"image": XPath ,"label": XPath})
        preprocess = dp.preprocessing(("image", "label"), self.transformation)
        data_dict_processed = preprocess(data_dict)
        X=data_dict_processed["image"]
        X=X.expand(1,X.shape[0],X.shape[1],X.shape[2],X.shape[3])
        self.eval()
        with torch.no_grad():
            pred = self(X.to(self.Device))
        return pred