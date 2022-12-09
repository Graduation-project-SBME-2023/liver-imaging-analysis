import torch
from torch import nn

import dataloader


class Engine(nn.Module):
    """Class that implements the basic PyTorch methods for neural network
    Neural Networks should inherit from this class
    """

    def __init__(self):
        # self.Device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        super(Engine, self).__init__()

    def load_data(
        self,
        dataset_path,
        transformation_flag,
        transformation,
        batchsize=1,
        test_valid_split=0,
    ):
        """
        Loads and saves the data to the data attribute

        Parameters
        ----------
        dataset_path: str,
              String containing paths of the volumes at the folder Path and
              masks at the folder Path2.
        transformation_flag: bool,
              indicates if data preprocessing should be performed or not
              False will ignore "transformation" argument
        transformation: array_like,
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
        self.test_dataloader = []
        loader = dataloader.DataLoader(
            dataset_path,
            batchsize,
            0,
            False,
            test_valid_split,
            transformation_flag,
            dataloader.keys,
            transformation,
        )
        self.train_dataloader = loader.get_training_data()
        self.test_dataloader = loader.get_testing_data()

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
        for batch in self.test_dataloader:
            print(
                f"Batch Shape of Testing Features:"
                f" {batch['image'].shape} {batch['image'].dtype}"
            )
            print(
                f"Batch Shape of Testing Labels:"
                f" {batch['label'].shape} {batch['label'].dtype}"
            )
            break

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
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            size = self.train_dataloader.__len__()
            self.train()  # from pytorch
            for batch_num, batch in enumerate(self.train_dataloader):
                volume, mask = \
                    batch["image"].to(self.device),\
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
                    print(f"loss: {loss.item():>7f}  "
                          f"      [{current:>5d}/{size:>5d}]")
                if "dice_score" in self.metrics:
                    print(
                        f"Dice Score: "
                        f"{(1-loss.item()):>7f}  [{current:>5d}/{size:>5d}]"
                    )
                # if 'accuracy' in self.Metrics:
                #     self.eval()
                #     with torch.no_grad():
                #             pred = self(X)
                #     correct = int((pred.round()==y).sum())
                #     correct /= math.prod(pred.shape)
                #     print(f"Accuracy: {(100*correct):>0.1f}%"
                #           f"[{current:>5d}/{size:>5d}]")

    def test(self, dataloader):
        """
        function that calculates metrics without updating weights

        Parameters
        ----------
        dataloader: dict
                the dataset to evaluate on
        """
        num_batches = len(dataloader)
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                volume, mask = \
                    batch["image"].to(self.device),\
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
                    test_loss += \
                        self.loss(pred, mask).item()
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
        preprocess = dataloader.Preprocessing(
            ("image", "label"), self.transformation)
        data_dict_processed = preprocess(data_dict)
        volume = data_dict_processed["image"]
        volume = volume.expand(
            1, volume.shape[0], volume.shape[1],
            volume.shape[2], volume.shape[3]
        )
        self.eval()
        with torch.no_grad():
            pred = self(volume.to(self.device))
        return pred
