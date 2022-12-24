import os
import sys
sys.path.append('/Users/mn3n3/Documents/GP/liver-imaging-analysis/liver-imaging-analysis/')
from configs.config import configuration
from sklearn.model_selection import train_test_split
import monai
from monai.data import Dataset


class DataLoader:
    def __init__(
            self,
            dataset_path,
            batch_size,
            transforms,
            num_workers = 0,
            pin_memory = False,
            test_size = 0.15,
            keys = ("image", "label"),
            shuffle =  False,
    ):
        """Initializes and saves all the parameters required for creating
        transforms as well as initializing two dataset instances to be
        used for loading the testing and the training data

         Parameters
         ----------
         dataset_path: str
              String containing paths of the volumes at the folder Path and
              masks at the folder Path2.
         batch_size: int
             Integer size of batches to be returned
         num_workers : int, optional
             Integer that specifies how many sub-processes to use for data
             loading and is set by default to 0.
         pin_memory : bool, optional
             If True, the data loader will copy tensors into CUDA pinned
             memory before returning them.
         test_size : float
             proportion of the test size to the whole dataset.
             A number between 0 and 1.
         transforms: list
             List of transforms to be applied on the data
         keys: dict
              Dictionary of the corresponding items to be loaded.
              set by default to ("image","label")
        shuffle: bool
              If, True will shuffle the loaded images and masks before returning them 
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        volume_names = os.listdir(os.path.join(dataset_path, "volume"))
        mask_names = os.listdir(os.path.join(dataset_path, "mask"))

        volume_paths = [os.path.join(dataset_path, "volume", fname) for fname in volume_names]
        mask_paths = [os.path.join(dataset_path, "mask", fname) for fname in mask_names]

        volume_paths.sort()
        mask_paths.sort()

        if test_size == 0:
            training_volume_path, training_mask_path = volume_paths, mask_paths
            test_volume_path = []
            test_mask_path = []
        else:
            (
                training_volume_path,
                test_volume_path,
                training_mask_path,
                test_mask_path,
            ) = train_test_split(
                volume_paths, mask_paths, test_size=test_size,
            )

        train_files = [{keys[0]: image_name, keys[1]: label_name} for image_name, label_name in
                       zip(training_volume_path, training_mask_path)]
        test_files = [{keys[0]: image_name, keys[1]: label_name} for image_name, label_name in
                      zip(test_volume_path, test_mask_path)]

        self.train_ds = Dataset(data=train_files, transform = transforms)
        self.test_ds = Dataset(data=test_files, transform=transforms)
        

    def get_training_data(self):
        """Loads the training dataset.

        Returns
        -------
        dict
            Dictionary containing the training volumes and masks
            that can be called using their specified keys.
        """
        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = self.shuffle,
        )
        return train_loader

    def get_testing_data(self):
        """Loads the testing data set.

        Returns
        -------
        dict
        Dictionary containing the testing volumes and masks
        that can be called using their specified keys.
        """
        test_loader = monai.data.DataLoader(
            self.test_ds,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            shuffle = self.shuffle,
        )

        return test_loader
