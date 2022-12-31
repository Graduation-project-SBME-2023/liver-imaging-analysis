import os
import sys
from engine.config import config
from sklearn.model_selection import train_test_split
import monai

def slices_paths_reader(volume_text_path,mask_text_path):
    """ Read two paths contain txt files and return two lists contain the content of txt files

    Parameters
    ----------
    volume_text_path: str
        String containing the path of txt file which contains the paths of the volumes and slices
    mask_text_path: str
        String containing the path of txt file which contains the paths of the masks and slices

    Returns
    ----------
    volume_paths: list
        a list contains the paths of all the volumes and slices
    mask_paths: list
        a list contains the paths of all the masks and slices    
    """

  # empty list to read list from a file
    volume_paths = []
    mask_paths = []
    # open file and read the content in a list
    with open(volume_text_path, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]
            # add current item to the list
            volume_paths.append(x)
    with open(mask_text_path, 'r') as fp:
        for line in fp:
            x = line[:-1]
            mask_paths.append(x)
    return volume_paths,mask_paths

class DataLoader:
    def __init__(
            self,
            dataset_path,
            batch_size,
            transforms,
            num_workers=0,
            pin_memory=False,
            test_size=0.1,
            keys=("image", "label"),
            shuffle=False,
    ):
        """Initializes and saves all the parameters required for creating
        transforms as well as initializing two dataset instances to be
        used for loading the testing and the training data

         Parameters
         ----------
         dataset_path: str
              String containing paths of the volumes at the folder volume and
              masks at the folder mask.
         batch_size: int
             Size of batches to be returned
         transforms: list
             List of transforms to be applied on the data
         num_workers : int, optional
             Integer that specifies how many sub-processes to use for data
             loading and is set by default to 0.
         pin_memory : bool, optional
             If True, the data loader will copy tensors into CUDA pinned
             memory before returning them. Default is False.
         test_size : float
             proportion of the test size to the whole dataset.
             A number between 0 and 1. Default is 0.1
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
        self.mode='2D'
        if (self.mode=='2D'): # naive way will be optimized later 
            volume_names = os.path.join(dataset_path, "volume.txt")
            mask_names = os.path.join(dataset_path, "mask.txt")
            volume_paths,mask_paths = slices_paths_reader(volume_names,mask_names)

        else:
            volume_paths = [os.path.join(dataset_path, "volume", file_name) for file_name in volume_names]
            mask_paths = [os.path.join(dataset_path, "mask", file_name) for file_name in mask_names]


        volume_paths.sort()  # to be sure that the paths are sorted so every volume corresponds to the correct mask
        mask_paths.sort()

        if test_size == 0:  # train_test_split does not take 0 as a valid test_size, so we have to implement manually.
            training_volume_path, training_mask_path = volume_paths, mask_paths
            test_volume_path = []
            test_mask_path = []
        elif test_size == 1:  # train_test_split does not take 0 as a valid test_size, so we have to implement manually.
            test_volume_path, test_mask_path = volume_paths, mask_paths
            training_volume_path = []
            training_mask_path = []
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

        self.train_ds = monai.data.Dataset(data=train_files, transform=transforms)
        self.test_ds = monai.data.Dataset(data=test_files, transform=transforms)

    def get_training_data(self):
        """Loads the training dataset.

        Returns
        -------
        dict
            Dictionary containing the training volumes and masks
            that can be called using their specified keys.
            An iterable object over the training data
        """
        train_loader = monai.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
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
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return test_loader
