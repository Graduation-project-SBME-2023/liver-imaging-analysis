import os
from glob import glob
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from monai.data.utils import decollate_batch, pad_list_data_collate
from monai.transforms import (
LoadImageD,
ForegroundMaskD,
EnsureChannelFirstD,
AddChannelD,
ScaleIntensityD,
ToTensorD,
Compose,
NormalizeIntensityD,
AsDiscreteD,
SpacingD,
OrientationD,
ResizeD,
RandSpatialCropd,
Spacingd,
RandFlipd,
RandScaleIntensityd,
RandShiftIntensityd,
RandRotated,
SqueezeDimd,
CenterSpatialCropD,
)
import monai
import numpy as np
import matplotlib.pyplot as plt



roi_size=[500, 500, 30]
keys=("image", "label")


class Preprocessing():
    def __init__(self, keys, size):
        """Makes an instance of the preprocessing transforms.

            Parameters
            ----------
            keys: keys, optional
                 keys of the corresponding items to be loaded containing two strings
                 the first refers to the key for image and the second refers to the key for label.
            size: array_like
                Array of the wanted volume size
        """

        self.transform = Compose([
            EnsureChannelFirstD(keys),
            # AddChannelD(keys),
            #assumes label is not rgb
            #will need to manually implement a class for multiple segments
            #OrientationD(keys, axcodes='LAS'), #preferred by radiologists
            #SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
            # CenterSpatialCropD(keys,size),
            ResizeD(keys, size , mode=('trilinear', 'nearest')),
            #RandFlipd(keys, prob=0.5, spatial_axis=1),
            #RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, keep_size=True),
            #RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            #RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            NormalizeIntensityD(keys=keys[0], channel_wise=True),
            ForegroundMaskD(keys[1],threshold=0.5,invert=True),
            #normalize intensity to have mean = 0 and std = 1.
            # SqueezeDimd(keys),
            ToTensorD(keys),
        ])
    def __call__(self,data_dict):
        """Applies transforms to the loaded data in the dictionary.

            Parameters
            ----------
            data_dict: dict
                     Dictionary of images and masks.

            Returns
            -------
            dict
                Dictionary containing data after applying transformations.
        """

        data_dict = self.transform(data_dict)
        return data_dict




class CustomData(Dataset):
    def __init__(self, volume_path, mask_path, keys, size, transform):
        """Initializes and saves all the parameters required for creating transforms and datasets

            Parameters
            ----------
            volume_path: array_like(str)
                 Array containing paths of the volumes.
            mask_path: array_like(str)
                 Array containing paths of the masks.
            keys: keys, optional
                 keys of the corresponding items to be loaded containing two strings
                 the first refers to the key for image and the second refers to the key for label.
            size: array_like
                Array of the wanted volume size
            transform: bool
                True if data needs preprocessing, False otherwise.
        """
        self.volume_path = volume_path
        # self.volume_path = self.volume_path.sort()

        self.mask_path = mask_path
        self.keys = keys
        self.transform = transform
        self.preprocess = Preprocessing(keys, size)

    def __len__(self):
        """Calculates the length of the dataset

            Returns
            -------
            int
                Length of the whole dataset.
        """
        return len(self.volume_path)

    def __getitem__(self, index):
        """Gets the item with the given index from the dataset.

            Parameters
            ----------
            index : int
                index of the required volume and mask

            Returns
            -------
            dict
                Dictionary containing the volume and
                the mask that can be called using their specified keys.
        """

        dict_loader = LoadImageD(keys=self.keys)
        # print(f"volume path: {self.volume_path[index]} mask path: {self.mask_path[index]}")
        data_dict = dict_loader({self.keys[0]: self.volume_path[index],
                                 self.keys[1]: self.mask_path[index]})
        # print(self.volume_path[index],self.mask_path[index])

        if self.transform is True:
            data_dict = self.preprocess(data_dict)
        return data_dict




class DataLoader():
    def __init__(self, dataset_path, batch_size, num_workers=0,
                 pin_memory=False, test_size=0.15, transform=False,
                 keys=("image", "label"), size=[500, 500, 30]):
        """Initializes and saves all the parameters required for creating transforms as well as
           initializing two dataset instances to be used for loading the testing and
           the training data

            Parameters
            ----------
            dataset_path: str
                 String containing paths of the volumes at the folder Path and
                 masks at the folder Path2.
            batch_size: int
                Integer size of batches to be returned
            num_workers : int, optional
                Integer that specifies how many sub-processes to use for data loading
                Set by default to 0.
            pin_memory : bool, optional
                If True, the data loader will copy tensors into CUDA pinned memory
                before returning them.
            test_size : float
                proportion of the test size to the whole dataset. A number between 0 and 1.
            transform: bool
                True if data needs preprocessing, False otherwise.
            keys: keys, optional
                 keys of the corresponding items to be loaded.
                 set by default to ("image","label")
            size: array_like
                Array of the wanted volume size
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        volume_names = os.listdir(os.path.join(dataset_path, "volume"))
        volume_names.sort()
        mask_names = os.listdir(os.path.join(dataset_path, "mask"))
        mask_names.sort()

        volume_paths = []
        mask_paths = []

        for fname in volume_names:
            volume_paths.append(os.path.join(os.path.join(dataset_path, "volume"),fname))
           
        for fname in mask_names:
            mask_paths.append(os.path.join(os.path.join(dataset_path, "mask"),fname))

        volume_paths.sort()
        mask_paths.sort()
        # print(len(mask_paths))

        if test_size == 0:
          test_size = 0.00001

        # training_volume_path, test_volume_path, training_mask_path, test_mask_path = \
        #  train_test_split(volume_paths, mask_paths, test_size=test_size, random_state=42)
        

        training_volume_path=volume_paths
        training_mask_path=mask_paths

        training_volume_path.sort()
        training_mask_path.sort()
        # test_volume_path.sort()
        # test_mask_path.sort()
        
        self.train_ds = CustomData(
        volume_path=training_volume_path,
        mask_path=training_mask_path,
        transform=transform,
        keys=keys,
        size=size
        )
        # self.test_ds = CustomData(
        # volume_path=test_volume_path,
        # mask_path=test_mask_path,
        # transform=transform,
        # keys=keys,
        # size=size
        # )
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
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        )

        return train_loader

    def get_testing_data(self):
        """Loads the testing dataset.

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
