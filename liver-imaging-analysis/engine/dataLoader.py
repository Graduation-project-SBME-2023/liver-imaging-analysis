import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from monai.transforms import LoadImageD
from monai.transforms import \
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose,NormalizeIntensityD, \
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandSpatialCropd, Spacingd,RandFlipd, RandScaleIntensityd,RandShiftIntensityd, \
    RandSpatialCropd, RandRotated
import monai
# from unittest_loader import TestStringMethods


roi_size=[500, 500, 30]
keys=("image", "label")

class Preprocessing():
    """"
        A Class that preprocesses data
        Methods:
             __init__: creates a preprocessing instance and saves it
                Args:   keys : Dictionary labels for image and mask
                        size : 3d array of the wanted volume size 
                
                
            __call__: applies preprocessing on data dictionary and returns it 
                Args:    data_dict : Dictionary of paths for images and masks
                returns: the dictionary containing data after applying transformations.
    """
    def __init__(self, keys=("image", "label"), size=[500, 500, 30]):
        self.transform = Compose([
            EnsureChannelFirstD(keys),
#             AddChannelD("label"), #assumes label is not rgb - will need to manually implement a class for multiple segments
            OrientationD(keys, axcodes='LAS'), #preferred by radiologists
            SpacingD(keys, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
            ResizeD(keys, size , mode=('trilinear', 'nearest')),
            RandFlipd(keys, prob=0.5, spatial_axis=1),
            RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, keep_size=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            NormalizeIntensityD(keys, channel_wise=True), #normalize intensity to have mean = 0 and std = 1.
            ToTensorD(keys),
        ])
    def __call__(self,data_dict):
        data_dict = self.transform(data_dict)
        return data_dict


class CustomData(Dataset):
    """"
        A Class intends to create dataset and returns instances from it 
        Methods:
             __init__: creates the dataset
                Args:   volume_dir : string directory for all the volumes.
                        volume_names : array of names inside the volume directory.
                        mask_dir : string directory for all the masks
                        mask_names : string array of names inside the mask directorymask.
                        keys : dictionary labels for image and mask.
                        roi_size : 3d array of the wanted volume size the type above can either refer to an actual Python type.
                        transform: boolean to set true if you want to preprocess the data.
                
            __len__: calculates the length of the whole dataset
                returns: data length : int length of the whole dataset
            
            __getitem__: gets an item of the dataset
                Args:   index : int index of the required volume and mask
                returns: data_dict : dictionary containing the volume as the image and the mask as the label
            
    """
    def __init__(self, volume_dir, mask_dir, volume_names, mask_names, keys=("image", "label") ,roi_size=[500, 500, 30], 
                 transform = False):
       
        self.volume_dir = volume_dir
        self.mask_dir = mask_dir
        self.volume_names = volume_names
        
        self.mask_names=mask_names
        self.keys = keys

        #transforms from monai transform lib 
        self.transform = transform
        self.preprocess = Preprocessing(keys, roi_size)
        
        
            

    def __len__(self):
   
        return len(self.volume_names)

    def __getitem__(self, index):
        
        volume_path = os.path.join(self.volume_dir, self.volume_names[index])
        mask_path = os.path.join(self.mask_dir, self.mask_names[index])
         
        dict_loader = LoadImageD(keys=self.keys)
        data_dict = dict_loader({self.keys[0]: volume_path ,
                                 self.keys[1]: mask_path})
        if self.transform == True:
            data_dict = self.preprocess(data_dict) 
        
        return data_dict


class DataLoader():
    """"
        A Class intends the dataset in batches 
        Methods:
             __init__: creates the dataloader instance
                Args:   dataset_name: string name of the dataset to be loaded eg "MSD" or "MED_seg"
                        paths: dictionary with data paths and keys 
                        batch_size: int size of batches to be returned    
                        test_size : float proportion of the test size 
                        transform: boolean set true if you want to preprocess the data.
                        volume_names : array of names inside the volume directory.
                        keys : Dictionary labels for image and mask.
                        size : 3d array of the wanted volume size 
                
            get_training_data: gets the training dataloader
                returns: train_loader : data loader dictionary 
                        containing the training volumes as the image and the training masks as the label
            
            get_testing_data: gets the testing dataloader
                returns: test_loader : data loader dictionary
                        containing the testing volumes as the image and the testing masks as the label
            
    """
    def __init__(self, paths,dataset_name, batch_size, num_workers=0, pin_memory=False , test_size=0.15, transform = False,
                 keys=("image", "label"),size=[500, 500, 30]):
        
        print(paths[dataset_name])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.volume_dir = os.path.join(paths[dataset_name], "Path")
        self.mask_dir = os.path.join(paths[dataset_name], "Path2")
       
        self.volume_names = os.listdir(self.volume_dir)
        self.mask_names=os.listdir(self.mask_dir)
        
        test_size = int(test_size * len(self.volume_names))
        train_size = len(self.volume_names)-test_size 
        
        assert_greater = TestStringMethods()
        assert_greater.test_greater(train_size,test_size)
        
        self.train_volume_names, self.test_volume_names = torch.utils.data.random_split(self.volume_names, [train_size, test_size])
        self.train_mask_names, self.test_mask_names = torch.utils.data.random_split(self.mask_names, [train_size, test_size])
        
        
        self.train_ds = CustomData(
        volume_dir=self.volume_dir,
        mask_dir=self.mask_dir,
        volume_names =  self.train_volume_names, 
        mask_names =  self.train_mask_names,
        transform=transform,
        keys=keys,
        roi_size = size
        )
        
        self.test_ds = CustomData(
        volume_dir=self.volume_dir,
        mask_dir=self.mask_dir,
        volume_names =  self.test_volume_names, 
        mask_names =  self.test_mask_names,
        transform=transform,
        keys=keys,
        roi_size = size
        )
        
    def get_training_data(self):
        
        train_loader = monai.data.DataLoader(
        self.train_ds,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        )

        return train_loader
    
    def get_testing_data(self):
       
        test_loader = monai.data.DataLoader(
        self.test_ds,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        )

        return test_loader
