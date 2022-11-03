import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os 
from monai.transforms import LoadImageD
from monai.transforms import \
    LoadImageD, EnsureChannelFirstD, AddChannelD, ScaleIntensityD, ToTensorD, Compose,NormalizeIntensityD, \
    AsDiscreteD, SpacingD, OrientationD, ResizeD, RandSpatialCropd, Spacingd,RandFlipd, RandScaleIntensityd,RandShiftIntensityd, \
    RandSpatialCropd, RandRotated
import monai




roi_size=[500, 500, 30]
KEYS=("image", "label")

class preprocessing():
    def __init__(self, KEYS=("image", "label"), size=[500, 500, 30]):
        r"""A Class that preprocesses data
            __init__
            
            Parameters
            ----------
            KEYS : Dictionary labels for image and mask

            size : 3d array of the wanted volume size 
                The type above can either refer to an actual Python type

            Returns
            -------
            No returns

          """
        self.transform = Compose([
            EnsureChannelFirstD(KEYS),
#             AddChannelD("label"), #assumes label is not rgb - will need to manually implement a class for multiple segments
            OrientationD(KEYS, axcodes='LAS'), #preferred by radiologists
            SpacingD(KEYS, pixdim=(1., 1., 1.), mode=('bilinear', 'nearest')),
            ResizeD(KEYS, size , mode=('trilinear', 'nearest')),
            RandFlipd(KEYS, prob=0.5, spatial_axis=1),
            RandRotated(KEYS, range_x=0.1, range_y=0.1, range_z=0.1, prob=0.5, keep_size=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            NormalizeIntensityD(KEYS, channel_wise=True), #normalize intensity to have mean = 0 and std = 1.
            ToTensorD(KEYS),
        ])
    def __call__(self,data_dict):
        r"""__call__
        
            Parameters
            ----------
            data_dict : Dictionary of paths for images and masks

            Returns
            -------
            
            data_dict : Dictionary
                the dictionary containing data after applying transformations.
           
          """
        data_dict = self.transform(data_dict)
        return data_dict



class CustomData(Dataset):
    def __init__(self, volume_dir, mask_dir, volumeNames, maskNames, KEYS=("image", "label") ,roi_size=[500, 500, 30], 
                 transform = False):
       
        r"""__init__
            
            Parameters
            ----------
            volume_dir : string
                directory for all the volumes.

            volumeNames : Arr 
                Array of names inside the volume directory.
            
            mask_dir : string
                directory for all the masks

            maskNames : string 
                Array of names inside the mask directorymask.
            
            KEYS : Dictionary labels for image and mask.

            roi_size : 3d array of the wanted volume size 
                The type above can either refer to an actual Python type.
                
            transform: boolean
                 Set true if you want to preprocess the data.
                 
            Returns
            -------
            No returns
            
        """
 
        
        self.volume_dir = volume_dir
        self.mask_dir = mask_dir
        self.volumeNames = volumeNames
        self.maskNames=maskNames
        self.keys = KEYS

        #transforms from monai transform lib 
        self.transform = transform
        self.preprocess = preprocessing(KEYS, roi_size)
        
        
            

    def __len__(self):
        r"""__len__
        
            Parameters
            ----------
            no params 
            
            Returns
            -------
            
            data length : int
                Length of the whole dataset
           
          """
        return len(self.volumeNames)

    def __getitem__(self, index):
        
        r"""__getitem__
        
            Parameters
            ----------
            index : int 
                index of the required volume and mask
            Returns
            -------
            
            data_dict : dictionary
                Containing the volume as the image and the mask as the label
           
          """
        
               
        volume_path = os.path.join(self.volume_dir, self.volumeNames[index])
        mask_path = os.path.join(self.mask_dir, self.maskNames[index])
         
        dict_loader = LoadImageD(keys=self.keys)
        data_dict = dict_loader({self.keys[0]: volume_path ,
                                 self.keys[1]: mask_path})
        if self.transform == True:
            data_dict = self.preprocess(data_dict) 
        
        return data_dict




class DataLoader():
    def __init__(self, Paths,dataset_name, batch_size, num_workers=0, pin_memory=False , test_size=0.15, Transform = False,
                 Keys=("image", "label"),size=[500, 500, 30]):
        
        r"""__init__
            
            Parameters
            ----------
            dataset_name: string
                name of the dataset to be loaded eg "MSD" or "MED_seg"
                
            batch_size: int
                size of batches to be returned    
            
            test_size : float
                proportion of the test size 
                
            transform: boolean
                 Set true if you want to preprocess the data.
                 
            volumeNames : Arr 
                Array of names inside the volume directory.
            
            Keys : Dictionary labels for image and mask.

            size : 3d array of the wanted volume size 
                The type above can either refer to an actual Python type.
                
            
            Returns
            -------
            No returns
            
            Other Parameters
            ----------------
             num_workers : int, optional
                set by default to 0
            
             pin_memory : boolean, optional
                set by default to False

          """
        
        print(Paths[dataset_name])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.volume_dir = os.path.join(Paths[dataset_name], "Path")
        self.mask_dir = os.path.join(Paths[dataset_name], "Path2")
       
        self.volumeNames = os.listdir(self.volume_dir)
        self.maskNames=os.listdir(self.mask_dir)
        
        test_size = int(test_size * len(self.volumeNames))
        train_size = len(self.volumeNames)-test_size
        
        self.train_volumeNames, self.test_volumeNames = torch.utils.data.random_split(self.volumeNames, [train_size, test_size])
        self.train_maskNames, self.test_maskNames = torch.utils.data.random_split(self.maskNames, [train_size, test_size])
        
        
        self.train_ds = CustomData(
        volume_dir=self.volume_dir,
        mask_dir=self.mask_dir,
        volumeNames =  self.train_volumeNames, 
        maskNames =  self.train_maskNames,
        transform=Transform,
        KEYS=Keys,
        roi_size = size
        )
        
        self.test_ds = CustomData(
        volume_dir=self.volume_dir,
        mask_dir=self.mask_dir,
        volumeNames =  self.test_volumeNames, 
        maskNames =  self.test_maskNames,
        transform=Transform,
        KEYS=Keys,
        roi_size = size
        )
        
    def get_training_data(self):
        
        r"""get_training_data
            
            Parameters
            ----------
            None
            
            Returns
            -------
            train_loader : data loader dictionary
                Containing the training volumes as the image and the training masks as the label
           
          """
        
        train_loader = monai.data.DataLoader(
        self.train_ds,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        )

        return train_loader
    
    def get_testing_data(self):
        r"""get_testing_data
            
            Parameters
            ----------
            None
            
            Returns
            -------
            test_loader : data loader dictionary
                Containing the testing volumes as the image and the testing masks as the label
           
          """
        
        test_loader = monai.data.DataLoader(
        self.test_ds,
        batch_size=self.batch_size,
        num_workers=self.num_workers,
        pin_memory=self.pin_memory,
        )

        return test_loader
