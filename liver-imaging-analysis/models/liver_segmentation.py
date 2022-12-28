from engine.config import config
import monai
import torch
from monai.utils import ensure_tuple_rep
import numpy as np
from monai.transforms import (
MapTransform,
LoadImageD,
LoadImage,
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
from typing import Any, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import SimpleITK as sitk
import os
import natsort
import cv2 as cv
from engine.engine import Engine
volume_nii_path  = 'C:/dataset/volume'  # path to nii files
mask_nii_path    = 'C:/dataset/mask'  # path to nii files
volume_save_path = 'C:/dataset/nii2png/volume' # path to generated png images
mask_save_path   = 'C:/dataset/nii2png/mask' # path to generated png images


volume_folders = natsort.natsorted(os.listdir(volume_nii_path)) ## sort the directory of files
mask_folders   = natsort.natsorted(os.listdir(mask_nii_path))
class ShuffleSlices(MapTransform):
    def __init__(self, keys)->None:
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        MapTransform.__init__(self, keys, allow_missing_keys=False)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> None:
        volume=data[list(data.keys())[0]]
        mask= data[list(data.keys())[1]]
        volume=volume.permute(2,0,1)
        mask=mask.permute(2,0,1)
        data[list(data.keys())[0]]=volume
        data[list(data.keys())[1]]=mask
        return data

class nii2png(MapTransform):
    def __init__(self, keys,volume_folders=volume_folders,mask_folders=mask_folders)->None:
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        MapTransform.__init__(self, keys, allow_missing_keys=False)


    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> None:

        d = dict(data)
        for key  in self.key_iterator(d):
        
            # for i in range(len(volume_folders)): 

                # volume_path = os.path.join(volume_nii_path, volume_folders[i])
                # mask_path   = os.path.join(mask_nii_path, mask_folders[i])

                # img_volume = sitk.ReadImage(volume_path)
                # img_mask   = sitk.ReadImage(mask_path)

                img_volume_array = d[key]
                img_volume_array=torch.squeeze(img_volume_array,dim=0)
                # img_mask_array   = d[key][1]
            
                number_of_slices = img_volume_array.shape[2] 

                for slice_number in range(number_of_slices):

                    volume_silce = img_volume_array[:,:,slice_number]
                    # mask_silce   = img_mask_array[slice_number,:,:]

                    volume_file_name = "HI" ## delete extension from filename
                    mask_file_name   = "BYE" ## delete extension from filename
                    


                    ## name =  "defaultNameWithoutExtention_sliceNum.png"
                    volume_png_path = os.path.join(volume_save_path, volume_file_name + "_" + str(slice_number))+'.png'  
                    # mask_png_path   = os.path.join(mask_save_path, mask_file_name + "_" + str(slice_number))+'.png'  

                    cv.imwrite(volume_png_path , np.asarray(volume_silce))
                    # cv.imwrite(mask_png_path   , mask_silce)
        return d

class LiverSegmentation(Engine):
    def get_pretraining_transforms(self,transform_name,keys):
        resize_size = config.resize
        transforms= {
            '3DUnet_transform': Compose(
            [
                LoadImageD(keys),
                EnsureChannelFirstD(keys),
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                ResizeD(keys, resize_size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                ShuffleSlices(keys),
                ToTensorD(keys),
            ]
        ),

        '2DUnet_transform': Compose(
            [
                # LoadImage(image_only=True, ensure_channel_first=True),
                LoadImageD(keys),
                EnsureChannelFirstD(keys),
                nii2png(keys[0]),
                LoadImageD(keys),
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                ResizeD(keys, resize_size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                SqueezeDimd(keys),
                ToTensorD(keys),
                ShuffleSlices(keys),
            ]
        ),

        'custom_transform': Compose(
            [
                #Add your stack of transforms here
            ]
        )
        } 
        return transforms[transform_name]     


    def get_pretesting_transforms(self,transform_name,keys):
            resize_size = config.resize
            transforms= {
                '3DUnet_transform': Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                    ResizeD(keys, resize_size , mode=('trilinear', 'nearest')),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                    ToTensorD(keys),
                ]
            ),
            '2DUnet_transform': Compose(
                [
                    # LoadImage(image_only=True, ensure_channel_first=True),
                    LoadImageD(keys),
                    # ShuffleSlices(keys),
                    EnsureChannelFirstD(keys),
                    # OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                    ResizeD(keys, resize_size , mode=('bilinear', 'nearest')),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                    ToTensorD(keys),
            
                ]
            ),
            'custom_transform': Compose(
                [
                    #Add your stack of transforms here
                ]
            )
            } 
            return transforms[transform_name]     


liver_segm = LiverSegmentation()
liver_segm.fit()