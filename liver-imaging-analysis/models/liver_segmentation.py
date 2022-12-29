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


class LoadLITSLiverd(LoadImage):
    def __init__(self, keys) -> None:
        super().__init__()
        self.keys = keys
        # MapTransform.__init__(self, keys, allow_missing_keys=False)


    def __call__(self, data) -> None:
        d = dict(data)
        for key  in list(data.keys()):
        # keys loop
            file_path=d[key]
            if key=="image":
                data_2d_path = 'models/Temp2D/volume/'
            elif key=="label":
                data_2d_path = 'models/Temp2D/mask/'
            slice_id =  data_2d_path + '_'.join(file_path.split('/')[-1:]) #Temp2D/mask/segmentation-0_0.nii
            # if '2D': 
            current_paths=os.listdir(data_2d_path)
            if ((slice_id.split("/")[-1]).split(".")[0]+".png") in current_paths:
                image = super().__call__(slice_id.split(".")[0]+".png")[0]
            else:
                # print((slice_id.split("/")[-1]).split(".")[0]+".png")
                print("writing")
                vol_path = '_'.join(file_path.split('_')[:-1])+'.nii'
                slice_idx = int(file_path.split('_')[-1].split(".")[0])
                vol = super().__call__(vol_path)[0]
                image = vol[..., slice_idx]
                # check storage
                cv.imwrite(slice_id.split(".")[0]+".png" , np.asarray(image))

                
            # elif '3D':
            #     image = super().__call__(file_path)
            
            d[key] = image
        return d


# if '2D':
#     train_volume_paths,train_mask_paths = slices_paths_reader("C:/dataset/volumes.txt",'C:/dataset/masks.txt')
    # test_volume_paths, test_mask_paths = slices_paths_reader("test_volumes.txt",'test_masks.txt')



volume_folders = natsort.natsorted(os.listdir(volume_nii_path)) ## sort the directory of files
mask_folders   = natsort.natsorted(os.listdir(mask_nii_path))


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
                ToTensorD(keys),
            ]
        ),

        '2DUnet_transform': Compose(
            [
                LoadLITSLiverd(keys),
                EnsureChannelFirstD(keys),
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
                    LoadLITSLiverd(keys),
                    EnsureChannelFirstD(keys),
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

model=LiverSegmentation()
model.data_status()
model.fit()

# def segment_liver(*args):
#     liver_segm = LiverSegmentation()
#     liver_segm.fit()



# if __name__ == '__main__':
#     # args
#     segment_liver(args)
