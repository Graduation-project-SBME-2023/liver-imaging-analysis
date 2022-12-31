from engine.config import config
import monai
import torch
from monai.utils import ensure_tuple_rep
import numpy as np
from engine.preprocessing import LoadLITSLiverd
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


def segment_liver(*args):
    model=LiverSegmentation()
    model.data_status()
    model.fit()
