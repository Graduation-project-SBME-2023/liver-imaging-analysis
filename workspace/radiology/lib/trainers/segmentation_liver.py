# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss
from monai.optimizers import Novograd
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    CropForegroundd,
    EnsureChannelFirstD,
    EnsureTyped,
    LoadImageD,
    RandCropByPosNegLabeld,
    OrientationD,
    ScaleIntensityRanged,
    ForegroundMaskD,

    SelectItemsd,
    Spacingd,
    ToTensorD,
)

from monailabel.tasks.train.basic_train import BasicTrainTask, Context

logger = logging.getLogger(__name__)


class SegmentationLiver(BasicTrainTask):
    def __init__(
        self,
        model_dir,
        network,
        description="Train Segmentation model for Liver",
        **kwargs,
    ):
        self._network = network
        super().__init__(model_dir, description, **kwargs)

    def network(self, context: Context):
        return self._network

    def optimizer(self, context: Context):
        return torch.optim.Adam(context.network.parameters(), 0.0001)

    def loss_function(self, context: Context):
        return DiceLoss(batch=True, sigmoid=True, jaccard=1)

    def train_pre_transforms(self, context: Context):
        return [
                LoadImageD(keys=("image", "label")),
                EnsureChannelFirstD(keys=("image", "label")),
                OrientationD(keys=("image", "label"), axcodes="LAS"),  # preferred by radiologists
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-135,
                    a_max=215,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ForegroundMaskD(keys="label", threshold=0.5, invert=True),
                RandCropByPosNegLabeld(keys=("image", "label"),image_key="image",label_key="label",spatial_size=(128,128,32),num_samples=12,pos=5,neg=8,allow_smaller=True,image_threshold=0.3),
                ToTensorD(keys=("image", "label")),
        ]

    def train_post_transforms(self, context: Context):
        return [
            ToTensorD(keys=("pred", "label")),
            Activationsd(keys="pred", sigmoid=True),
            AsDiscreted(
                keys=("pred", "label"),
                threshold=0.5
            ),
        ]

    def val_pre_transforms(self, context: Context):
        return [
                LoadImageD(keys=("image", "label")),
                EnsureChannelFirstD(keys=("image", "label")),
                OrientationD(keys=("image", "label"), axcodes="LAS"),  # preferred by radiologists
                ScaleIntensityRanged(
                    keys="image",
                    a_min=-135,
                    a_max=215,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ForegroundMaskD(keys="label", threshold=0.5, invert=True),
                ToTensorD(keys=("image", "label")),
        ]

    def val_inferer(self, context: Context):
        return SlidingWindowInferer(roi_size=(128, 128, 32), sw_batch_size=6, overlap=0.25)
