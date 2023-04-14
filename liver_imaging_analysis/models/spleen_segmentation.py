from monai.data import DataLoader, Dataset, decollate_batch

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImage,
    LoadImageD,
    NormalizeIntensityD,
    OrientationD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
    EnsureTyped,
    RandSpatialCropd,
    RandAdjustContrastd,
    RandZoomd,
    Spacingd,
    CropForegroundd,
    Activationsd,
    AsDiscreted,
    KeepLargestConnectedComponent,
    RemoveSmallObjects,
    FillHoles,
    ScaleIntensityRanged,
    Invertd,
    ToTensor,
)
from monai.networks.nets import UNet
import os

from monai.data import Dataset, decollate_batch
import torch
from monai.inferers import sliding_window_inference


class SpleenSegmentation:
    """

    a class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """

    def __init__(self):

        self.network = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="BATCH",
        )
        self.network.load_state_dict(
            torch.load("D:/GP/Spleen/models/model.pt")
        )  # remove 'cpu' and path
        self.get_data("D:/GP/volumes/volume/")


    def get_data(self, data_path):
        volumes_name = os.listdir(data_path)
        volumes_paths = [
            os.path.join(data_path, volume_path) for volume_path in volumes_name
        ]
        volumes_paths.sort()
        self.data = [
            {"image": volume_name} for volume_name in volumes_paths
        ]  
        self.pre_transforms = self.get_pretesting_transforms()
        self.test_data = Dataset(data=self.data, transform=self.pre_transforms)
        self.test_loader = DataLoader(self.test_data, batch_size=1, num_workers=0)

    def get_pretesting_transforms(
        self, transform_name="3DUnet_transform", keys="image"
    ):
        """
        Function used to define the needed transforms for the training data

        Args:
             transform_name: string
             Name of the required set of transforms
             keys: list
             Keys of the corresponding items to be transformed.

        Return:
            transforms: compose
             Return the compose of transforms selected
        """

        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="RAS"),  # preferred by radiologists
                    Spacingd(keys, pixdim=[1.5, 1.5, 2.0], mode=("bilinear")),
                    ScaleIntensityRanged(
                        keys=keys,
                        a_min=-57,
                        a_max=164,
                        b_min=0,
                        b_max=1,
                        clip=True,
                    ),
                    EnsureTyped(keys),
                ]
            ),
        }
        return transforms[transform_name]

    def get_postprocessing_transforms(
        self, transform_name="3DUnet_transform", keys="pred"
    ):
        """
        Function used to define the needed post processing transforms for prediction correction

        Args:
             transform_name(string): name of the required set of transforms
        Return:
            transforms(compose): return the compose of transforms selected

        """
        trans = self.get_pretesting_transforms()
        transforms = {
            "3DUnet_transform": Compose(
                [
                    Invertd(
                        keys="pred",
                        transform=self.pre_transforms,
                        orig_keys="image",
                        meta_keys="pred_meta_dict",
                        orig_meta_keys="image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cuda",
                    ),
                    Activationsd(keys="pred", softmax=True),
                    AsDiscreted(keys="pred", argmax=True, to_onehot=2),
                ]
            )
        }
        return transforms[transform_name]

    def predict(self):
        """
        predicts the liver & lesions mask given the liver mask
        Parameters
        ----------
        data_dir: str
            path of the input directory. expects nifti or png files.
        liver_mask: tensor
            the liver mask predicted by the liver model

        Returns
        -------
        tensor
            tensor of the predicted labels
        """

        post_transforms = self.get_postprocessing_transforms()
        self.network.eval()
        device = "cuda"
        batches = []
        with torch.no_grad():
            for batch in self.test_loader:
                volume = batch["image"].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                batch["pred"] = sliding_window_inference(
                    volume, roi_size, sw_batch_size, self.network, overlap=0.5
                )
                batch = [post_transforms(i) for i in decollate_batch(batch)]
                batches.append(batch)
        return batches


def spleen_segmentation(*args):

    spleenseg = SpleenSegmentation()
    pred_list = spleenseg.predict()

    return [torch.argmax(spleen[0]["pred"], dim=0) for spleen in pred_list]
