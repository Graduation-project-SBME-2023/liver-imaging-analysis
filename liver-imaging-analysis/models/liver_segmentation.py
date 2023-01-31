from config import config
from preprocessing import LoadImageLocally
from monai.transforms import (
    LoadImageD,
    ForegroundMaskD,
    EnsureChannelFirstD,
    ToTensorD,
    Compose,
    NormalizeIntensityD,
    OrientationD,
    ResizeD,
    RandFlipd,
    RandRotated,
)
from engine import Engine
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter(config.save["tensorboard"])
class LiverSegmentation(Engine):
    """

    a class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """
    def get_pretraining_transforms(self, transform_name, keys):
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

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),  # preferred by radiologists
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(
                        keys,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        prob=0.5,
                        keep_size=True,
                    ),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest")),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

    def get_pretesting_transforms(self, transform_name, keys):
        """
        Function used to define the needed transforms for the training data

        Args:
             transform_name(string): name of the required set of transforms
             keys(list): keys of the corresponding items to be transformed.

        Return:
            transforms(compose): return the compose of transforms selected

        """

        resize_size = config.transforms["transformation_size"]
        transforms = {
            "3DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    OrientationD(keys, axcodes="LAS"),  # preferred by radiologists
                    ResizeD(keys, resize_size, mode=("trilinear", "nearest")),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "2DUnet_transform": Compose(
                [
                    LoadImageD(keys),
                    EnsureChannelFirstD(keys),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest")),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True),
                    ToTensorD(keys),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]

def per_batch_callback(batch_num,image,label,prediction):
    plot_2d_or_3d_image(
    data=image,
    step=0,
    writer=summary_writer,
    frame_dim=-1,
    tag=f"Batch{batch_num}:Volume",
    )
    plot_2d_or_3d_image(
        data=label,
        step=0,
        writer=summary_writer,
        frame_dim=-1,
        tag=f"Batch{batch_num}:Mask",
    )
    plot_2d_or_3d_image(
        data=prediction,
        step=0,
        writer=summary_writer,
        frame_dim=-1,
        tag=f"Batch{batch_num}:Prediction",
    )

def per_epoch_callback(epoch,training_loss,valid_loss):
    print("\nTraining Loss=", training_loss)
    summary_writer.add_scalar("\nTraining Loss", training_loss, epoch)
    if valid_loss is not None:
        print(f"Validation Loss={valid_loss}")
        summary_writer.add_scalar("Validation Loss", valid_loss, epoch)


def segment_liver(*args):
    """
    a function used to start the training of liver segmentation

    """
    model = LiverSegmentation()
    model.data_status()
    
    model.fit(
      evaluate_epochs=1,
      batch_callback_epochs=1,
      save_weight=True,
      per_batch_callback=per_batch_callback,
      per_epoch_callback=per_epoch_callback
    )
    print("final test loss:", model.test(model.test_dataloader))
