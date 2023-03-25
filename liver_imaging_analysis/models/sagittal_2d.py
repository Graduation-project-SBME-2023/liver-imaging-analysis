from config import config
from engine import Engine
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    ForegroundMaskD,
    LoadImageD,
    NormalizeIntensityD,
    OrientationD,
    RandFlipd,
    RandRotated,
    ResizeD,
    ToTensorD,
)
from monai.visualize import plot_2d_or_3d_image
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter(config.save["tensorboard"])
dice=DiceMetric(ignore_empty=False)

class SagittalSegmentation2D(Engine):
    """

    A class that must be used when you want to run the liver segmentation engine,
     contains the transforms required by the user and the function that is used to start training

    """
    def __init__(self):
        config.network_parameters['dropout']= 0
        config.network_parameters['num_res_units']=  2
        config.network_parameters['norm']= "batch"
        config.network_parameters['bias']= 0
        config.training['batch_size']=16
        config.training['optimizer_parameters']['lr']=.001
        config.training['scheduler_parameters']['step_size']=10
        config.training['scheduler_parameters']['verbose']=0
        config.training['scheduler_parameters']['gamma']=0
        super().__init__()

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
                    RandFlipd(keys, prob=0.5, spatial_axis=1),
                    RandRotated(keys,range_x=1.5, range_y=0, range_z=0, prob=0.5),
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
                    LoadImageD(keys, allow_missing_keys=True),
                    EnsureChannelFirstD(keys, allow_missing_keys=True),
                    ResizeD(keys, resize_size, mode=("bilinear", "nearest"), allow_missing_keys=True),
                    # RandFlipd(keys, prob=0.5, spatial_axis=1, allow_missing_keys=True),
                    # RandRotated(keys,range_x=1.5, range_y=0, range_z=0, prob=0.5, allow_missing_keys=True),
                    NormalizeIntensityD(keys=keys[0], channel_wise=True, allow_missing_keys=True),
                    ForegroundMaskD(keys[1], threshold=0.5, invert=True, allow_missing_keys=True),
                    ToTensorD(keys, allow_missing_keys=True),
                ]
            ),
            "custom_transform": Compose(
                [
                    # Add your stack of transforms here
                ]
            ),
        }
        return transforms[transform_name]


def per_batch_callback(batch_num, image, label, prediction):
    score = dice(prediction.bool(),label.bool())[0] #first slice as default in plot_2d_or_3d_image
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
        tag=f"Batch{batch_num}:Prediction:Dice Score={score}",
    )


def per_epoch_callback(epoch, training_loss, valid_loss, training_metric, valid_metric):
    print("\nTraining Loss=", training_loss)
    print("Training Metric=", training_metric)

    summary_writer.add_scalar("Training Loss", training_loss, epoch)
    summary_writer.add_scalar("Training Metric", training_metric, epoch)

    if valid_loss is not None:
        print(f"Validation Loss={valid_loss}")
        print(f"Validation Metric={valid_metric}")

        summary_writer.add_scalar("Validation Loss", valid_loss, epoch)
        summary_writer.add_scalar("Validation Metric", valid_metric, epoch)


def segment_liver(*args):
    """
    a function used to start the training of liver segmentation

    """
    model = SagittalSegmentation2D()
    model.set_seed()
    model.data_status()
    model.load_checkpoint(config.save["potential_checkpoint"])
    print("Initial Test Loss, Test Metric=", model.test(model.test_dataloader,epoch=1, per_batch_callback=per_batch_callback))
    model.fit(
        evaluate_epochs=1,
        batch_callback_epochs=per_batch_callback,
        save_weight=True,
    )
    print("test_loss, test_metric=", model.test(model.test_dataloader, per_batch_callback=per_batch_callback))
    
