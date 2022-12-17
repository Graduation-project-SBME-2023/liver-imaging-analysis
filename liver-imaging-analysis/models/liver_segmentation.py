

class LiverSegmentation(Engine):
    

    def get_pretraining_2d():
        # All 2D preprocessiong steps
        return Compose(
            [
                LoadLoadImageD(keys),
                EnsureChannelFirstD(keys),
                # AddChannelD(keys),
                # assumes label is not rgb
                # will need to manually implement a class for multiple segments
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                # SpacingD(keys, pixdim=(1., 1., 1.),
                # mode=('bilinear', 'nearest')),
                # CenterSpatialCropD(keys, size),
                ResizeD(keys, size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                # normalize intensity to have mean = 0 and std = 1.
                # SqueezeDimd(keys),
                ToTensorD(keys),
            ]
        )

    def get_pretraining_3d():
        # All 3D preprocessiong steps
        return Compose(
            [
                LoadLoadImageD(keys),
                EnsureChannelFirstD(keys),
                # AddChannelD(keys),
                # assumes label is not rgb
                # will need to manually implement a class for multiple segments
                OrientationD(keys, axcodes='LAS'), #preferred by radiologists
                # SpacingD(keys, pixdim=(1., 1., 1.),
                # mode=('bilinear', 'nearest')),
                # CenterSpatialCropD(keys, size),
                ResizeD(keys, size , mode=('trilinear', 'nearest')),
                # RandFlipd(keys, prob=0.5, spatial_axis=1),
                # RandRotated(keys, range_x=0.1, range_y=0.1, range_z=0.1,
                # prob=0.5, keep_size=True),
                # RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                # RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                NormalizeIntensityD(keys=keys[0], channel_wise=True),
                ForegroundMaskD(keys[1],threshold=0.5,invert=True),
                # normalize intensity to have mean = 0 and std = 1.
                # SqueezeDimd(keys),
                ToTensorD(keys),
            ]
        )

    def get_pretraining_transforms(self):
        if dim == '2D':
            return get_pretraining_2d()
        elif dim == '3D':
            return get_pretraining_3d()
        
    pass


liver_segm = LiverSegmentation()
liver_segm.fit()