import os
import pytest
import torch
from liver_imaging_analysis.engine.dataloader import DataLoader, Keys

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImageD,
    ToTensorD,
)
import monai.data
import nibabel as nib
import numpy as np


def test_data_loader():
    """
    Tests data splitting into train and validation,
    compares loaded batches with reference output.
    """
        batch_size=8
        test_size=0.2
        test_dir= "test/testdata/liverslices"
        image_batches_dir= "test/testdata/batches/image_batches"
        label_batches_dir=  "test/testdata/batches/label_batches"

        transforms =  Compose(
                [   LoadImageD(Keys.all(), allow_missing_keys = True),
                    EnsureChannelFirstD(Keys.all(), allow_missing_keys = True),
                    ToTensorD(Keys.all(), allow_missing_keys = True),
                ])
        trainloader = DataLoader(
            dataset_path =test_dir,
            batch_size = batch_size,
            train_transforms = transforms,
            test_transforms =transforms,
            num_workers = 0,
            pin_memory = False,
            test_size = test_size,
            mode = '2D',
            shuffle = False   
        )
        train_dataloader = trainloader.get_training_data()
        val_dataloader = trainloader.get_testing_data()
        assert isinstance(train_dataloader, monai.data.DataLoader)
        assert isinstance(val_dataloader, monai.data.DataLoader)

        # correct number of batches
        assert len(train_dataloader) == len(os.listdir(image_batches_dir))   

        for i,train_batch in enumerate(train_dataloader):  
            expected_image=np.load( os.path.join(image_batches_dir, f"batch{i}_.npy"))
            expected_label=np.load(os.path.join(label_batches_dir, f"batch{i}_.npy"))

            assert train_batch['image'].shape== expected_image.shape  #correct batch dimension (batch_size,1,length,width)
            assert train_batch['label'].shape== expected_label.shape  
            assert np.all(train_batch['image'].numpy()==expected_image)  
            assert np.all(train_batch['label'].numpy()==expected_label)