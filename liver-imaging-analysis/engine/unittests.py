import unittest
import numpy as np
import sys
import dataloader
import losses
import json
import os 

with open('config/configs.json') as f:
    config = json.load(f)
dataset_path=config['dataset']['Test']


class TestLoader(unittest.TestCase):
    def setUp(self):
        DataLoader = dataloader.DataLoader(dataset_path, 1, 0, False, 0, True, size=config['transformation_size']) #try removing size
        self.train_dataloader = DataLoader.get_training_data()
        self.data_length = len(self.train_dataloader)
        self.expected_length = 1 #try changing expected length
        for batch in self.train_dataloader:
            self.image_shape= list(batch['image'].shape[2:])
            self.label_shape= list(batch['label'].shape[2:])
            break
        self.expected_shape=config['transformation_size']

    def tearDown(self):
        print("\nTest case for dataloader is completed. Result:")

    def test_length(self):
        message_length = "data did not load properly! number of images and masks is different than expected test dataset"
        self.assertEqual(self.data_length, self.expected_length, msg=message_length)

    def test_shape(self):
        message_shape = "transformation did not occur properly! shape of images and/or masks is different than expected shape"
        self.assertEqual(self.image_shape, self.expected_shape, msg=message_shape)
        self.assertEqual(self.label_shape, self.expected_shape, msg=message_shape)


class TestLoss(unittest.TestCase):
    def setUp(self):
        DataLoader = dataloader.DataLoader(dataset_path, 1, 0, False, 0, False, size=config['transformation_size']) #transformation set to false
        self.train_dataloader = DataLoader.get_training_data()
        for batch in self.train_dataloader:
            self.dice_loss_instance=losses.DiceLoss()
            self.diceloss=self.dice_loss_instance.forward(batch['label'],batch['label']).item()
            break
        self.expected_diceloss=0 #try changing expected diceloss

    def tearDown(self):
        print("\nTest case for losses is completed. Result:")    

    def test_loss(self):
        message_loss="loss wasn't calculated properly! calculated loss is different than expected loss"
        self.assertEqual(self.diceloss, self.expected_diceloss, msg=message_loss)

class TestEngine(unittest.TestCase):
    pass


if __name__ == '__main__':

    unittest.main() 
