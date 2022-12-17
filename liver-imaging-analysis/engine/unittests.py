#global
import unittest
import numpy as np
import sys
import os 
import torch
import json
#local
import dataloader
import diceloss
import engine
import unet

with open('config/configs.json') as f:
    config = json.load(f)
dataset_path=config['dataset']['unittest']


class TestLoader(unittest.TestCase):
    def setUp(self):
        DataLoader = dataloader.DataLoader(
            dataset_path,
            1,
            0,
            False,
            0,
            True,
            size=config['transformation_size']#try removing size
            ) 
        self.train_dataloader = DataLoader.get_training_data()
        self.data_length = len(self.train_dataloader)
        self.expected_length = 2 #2 volumes in dataset_path
        for batch in self.train_dataloader:
            self.image_shape= list(batch['image'].shape[2:])
            self.label_shape= list(batch['label'].shape[2:])
            break
        self.expected_shape=config['transformation_size']

    def tearDown(self):
        print("\nTest case for dataloader is completed. Result:")

    def test_length(self):
        message_length = "data did not load properly! number of images\
                         and masks is different than expected test dataset"
        self.assertEqual(
            self.data_length,
            self.expected_length,
             msg=message_length
             )

    def test_shape(self):
        message_shape = "transformation did not occur properly! shape of\
                         images and/or masks is different than expected shape"
        self.assertEqual(
            self.image_shape,
            self.expected_shape,
            msg=message_shape
            )
        self.assertEqual(
            self.label_shape,
            self.expected_shape,
            msg=message_shape
            )


class TestLoss(unittest.TestCase):
    def setUp(self):
        DataLoader = dataloader.DataLoader(
            dataset_path,
            1,
            0,
            False,
            0,
            False,
            size=config['transformation_size']#transformation set to false
            )
        self.train_dataloader = DataLoader.get_training_data()
        for batch in self.train_dataloader:
            self.dice_loss_instance=diceloss.DiceLoss()
            self.diceloss=self.dice_loss_instance.forward(
                                batch['label'],
                                batch['label']
                            ).item()
            break
        self.expected_diceloss=0 #try changing expected diceloss

    def tearDown(self):
        print("\nTest case for losses is completed. Result:")    

    def test_loss(self):
        message_loss="loss wasn't calculated properly!\
                    calculated loss is different than expected loss"
        self.assertEqual(
            self.diceloss,
            self.expected_diceloss,
            msg=message_loss
            )



class TestEngine(unittest.TestCase):
    def setUp(self):
        device= "cuda" if torch.cuda.is_available() else "cpu"
        class NeuralNetwork(unet.UNet3D,engine.Engine):
            def __init__(self):
                engine.Engine.__init__(
                    self,
                    device= device,
                    loss=diceloss.DiceLoss(),
                    optimizer= config["training"]['optimizer']['name'],
                    metrics=['dice_score','loss'],
                    training_data_path=dataset_path,
                    testing_data_path=dataset_path,
                    transformation_flag=True,
                    data_size=config['unittest_size'],
                    batchsize=config["unittest_batch"],
                    train_valid_split=0,
                )        
                unet.UNet3D.__init__(self,1,1,device=device)
        self.unittest_model=NeuralNetwork().to(device)

    def tearDown(self):
        print("\nTest case for Engine is completed. Result:")    

    def test_fitting(self):
        self.unittest_model.fit(
            epochs=1,
            evaluation_set=self.unittest_model.test_dataloader,
            evaluate_epochs=1,
            visualize_epochs=1,
            save_flag=True,
            save_path="unittest_weights"
        )

    def test_testing(self):
        first_test=self.unittest_model.test(self.unittest_model.train_dataloader)
        second_test=self.unittest_model.test(self.unittest_model.test_dataloader)
        message_loss="Testing Error, Two tests yielded different losses"
        #both training set and testing set are the same, so loss must be the same!
        self.assertEqual(first_test, second_test, msg=message_loss) 

if __name__ == '__main__':

    unittest.main() 
