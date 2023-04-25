import torch
import numpy as np
from monai.transforms import Resize
from axial_2d import AxialSegmentation2D
from sagittal_2d import SagittalSegmentation2D
from coronal_2d import CoronalSegmentation2D
from config import config
from utils import nii2png_XYZ
from monai.transforms import (Resize,EnsureChannelFirst, Compose)
import os
import glob
import shutil

class ViewSegmentation():
    def slice(self, volume_nii_path):
        
        self.slices_2d_path = config.dataset["slices_2d_path"]

        if os.path.exists(self.slices_2d_path) == False:
            os.mkdir(self.slices_2d_path)
        
        self.orig_size = nii2png_XYZ("xy",
            volume_nii_path = volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_axial/",
            
        )
        self.orig_size = nii2png_XYZ("xz",
            volume_nii_path = volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_coronal/",
        )
        
        self.orig_size = nii2png_XYZ("yz",
            volume_nii_path = volume_nii_path,
            volume_save_path=f"{self.slices_2d_path}/volume_sagittal/",
        )


    def delete_temp_slices(self):
        shutil.rmtree(self.slices_2d_path)


    def reorder_dims(self, plane, predicted_mask):
        '''
        A method that takes the predicted mask and reorders it to returns it to 
        it's original orientation 
        
        '''
        if plane == "xy":
            predicted_mask=np.transpose(predicted_mask,(1,0,2,3)) # axial reshaping
            predicted_mask = torch.from_numpy(predicted_mask)
            predicted_mask = predicted_mask[0,:,:,:]
            predicted_mask = torch.flip(predicted_mask , [1])
            predicted_mask = torch.rot90(predicted_mask ,dims =  [2,1] )
            
        elif plane == "xz":
            predicted_mask=np.transpose(predicted_mask,(1,2,0,3)) # coronal reshaping
            predicted_mask = torch.from_numpy(predicted_mask[0,:,:,:])
            predicted_mask = torch.rot90(predicted_mask ,dims =  [2,0] ) # assuming zyx , correct
            predicted_mask = torch.flip(predicted_mask , [1])
            predicted_mask = torch.rot90(predicted_mask ,dims =  [1,2] , k =2 ) # assuming zyx , correct

        elif plane == "yz":
            predicted_mask=np.transpose(predicted_mask,(1,2,3,0)) # saggital reshaping
            predicted_mask = torch.from_numpy(predicted_mask[0,:,:,:])
            predicted_mask = torch.rot90(predicted_mask ,dims =  [1,0] ) # assuming zyx , correct
            predicted_mask = torch.flip(predicted_mask , [2])
            predicted_mask = torch.rot90(predicted_mask ,dims =  [2,1] , k = 2)

        return predicted_mask    
    
    def get_segmentation(self):
        sagittal_plane = "yz" 
        sagittal_model=SagittalSegmentation2D()
        sagittal_model.load_checkpoint("/Users/mn3n3/Documents/GP/model_checkpoints/sagittal_checkpoint")
        sagittal_prediction = sagittal_model.predict(f"{self.slices_2d_path}/volume_sagittal")
        sagittal_prediction = self.reorder_dims(sagittal_plane , sagittal_prediction)

        coronal_plane = "xz"
        coronal_model = CoronalSegmentation2D()
        coronal_model.load_checkpoint("/Users/mn3n3/Documents/GP/model_checkpoints/coronal_checkpoint")
        coronal_prediction = coronal_model.predict(f"{self.slices_2d_path}/volume_coronal")
        coronal_prediction = self.reorder_dims(coronal_plane , coronal_prediction)

        axial_plane = "xy"
        axial_model = AxialSegmentation2D()
        axial_model.load_checkpoint("/Users/mn3n3/Documents/GP/model_checkpoints/axial_checkpoint")
        axial_prediction = axial_model.predict(f"{self.slices_2d_path}/volume_axial")
        axial_prediction = self.reorder_dims(axial_plane ,axial_prediction )

        resize_to_original_shape = Compose([
                    EnsureChannelFirst(),
                    Resize(self.orig_size, mode=("nearest-exact")),
                ])

        axial_prediction = resize_to_original_shape(axial_prediction)
        coronal_prediction = resize_to_original_shape(coronal_prediction)
        sagittal_prediction = resize_to_original_shape(sagittal_prediction)

        assert coronal_prediction.shape == sagittal_prediction.shape == axial_prediction.shape

        volume_3D = coronal_prediction[0,:,:,:] + sagittal_prediction[0,:,:,:] + axial_prediction[0,:,:,:]
        volume_3D = volume_3D.cpu()
        volume_3D[volume_3D < 2] = 0
        volume_3D[volume_3D >= 2] = 1

        return volume_3D
    



