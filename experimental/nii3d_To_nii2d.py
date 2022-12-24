'''
this script converts 3d volumes to 2d nii files to train on 2d architecture like 2DUnet
'''


import SimpleITK as sitk
import os
import natsort
import cv2 as cv
import nibabel as nib
import numpy as np



volume_nii_path  = '/Users/mn3n3/Downloads/experimental/volume/'  # path to nii files
volume_save_path = '/Users/mn3n3/Downloads/experimental/volume2d' # path to generated 2d nii images
mask_nii_path    = '/Users/mn3n3/Downloads/experimental/mask/'  # path to nii files
mask_save_path   = '/Users/mn3n3/Downloads/experimental/mask2d' # path to generated 2d nii images


volume_folders = natsort.natsorted(os.listdir(volume_nii_path)) ## sort the directory of files
mask_folders   = natsort.natsorted(os.listdir(mask_nii_path))

for i in range(len(volume_folders)): 

    volume_path = os.path.join(volume_nii_path, volume_folders[i])
    mask_path   = os.path.join(mask_nii_path, mask_folders[i])

    img_volume = sitk.ReadImage(volume_path)
    img_mask   = sitk.ReadImage(mask_path)

    img_volume_array = sitk.GetArrayFromImage(img_volume)
    img_mask_array   = sitk.GetArrayFromImage(img_mask)

    number_of_slices = img_volume_array.shape[0] 

    for slice_number in range(number_of_slices):

        volume_silce = img_volume_array[slice_number,:,:]
        mask_silce   = img_mask_array[slice_number,:,:]

    
        volume_file_name = os.path.splitext(volume_folders[i])[0] ## delete extension from filename
        mask_file_name   = os.path.splitext(mask_folders[i])[0] ## delete extension from filename
        

        ## nameConvention =  "defaultNameWithoutExtention_sliceNum.nii.gz"
        nii_volume_path = os.path.join(volume_save_path, volume_file_name + "_" + str(slice_number))+'.nii.gz'
        nii_mask_path   = os.path.join(mask_save_path, mask_file_name + "_" + str(slice_number))+'.nii.gz'

        new_nii_volume = nib.Nifti1Image(volume_silce, affine=np.eye(4))  ## ref : https://stackoverflow.com/questions/28330785/creating-a-nifti-file-from-a-numpy-array
        new_nii_mask   = nib.Nifti1Image(mask_silce  , affine=np.eye(4)) 


        nib.save(new_nii_volume , nii_volume_path) 
        nib.save(new_nii_mask   , nii_mask_path) 

