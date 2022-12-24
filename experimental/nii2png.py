'''
this script converts 3d volumes to 2d png images to train on 2d architecture like 2DUnet
'''


import SimpleITK as sitk
import os
import natsort
import cv2 as cv



volume_nii_path  = '/Users/mn3n3/Documents/GP/toy_data/volume'  # path to nii files
mask_nii_path    = '/Users/mn3n3/Documents/GP/toy_data/mask'  # path to nii files
volume_save_path = '/Users/mn3n3/Documents/GP/toy_data/Train' # path to generated png images
mask_save_path   = '/Users/mn3n3/Documents/GP/toy_data/Test' # path to generated png images


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
        


        ## name =  "defaultNameWithoutExtention_sliceNum.png"
        volume_png_path = os.path.join(volume_save_path, volume_file_name + "_" + str(slice_number))+'.png'  
        mask_png_path   = os.path.join(mask_save_path, mask_file_name + "_" + str(slice_number))+'.png'  

        cv.imwrite(volume_png_path , volume_silce)
        cv.imwrite(mask_png_path   , mask_silce)