# volume_nii_path  = 'C:/dataset/volume'  # path to nii files
# mask_nii_path    = 'C:/dataset/mask'  # path to nii files
# volume_save_path = 'C:/dataset/nii2png/volume' # path to generated png images
# mask_save_path   = 'C:/dataset/nii2png/mask' # path to generated png images





import os
import numpy as np
# volume_folders = natsort.natsorted(os.listdir(volume_nii_path)) ## sort the directory of files
# mask_folders   = natsort.natsorted(os.listdir(mask_nii_path))
from engine.config import config
from monai.transforms import LoadImage
import cv2 as cv


class LoadLITSLiverd(LoadImage):

    """ a class that takes the path of volume with a specific slice and saves it localy if not saved then reads it, 
    if saved it reads it only



    """
    def __init__(self, keys) -> None:
        super().__init__()


    def __call__(self, data) -> None:
        d = dict(data)
        for key  in list(data.keys()):
        # keys loop
            input_data_path=d[key] # ".../training/volume/volume-0_0.nii"
            if key=="image":
                model_png_path = config.volume_png_path # ".../Temp2D/volume/"
            elif key=="label":
                model_png_path = config.mask_png_path # ".../Temp2D/volume/"
            slice_id =  model_png_path + input_data_path.split('/')[-1] # "/Temp2D/volume/volume-0_0.nii" 
            # if '2D': 
            current_model_paths=os.listdir(model_png_path)
            if ((slice_id.split("/")[-1]).split(".")[0]+".png") in current_model_paths: # "volume-0_0.png"
                image = super().__call__(slice_id.split(".")[0]+".png")[0] # "/Temp2D/volume/volume-0_0.png"
            else:
                vol_path = '_'.join(input_data_path.split('_')[:-1])+'.nii' # "training/volume/volume-0.nii"
                slice_idx = int(input_data_path.split('_')[-1].split(".")[0]) # '0'
                vol = super().__call__(vol_path)[0]
                image = vol[..., slice_idx]
                # check storage
                cv.imwrite(slice_id.split(".")[0]+".png" , np.asarray(image)) # /Temp2D/volume/volume-0_0.png
            
            d[key] = image
        return d