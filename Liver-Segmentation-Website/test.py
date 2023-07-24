# import shutil
# import os
# import sys
# import gc
# import torch
# import matplotlib.pyplot as plt
# from flask import (
#     Flask,
#     render_template,
#     request,
#     send_file,
#     jsonify,
#     make_response
#     )
# import nibabel as nib
# import monai
# import pdfkit
# import json
# import subprocess

# sys.path.append(".")
# plt.switch_backend("Agg")
# gc.collect()
# torch.cuda.empty_cache()

# from liver_imaging_analysis.models.lesion_segmentation import segment_lesion
# from liver_imaging_analysis.models.lobe_segmentation import segment_lobe
# from liver_imaging_analysis.models.spleen_segmentation import segment_spleen
# from liver_imaging_analysis.engine.config import config
# from visualize_tumors import visualize_tumor, parameters
# from liver_imaging_analysis.engine.utils import (
#     Overlay,
#     Report,
#     create_image_grid,
#     round_dict
# )
# from monai.transforms import (
#     ToTensor,
#     Compose,
#     ScaleIntensityRange
# )


# volume_processing = Compose(
#     [
#         ScaleIntensityRange(
#             a_min=-135,
#             a_max=215,
#             b_min=0.0,
#             b_max=1.0,
#             clip=True,
#         )
#     ]
# )


# volume_location = "C:/Users/youse/Downloads/img50.nii"
# mask_location = "C:/Users/youse/Downloads/mask50.nii"

# volume = nib.load(volume_location).get_fdata()
# header = nib.load(volume_location).header
# affine = nib.load(volume_location).affine

# volume = ToTensor()(volume).unsqueeze(dim=0).unsqueeze(dim=0)
# volume = volume_processing(volume).squeeze(dim=0).squeeze(dim=0)

# liver_lesion = nib.load(mask_location).get_fdata()

# visualize_tumor(volume_location, liver_lesion, mode="contour")
# visualize_tumor(volume_location, liver_lesion, mode="box")
# visualize_tumor(volume_location, liver_lesion, mode="zoom")










import matplotlib.pyplot as plt
import numpy as np
import torch
from visualize_tumors import plot_bbox_image_call # Import the relevant function from your original file
import nibabel as nib



from itertools import permutations 
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
import SimpleITK as sitk
import nibabel as nib
import cv2
from monai.transforms import (
    EnsureChannelFirst,
    AsDiscrete,
    KeepLargestConnectedComponent,
    ToTensor,
)



if __name__ == "__main__":

    volume_to_pix_dim = nib.load("C:/Users/youse/Downloads/img50.nii")
    volume = nib.load("C:/Users/youse/Downloads/img50.nii").get_fdata()
    mask_location = "C:/Users/youse/Downloads/mask50.nii"
    mask = nib.load(mask_location).get_fdata()
    mask= AsDiscrete(threshold=1.5)(mask)

    
    volume = np.rot90(volume)
    mask = np.rot90(mask)
    mask = torch.from_numpy(mask.copy())




    plot_bbox_image_call(volume,mask,volume_to_pix_dim,crop_margin=0)

