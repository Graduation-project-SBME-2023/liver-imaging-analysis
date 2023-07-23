import shutil
import os
import sys
import gc
import torch
import matplotlib.pyplot as plt
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    make_response
    )
import nibabel as nib
import monai
import pdfkit
import json
import subprocess

sys.path.append(".")
plt.switch_backend("Agg")
gc.collect()
torch.cuda.empty_cache()

from liver_imaging_analysis.models.lesion_segmentation import segment_lesion
from liver_imaging_analysis.models.lobe_segmentation import segment_lobe
from liver_imaging_analysis.models.spleen_segmentation import segment_spleen
from liver_imaging_analysis.engine.config import config
from visualize_tumors import visualize_tumor, parameters
from liver_imaging_analysis.engine.utils import (
    Overlay,
    Report,
    create_image_grid,
    round_dict
)
from monai.transforms import (
    ToTensor,
    Compose,
    ScaleIntensityRange
)


volume_processing = Compose(
    [
        ScaleIntensityRange(
            a_min=-135,
            a_max=215,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        )
    ]
)


volume_location = "C:/Users/youse/Downloads/img50.nii"
mask_location = "C:/Users/youse/Downloads/mask50.nii"

volume = nib.load(volume_location).get_fdata()
header = nib.load(volume_location).header
affine = nib.load(volume_location).affine

volume = ToTensor()(volume).unsqueeze(dim=0).unsqueeze(dim=0)
volume = volume_processing(volume).squeeze(dim=0).squeeze(dim=0)

liver_lesion = nib.load(mask_location).get_fdata()
print(liver_lesion.shape)
# liver_lesion = segment_lesion(volume_location)[0][0]
# lobes = segment_lobe(volume_location)[0][0]
# spleen = segment_spleen(volume_location)[0][0]

# new_nii_volume = nib.Nifti1Image(volume, affine=affine, header=header)
# nib.save(new_nii_volume, volume_location)
# new_nii_mask = nib.Nifti1Image(liver_lesion, affine=affine, header=header)
# nib.save(new_nii_mask, mask_location)

# report = Report(volume, mask=liver_lesion, lobes_mask=lobes, spleen_mask=spleen)
# report = report.build_report()
# global report_json
# report_json = round_dict(report)
# print (report_json)

visualize_tumor(volume_location, liver_lesion, mode="contour")
visualize_tumor(volume_location, liver_lesion, mode="box")
visualize_tumor(volume_location, liver_lesion, mode="zoom")