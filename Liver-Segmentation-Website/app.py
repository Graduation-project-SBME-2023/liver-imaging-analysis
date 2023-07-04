"""
Flask website app module
"""

import os
import sys
import gc
import torch
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify, make_response
import nibabel as nib
import monai
# import pdfkit

sys.path.append(".")

from liver_imaging_analysis.models.lesion_segmentation import segment_lesion
from liver_imaging_analysis.models.lobe_segmentation import segment_lobe
from liver_imaging_analysis.models.spleen_segmentation import segment_spleen
from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.utils import Overlay, Report, create_image_grid
from visualize_tumors import visualize_tumor, parameters
import json
from monai.transforms import( ToTensor, NormalizeIntensity, Compose
                             ,ScaleIntensityRange)


report_json = {}

# paths
lobes_img_path = "../static/images/lobes.PNG"
segmented_slice_path = "../static/images/liver_slice.png"

plt.switch_backend("Agg")
gc.collect()
torch.cuda.empty_cache()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
save_folder = "Liver-Segmentation-Website/static/img/"

def round_dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            round_dict(v)
        elif isinstance(v, float):
            d[k] = round(v, 2)

    return d

longest_diameter_sum = 0  # initialize
data = 0  # initialize
volume_processing = Compose(
                [

                     ScaleIntensityRange(
                        a_min = -135,
                        a_max = 215,
                        b_min = 0.0,
                        b_max = 1.0,
                        clip = True,
                    )
        
                ]
            )

@app.route("/")
def index():
    """
    main page
    """
    return render_template("index.html")


@app.route("/segment", methods = ["GET", "POST"])
def success():
    """
    Start segmentation , create and display the gifs
    """
    if request.method == "GET":
        global data
        return render_template(
            "segmentation.html", data=data
        )
    elif request.method == "POST":
        file = request.files["file"]
        volume_location = save_folder + "volume.nii"
        mask_location = save_folder + "mask.nii"
        file.save(volume_location)


        # file_list = os.listdir("../static/contour")
        # file_list = os.listdir("../static/contour")
        # file_list = os.listdir("../static/contour")

        # # Loop through the list and delete each file
        # for filename in file_list:
        #     os.remove(filename)


        volume = nib.load(volume_location).get_fdata()
        header = nib.load(volume_location).header
        affine = nib.load(volume_location).affine

        volume = ToTensor()(volume).unsqueeze(dim = 0).unsqueeze(dim = 0)
        volume = volume_processing(volume).squeeze(dim = 0).squeeze(dim = 0)

        liver_lesion = segment_lesion(volume_location)[0][0]
        lobes  = segment_lobe(volume_location)[0][0]
        spleen = segment_spleen(volume_location)[0][0]

        new_nii_volume = nib.Nifti1Image(volume, affine=affine, header=header)
        nib.save(new_nii_volume, volume_location)
        new_nii_mask = nib.Nifti1Image(liver_lesion, affine=affine, header=header)
        nib.save(new_nii_mask, mask_location)

        global report_json
   
        report = Report(volume, mask=liver_lesion, lobes_mask=lobes, spleen_mask=spleen)
        rep = report.build_report()
        report_json = round_dict(rep)

        visualize_tumor(volume_location, liver_lesion, mode='contour')
        visualize_tumor(volume_location, liver_lesion, mode='box')
        visualize_tumor(volume_location, liver_lesion, mode='zoom')

        create_image_grid("Liver-Segmentation-Website/static/contour","Liver-Segmentation-Website/static/images/contour_grid.jpg")

        transform = monai.transforms.Resize((256, 256, 256), mode = "nearest")
        volume = transform(volume[None]).squeeze(0)
        liver_lesion = transform(liver_lesion[None]).squeeze(0)
        lobes = transform(lobes[None]).squeeze(0)

        original_volume = Overlay( volume, torch.zeros(volume.shape), mask2_path = None, alpha = 0.2)
        original_volume.generate_animation("Liver-Segmentation-Website/static/axial/original.gif", 2)
        original_volume.generate_animation("Liver-Segmentation-Website/static/coronal/original.gif", 1)
        original_volume.generate_animation("Liver-Segmentation-Website/static/sagittal/original.gif", 0)

        liver_lesion_overlay = Overlay( volume, liver_lesion ,mask2_path = None, alpha = 0.2)
        liver_lesion_overlay.generate_animation("Liver-Segmentation-Website/static/axial/liver_lesion.gif", 2)
        liver_lesion_overlay.generate_animation("Liver-Segmentation-Website/static/coronal/liver_lesion.gif", 1)
        liver_lesion_overlay.generate_animation("Liver-Segmentation-Website/static/sagittal/liver_lesion.gif", 0)
        liver_lesion_overlay.generate_slice(liver_lesion,"Liver-Segmentation-Website/static/images/liver_slice.png")

        lobes_overlay = Overlay( volume, lobes ,mask2_path = None, alpha = 0.2)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/axial/lobes.gif", 2)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/coronal/lobes.gif", 1)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/sagittal/lobes.gif", 0)
        lobes_overlay.generate_slice(liver_lesion, "Liver-Segmentation-Website/static/images/lobes.PNG")

        global longest_diameter_sum
        for item in parameters:
            max_axis = max(item[0], item[1])
            longest_diameter_sum += max_axis
        longest_diameter_sum = round(longest_diameter_sum,2)
        data = {"Data": parameters, "sum_longest": longest_diameter_sum}
        report_json["Sum Of Longest Diameters"] = longest_diameter_sum

        return render_template(
            "segmentation.html", data=data 
        )


@app.route("/download")
def download_file():
    """
    download the pdf report
    """
    path = "./assets/paper.pdf"
    return send_file(path, as_attachment=True)


@app.route("/views")
def show_views():
    """
    choose from multiple views
    """
    path_overlay = ""
    path_original = ""
    args = request.args
    view = int(args.get("view"))

    if view == 2:
        path_original = "static/axial/original.gif"
        path_overlay = "static/axial/liver_lesion.gif"
        path_lobes = "static/axial/lobes.gif"
    elif view == 1:
        path_original = "static/coronal/original.gif"
        path_overlay = "static/coronal/liver_lesion.gif"
        path_lobes = "static/coronal/lobes.gif"
    else:
        path_original = "static/sagittal/original.gif"
        path_overlay = "static/sagittal/liver_lesion.gif"
        path_lobes = "static/sagittal/lobes.gif"

    return jsonify(path_overlay, path_original, path_lobes)


@app.route("/report", methods=["GET", "POST"])
def report():
    """
    parsing patient data
    """

    if request.method == "POST":
        if ('Lesions Information' in report_json and len(report_json['Lesions Information']) ) > 0:
            flag = True
            tumor_img_path = "../static/zoom/tumor_1.png"
        else:
            flag = False
            tumor_img_path = ""

        id = request.form.get("id")
        age = request.form.get("age")
        phone_number = request.form.get("phone")
        gender = request.form.get("gender")
        patient_data = [
            ["Id", id],
            ["Age", age],
            ["Phone number", phone_number],
            ["Gender", gender],
        ]
        global patient_info
        patient_info = [id,age,phone_number,gender]

        return render_template(
            "report.html",
            patient_data=patient_data,
            my_flag=flag,
            tumor_path=tumor_img_path,
            lobes_path=lobes_img_path,
            rep=report_json,
            liver_slice=segmented_slice_path
        )


    return render_template("form.html")


@app.route('/pdf')
def pdf():
    if len(report_json['Lesions Information'])>0:
        flag = True
        tumor_img_path = "C:/Users/roro1/PycharmProjects/pythonProject5/Liver-Segmentation-Website/static/zoom/tumor_1.png"
    else:
        flag = False
        tumor_img_path = ""
    lobes_img_path_global = "C:/Users/roro1/PycharmProjects/pythonProject5/Liver-Segmentation-Website/static/images/lobes.PNG"
    segmented_slice_path_global = "C:/Users/roro1/PycharmProjects/pythonProject5/Liver-Segmentation-Website/static/images/liver_slice.png"
    rendered = render_template('pdf.html' ,rep = report_json,
                               longest_diam=longest_diameter_sum, my_flag = flag,
                               tumor_path = tumor_img_path, lobes_path = lobes_img_path_global,
                               liver_slice = segmented_slice_path_global,
                               patient_data = patient_info )
    config = pdfkit.configuration(wkhtmltopdf="C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe")
    pdf = pdfkit.from_string(rendered, False, configuration=config,options={"enable-local-file-access": ""})

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=patient report.pdf'
    return response

if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host = "0.0.0.0", port = port, debug = True)