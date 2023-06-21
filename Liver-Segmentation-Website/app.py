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
from liver_imaging_analysis.models import liver_segmentation, lesion_segmentation , lobe_segmentation
from liver_imaging_analysis.engine.config import config
from liver_imaging_analysis.engine.utils import Overlay
from visualize_tumors import visualize_tumor, parameters


plt.switch_backend("Agg")
gc.collect()
torch.cuda.empty_cache()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
save_folder = "Liver-Segmentation-Website/static/img/"

def create_models ():
    """
    Function used to create the segmentation models' objects , each one of them is loaded 
    with its weights

    Returns:
    -------
        LiverSegmentation instance
            
        LesionSegmentation instance
        
        LobeSegmentation instance

    """
    liver_model = liver_segmentation.LiverSegmentation(mode = '3D')
    lesion_model = lesion_segmentation.LesionSegmentation(mode = '3D')
    lobe_model = lobe_segmentation.LobeSegmentation(mode = '3D')

    liver_model.load_checkpoint(config.save["liver_checkpoint"])
    lesion_model.load_checkpoint(config.save["lesion_checkpoint"])
    lobe_model.load_checkpoint(config.save["lobe_checkpoint"])

    return liver_model, lesion_model, lobe_model


def segment_3d(volume_path):
    '''
    Uses the liver, lesion and lobes segmentation models to segment the 3d volume given 

    Args:

    volume_path: str
    path to the 3d volume , expected nii or nii.gz file
    --------
    Returns:
    liver_lesion_prediction : tensor
    3D tensor for the volume with the liver and lesion segmented

    lobe_prediction : tensor
    3D tensor for the volume with the lobes segmented

    '''
    liver_prediction = liver_model.predict(volume_path = volume_path)
    lesion_prediction = lesion_model.predict(
                            volume_path = volume_path,
                            liver_mask = liver_prediction[0].permute(3, 0, 1, 2)
                            )
    lesion_prediction = lesion_prediction * liver_prediction  # no liver -> no lesion
    liver_lesion_prediction = lesion_prediction + liver_prediction  # lesion label is 2

    lobe_prediction = lobe_model.predict(
                            volume_path =volume_path,
                            liver_mask = liver_prediction[0].permute(3,0,1,2)
                            )
    lobe_prediction = lobe_prediction * liver_prediction  # no liver -> no lobe

    return liver_lesion_prediction[0][0], lobe_prediction[0][0]


liver_model, lesion_model, lobe_model  = create_models()
longest_diameter_sum = 0 

@app.route("/")
def index():
    """
    main page
    """
    return render_template("index.html")


@app.route("/infer", methods = ["GET", "POST"])
def success():
    """
    Start segmentation , create and display the gifs
    """
    if request.method == "GET":
        return render_template(
            "segmentation.html",
        )
    elif request.method == "POST":
        file = request.files["file"]
        volume_filename = file.filename
        volume_location = save_folder + volume_filename
        file.save(volume_location)

        volume = nib.load(volume_location).get_fdata()
        liver_lesion , lobes  = segment_3d(volume_location)

        visualize_tumor(volume_location, liver_lesion, mode='contour')
        visualize_tumor(volume_location, liver_lesion, mode='box')
        visualize_tumor(volume_location, liver_lesion, mode='zoom')

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

        lobes_overlay = Overlay( volume, lobes ,mask2_path = None, alpha = 0.2)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/axial/lobes.gif", 2)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/coronal/lobes.gif", 1)
        lobes_overlay.generate_animation("Liver-Segmentation-Website/static/sagittal/lobes.gif", 0)

        return render_template(
            "segmentation.html",
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


@app.route("/TumorRoute")
def tumor_analysis():
    """
    parsing Tumor analysis
    """

    global longest_diameter_sum
    longest_diameter_sum = 0

    for item in parameters:
        max_axis = max(item[0], item[1])
        longest_diameter_sum += max_axis

    data = {"Data": parameters, "sum_longest": longest_diameter_sum}
    return render_template("visualization.html", data = data)


@app.route("/report", methods=["GET", "POST"])
def report():
    """
    parsing patient data
    """

    if request.method == "POST":
        name = request.form.get("name")
        age = request.form.get("age")
        phone_number = request.form.get("phone")
        gender = request.form.get("gender")
        patient_data = [
            ["Name", name],
            ["Age", age],
            ["Phone number", phone_number],
            ["Gender", gender],
        ]

        analysis_headings = (
            "Lesion",
            "axis_1",
            "axis_2",
            "Volume",
        )
        return render_template(
            "report.html",
            headings = analysis_headings,
            data = parameters,
            longest_diam = '%.3f'% longest_diameter_sum,
            patient_data = patient_data,
        )

    return render_template("form.html")


# @app.route('/pdf')
# def pdf():
#     rendered = render_template('pdf.html' ,out_arr=parameters, longest_diam=longest_diameter_sum)
#     config = pdfkit.configuration(wkhtmltopdf="C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe")
#     pdf = pdfkit.from_string(rendered, False, configuration=config,options={"enable-local-file-access": ""})

#     response = make_response(pdf)
#     response.headers['Content-Type'] = 'application/pdf'
#     response.headers['Content-Disposition'] = 'attachment; filename=patient report.pdf'
#     return response

if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host = "0.0.0.0", port = port, debug = True)
