"""
Flask website app module
"""

import os
import sys
sys.path.append(".")
import gc
import torch
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify, make_response
from liver_imaging_analysis.models import liver_segmentation, lesion_segmentation
from liver_imaging_analysis.engine.utils import gray_to_colored, animate
from visualize_tumors import visualize_tumor, parameters
# import pdfkit
plt.switch_backend("Agg")
gc.collect()
torch.cuda.empty_cache()


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
save_folder = "Liver-Segmentation-Website/static/img/"

liver_model = liver_segmentation.LiverSegmentation()
liver_model.load_checkpoint("Liver-Segmentation-Website/models_checkpoints/liver_cp")
lesion_model = lesion_segmentation.LesionSegmentation()


data_arr = [[]]
sum_longest = 0


@app.route("/")
def index():
    """
    main page
    """
    return render_template("index.html")


@app.route("/about")
def about():
    """
    about page , continas some information about the website
    """
    return render_template("about.html")


@app.route("/infer", methods=["POST"])
def success():
    """
    Start segmentation , create and display the gifs
    """
    global save_folder
    if request.method == "POST":
        nifti_file = request.files["file"]
        save_location = nifti_file.filename
        save_location = save_folder + save_location
        nifti_file.save(save_location)
        # volume, prediction = liver_model.predict_with_lesions(
        #     save_location, network_lesions
        # )

        # visualize_tumor(save_location,prediction,mode='contour')
        # visualize_tumor(save_location,prediction,mode='box')
        # visualize_tumor(save_location,prediction,mode='zoom')

        # overlay_3D , original_3D = gray_to_colored(volume,prediction)        
        # animate(original_3D ,"Liver-Segmentation-Website/static/axial/OriginalGif.gif",2)
        # animate(overlay_3D,"Liver-Segmentation-Website/static/axial/OverlayGif.gif",2)
        # animate(original_3D ,"Liver-Segmentation-Website/static/coronal/OriginalGif.gif",1)
        # animate(overlay_3D,"Liver-Segmentation-Website/static/coronal/OverlayGif.gif",1)
        # animate(original_3D ,"Liver-Segmentation-Website/static/sagittal/OriginalGif.gif",0)
        # animate(overlay_3D,"Liver-Segmentation-Website/static/sagittal/OverlayGif.gif",0)

        return render_template(
            "inference.html",
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
        path_original = "static/axial/OriginalGif.gif"
        path_overlay = "static/axial/OverlayGif.gif"
    elif view == 1:
        path_original = "static/coronal/OriginalGif.gif"
        path_overlay = "static/coronal/OverlayGif.gif"
    else:
        path_original = "static/sagittal/OriginalGif.gif"
        path_overlay = "static/sagittal/OverlayGif.gif"

    return jsonify(path_overlay, path_original)


@app.route("/TumorRoute")
def tumor_analysis():
    """
    parsing Tumor analysis
    """
    global data_arr
    global sum_longest
    sum_longest_diameter = 0

    data_arr = []
    for item in parameters:
        max_axis = max(item[0], item[1])
        sum_longest_diameter += max_axis
    # data_arr = parameters
    data_arr = [[1, 1, 2, 3], [2, 4, 5, 6], [3, 7, 8, 9]]
    sum_longest = sum_longest_diameter
    data = {"Data": parameters, "sum_longest": sum_longest}
    return render_template("visualization.html", data=data)


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
            "Longest diameter",
            "Shortest diameter",
            "Volume",
        )
        return render_template(
            "report.html",
            headings=analysis_headings,
            data=data_arr,
            longest_diam=sum_longest,
            patient_data=patient_data,
        )

    return render_template("form.html")


# @app.route('/pdf')
# def pdf():
#     rendered = render_template('pdf.html' ,out_arr=data_arr, longest_diam=sum_longest)
#     config = pdfkit.configuration(wkhtmltopdf="C:/Program Files (x86)/wkhtmltopdf/bin/wkhtmltopdf.exe")
#     pdf = pdfkit.from_string(rendered, False, configuration=config,options={"enable-local-file-access": ""})

#     response = make_response(pdf)
#     response.headers['Content-Type'] = 'application/pdf'
#     response.headers['Content-Disposition'] = 'attachment; filename=patient report.pdf'
#     return response

if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get("PORT", 8001))
    app.run(host="0.0.0.0", port=port, debug=True)
