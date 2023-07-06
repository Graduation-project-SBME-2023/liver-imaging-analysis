"""
A module responsible for calculating the patient clinical parameters and generating an automatic patient report.

"""

import json
import openai
import numpy as np
from utils import stringify_dictionary,find_pix_dim,transform_to_hu,mask_average
import visualization

class Report:
    """
    A class used to calculate all the patient vital clinical parameters based on the segmentation masks
    and generate a detailed report for the patient health condition.

    Attributes:
        volume (ndarray): The volume of the patient abdomen scan.
        volume_hu_transformed (ndarray): The volume transformed to Hounsfield units.
        liver_mask (ndarray): The liver mask array. labeled as 0 for background , 1 for liver
        lesions_mask (ndarray): The lesions mask array.  labeled as 0 for background , 1 for lesions
        lobes_mask (ndarray): The lobes mask array.
        spleen_mask (ndarray): The spleen mask array.
        x (float): The X-dimension pixel dimension.
        y (float): The Y-dimension pixel dimension.
        z (float): The Z-dimension pixel dimension.

    Methods:
        liver_analysis(): Performs liver analysis and calculate liver parameters.
        lesions_analysis(): Performs lesions analysis and calculate lesions parameters.
        lobes_analysis(): Performs lobes analysis and calculate lobes parameters.
        spleen_analysis(): Performs spleen analysis and calculate spleen parameters.
        build_report(): Builds the report by calling all analysis functions and saves the report to a JSON file.


    """

    def __init__(self, volume_nfti, mask=None, lobes_mask=None, spleen_mask=None):
        """
        Initializes a Report instance with the given volume and masks.

        Args:
            volume (NFTI object): The nfti object of the patient abdomen scan.
            mask (ndarray, optional): The liver mask array of patient,labeled as 0 for background, 1 for liver, 2 for lesions.
            lobes_mask (ndarray, optional): The lobes mask array.
            spleen_mask (ndarray, optional): The spleen mask array.

        """
        self.volume_nfti=volume_nfti
        self.volume = volume_nfti.get_fdata()
        self.volume_hu_transformed = transform_to_hu(volume_nfti)
        self.liver_lesions_mask=mask
        self.liver_mask = np.where(mask == 1, 1, 0)
        self.lesions_mask = np.where(mask == 2, 1, 0)
        self.lobes_mask = lobes_mask
        self.spleen_mask = spleen_mask
        self.x, self.y, self.z = find_pix_dim(volume_nfti)

    def liver_analysis(self):
        """
        Performs liver analysis by calculating the liver volume in cm3 and attenuation.
        """
        self.liver_volume = (
            np.unique(self.liver_mask, return_counts=True)[1][1]
            * self.x
            * self.y
            * self.z
            / 1000
        )

        self.liver_attenuation = mask_average(
            self.volume_hu_transformed, self.liver_mask
        )

    def lesions_analysis(self):
        """
        Performs lesions analysis by calculating the volume in cm3 and principle axes in mm for each lesion.
        """
        vis=visualization.Lesions_Visualization(self.volume_nfti,self.liver_lesions_mask,idx=None, mode="contour", plot=False)
        self.lesions_calculations = vis.visualize_tumor()

    def lobes_analysis(self):
        """
        Performs lobes analysis by calculating the volume in cm3 and attenuation for each lobe. In addition, LSVR Metric
        which is a measure for the change in the geomtry of the liver.
        """
        values, total_pixels = np.unique(self.lobes_mask, return_counts=True)
        values, total_pixels = values[1:], total_pixels[1:]
        self.lobes_volumes = total_pixels * self.x * self.y * self.z / 1000

        self.lobes_average = [
            mask_average(
                volume=self.volume_hu_transformed,
                mask=np.where(self.lobes_mask == i, 1, 0),
            )
            for i in values
        ]
        self.metric = np.sum(self.lobes_volumes[:3]) / np.sum(self.lobes_volumes[3:])

    def spleen_analysis(self):
        """
        Performs spleen analysis by calculating the spleen volume in cm3 and attenuation.

        """
        self.spleen_volume = (
            np.unique(self.spleen_mask, return_counts=True)[1][1]
            * self.x
            * self.y
            * self.z
            / 1000
        )
        self.spleen_attenuation = mask_average(
            self.volume_hu_transformed, self.spleen_mask
        )

    def build_report(self):
        """
        Builds the report by calling the analysis functions, aggregating various analysis results and saves it to a JSON file.

        Returns:
            dict: The generated report with analysis information.

        """

        report = {}

        if self.spleen_mask is not None:
            self.spleen_analysis()
            report["Spleen Volume"] = self.spleen_volume
            report["Spleen Attenuation"] = self.spleen_attenuation

        if self.liver_mask is not None:
            self.liver_analysis()
            report["Liver Volume"] = self.liver_volume
            if self.spleen_mask is not None:
                report["Liver/Spleen Attenuation Ratio"] = (
                    self.liver_attenuation / self.spleen_attenuation
                )
            else:
                report["Liver Attenuation"] = self.liver_attenuation

        if self.lesions_mask is not None:
            self.lesions_analysis()
            lesions = {}
            for i, calc in enumerate(self.lesions_calculations):
                if calc[0] > calc[1]:

                    lesions[f"{i}"] = {
                        "Major Axis": calc[0],
                        "Minor Axis": calc[1],
                        "Volume": calc[2],
                    }
                else:
                    lesions[f"{i}"] = {
                        "Major Axis": calc[1],
                        "Minor Axis": calc[0],
                        "Volume": calc[2],
                    }
            report["Lesions Information"] = lesions

        if self.lobes_mask is not None:
            self.lobes_analysis()
            lobes_volume = {}
            lobes_attenuation = {}

            for i, volume in enumerate(self.lobes_volumes):
                lobes_volume[f" Lobe {i+1} "] = volume
            if self.spleen_mask is not None:
                for i, attenuation in enumerate(self.lobes_average):
                    lobes_attenuation[f" Lobe {i+1} "] = (
                        attenuation / self.spleen_attenuation
                    )
            else:
                for i, attenuation in enumerate(self.lobes_average):
                    lobes_attenuation[f" Lobe {i+1} "] = attenuation

            report["Each Lobe Volume"] = lobes_volume
            report["Each Lobe/Spleen Attenuation Ratio"] = lobes_attenuation
            report["LSVR Metric"] = self.metric

            msg, tokens = generate(report)
            report["msg"] = msg

        with open("liver_imaging_analysis/resources/report.json", "w") as json_file:
            json.dump(report, json_file)

        return report


def generate(calculations_dict, max_retries=5):
    """
    Creates an AI generated detailed report describing the patient health status based on the calculated clinical parameters

    Args:
        calculations_dict (dict): The dictionary that contains the patient clinical parameters
        max_retries (int): Maximum server retry attempts.

    Return:
        result (string): The Patient AI Generated Report
        tokens (int): The consumed tokens

    """
    calculations = stringify_dictionary(calculations_dict)
    key = "ENTER_YOU_API_KEY"
    openai.api_key = key

    with open("liver_imaging_analysis/resources/prompt.json", "r") as json_file:
        prompt = json.load(json_file)

    instruction = prompt['instruction']
    message = prompt['message']
    message = message.format(calculations=calculations)

    retries = 0
    tokens = 0
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": message},
                ],
                temperature=0.15,
            )

            tokens = response["usage"]["total_tokens"]
            result = response["choices"][0]["message"]["content"]
            return result, tokens
        except:
            if retries >= max_retries:
                return None, tokens
            else:
                retries += 1

# CONFIG PATH
#PATH HERE