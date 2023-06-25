"""
a module contains the implemented preprocessing classes

"""
import os
from scipy.ndimage.morphology import binary_closing
import cv2 as cv
import torch
import numpy as np
from liver_imaging_analysis.engine.config import config
from monai.transforms import LoadImage
from liver_imaging_analysis.engine.dataloader import Keys
from monai.utils import ensure_tuple

class LoadImageLocally(LoadImage):
    """
    a class that takes the path of volume with a specific slice and
     saves it localy if not saved then reads it, if saved it reads it only

    Return:
        d (list): loaded image
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, data) -> None:
        d = dict(data)
        for key in list(data.keys()):
            # keys loop
            input_data_path = d[key]  # ".../training/volume/volume-0_0.nii"
            if key == Keys.IMAGE:
                model_png_path = config.save["volume_png_path"]  # ".../Temp2D/volume/"
            elif key == Keys.LABEL:
                model_png_path = config.save["mask_png_path"]  # ".../Temp2D/volume/"
            slice_id = (
                model_png_path + input_data_path.split("/")[-1]
            )  # "/Temp2D/volume/volume-0_0.nii"
            # if '2D':
            current_model_paths = os.listdir(model_png_path)
            if (
                (slice_id.split("/")[-1]).split(".")[0] + ".png"
            ) in current_model_paths:  # "volume-0_0.png"
                image = super().__call__(slice_id.split(".")[0] + ".png")[
                    0
                ]  # "/Temp2D/volume/volume-0_0.png"
            else:
                vol_path = (
                    "_".join(input_data_path.split("_")[:-1]) + ".nii"
                )  # "training/volume/volume-0.nii"
                slice_idx = int(input_data_path.split("_")[-1].split(".")[0])  # '0'
                vol = super().__call__(vol_path)[0]
                image = vol[..., slice_idx]
                # check storage
                cv.imwrite(
                    slice_id.split(".")[0] + ".png", np.asarray(image)
                )  # /Temp2D/volume/volume-0_0.png

            d[key] = image
        return d


class ClosingD:
    '''
    A MONAI-like dictionary-based transform that applies morphological closing.
    Morphological closing consists of applying dilation followed by erosion 
    using the same structuring element. Usually used to fill small gaps in binary masks.
    Can be used individually or in a Compose stack of transforms.

    See also
    ----------
    .. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    '''
    def __init__(self, keys, iters = 1, struct = None):
        '''
        Initialize the structuring element, number of closing iterations, 
        and keys of corresponding masks to apply morphological closing to.

        Parameters
        ----------
        keys : tuple
            keys of the corresponding items to apply morphological closing to.
        iters : int, optional
            The dilation step of the closing, then the erosion step are each
            repeated `iterations` times (one, by default). If iterations is
            less than 1, each operations is repeated until the result does
            not change anymore. Only an integer of iterations is accepted.
        struct : array_like, optional
            Structuring element used for the closing. Non-zero elements are
            considered True. If no structuring element is provided an element
            is generated with a square connectivity equal to one (i.e., only
            nearest neighbors are connected to the center, diagonally-connected
            elements are not considered neighbors).
        '''
        self.keys = ensure_tuple(keys)
        self.iters = iters
        self.struct = struct

    def closing(self, mask):
        '''
        Applies morphological closing to a pytorch tensor. Expects a channel first shape.

        Parameters
        ----------
        mask : tensor
            binary mask to apply morphological closing to.

        Returns
        -------
        tensor
            resultant mask after applying closing to the input by the structuring element.
        '''
        device = mask.get_device()
        dtype = mask.dtype
        mask = mask.cpu().to(torch.uint8)[0] # remove channel dim
        mask = binary_closing(mask, iterations = self.iters, structure = self.struct)
        mask = torch.tensor(mask).to(device).unsqueeze(dim = 0).to(dtype)
        return mask

    def __call__(self, data):
        """
        Applies closing to the corresponding masks of the object initialized keys.

        Parameters
        ----------
        data : dict
            a dictionary containing the binary mask to apply morphological closing to.
            
        Returns
        -------
        dict
            resultant dictionary after applying closing to the corresponding masks.
        """
        d = dict(data)
        for key in self.keys:
            d[key] = self.closing(d[key])
        return d
