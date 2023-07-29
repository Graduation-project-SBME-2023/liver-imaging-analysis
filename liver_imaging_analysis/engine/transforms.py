"""
a module contains the implemented transformation classes

"""
import os
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull as Hull, Delaunay
import cv2 as cv
import torch
import numpy as np
from liver_imaging_analysis.engine.config import config
from monai.transforms import LoadImage, MapTransform
from monai.transforms.transform import Transform
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


class ConvexHull(Transform):
    """
    This transform applies convex hull to a given binary mask.
    Convex Hull is the smallest convex polygon that contains all 
    the points with a true value in the input mask.  

    See also
    ----------
    .. [1] https://en.wikipedia.org/wiki/Convex_hull
    .. [2] http://www.qhull.org/
    """
    def __init__(self, channels = None):
        '''
        Initialize channels of corresponding mask to apply convex hull to.

        Parameters
        ----------
        channels : list
            Sequence of channels to apply convex hull to. 
            If not provided, convex hull will be applied to all channels.
        '''
        super().__init__()
        self.channels = channels


    def __call__(self, mask):
        """
        Applies convex hull to the input mask.

        Parameters
        ----------
        mask : tensor
            a tensor of the binary mask to apply convex hull to.
            
        Returns
        -------
        tensor
            resultant tensor after applying convex hull to the input mask.
        """
        channels = self.channels if self.channels is not None else range(mask.shape[0])
        device = mask.device
        dtype = mask.dtype
        for channel in channels:
            current_channel = mask.cpu().to(torch.uint8)[channel] # remove channel dim
            points = np.transpose(np.where(current_channel))
            hull = Hull(points)
            deln = Delaunay(points[hull.vertices]) 
            idx = np.stack(np.indices(current_channel.shape), axis = -1)
            out_idx = np.nonzero(deln.find_simplex(idx) + 1)
            out_mask = np.zeros(current_channel.shape)
            out_mask[out_idx] = 1
            mask[channel] = torch.tensor(out_mask).to(device).to(dtype)
        return mask
    

class Dilation(Transform):
    """
    This transform applies dilation to a given binary mask.
    Dilation enlarges the boundaries of regions of foreground pixels.

    See also
    ----------
    .. [1] https://en.wikipedia.org/wiki/Dilation_(morphology)
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology
    """
    def __init__(self, channels = None, iters = 1, struct = None):
        '''
        Initialize the structuring element, number of dilation iterations, 
        and channels of input mask to apply dilation to.

        Parameters
        ----------
        channels : list
            Sequence of channels to apply dilation to. 
            If not provided, dilation will be applied to all channels.
        iters : int, optional
            The dilation is repeated `iterations` times (one, by default). 
            If `iterations` is less than 1, dilation is repeated until the result 
            does not change anymore. Only an integer of iterations is accepted.
        struct : array_like, optional
            Structuring element used for the dilation. Non-zero elements are
            considered True. If no structuring element is provided an element
            is generated with a square connectivity equal to one (i.e., only
            nearest neighbors are connected to the center, diagonally-connected
            elements are not considered neighbors).
        '''
        super().__init__()
        self.iters = iters
        self.struct = struct
        self.channels = channels

    def __call__(self, mask):
        """
        Applies dilation to the input mask.

        Parameters
        ----------
        mask : tensor
            a tensor of the binary mask to apply dilation to.
            
        Returns
        -------
        tensor
            resultant tensor after applying dilation to the input mask.
        """
        channels = self.channels if self.channels is not None else range(mask.shape[0])
        device = mask.device
        dtype = mask.dtype
        for channel in channels:
            current_channel = mask.cpu().to(torch.uint8)[channel] # remove channel dim
            current_channel = binary_dilation(current_channel, iterations = self.iters, structure = self.struct)
            mask[channel] = torch.tensor(current_channel).to(device).to(dtype)
        return mask
    
    
class MorphologicalClosing(Transform):
    '''
    this transform applies morphological closing.
    Morphological closing consists of applying dilation followed by erosion 
    using the same structuring element. Usually used to fill small gaps in binary masks.
    Can be used individually or in a Compose stack of transforms.

    See also
    ----------
    .. [1] https://en.wikipedia.org/wiki/Closing_%28morphology%29
    .. [2] https://en.wikipedia.org/wiki/Mathematical_morphology

    '''
    def __init__(self, channels = None, iters = 1, struct = None):
        '''
        Initialize the structuring element, number of closing iterations, 
        and channels of input mask to apply morphological closing to.

        Parameters
        ----------
        channels : list
            Sequence of channels to apply morphological closing to. 
            If not provided, morphological closing will be applied to all channels.
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
        super().__init__()
        self.iters = iters
        self.struct = struct
        self.channels = channels
        
    def __call__(self, mask):
        """
        Applies morphological closing to the input mask.

        Parameters
        ----------
        mask : tensor
            a tensor of the binary mask to apply closing to.
            
        Returns
        -------
        tensor
            resultant tensor after applying closing to the input mask.
        """
        channels = self.channels if self.channels is not None else range(mask.shape[0])
        device = mask.device
        dtype = mask.dtype
        for channel in channels:
            current_channel = mask.cpu().to(torch.uint8)[channel] # remove channel dim
            current_channel = binary_closing(current_channel, iterations = self.iters, structure = self.struct)
            mask[channel] = torch.tensor(current_channel).to(device).to(dtype)
        return mask
    


class ConvexHulld(MapTransform):
    '''
    A MONAI-like dictionary-based wrapper of :py:class:`ConvexHull`.
    Can be used individually or in a Compose stack of transforms.
    '''
    def __init__(self, keys, channels = None, allow_missing_keys = False):
        '''
        Initialize keys and channels of corresponding masks to apply convex hull to.

        Parameters
        ----------
        keys : tuple
            Keys of the corresponding items to apply convex hull to.
        channels : list
            Sequence of channels to apply convex hull to. 
            If not provided, convex hull will be applied to all channels.
        allow_missing_keys : bool
            don't raise exception if key is missing, deactivated by default.
        '''
        super().__init__(keys, allow_missing_keys)
        self.converter = ConvexHull(channels = channels)

    def __call__(self, data):
        """
        Applies convex hull to the corresponding masks of the object initialized keys.

        Parameters
        ----------
        data : dict
            a dictionary containing the binary mask to apply convex hull to.
            
        Returns
        -------
        dict
            resultant dictionary after applying convex hull to the corresponding masks.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    

class Dilationd(MapTransform):
    '''
    A MONAI-like dictionary-based wrapper of :py:class:`Dilation`.
    Can be used individually or in a Compose stack of transforms.
    '''
    def __init__(self, keys, channels = None, iters = 1, struct = None, allow_missing_keys = False):
        '''
        Initialize the structuring element, number of dilation iterations, 
        and keys of corresponding masks to apply dilation to.

        Parameters
        ----------
        keys : tuple
            Keys of the corresponding items to apply dilation to.
        channels : list
            Sequence of channels to apply dilation to. 
            If not provided, dilation will be applied to all channels.
        iters : int, optional
            The dilation is repeated `iterations` times (one, by default). 
            If `iterations` is less than 1, dilation is repeated until the result 
            does not change anymore. Only an integer of iterations is accepted.
        struct : array_like, optional
            Structuring element used for the dilation. Non-zero elements are
            considered True. If no structuring element is provided an element
            is generated with a square connectivity equal to one (i.e., only
            nearest neighbors are connected to the center, diagonally-connected
            elements are not considered neighbors).
        allow_missing_keys : bool
            don't raise exception if key is missing, deactivated by default.
        '''
        super().__init__(keys, allow_missing_keys)
        self.converter = Dilation(channels = channels, iters = iters, struct = struct)

    def __call__(self, data):
        """
        Applies dilation to the corresponding masks of the object initialized keys.

        Parameters
        ----------
        data : dict
            a dictionary containing the binary mask to apply dilation to.
            
        Returns
        -------
        dict
            resultant dictionary after applying dilation to the corresponding masks.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
    

class MorphologicalClosingd(MapTransform):
    '''
    A MONAI-like dictionary-based wrapper of :py:class:`MorphologicalClosing`.
    Can be used individually or in a Compose stack of transforms.
    '''
    def __init__(self, keys, channels = None, iters = 1, struct = None, allow_missing_keys = False):
        '''
        Initialize the structuring element, number of closing iterations, 
        and keys of corresponding masks to apply morphological closing to.

        Parameters
        ----------
        keys : tuple
            Keys of the corresponding items to apply morphological closing to.
        channels : list
            Sequence of channels to apply morphological closing to. 
            If not provided, morphological closing will be applied to all channels.
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
        allow_missing_keys : bool
            don't raise exception if key is missing, deactivated by default.
        '''
        super().__init__(keys, allow_missing_keys)
        self.converter = MorphologicalClosing(channels = channels, iters = iters, struct = struct)

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
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d
