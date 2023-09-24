"""
This module provides various visualization functions for tumors.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from skimage.measure import find_contours
import cv2 as cv
import matplotlib.pyplot as plt
from monai.transforms import KeepLargestConnectedComponent, EnsureChannelFirst
from liver_imaging_analysis.engine.utils import (
    visualization,
    calculate_largest_tumor,
    find_pix_dim,
    get_colors,
)
import numpy as np
from torch import subtract
import SimpleITK as sitk

first_tumor = False  # to add plot titles for first tumor only not on every plot


def visualize_tumor(volume, mask, idx=None, mode="contour", save_path=None):
    """
    Initiate the tumor visualization based on the chosen mode and whether to calculate the tumor
    for the whole volume or a specific slice.

    Parameters
    ----------
    volume : np.ndarray
        The NIfTI volume data.
    mask : np.ndarray
        The NIfTI tumor mask data (0: background, 1: tumor).
    idx : int, optional
        The index of the slice to calculate all tumors in it. If None, the calculations are done for all slices.
    mode : str, optional
        The visualization mode:
            - 'contour': Draws the major axis and contour around the tumor.
            - 'box': Draws a bounding box around the tumor.
            - 'zoom': Zooms in and draws the major axis and contours on the tumor.
    save_path : str, optional
        The path to save the figure(s) in. If None, the figure(s) are not saved.

    """
    fig, ax = plt.subplots()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if mode == "contour":
        if idx is not None:
            major_axis_recursive(
                volume[:, :, idx], mask[:, :, idx], mode="slice", save_path=save_path
            )

        else:
            major_axis_recursive(volume, mask, save_path=save_path)

    if mode == "box":
        if idx is not None:
            ax.imshow(volume[:, :, idx], interpolation=None, cmap="gray")

            plot_bbox_image(mask[:, :, idx], crop_margin=0, save_path=save_path)
        else:
            plot_bbox_image_volume(volume, mask, crop_margin=0, save_path=save_path)

    if mode == "zoom":
        if idx is not None:
            # ax.imshow(volume[:,:,idx], interpolation=None, cmap=plt.cm.Greys_r)

            plot_tumor(volume[:, :, idx], mask[:, :, idx], save_path=save_path)

        else:
            plot_tumor_volume(volume, mask, save_path=save_path)
    fig.show()


def major_axis_recursive(volume, mask, mode="volume", save_path=None):
    """
    Calls the 'major_axis' function independently for each tumor in the volume.


    Parameters
    ----------
    volume : np.ndarray
        The NIfTI volume data.
    mask : np.ndarray
        The NIfTI tumor mask data (0: background, 1: tumor).
    mode : str, optional
        Determines whether the input is a 3D volume or a 2D slice.
        If 'volume', the code is performed for each tumor in the volume,
          and each tumor is plotted individually in the slice with the largest volume.
        If 'slice', the code is performed on all tumors in a specific slice.
    save_path : str, optional
        The path to save the figure(s) in. If None, the figure(s) are not saved.
    """

    global first_tumor
    first_tumor = True
    fig, ax = plt.subplots()
    idx_tumor_counter = {}
    #    It removes all tumors in the volume/slice except the largest one and then calls the 'major_axis' function for it.
    #    This process is repeated until all tumors have been processed.
    # Initialize a counter for each idx

    while np.unique(mask).any() == 1:
        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        if mode == "volume":
            idx = calculate_largest_tumor(largest_tumor[0])

            major_axes(volume[:, :, idx], largest_tumor[0][:, :, idx], ax)
            contours = find_contours(largest_tumor[0][:, :, idx], 0)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")

            if save_path is not None:
                # Increment the counter for each new tumor in the same idx
                if idx in idx_tumor_counter:
                    idx_tumor_counter[idx] += 1
                else:
                    idx_tumor_counter[idx] = 1
                filename = os.path.join(
                    "{0}/slice_{1}_tumor_{2}.png".format(
                        save_path, idx, idx_tumor_counter[idx]
                    )
                )
                plt.savefig(filename)

            plt.show()
            fig, ax = plt.subplots()

        elif mode == "slice":
            major_axes(volume, largest_tumor[0], ax)
            print(mask.shape)
            contours = find_contours(mask.numpy(), 0)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")

            if save_path is not None:
                plt.savefig(os.path.join("{0}/slice.png".format(save_path)))
        mask = subtract(mask, largest_tumor[0])
        first_tumor = False
    plt.show()


def major_axes(volume_slice, mask_slice, ax):
    """
    Calculates the major axes length for each tumor in a given slice and draws them on the tumors.

    Parameters
    ----------
    volume_slice : np.ndarray
        A specific slice from the patient's volume.
    mask_slice : np.ndarray
        A specific slice from the patient's tumor mask. (0: background, 1: tumor)
    ax : matplotlib axis
        The axis to plot the axes on, in order to align all axes.
    """
    img = sitk.GetImageFromArray(mask_slice.astype(int))

    # generate label
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label.SetComputeFeretDiameter(True)
    filter_label.Execute(img)

    # compute the Feret diameter
    # the 1 means we are computing for the label with value 1
    filter_label.GetFeretDiameter(1)
    # we have to get a bit smarter for the principal moments
    pc1_x, pc1_y, pc2_x, pc2_y = filter_label.GetPrincipalAxes(1)

    # get the center of mass
    com_y, com_x = filter_label.GetCentroid(1)

    # now trace the distance from the centroid to the edge along the principal axes
    # we use some linear algebra

    # get the position of each point in the image
    v_x, v_y = np.where(mask_slice)

    # convert these positions to a vector from the centroid
    v_pts = np.array((v_x - com_x, v_y - com_y)).T

    # project along the first principal component
    distances_pc1 = np.dot(v_pts, np.array((pc1_x, pc1_y)))

    # get the extent
    dmax_1 = distances_pc1.max()
    dmin_1 = distances_pc1.min()

    # project along the second principal component
    distances_pc2 = np.dot(v_pts, np.array((pc2_x, pc2_y)))

    # get the extent
    dmax_2 = distances_pc2.max()
    dmin_2 = distances_pc2.min()

    # the total diameter is the difference in these distances
    (
        x,
        y,
        z,
    ) = find_pix_dim()
    print("Distance along major axis:", (dmax_1 - dmin_1) * x)
    print("Distance along minor axis:", (dmax_2 - dmin_2) * y)

    # display
    # fig, ax = plt.subplots(1,1,figsize=(5,5))
    # ax.imshow(arr, interpolation=None, cmap=plt.cm.Greys_r)
    ax.imshow(volume_slice, cmap="gray")

    if first_tumor:
        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99, label="Center")
    else:
        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99)

    ax.plot(
        (com_y, com_y + dmax_1 * pc1_y),
        (com_x, com_x + dmax_1 * pc1_x),
        linestyle="dashed",
        lw=1,
        c="b",
    )

    if first_tumor:
        ax.plot(
            (com_y, com_y + dmin_1 * pc1_y),
            (com_x, com_x + dmin_1 * pc1_x),
            lw=1,
            c="b",
            linestyle="dashed",
            label="Major Axis",
        )
    else:
        ax.plot(
            (com_y, com_y + dmin_1 * pc1_y),
            (com_x, com_x + dmin_1 * pc1_x),
            linestyle="dashed",
            lw=1,
            c="b",
        )
    ax.plot(
        (com_y, com_y + dmax_2 * pc2_y),
        (com_x, com_x + dmax_2 * pc2_x),
        linestyle="dashed",
        lw=1,
        c="g",
    )

    if first_tumor:
        ax.plot(
            (com_y, com_y + dmin_2 * pc2_y),
            (com_x, com_x + dmin_2 * pc2_x),
            lw=1,
            c="g",
            linestyle="dashed",
            label="Minor Axis",
        )
    else:
        ax.plot(
            (com_y, com_y + dmin_2 * pc2_y),
            (com_x, com_x + dmin_2 * pc2_x),
            lw=1,
            linestyle="dashed",
            c="g",
        )

    ax.legend(fontsize="small")


def get_bounding_box(mask_slice, crop_margin=0):
    """
    Retrieves the vertices of the bounding box around the tumor mask.

    Parameters
    ----------
    mask_slice : np.ndarray
        A specific slice from the patient's tumor mask. (0: background, 1: tumor)
    crop_margin : int, optional
        The margin of the bounding box.

    Returns
    -------
    tuple
        A tuple containing the vertices of the bounding box: (xmin, xmax, ymin, ymax).
    """
    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    for row in range(mask_slice.shape[0]):
        if mask_slice[row, :].max() != 0:
            ymin = row + crop_margin
            break

    for row in range(mask_slice.shape[0] - 1, -1, -1):
        if mask_slice[row, :].max() != 0:
            ymax = row + crop_margin
            break

    for col in range(mask_slice.shape[1]):
        if mask_slice[:, col].max() != 0:
            xmin = col + crop_margin
            break

    for col in range(mask_slice.shape[1] - 1, -1, -1):
        if mask_slice[:, col].max() != 0:
            xmax = col + crop_margin
            break
    return xmin, ymin, xmax, ymax


def plot_bbox_image(mask_slice, crop_margin=0, j=-1, save_path=None):
    """
    Plots the bounding box around tumors in a specific slice.

    Parameters
    ----------
    mask_slice : np.ndarray
        A specific slice from the patient's tumor mask. (0: background, 1: tumor)
    crop_margin : int, optional
        The margin of the bounding box.
    j : int, optional
        An additional parameter for the function to track and number each tumor in volume. (Default: -1)
    save_path : str, optional
        The path to save the figure in. If None, the figure is not saved.
    """

    dimensions = []

    while np.unique(mask_slice).any() == 1:
        temp_mask = mask_slice.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        total_pixels = np.unique(largest_tumor[0], return_counts=True)[1][1]

        xmin, ymin, xmax, ymax = get_bounding_box(largest_tumor[0], crop_margin)

        dimensions.append((xmin, ymin, xmax, ymax, total_pixels))

        mask_slice = subtract(mask_slice, largest_tumor[0])

    colors = get_colors()
    print("Tumors Number = ", len(dimensions))
    x_dim, y_dim, z_dim = find_pix_dim()  # again same calculation

    for i in range(len(dimensions)):
        if (
            j != -1
        ):  # if j is not equal to -1, then it's the number of tumor in volume and should be tracked.
            n = j
        else:
            n = i  # if j equals -1 then it's a single tumor
        xmin, ymin, xmax, ymax, pixels = dimensions[i]

        plt.plot([xmin, xmax], [ymin, ymin], color=colors[i], label=f"Lesion {n+1}")
        plt.plot([xmax, xmax], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmin], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmax], [ymax, ymax], color=colors[i])
        plt.legend(fontsize="small")
        print(
            f"Lesion {n+1} Length = {(xmax-xmin)*x_dim}, Width = {(ymax-ymin) * y_dim }, Tumor Volume = {pixels*x_dim*y_dim*z_dim}"
        )

    if save_path is not None :
        if not save_path.endswith(".png"):
            plt.savefig(os.path.join("{0}/slice.png".format(save_path)))
        else:
            plt.savefig(save_path)

    plt.show()


def plot_bbox_image_volume(volume, mask, crop_margin=0, save_path=None):
    """
    Plots the bounding box around tumors in the volume by calculating for each tumor independently.

    Parameters
    ----------
    volume : np.ndarray
        The volume to calculate all tumors in.
    mask : np.ndarray
        The tumor mask. (0: background, 1: tumor)
    crop_margin : int, optional
        The margin of the bounding box.
    save_path : str, optional
        The path to save the figure(s) in. If None, the figure(s) are not saved.
    """
    idx_tumor_counter = {}
    tumor_save_path = None
    i = 0
    while np.unique(mask).any() == 1:
        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        idx = calculate_largest_tumor(largest_tumor[0])
        plt.imshow(volume[:, :, idx], cmap="gray")

        if save_path is not None:
            # Increment the counter for each new tumor in the same idx
            if idx in idx_tumor_counter:
                idx_tumor_counter[idx] += 1
            else:
                idx_tumor_counter[idx] = 1
            tumor_filename = "slice_{0}_tumor_{1}.png".format(idx, idx_tumor_counter[idx])
            tumor_save_path = os.path.join(save_path, tumor_filename)

        plot_bbox_image(largest_tumor[0][:, :, idx], crop_margin, i, tumor_save_path)
        i = i + 1

        mask = subtract(mask, largest_tumor[0])


def crop_to_bbox(image, bbox, crop_margin=0, pad=40):
    """
    Crops the image to the bounding box to zoom in.

    Parameters
    ----------
    image : np.ndarray
        The image to zoom in on.
    bbox : list
        The vertices of the bounding box to crop.
    crop_margin : int, optional
        The margin for cropping. (Default: 0)
    pad : int, optional
        Padding for zooming in. (Default: 40)
    """
    x1, y1, x2, y2 = bbox

    # force a squared image
    max_width_height = np.maximum(y2 - y1, x2 - x1)
    y2 = y1 + max_width_height
    x2 = x1 + max_width_height

    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0) - pad
    y2 = np.minimum(y2 + crop_margin, image.shape[0]) + pad
    x1 = np.maximum(x1 - crop_margin, 0) - pad
    x2 = np.minimum(x2 + crop_margin, image.shape[1]) + pad

    return image[y1:y2, x1:x2]


def plot_tumor(volume_slice, mask_slice, save_path=None):
    """
    Draws a box around the tumor, zooms in, and plots contours and major axes for the tumor in a specific slice.

    Parameters
    ----------
    volume_slice : np.ndarray
        Specific slice of the patient's volume.
    mask_slice : np.ndarray
        Specific slice of the patient's tumor mask (0: background, 1:tumor).
    save_path : str, optional
        The path to save the figure in. If None, the figure is not saved.
    """
    image = np.asarray(volume_slice)
    mask = np.asarray(mask_slice)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # show image
    ax[0].imshow(image, cmap="gray")
    xmin, ymin, xmax, ymax = get_bounding_box(mask, crop_margin=0)
    ax[0].plot([xmin, xmax], [ymin, ymin], color="red")
    ax[0].plot([xmax, xmax], [ymin, ymax], color="red")
    ax[0].plot([xmin, xmin], [ymin, ymax], color="red")
    ax[0].plot([xmin, xmax], [ymax, ymax], color="red")
    ax[0].plot([xmax, 511], [ymax, 511], color="red")
    ax[0].plot([xmax, 511], [ymin, 0], color="red")

    # show image cropped around the tumor
    bbox = get_bounding_box(mask)
    croped_image = crop_to_bbox(image, bbox, crop_margin=0)
    croped_image = cv.resize(
        croped_image, dsize=(512, 512), interpolation=cv.INTER_CUBIC
    )
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0, pad=5)
    
    croped_masks = np.array(croped_masks, dtype='uint8')
    croped_masks = cv.resize(
        croped_masks, dsize=(512, 512), interpolation=cv.INTER_CUBIC
    )

    contours = find_contours(croped_masks, 0.1)
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")
    ax[1].imshow(croped_image, cmap="gray")

    major_axes(croped_image, croped_masks, ax[2])

    for axis in ["top", "bottom", "left", "right"]:
        ax[1].spines[axis].set_color("red")
        ax[2].spines[axis].set_color("red")

        ax[1].spines[axis].set_linewidth(1)
        ax[2].spines[axis].set_linewidth(1)

    ax[1].axes.get_yaxis().set_visible(False)
    ax[1].axes.get_xaxis().set_visible(False)
    ax[2].axes.get_yaxis().set_visible(False)
    ax[2].axes.get_xaxis().set_visible(False)

    plt.subplots_adjust(wspace=0.02)
    if save_path is not None :
        if not save_path.endswith(".png"):
            plt.savefig(os.path.join("{0}/slice.png".format(save_path)))
        else:
            plt.savefig(save_path)
    plt.show()


def plot_tumor_volume(volume, mask, save_path=None):
    """
    Draws a box around tumors, zooms in, and plots contours and major axes for all tumors in the volume.

    Parameters
    ----------
    volume : np.ndarray
        Patient volume.
    mask : np.ndarray
        Tumor mask volume.
    save_path : str, optional
        The path to save the figure(s) in. If None, the figure(s) are not saved.

    """
    idx_tumor_counter = {}
    global first_tumor
    first_tumor = True
    tumor_save_path = None
    while np.unique(mask).any() == 1:
        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        idx = calculate_largest_tumor(largest_tumor[0])

        if save_path is not None:
            # Increment the counter for each new tumor in the same idx
            if idx in idx_tumor_counter:
                idx_tumor_counter[idx] += 1
            else:
                idx_tumor_counter[idx] = 1
            tumor_filename = "slice_{0}_tumor_{1}.png".format(idx, idx_tumor_counter[idx])
            tumor_save_path = os.path.join(save_path, tumor_filename)
        plot_tumor(volume[:, :, idx], largest_tumor[0][:, :, idx] , save_path=tumor_save_path)
        mask = subtract(mask, largest_tumor[0])
        first_tumor = False
