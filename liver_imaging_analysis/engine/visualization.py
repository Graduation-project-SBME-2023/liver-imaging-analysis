"""
a module to perform different visualization functions on tumors
"""

from skimage.measure import find_contours
import cv2 as cv
import matplotlib.pyplot as plt
from monai.transforms import KeepLargestConnectedComponent, EnsureChannelFirst
from utils import visualization
import numpy as np
from torch import subtract
import SimpleITK as sitk

first_tumor = False  # to add plot titles for first tumor only not on every plot


def visualize_tumor(volume, mask, idx, mode):
    """
    initiate the visualization for the chose mood, and whether we will calculate tumor for whole volume or specific slice

    Parameters
    ----------
    volume: nparray
            the nfti volume data
    mask: nparray
            the nfti tumor mask data ( 0: background , 1: tumor)
    idx: int
            the index of slice to calculate all tumors in it. If None, the calculations are done for all slices
    mode: string
            the visualization mood
                contour: draw the major axis and contour around the tumor
                box: draw a bounding box around the tumor
                zoom: zoomin, draw major axis and contours on tumor
    """
    fig, ax = plt.subplots()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if mode == "contour":

        if idx is not None:

            major_axis_recursive(volume[:, :, idx], mask[:, :, idx], mode="slice")

        else:
            major_axis_recursive(volume, mask)

    if mode == "box":
        if idx is not None:

            ax.imshow(volume[:, :, idx], interpolation=None, cmap="gray")

            plot_bbox_image(mask[:, :, idx], crop_margin=0)
        else:
            plot_bbox_image_volume(volume, mask, crop_margin=0)

    if mode == "zoom":
        if idx is not None:
            # ax.imshow(volume[:,:,idx], interpolation=None, cmap=plt.cm.Greys_r)

            plot_tumor(
                volume[:, :, idx], mask[:, :, idx]
            )  # FIXED LATER ( FAILS WHEN MORE THAN 1 TUMOR )

        else:
            plot_tumor_volume(volume, mask)
    fig.show()


def major_axis_recursive(volume, mask, mode="volume"):
    """
    in order to call major_axis independently for each tumor in volume,
    remove all tumors in volume/slice except the largest one and call major_axis function for it,
    then removes it and repeat.

    Parameters
    ----------
    volume: np array
            the nfti volume data
    mask: nparray
            the nfti tumor mask data ( 0: background , 1: tumor)
    mode: string
            whether the input is 3D volume or 2D slice, if 3D volume the code is performed for each
            tumor in the volume and plotted individually in slice with largest volume,
            if slice the code is performed on all tumors in a specific slice
    """

    global first_tumor
    first_tumor = True
    fig, ax = plt.subplots()

    while np.unique(mask).any() == 1:

        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        if mode == "volume":
            idx = visualization.calculate_largest_tumor(largest_tumor[0])

            major_axis(volume[:, :, idx], largest_tumor[0][:, :, idx], ax)

            contours = find_contours(largest_tumor[0][:, :, idx], 0)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")
            plt.show()
            fig, ax = plt.subplots()

        elif mode == "slice":
            major_axis(volume, largest_tumor[0], ax)

            contours = find_contours(mask, 0)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")

        mask = subtract(mask, largest_tumor[0])
        first_tumor = False
    plt.show()


def major_axis(volume_slice, mask_slice, ax):
    """
    calculate major axes length and draw them on the tumors for a given slice

    Parameters
    ----------
    volume_slice: np array
            a specific slice from patient volume
    mask_slice: nparray
            a specific slice from patient tumor mask ( 0: background , 1: tumor)
    ax: matplotlip axis
            the axis to plot the axes on to make all axes on same axis

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
    ) = visualization.find_pix_dim()  # may Be optimized later ( redundant calculation )
    print("Distance along major axis:", (dmax_1 - dmin_1) * x)
    print("Distance along minor axis:", (dmax_2 - dmin_2) * y)

    # display
    # fig, ax = plt.subplots(1,1,figsize=(5,5))
    # ax.imshow(arr, interpolation=None, cmap=plt.cm.Greys_r)
    ax.imshow(volume_slice, cmap="gray")

    if first_tumor:  # FIXED LATER

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
            label="Shortest Diameter",
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
            label="Longest Diameter",
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
    get the vertices of bounding box around mask

    Parameters
    ----------
    mask_slice: np array
            a specific slice from patient tumor mask ( 0: background , 1: tumor)
    crop_maring: int
            the margin of the box
    Returns
    ----------
    tuple:
            returns tuple of xmin,xmax,ymin,ymax which are the vertices of box

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


def plot_bbox_image(mask_slice, crop_margin=0, j=-1):
    """
    plot bounding box around tumors in a specific slice

    Parameters
    ----------
    mask_slice: np array
            a specific slice from patient tumor mask ( 0: background , 1: tumor)
    crop_maring: int
            the margin of the box
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

    colors = visualization.get_colors()
    print("Tumors Number = ", len(dimensions))
    x_dim, y_dim, z_dim = visualization.find_pix_dim()  # again same calculation
    for i in range(len(dimensions)):

        if j != -1:
            n = j
        else:
            n = i
        xmin, ymin, xmax, ymax, pixels = dimensions[i]

        plt.plot([xmin, xmax], [ymin, ymin], color=colors[i], label=f"Lesion {n+1}")
        plt.plot([xmax, xmax], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmin], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmax], [ymax, ymax], color=colors[i])
        plt.legend(fontsize="small")
        print(
            f"Lesion {n+1} Length = {(xmax-xmin)*x_dim}, Width = {(ymax-ymin) * y_dim }, Tumor Volume = {pixels*x_dim*y_dim*z_dim}"
        )

    plt.show()


def plot_bbox_image_volume(volume, mask, crop_margin=0):
    """
    plot bounding box around tumors in a the volume by calculating for each tumor independently

    Parameters
    ----------
    volume: np array
            the volume to calculate all tumors in
    mask: np array
            the tumors mask (0: background, 1:tumor)
    """
    i = 0
    while np.unique(mask).any() == 1:
        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        idx = visualization.calculate_largest_tumor(largest_tumor[0])
        plt.imshow(volume[:, :, idx], cmap="gray")
        plot_bbox_image(largest_tumor[0][:, :, idx], crop_margin, i)
        i = i + 1

        mask = subtract(mask, largest_tumor[0])


def crop_to_bbox(image, bbox, crop_margin=0, pad=40):
    """
    crop the box of the image to zoom in

    Parameters
    ----------
    image: np array
            the image to zoom in it
    bbox: list
            the vertices of bounding box to crop
    crop_margin: int
            the margin of crop
    pad: int
            zoomin padding
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


def plot_tumor(volume_slice, mask_slice):
    """
    draw a box around the tmor, zoomin, draw contours and major axes for tumor in a slice

    Parameters
    ----------
    volume_slice: np array
            specific slice of patient volume
    mask_slice: np array
            tumor mask

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
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0)
    croped_masks = cv.resize(
        croped_masks, dsize=(512, 512), interpolation=cv.INTER_CUBIC
    )

    contours = find_contours(croped_masks, 0.1)
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=0.5, c="r")
    ax[1].imshow(croped_image, cmap="gray")

    major_axis(croped_image, croped_masks, ax[2])

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
    plt.show()


def plot_tumor_volume(volume, mask):
    """
    draw a box around the tmor, zoomin, draw contours and major axes for all tumors in volume

    Parameters
    ----------
    volume: np array
            patient volume
    mask: np array
            tumor mask volume

    """
    global first_tumor
    first_tumor = True
    while np.unique(mask).any() == 1:

        temp_mask = mask.clone()
        temp_mask = EnsureChannelFirst()(temp_mask)
        largest_tumor = KeepLargestConnectedComponent()(temp_mask)
        idx = visualization.calculate_largest_tumor(largest_tumor[0])
        plot_tumor(volume[:, :, idx], largest_tumor[0][:, :, idx])
        mask = subtract(mask, largest_tumor[0])
        first_tumor = False
