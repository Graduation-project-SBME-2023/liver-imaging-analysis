from skimage.measure import find_contours
import cv2 as cv
import matplotlib.pyplot as plt
from monai.transforms import KeepLargestConnectedComponent,AsDiscrete,EnsureChannelFirst
from utils import find_pix_dim,get_colors,calculate_largest_tumor
import numpy as np
from torch import subtract
import SimpleITK as sitk


def visualize_tumor(volume,mask,idx,mode):
    first_tumor=True # FIXED LATER ( global variable )

    mask= AsDiscrete(threshold=1.5)(mask) # FIXED LATER

    fig, ax = plt.subplots()
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    if mode=='contour':

        if(idx is not None):

            ax.imshow(volume[:,:,idx], interpolation=None, cmap=plt.cm.Greys_r)

            major_axis(volume[:,:,idx],mask[:,:,idx],ax) # FIXED LATER ( NOT WORKING WITH MORE THAN ONE TUMOR IN SLICE )

            contours = find_contours(mask[:,:,idx],0)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c='r')
        else:
            plot_axis_call(volume,mask)
        
    if mode=='box':
        if (idx is not None):

            ax.imshow(volume[:,:,idx], interpolation=None, cmap=plt.cm.Greys_r)

            plot_bbox_image(mask[:,:,idx],crop_margin=0)
        else:
            plot_bbox_image_call(volume,mask,crop_margin=0)

    
    

    if mode=='zoom':
        if ( idx is not None):
            # ax.imshow(volume[:,:,idx], interpolation=None, cmap=plt.cm.Greys_r)

            plot_tumor(volume[:,:,idx],mask[:,:,idx]) # FIXED LATER ( FAILS WHEN MORE THAN 1 TUMOR )

        else:
            plot_tumor_call(volume,mask)
    fig.show()  


def plot_axis_call(volume,mask):

    first_tumor=True # FIXED LATER


    while(np.unique(mask).any()==1):
        fig, ax1 = plt.subplots()

        temp_mask=mask.clone()
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx=calculate_largest_tumor(volume,largest_tumor[0])

        major_axis(volume[:,:,idx],largest_tumor[0][:,:,idx],ax1)
        
        contours = find_contours(largest_tumor[0][:,:,idx],0)
        ax1.imshow(volume[:,:,idx],cmap='gray')
        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c='r')
        plt.show()

        mask=subtract(mask,largest_tumor[0])
        first_tumor=False # fixed LATER


def major_axis(volume,mask,ax):
    img = sitk.GetImageFromArray(mask.astype(int))
    first_tumor=True

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
    v_x, v_y = np.where(mask)

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
    x,y,z=find_pix_dim(volume)
    print("Distance along major axis:", (dmax_1 - dmin_1)*x)
    print("Distance along minor axis:", (dmax_2 - dmin_2)*y)

    # display
    # fig, ax = plt.subplots(1,1,figsize=(5,5))
    # ax.imshow(arr, interpolation=None, cmap=plt.cm.Greys_r)
    ax.imshow(volume,cmap='gray')

    if(first_tumor): # FIXED LATER

        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99, label="Center")
    else:
        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99)

    ax.plot((com_y, com_y+dmax_1*pc1_y), (com_x, com_x+dmax_1*pc1_x),linestyle='dashed', lw=1, c='b')

    if(first_tumor):
        ax.plot((com_y, com_y+dmin_1*pc1_y), (com_x, com_x+dmin_1*pc1_x), lw=1, c='b', linestyle='dashed', label="Shortest Diameter")
    else:
        ax.plot((com_y, com_y+dmin_1*pc1_y), (com_x, com_x+dmin_1*pc1_x),linestyle='dashed', lw=1, c='b')
    ax.plot((com_y, com_y+dmax_2*pc2_y), (com_x, com_x+dmax_2*pc2_x),linestyle='dashed', lw=1, c='g')

    if(first_tumor):
        ax.plot((com_y, com_y+dmin_2*pc2_y), (com_x, com_x+dmin_2*pc2_x), lw=1, c='g',linestyle='dashed', label="Longest Diameter")
    else:
        ax.plot((com_y, com_y+dmin_2*pc2_y), (com_x, com_x+dmin_2*pc2_x), lw=1,linestyle='dashed', c='g')

    ax.legend(fontsize='small')

def get_bounding_box(mask, crop_margin=0):
    """
    Return the bounding box of a mask image.
    slightly modify from https://github.com/guillaumefrd/brain-tumor-mri-dataset/blob/master/data_visualization.ipynb
    
    """
    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    for row in range(mask.shape[0]):
        if mask[row, :].max() != 0:
            ymin = row + crop_margin
            break

    for row in range(mask.shape[0] - 1, -1, -1):
        if mask[row, :].max() != 0:
            ymax = row + crop_margin
            break

    for col in range(mask.shape[1]):
        if mask[:, col].max() != 0:
            xmin = col + crop_margin
            break

    # This code is looping through the columns of a two-dimensional array called "mask" from right to left. The range() function is used to specify the starting column (mask.shape[1] - 1), the ending column (0), and the step size (-1). This loop will iterate through each column of the array from right to left.
    for col in range(mask.shape[1] - 1, -1, -1):
        if mask[:, col].max() != 0:
            xmax = col + crop_margin
            break

    return xmin, ymin, xmax, ymax

def plot_bbox_image(mask, crop_margin=0):

        dimensions=[]

        while(np.unique(mask).any()==1):
            temp_mask=mask.clone()
            temp_mask=EnsureChannelFirst()(temp_mask)
            largest_tumor=KeepLargestConnectedComponent()(temp_mask)
            total_pixels=np.unique(largest_tumor[0],return_counts=True)[1][1]

            xmin, ymin, xmax, ymax = get_bounding_box(largest_tumor[0], crop_margin)

            dimensions.append((xmin,ymin,xmax,ymax,total_pixels))
            

            mask=subtract(mask,largest_tumor[0])

        colors=get_colors()
        print("Tumors Number = ", len(dimensions))
        for i in range(len(dimensions)):
            xmin,ymin,xmax,ymax,pixels=dimensions[i]
            plt.plot([xmin, xmax], [ymin, ymin], color=colors[i], label=f"Lesion {i+1}")
            plt.plot([xmax, xmax], [ymin, ymax], color=colors[i])
            plt.plot([xmin, xmin], [ymin, ymax], color=colors[i])
            plt.plot([xmin, xmax], [ymax, ymax], color=colors[i])
            plt.legend(fontsize='small')
            


        x_dim,y_dim,z_dim=find_pix_dim(mask)
        for i in range(len(dimensions)):
            xmin,ymin,xmax,ymax,pixels=dimensions[i]
            print(f"Lesion {i+1} Length = {(xmax-xmin)*x_dim}, Width = {(ymax-ymin) * y_dim }, Tumor Volume = {pixels*x_dim*y_dim*z_dim}")
        plt.show()
        return dimensions

def plot_bbox_image_call(image, mask, crop_margin=0):
    i=0
    while(np.unique(mask).any()==1):
        temp_mask=mask.clone()
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx=calculate_largest_tumor(image,largest_tumor[0])
        plt.imshow(image[:,:,idx], cmap='gray')
        plot_bbox_image(largest_tumor[0][:,:,idx], crop_margin)            
        i=i+1

        mask=subtract(mask,largest_tumor[0])

def crop_to_bbox(image, bbox, crop_margin=0,pad=40):

    x1, y1, x2, y2 =  bbox

    # force a squared image
    max_width_height = np.maximum(y2 - y1, x2 - x1)
    y2 = y1 + max_width_height
    x2 = x1 + max_width_height

    # in case coordinates are out of image boundaries
    y1 = np.maximum(y1 - crop_margin, 0)-pad
    y2 = np.minimum(y2 + crop_margin, image.shape[0])+pad
    x1 = np.maximum(x1 - crop_margin, 0)-pad
    x2 = np.minimum(x2 + crop_margin, image.shape[1])+pad

    return image[y1:y2, x1:x2]

def plot_tumor(images, masks):


    image = np.asarray(images)
    mask = np.asarray(masks)

    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    # show image
    ax[0].imshow(image, cmap='gray')
    xmin, ymin, xmax, ymax = get_bounding_box(mask, crop_margin=0)
    ax[0].plot([xmin, xmax], [ymin, ymin],color='red' )
    ax[0].plot([xmax, xmax], [ymin, ymax], color='red')
    ax[0].plot([xmin, xmin], [ymin, ymax], color='red')
    ax[0].plot([xmin, xmax], [ymax, ymax], color='red')
    ax[0].plot([xmax, 511], [ymax, 511], color='red')
    ax[0].plot([xmax, 511], [ymin, 0], color='red')

    # show image cropped around the tumor
    bbox = get_bounding_box(mask)
    croped_image = crop_to_bbox(image, bbox, crop_margin=0)
    croped_image = cv.resize(croped_image, dsize=(512,512), interpolation=cv.INTER_CUBIC)
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0)
    croped_masks = cv.resize(croped_masks, dsize=(512,512), interpolation=cv.INTER_CUBIC)


    contours = find_contours(croped_masks,0.1)
    for contour in contours:
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=0.5, c='r')
    ax[1].imshow(croped_image, cmap='gray')

    major_axis(croped_image,croped_masks,ax[2])


    # contours = find_contours(croped_masks,0.5)
    # for contour in contours:
    #     ax[2].plot(contour[:, 1], contour[:, 0], linewidth=0.7, c='r')



    # show only the tumor
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0)
    croped_masks = cv.resize(croped_masks, dsize=(512,512), interpolation=cv.INTER_CUBIC)
    croped_tumor = np.ma.masked_where(croped_masks == False, croped_image)
    croped_tumor_background = np.ma.masked_where(croped_masks == True, np.zeros((512,512)))
    # ax[2].imshow(croped_tumor, cmap='gray')
    # ax[2].imshow(croped_tumor_background, cmap='gray')

    # lighten ticks and labels
    for axis in ['top','bottom','left','right']:
        ax[1].spines[axis].set_color('red')
        ax[2].spines[axis].set_color('red')

        ax[1].spines[axis].set_linewidth(1)
        ax[2].spines[axis].set_linewidth(1)

        ax[1].axes.get_yaxis().set_visible(False)
        ax[1].axes.get_xaxis().set_visible(False)
        ax[2].axes.get_yaxis().set_visible(False)
        ax[2].axes.get_xaxis().set_visible(False)  
        
    plt.subplots_adjust(wspace=0.02)
    plt.show()
def plot_tumor_call(volume,mask):
    first_tumor=True
    while(np.unique(mask).any()==1):
        
        temp_mask=mask.clone()
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx=calculate_largest_tumor(volume,largest_tumor[0])
        plot_tumor(volume[:,:,idx],largest_tumor[0][:,:,idx])
        mask=subtract(mask,largest_tumor[0])