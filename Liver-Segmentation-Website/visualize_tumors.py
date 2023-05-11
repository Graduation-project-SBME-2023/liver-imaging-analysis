from itertools import permutations 
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
import SimpleITK as sitk
import nibabel as nib
import cv2
from monai.transforms import (
    EnsureChannelFirst,
    AsDiscrete,
    KeepLargestConnectedComponent,
    ToTensor,
)


parameters = []

def visualize_tumor(volume_path,mask_path,mode):
    volume_to_pix_dim = nib.load(volume_path)
    volume = nib.load(volume_path).get_fdata()
    mask = mask_path
    mask= AsDiscrete(threshold=1.5)(mask)
    print(mask.shape)
    print(volume.shape)
    if mode=='contour':
        plot_axis_call(mask,volume,volume_to_pix_dim)    
    if mode=='box':
        plot_bbox_image_call(volume,mask,volume_to_pix_dim,crop_margin=0)
    if mode=='zoom':
        plot_tumor_call(volume,mask,volume_to_pix_dim)

def plot_axis_call(mask,volume,volume_to_pix_dim):

    first_tumor=True
    i = 0
    # global parameters
    arr = []
    while(np.unique(mask).any()==1):
        fig, ax1 = plt.subplots()
        temp_mask=ToTensor()(np.copy(mask)).to('cpu') # change to your device(CUDA/cpu)
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx , max_volume =calculate_largest_tumor(largest_tumor[0],volume_to_pix_dim)
        mask=torch.subtract(mask,largest_tumor[0])
        if (idx == -1):
            continue

        axis1, axis2 = major_axis(largest_tumor[0][:,:,idx],volume[:,:,idx],ax1,first_tumor,volume_to_pix_dim)
        contours = find_contours(largest_tumor[0][:,:,idx],0)

        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=0.5, c='r')
            
        first_tumor=False
        fig.savefig(f'Liver-Segmentation-Website/static/contour/tumor_{i}.png')
        i += 1
        arr.append([axis1,axis2,max_volume])
        parameters[:] = arr
    plt.close('all')        

def plot_bbox_image_call(image, mask,volume_to_pix_dim, crop_margin=0):
    i=0
    while(np.unique(mask).any()==1):
        temp_mask=ToTensor()(np.copy(mask)).to('cpu') # change to your device(CUDA/cpu)
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx , _ =calculate_largest_tumor(largest_tumor[0],volume_to_pix_dim)
        mask=torch.subtract(mask,largest_tumor[0])
        if(idx == -1 ):
            continue
        plt.imshow(image[:,:,idx], cmap='gray')
        plot_bbox_image(largest_tumor[0][:,:,idx],volume_to_pix_dim,crop_margin)            
        i=i+1
        plt.savefig(f'Liver-Segmentation-Website/static/box/tumor_{i}.png')
        plt.close('all')

def plot_tumor_call(volume,mask,volume_to_pix_dim):

    first_tumor=True
    i = 0
    while(np.unique(mask).any()==1): 
        temp_mask=ToTensor()(np.copy(mask)).to('cpu') # change to your device(cuda/cpu)
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        idx , _=calculate_largest_tumor(largest_tumor[0],volume_to_pix_dim)
        mask=torch.subtract(mask,largest_tumor[0])
        if (idx == -1):
            continue
        plot_tumor(volume[:,:,idx],largest_tumor[0][:,:,idx],volume_to_pix_dim,first_tumor,idx,i)
        i += 1

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

def get_colors():
    numbers = [0, 0.5, 1]
    perm = permutations(numbers)
    colors = [color for color in perm]
    return colors
        
def plot_bbox_image(mask,volume_to_pix_dim, crop_margin=0):

    dimensions=[]
    while(np.unique(mask).any()==1):
        temp_mask=ToTensor()(np.copy(mask)).to('cpu') # change to your device(CUDA/cpu)
        temp_mask=EnsureChannelFirst()(temp_mask)
        largest_tumor=KeepLargestConnectedComponent()(temp_mask)
        total_pixels=np.unique(largest_tumor[0],return_counts=True)[1][1]
        xmin, ymin, xmax, ymax = get_bounding_box(largest_tumor[0], crop_margin)
        dimensions.append((xmin,ymin,xmax,ymax,total_pixels))
        mask=torch.subtract(mask,largest_tumor[0])

    colors=get_colors()
    for i in range(len(dimensions)):
        xmin,ymin,xmax,ymax,_ =dimensions[i]
        plt.plot([xmin, xmax], [ymin, ymin], color=colors[i], label=f"Lesion {i+1}")
        plt.plot([xmax, xmax], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmin], [ymin, ymax], color=colors[i])
        plt.plot([xmin, xmax], [ymax, ymax], color=colors[i])
        plt.legend(fontsize='small')
        
    x_dim,y_dim,z_dim=find_pix_dim(volume_to_pix_dim)
    for i in range(len(dimensions)):
        xmin,ymin,xmax,ymax,_ =dimensions[i]
    return dimensions

def calculate_largest_tumor(img_mask_array,volume_to_pix_dim):

    max_volume=-1
    idx=-1
    x,y,z=find_pix_dim(volume_to_pix_dim)
    largest_tumor=KeepLargestConnectedComponent()(img_mask_array)
    for i in range(largest_tumor.shape[2]):
        slice=largest_tumor[:,:,i]
        if(slice.any()==1):
            count=np.unique(slice,return_counts=True)[1][1]
            if(count>max_volume):
                max_volume=count
                idx=i
    max_volume=max_volume*x*y*z
    if(max_volume < 6):
        return -1 , -1
    
    return idx , max_volume
    
def major_axis(arr,volume,ax,first_tumor,volume_to_pix_dim):
    img = sitk.GetImageFromArray(arr.astype(int))


    # generate label 
    filter_label = sitk.LabelShapeStatisticsImageFilter()
    filter_label.SetComputeFeretDiameter(True)
    filter_label.Execute(img)

    # compute the Feret diameter
    # the 1 means we are computing for the label with value 1
    # filter_label.GetFeretDiameter(1)

    # we have to get a bit smarter for the principal moments
    pc1_x, pc1_y, pc2_x, pc2_y = filter_label.GetPrincipalAxes(1)

    # get the center of mass
    com_y, com_x = filter_label.GetCentroid(1)

    # now trace the distance from the centroid to the edge along the principal axes

    # get the position of each point in the image
    v_x, v_y = np.where(arr)

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
    x,y,z=find_pix_dim(volume_to_pix_dim)
    axis_1 = (dmax_1 - dmin_1)*x
    axis_2 = (dmax_2 - dmin_2)*y

    ax.imshow(volume,cmap='gray')

    if(first_tumor):

        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99, label="Center")
    else:
        ax.scatter(com_y, com_x, c="g", marker="o", s=2, zorder=99)
    ax.plot((com_y, com_y+dmax_1*pc1_y), (com_x, com_x+dmax_1*pc1_x),linestyle='dashed', lw=1, c='b')

    if(first_tumor):
        ax.plot((com_y, com_y+dmin_1*pc1_y), (com_x, com_x+dmin_1*pc1_x), lw=1, c='b', linestyle='dashed', label="Major axis")
    else:
        ax.plot((com_y, com_y+dmin_1*pc1_y), (com_x, com_x+dmin_1*pc1_x),linestyle='dashed', lw=1, c='b')

    ax.plot((com_y, com_y+dmax_2*pc2_y), (com_x, com_x+dmax_2*pc2_x),linestyle='dashed', lw=1, c='g')

    if(first_tumor):
        ax.plot((com_y, com_y+dmin_2*pc2_y), (com_x, com_x+dmin_2*pc2_x), lw=1, c='g',linestyle='dashed', label="Minor axis")
    else:
        ax.plot((com_y, com_y+dmin_2*pc2_y), (com_x, com_x+dmin_2*pc2_x), lw=1,linestyle='dashed', c='g')

    ax.legend(fontsize='small')
    return axis_1 , axis_2

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
        
def plot_tumor(images, masks,volume_to_pix_dim,first_tumor, idx,i,save=False):

    image = np.asarray(images)
    # mask = np.asarray(masks.to('cpu')) # DON't change this to CUDA, it must stay as cpu
    mask = np.asarray(masks.cpu()) # DON't change this to CUDA, it must stay as cpu
    fig, ax = plt.subplots(1, 3, figsize=(12,4))

    ax[0].imshow(image, cmap='gray') # show image  (1)
    xmin, ymin, xmax, ymax = get_bounding_box(mask, crop_margin=0) # remember to put this box over the image
    ax[0].plot([xmin, xmax], [ymin, ymin],color='red' )
    ax[0].plot([xmax, xmax], [ymin, ymax], color='red')
    ax[0].plot([xmin, xmin], [ymin, ymax], color='red')
    ax[0].plot([xmin, xmax], [ymax, ymax], color='red')
    ax[0].plot([xmax, 511], [ymax, 511], color='red')
    ax[0].plot([xmax, 511], [ymin, 0], color='red')
    
    bbox = get_bounding_box(mask) # show image cropped around the tumor(2)
    croped_image = crop_to_bbox(image, bbox, crop_margin=0)
    croped_image = cv2.resize(croped_image, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0)
    croped_masks = cv2.resize(croped_masks, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

    contours = find_contours(croped_masks,0.1)
    for contour in contours: ## middle image
        ax[1].plot(contour[:, 1], contour[:, 0], linewidth=0.5, c='r')
    ax[1].imshow(croped_image, cmap='gray')
    major_axis(croped_masks,croped_image,ax[2],first_tumor,volume_to_pix_dim)
    # show only the tumor
    croped_masks = crop_to_bbox(mask, bbox, crop_margin=0)
    croped_masks = cv2.resize(croped_masks, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
    croped_tumor = np.ma.masked_where(croped_masks == False, croped_image)
    croped_tumor_background = np.ma.masked_where(croped_masks == True, np.zeros((512,512)))

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
    fig.savefig(f'Liver-Segmentation-Website/static/zoom/tumor_{i}.png')
    plt.close('all')

def find_pix_dim(volume_to_pix_dim):
    volume=volume_to_pix_dim
    dim = volume.header["dim"] # example [1,512,512,63,1]
    pix_dim = volume.header["pixdim"] # example [1,2,1.5,3,1]

    max_indx = np.argmax(dim)
    pixdimX = pix_dim[max_indx]

    dim = np.delete(dim, max_indx)
    pix_dim = np.delete(pix_dim, max_indx)

    max_indy = np.argmax(dim)
    pixdimY = pix_dim[max_indy]

    dim = np.delete(dim, max_indy)
    pix_dim = np.delete(pix_dim, max_indy)

    max_indZ= np.argmax(dim)
    pixdimZ = pix_dim[max_indZ]

    return [pixdimX, pixdimY,pixdimZ] # example [2, 1.5, 3]