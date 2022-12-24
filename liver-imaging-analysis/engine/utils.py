import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import animation, rc
from matplotlib.animation import PillowWriter
rc('animation', html='html5')


""""
    A module that display the original CT volume and the segmented mask and its labels
    with different color and opacity on it and saves the overlay into a gif file
    Methods:
        gray_to_colored: changes the input nfti from 3 channels gray scale ( L , W , N) to 4 channels RGB ( L , W , N , 3)
        by stacking the volume array and perform weighted add to put the segmented mask over the volume in one array
            Args:   Volume Path: the directory that includes the volume nii file
                    Mask Path: the directory that includes the segmented mask nii file
                    alpha: the opacity of the displayed mask
            Return: The Stacked 4 channels array of the nfti input
        normalize: normalize the input value to be in range 0:255
        animate: create the animated overlay and saves it as GIF
            Args: volume: the required array input to be animated
                  volumename: the name of the output gif file to be saved
"""


def gray_to_colored (VolumePath,MaskPath,alpha=0.2):
    def normalize(arr):
        return (255*(arr - np.min(arr)) / (np.max(arr) - np.min(arr)))

    Volume = nib.load(VolumePath).get_fdata()
    Mask = nib.load(MaskPath).get_fdata()
    Masklabel=[]
    masksNo=np.unique(Mask)[1:]
    dest=np.stack((normalize(Volume).astype(np.uint8),)*3,axis=-1) # stacked array of volume

    if masksNo.shape[0]<7:  # a loop to generate an array of unique rgb colors to be used for each label 
        numbers=[0,1]
    else:
        numbers=[0,0.5,1]
    colors=[]
    for i in numbers:
        for j in numbers:
            for k in numbers:
                if(i==j==k):
                    continue
                colors.append([i,j,k])
                
    colors= np.asarray((colors))
    for i,label in enumerate(masksNo):     # a loop to iterate over each label in the mask and perform weighted add for each
                                            # label with a unique color for each one
        Masklabel.append(Mask==label)
        Masklabel[i]=np.stack((Masklabel[i],)*3,axis=-1)
        Masklabel[i]=np.multiply((Masklabel[i].astype(np.uint8)*255),colors[i]).astype(np.uint8)
        dest = cv.addWeighted(dest, alpha, Masklabel[i],alpha, 0.0)
    return dest              # return an array of the volume with the mask overlayed on it with different label colors



def animate(volume,outputName):
    fig = plt.figure()
    ims = []
    for i in range(volume.shape[2]):      # generate an animation over the slices of the array 
        plt.axis('off')
        im = plt.imshow(volume[:,:,i],animated=True);
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True,
                                    repeat_delay=100)
    ani.save(outputName, dpi=300, writer=PillowWriter(fps=5))





def gray_to_colored_from_array (Volume,Mask,mask2=None,alpha=0.2):
    def normalize(arr):
        return (255*(arr - np.min(arr)) / (np.max(arr) - np.min(arr)))

    Masklabel=[]
    masksNo=np.unique(Mask)[1:]
    if mask2 != None:
        mask_label2=[]
        masks_number2=np.unique(mask2)[1:0]
    dest=np.stack((normalize(Volume).astype(np.uint8),)*3,axis=-1) # stacked array of volume

    if masksNo.shape[0]<7:  # a loop to generate an array of unique rgb colors to be used for each label 
        numbers=[0,1]
    else:
        numbers=[0,0.5,1]
    colors=[]
    for i in numbers:
        for j in numbers:
            for k in numbers:
                if(i==j==k):
                    continue
                colors.append([i,j,k])
                
    colors= np.asarray((colors))

    for i,label in enumerate(masksNo):     # a loop to iterate over each label in the mask and perform weighted add for each
                                            # label with a unique color for each one
        Masklabel.append(Mask==label)
        Masklabel[i]=np.stack((Masklabel[i],)*3,axis=-1)
        Masklabel[i]=np.multiply((Masklabel[i].astype(np.uint8)*255),colors[i]).astype(np.uint8)
        dest = cv.addWeighted(dest, 1, Masklabel[i],alpha, 0.0)
    if mask2 != None: 
        colors = np.flip(colors)
        for i,label in enumerate(masks_number2):     # a loop to iterate over each label in the mask and perform weighted add for each
                                                    # label with a unique color for each one
            mask_label2.append(mask2==label)
            mask_label2[i]=np.stack((mask_label2[i],)*3,axis=-1)
            mask_label2[i]=np.multiply((mask_label2[i].astype(np.uint8)*255),colors[i]).astype(np.uint8)
            dest = cv.addWeighted(dest, 1, mask_label2[i],alpha, 0.0)


    return dest              # return an array of the volume with the mask overlayed on it with different label colors





# volume=gray_to_colored('C:/dataset/Path/liver-orig002.nii','C:/dataset/Path2/liver-seg002.nii')
# animate(volume,'Vol_Mask_Overlay.gif')


def progress_bar (progress, total):
    percent = 100 * (progress / float (total))
    bar = '#'  * int(percent) + '_' * (100 - int (percent))
    print (f'\r|{bar}| {percent: .2f}%', end=f"  ---> {progress}/{total}")