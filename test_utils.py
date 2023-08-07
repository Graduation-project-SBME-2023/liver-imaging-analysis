import os
import pytest
import torch
from liver_imaging_analysis.engine.utils import VolumeSlicing,Overlay ,calculate_largest_tumor,liver_isolate_crop,find_pix_dim
import nibabel as nib
import numpy as np



# test input files
volume_dir="test/liver/volume"
mask_dir="test/liver/mask"
vol_path=os.path.join(volume_dir,'resized_liver.nii')
mask_path= os.path.join(mask_dir,'resized_mask.nii')   # 0 background, 1 liver, 2 lesion
mask=nib.load(mask_path)
volume=nib.load(vol_path)

@pytest.fixture
def temp_dir(tmpdir):
    """
    provides temporary directories for some test functions
    """
    temp_vol=tmpdir.mkdir("vol")
    temp_mask=tmpdir.mkdir("mask")
    return  temp_vol,temp_mask



def test_liver_isolate_crop(temp_dir):
    """
    Tests volume and mask are cropped in z direcrion,
    volume contains only liver while mask contains only lesions.


    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving new_volume and new_mask
    """   
    new_vol_dir=temp_dir[0]
    new_mask_dir=temp_dir[1]
    min_slice=10
    max_slice=32
    cropped_mask=mask.get_fdata()[:,:,min_slice:max_slice]
    cropped_vol=volume.get_fdata()[:,:,min_slice:max_slice]

    liver_isolate_crop(volume_dir, mask_dir,
                       new_vol_dir, new_mask_dir)
    
    #testing on only one volume and one mask
    assert os.path.exists(new_vol_dir.join(os.listdir(new_vol_dir)[0])) 
    assert os.path.exists(new_mask_dir.join(os.listdir(new_mask_dir)[0]))
    
    new_volume = nib.load(new_vol_dir.join(os.listdir(new_vol_dir)[0])).get_fdata()
    new_mask = nib.load(new_mask_dir.join(os.listdir(new_mask_dir)[0])).get_fdata()

    assert new_volume.shape==new_mask.shape==(64,64,max_slice-min_slice)         #shape after cropping
    assert np.all(new_mask==np.where(cropped_mask == 2, 1, 0))          # 0 background, 1 lesions
    assert np.all( new_volume==  np.where(
                (cropped_mask > 0.5),
                cropped_vol.astype(int),
                cropped_vol.astype(int).min()) )            #liver isolated





@pytest.mark.parametrize("input,expected", [(np.where(mask.get_fdata() == 2, 1, 0), 16), (np.zeros_like(mask.get_fdata(),dtype=int), -1)])
def test_largest_tumor(input,expected):
    """
    Tests index where largest tumor slice is,
    if no tumor ,slice_idx= -1
   
    Parameters
    ----------
    input: numpy.ndarray
        A NumPy array containing the mask
    expected: int
        expected slice number
    """
    slice_idx=calculate_largest_tumor( torch.from_numpy(input)) 
    assert slice_idx==expected  



def test_overlay(temp_dir):
    """
    Tests overlay gif is saved

    
    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            temporary directory for saving the gif
    """
    path=temp_dir[0].join("gif.gif")
    overlay=Overlay(vol_path,mask_path,str(path))
    overlay.generate_animation()
    
    assert os.path.exists(path) 

   
    

@pytest.mark.parametrize("extension", ['.nii.gz','.png'])
def test_volumeslicing(temp_dir,extension):
    """
    Tests all volume slices are saved

    Parameters
    ----------
    temp_dir: tuple of py._path.local.LocalPath
            Two temporary directories for saving volume slices & mask slices
    extension: str
            extension of the slice file
    """
    
    temp_vol_slices=temp_dir[0]
    temp_mask_slices=temp_dir[1]
    test_shape=(64,64)
    test_num_of_slices= volume.get_fdata().shape[2]
    
    if extension==".png":
        VolumeSlicing.nii2png( volume_dir, mask_dir, temp_vol_slices, temp_mask_slices)
    else:
        VolumeSlicing.nii3d_To_nii2d( volume_dir, mask_dir, temp_vol_slices, temp_mask_slices)

    #correct number of slices
    assert len(os.listdir(temp_vol_slices)) == len(os.listdir(temp_mask_slices)) == test_num_of_slices  
    #correct file type 
    assert all(file_name.endswith(extension)== True       
               for file_name in os.listdir(temp_vol_slices))  
    assert all(file_name.endswith(extension)== True 
               for file_name in os.listdir(temp_mask_slices)) 
    #correct slice dimenisons 
    assert (nib.load(temp_vol_slices/file_name).get_fdata().shape==test_shape 
            for file_name in os.listdir(temp_vol_slices))   
    assert (nib.load(temp_mask_slices/file_name).get_fdata().shape==test_shape 
            for file_name in os.listdir(temp_mask_slices))


