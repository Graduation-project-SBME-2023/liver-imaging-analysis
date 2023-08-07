import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  #remove
import pytest
import sys   #remove
sys.path.insert(0, 'D:/2023_GP/liver-imaging-analysis')   #remove
import torch
from liver_imaging_analysis.engine.utils import VolumeSlicing,Overlay ,calculate_largest_tumor,liver_isolate_crop,find_pix_dim
import nibabel as nib
import numpy as np



# test input files
volume_dir="C:/Users/Hager/Downloads/test/liver/volume"
mask_dir="C:/Users/Hager/Downloads/test/liver/mask"
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
    volume contains only liver while mask contains only lesions

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


#if no tumor ,slice_idx= -1
@pytest.mark.parametrize("input,expected", [(np.where(mask.get_fdata() == 2, 1, 0), 16), (np.zeros_like(mask.get_fdata(),dtype=int), -1)])
def test_largest_tumor(input,expected):
    """
    Tests index where largest tumor slice is,
    """
    slice_idx=calculate_largest_tumor( torch.from_numpy(input)) 
    assert slice_idx==expected  


# def test_find_pix_dim():
#     """
#     Tests nifti file pixel dimensions
#     """
#     x,y,z=find_pix_dim(volume)
#     assert np.allclose((x,y,z),(0.664062, 0.664062, 5.0))


def test_overlay(temp_dir):
    """
    Tests overlay gif is saved
    """
    path=temp_dir[0].join("gif.gif")
    overlay=Overlay(vol_path,mask_path,str(path))
    overlay.generate_animation()
    
    assert os.path.exists(path)  #check gif was saved

@pytest.mark.parametrize("extension", ['.nii.gz','.png'])
def test_volumeslicing(temp_dir,extension):
    """
    Tests all volume slices are saved
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



















 ################################################### getter functions  ###################################################################   

# @pytest.mark.parametrize("mode", [
#     ("3D"),
#     ("2D"),
#     ('sliding_window')
# ])
# def test_set_configs(mode):
#     LiverSeg = LiverSegmentation(mode)

#     assert isinstance(LiverSeg.scheduler, lr_scheduler.StepLR)
#     assert LiverSeg.scheduler.step_size== 20
#     assert LiverSeg.scheduler.gamma== 0.5

    
#     assert isinstance(LiverSeg.network , nets.UNet)
#     assert LiverSeg.network.strides == [2, 2, 2]
#     assert LiverSeg.network.channels == [64, 128, 256, 512]

#     if mode=="sliding_window":
#          assert LiverSeg.batch_size ==  1
#     else:
#         assert LiverSeg.batch_size ==  8

    
#     assert isinstance( LiverSeg.optimizer,torch.optim.Adam)
#     assert LiverSeg.optimizer.defaults['lr'] ==  0.01
    
#     assert LiverSeg.device =="cuda"
#     assert isinstance( LiverSeg.loss ,losses.DiceLoss)
#     assert LiverSeg.metrics.include_background== 1



# ########################################## Pretraining ##################################################################

# @pytest.mark.parametrize("transform_name", ["3DUnet_transform", "2DUnet_transform"])
# def test_get_pretraining_transforms(transform_name):
#     LiverSeg = LiverSegmentation()
#     transforms = LiverSeg.get_pretraining_transforms(transform_name)
#     assert isinstance(transforms, Compose)

#     if transform_name == "3DUnet_transform":
#         expected_cls = [LoadImageD, EnsureChannelFirstD, NormalizeIntensityD, 
#                         ForegroundMaskD, ToTensorD]
    
#     elif transform_name == "2DUnet_transform":
#         expected_cls = [LoadImageD, EnsureChannelFirstD, ResizeD, RandZoomd, RandFlipd,
#                         RandRotated, RandAdjustContrastd, NormalizeIntensityD, 
#                         ForegroundMaskD, ToTensorD]


#     for t, c in zip(transforms.transforms, expected_cls):
#         assert isinstance(t, c)

# ######################################## Pretesting ##################################################################

# @pytest.mark.parametrize("transform_name", ["3DUnet_transform", "2DUnet_transform"])
# def test_get_pretesting_transforms(transform_name):
#     LiverSeg = LiverSegmentation()
#     transforms = LiverSeg.get_pretesting_transforms(transform_name)
#     assert isinstance(transforms, Compose)

#     if transform_name == "3DUnet_transform":
#         expected_cls = [LoadImageD, EnsureChannelFirstD, NormalizeIntensityD,
#                         ForegroundMaskD, ToTensorD]

#     elif transform_name == "2DUnet_transform":
#         expected_cls = [LoadImageD, EnsureChannelFirstD, ResizeD, 
#                         NormalizeIntensityD, ForegroundMaskD, ToTensorD]


#     for t, c in zip(transforms.transforms, expected_cls):
#         assert isinstance(t, c)

# ######################################## Postprocessing ###################################################################

# @pytest.mark.parametrize("transform_name", ["3DUnet_transform", "2DUnet_transform"])
# def test_get_postprocessing_transforms(transform_name):
#     LiverSeg = LiverSegmentation()
#     transforms = LiverSeg.get_postprocessing_transforms(transform_name)

#     assert isinstance(transforms, Compose)

#     if transform_name == "3DUnet_transform":
#         expected_cls =  [ActivationsD, AsDiscreteD, FillHolesD, KeepLargestConnectedComponentD]

#     elif transform_name == "2DUnet_transform":
#         expected_cls = [ActivationsD, AsDiscreteD, FillHolesD, KeepLargestConnectedComponentD]


#     for t, c in zip(transforms.transforms, expected_cls):
#         assert isinstance(t, c)



# ######################################## Segmentation  ###################################################################\

# @pytest.mark.parametrize(("mode","prediction_path"), [

#      ("2D", "testcases/vol_slice.nii"),
#     ("3D","testcases/vol.nii"),
#     ("sliding_window","testcases/vol.nii")
#     ]) 
# class TestSegmentation:
#     def test_segment_liver(self,mode,prediction_path):
        
#         if mode=='2D':         
#             prediction = segment_liver(prediction_path)
#         if mode=='3D':
#             prediction = segment_liver_3d(prediction_path)
#         else:
#             prediction = segment_liver_sliding_window(prediction_path)


#         assert isinstance(prediction, torch.Tensor)
#         assert prediction.shape[0] == 1 #1 final concatenated batches 
#         assert prediction.shape[1] ==1 # number of channels
#         assert prediction.shape[2:] == nib.load(prediction_path).get_fdata().shape # original volume dimension
        
#         assert torch.min(prediction) >= 0
#         assert torch.max(prediction) <= 1

#     def test_segment_lesions(self,mode,prediction_path):   

#         if mode=='2D':
#             prediction = segment_lesion(prediction_path)
#         elif '3D':
#             prediction = segment_lesion_3d(prediction_path)
#         else:
#             return

#         assert isinstance(prediction, torch.Tensor)
#         assert prediction.shape[0] == 1 #1 final concatenated batches 
#         assert prediction.shape[1] ==1 # number of channels
#         assert prediction.shape[2:] == nib.load(prediction_path).get_fdata().shape # original volume dimension
        
#         assert torch.min(prediction) >= 0
#         assert torch.max(prediction) <= 2 


#     def test_segment_lobes(self,mode,prediction_path):
        
#         if mode=='2D':
#             prediction = segment_lobe(prediction_path)
#         elif '3D':
#             prediction = segment_lobe_3d(prediction_path)
#         else:
#             return

#         assert isinstance(prediction, torch.Tensor)
#         assert prediction.shape[0] == 1 #1 final concatenated batches 
#         assert prediction.shape[1] ==10 # number of channels
#         assert prediction.shape[2:] == nib.load(prediction_path).get_fdata().shape # original volume dimension
        
#         assert torch.min(prediction) >= 0
#         assert torch.max(prediction) <= 1         

####################################### Test_liver #####################################################################

# @pytest.mark.parametrize("modality", [
#     ("2D"), 
#     ("3D"),
#     ("sliding_window") 
#     ])
# def test_loss(mode):

#     # Train model
#     train_liver(mode = mode,  epochs=2,
#                 cp_path='/content/drive/MyDrive/liver-imaging-analysis/liver_imaging_analysis/pck')
    
#     # Load latest checkpoint
#     LiverSeg = LiverSegmentation(mode) 
#     LiverSeg.load_data()
#     LiverSeg.data_status()
#     LiverSeg.load_checkpoint('/content/drive/MyDrive/liver-imaging-analysis/liver_imaging_analysis/pck')

#     #Test on sample data
#     test_loss = LiverSeg.test(LiverSeg.test_dataloader)
#     # Assert loss decreased
#     assert test_loss[1] != 0

######################################################################################################################

