import os
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
from skimage import measure

import radiomics
import torchio as tio
import torch

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
from ipywidgets import interact, widgets

NUM_WORKERS = 14

AUG_COUNT = 5
AUG_TYPE = "inout_plane"
BIAS = "random"

PARAM_settingsFile = r"/home/thulasiseetha/research/sithin_projects/ACCRadiomics_CT/radiomicsFeatures/radiomicsSettingsCT.yaml"

IN_AUG_PARAMS = {'w_stdMM':5, 'h_stdMM':5, 'angle':5, 'ob_type':None}
OUT_AUG_PARAMS = {'scale_a':0.6, 'scale_b':0.8, 'angle':5, 'delta_z':2}

def vol_dice_score(y_pred,y_true, smooth=1):

    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    #wp = (1 - y_true).sum() / len(y_true)
    #wn = y_true.sum() / len(y_true)

    wp = 1


    tp = wp * ((y_pred * y_true).sum())
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    return (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

class ContourInPlaneAug(object):
    
    def __init__(self, w_stdMM,h_stdMM, angle, ob_type="random"): #w_aMM, w_bMM are the measurements in MM
        
        self.w_stdMM = w_stdMM
        self.h_stdMM = h_stdMM
        self.angle = angle
        
        self.ob_type = ob_type
    
    def __call__(self, mask):  
        
        try:
     
            origin = mask.GetOrigin()
            spacing = mask.GetSpacing()
            direction = mask.GetDirection()

            w_spacing, h_spacing,_ = spacing

            mask_arr = sitk.GetArrayFromImage(mask)

            sys_type = np.random.choice(["inc","dec"])#behavior of the annotator

            out_mask = np.zeros_like(mask_arr)

            z_indeces = [i for i,mask_slice in enumerate(mask_arr) if mask_slice.sum()>0]

            for z in z_indeces:


                if self.ob_type=="random":

                    w_stdVOX = np.ceil(np.random.uniform(-self.w_stdMM,self.w_stdMM)/w_spacing)
                    h_stdVOX = np.ceil(np.random.uniform(-self.h_stdMM,self.h_stdMM)/h_spacing)


                elif self.ob_type=="systematic":

                    if sys_type == "inc":
                        w_stdVOX = np.ceil(np.random.uniform(0,self.w_stdMM)/w_spacing)
                        h_stdVOX = np.ceil(np.random.uniform(0,self.h_stdMM)/h_spacing)
                    else:
                        w_stdVOX = np.ceil(np.random.uniform(-self.w_stdMM,0)/w_spacing)
                        h_stdVOX = np.ceil(np.random.uniform(-self.h_stdMM,0)/h_spacing)

                props = measure.regionprops(mask_arr[z])
                w_min,h_min,w_max,h_max = props[0].bbox

                dw = w_max - w_min
                dh = h_max - h_min

                aug_dw = dw + w_stdVOX 
                aug_dh = dh + h_stdVOX 

                factor_w  = np.round(aug_dw/dw,2)
                factor_h = np.round(aug_dh/dh,2)

                if factor_w<=0:
                    continue;#donot execute further - no need to augment

                if factor_h<=0:
                    continue;

                mask_slice = sitk.GetImageFromArray(mask_arr[z])

                scales = (factor_w,factor_w,factor_h,factor_h, 1, 1)
                degrees = (0, 0, 0, 0, -self.angle, self.angle)
                transform = tio.RandomAffine(scales=scales,degrees=degrees,image_interpolation='nearest',p=1)

                aug_mask_slice = transform(mask_slice)
                out_mask[z] = sitk.GetArrayFromImage(aug_mask_slice)

            out_mask = sitk.GetImageFromArray(out_mask)
            out_mask.SetOrigin(origin)
            out_mask.SetSpacing(spacing)
            out_mask.SetDirection(direction)
        except Exception as e:
            print("Error with InPlane Augmentation",e)
            out_mask = mask
        
 
        return out_mask
    
class ContourOutPlaneAug(object):
    
    def __init__(self, scale_a, scale_b,angle, delta_z):
        
        self.scale_a = scale_a
        self.scale_b = scale_b
        
        self.delta_z = delta_z
        self.angle = angle
        
    def __call__(self, mask):
        
        try:
        
            origin = mask.GetOrigin()
            spacing = mask.GetSpacing()
            direction = mask.GetDirection()

            mask_arr = sitk.GetArrayFromImage(mask)

            aug_num_slices = np.random.randint(0, self.delta_z+1) #low, high (excluded high)

            z_indeces = [i for i,mask_slice in enumerate(mask_arr) if mask_slice.sum()>0]

            z_min, z_max = min(z_indeces), max(z_indeces)

            for i in range(aug_num_slices):

                dz = z_max-z_min

                if dz>0:

                    ref_z = np.random.choice([z_min,z_max])

                    offset = -1 if ref_z==z_min else 1

                    aug_type = "del" if np.random.uniform()>0.5 else "add"

                    if mask_arr[ref_z].sum()>0: #which means contour exists in that place, possible that it got deleted during iteration

                        if aug_type=="del":

                            if ref_z+offset in range(len(mask_arr)):

                                if mask_arr[ref_z+offset].sum()==0:#to check if new contours were already inserted up or down
                                    mask_arr[ref_z] = np.zeros(mask_arr[ref_z].shape)

                            else:#if ref_z+offset is outside the boundary
                                mask_arr[ref_z] = np.zeros(mask_arr[ref_z].shape)

                        elif aug_type=="add":

                            if ref_z+offset in range(len(mask_arr)):

                                if mask_arr[ref_z+offset].sum()==0:

                                    scales =(self.scale_a,self.scale_b, self.scale_a,self.scale_b, 1,1)
                                    degrees = (0,0, 0,0, -self.angle, self.angle)

                                    transform = tio.RandomAffine(scales=scales,degrees=degrees,image_interpolation='nearest',p=1)

                                    mask_slice = sitk.GetImageFromArray(mask_arr[ref_z])
                                    mask_arr[ref_z+offset] = sitk.GetArrayFromImage(transform(mask_slice))

            mask = sitk.GetImageFromArray(mask_arr)
            mask.SetOrigin(origin)
            mask.SetSpacing(spacing)
            mask.SetDirection(direction)
        
        except Exception as e:
            print("Error with OutPlane Augmentation",e)
      
        return mask



def extract_features(pid, aug_type = "", ob_type = ""):
    
    global FEATURE_ROWS

    for sequence in SOI:

        img = sitk.ReadImage(os.path.join(DATA_DIR, pid,sequence, "img.nii.gz"))
        mask = sitk.ReadImage(os.path.join(DATA_DIR, pid,sequence, "mask.nii.gz"))

        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()

        if aug_type=="in_plane":
            
            IN_AUG_PARAMS["ob_type"] = ob_type

            ContourAug = ContourInPlaneAug(**IN_AUG_PARAMS)

        elif aug_type=="out_plane":

            ContourAug = ContourOutPlaneAug(**OUT_AUG_PARAMS)

        elif aug_type=="inout_plane":

            IN_AUG_PARAMS["ob_type"] = ob_type

            ContourAug = tio.Compose([
                ContourInPlaneAug(**IN_AUG_PARAMS),
                ContourOutPlaneAug(**OUT_AUG_PARAMS)
            ])

        else:
            #normal radiomics feature extraction; without mask augmentations
            ContourAug = None
            
            
       
        #Resampling - nearest neighbor for slice spacing, BSpline for axial spacing
        img = tio.Resample([spacing[0],spacing[1],OUT_SPACING[2]],'nearest')(img)#nearest neighbor for outspacing
        img = tio.Resample(OUT_SPACING,'bspline')(img)#bspline for inspacing
        mask = tio.Resample(OUT_SPACING,'nearest')(mask)
        
        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        direction = img.GetDirection()
        
        #Normalizing & Shifting Sequences
        
        img_arr = sitk.GetArrayFromImage(img)
        mask_arr = sitk.GetArrayFromImage(mask)

        mean = ADC_MEAN if sequence=="adc" else img_arr[mask_arr==1].mean()
        std = ADC_STD if sequence=="adc" else img_arr[mask_arr==1].std()
        
        img_arr = (img_arr-mean)/std
        img_arr[img_arr<-3] = -3
        img_arr[img_arr>3] = 3
        
        #shifting
        img_arr = (img_arr*SHIFT_STD)+SHIFT_MEAN#Also helps with fixing the bin_width
        
        img = sitk.GetImageFromArray(img_arr)
        img.SetOrigin(origin)
        img.SetSpacing(spacing)
        img.SetDirection(direction)
        
        if ContourAug:
            
            for i in range(AUG_COUNT):

                aug_mask = ContourAug(mask)

                dice = vol_dice_score(sitk.GetArrayFromImage(aug_mask),sitk.GetArrayFromImage(mask))

                extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(PARAM_settingsFile,verbosity=False)

                featureVector = extractor.execute(img,aug_mask)

                featureVector['id'] = pid
                featureVector['sequence'] = sequence
                featureVector['dice'] = dice
                featureVector['judge'] = i+1

                FEATURE_ROWS.append(featureVector)

        else:
            
            extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(PARAM_settingsFile,verbosity=False)

            featureVector = extractor.execute(img,mask)

            featureVector['id'] = pid
            featureVector['sequence'] = sequence

            FEATURE_ROWS.append(featureVector)

            
    pbar.update()