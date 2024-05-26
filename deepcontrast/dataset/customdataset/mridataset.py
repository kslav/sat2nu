#!/usr/bin/env python
"""A dataset object for MRI data that enables compatability with PyTorch."""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pathlib
#from SimpleITK import GetArrayFromImage
#from SimpleITK import ReadImage
import SimpleITK as sitk
import cv2 # OpenCV for image manipulation

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class CustomMRIDataset(Dataset):
# PURPOSE: Define a Dataset for MRIs that loads in pairs of ground truth 
# and input pairs from a CSV file. 

# FILE TYPES: Images are all loaded as dicoms from the directories in the CSV 
# file and are loaded as such before being converted to torch Tensors

# TRANSFORMS: NULL for now, consider rotaitons and translations

    # A custom Dataset class must have the following three functions
    def __init__(self, img_dirs_csv, target_size, limit_range = False, transform=None):
        self.img_dirs_csv = img_dirs_csv # CSV of ground truth and input pairs of 2D dicoms
        self.img_dirs = pd.read_csv(img_dirs_csv) # data frame of img_dirs_csv loaded with pandas
        self.transform = transform # transformations
        self.target_size = target_size # when doing reshaping
        self.limit_range = limit_range

    def __len__(self):
        return len(self.img_dirs_csv) # how many ground truth and input pairs

    def __getitem__(self, idx):
        img_path_gt = self.img_dirs.loc[idx,'GT_Nfs']   #ground truth dicom path (non-fat-sat)
        img_path_inp = self.img_dirs.loc[idx,'Inp_Pre'] #input into the model dicom path (pre-contrast fat-sat)
        
        ### This isn't quite working out; gt and inp don't correspond ###
        ### Model still converges, though, so that says something! ###

        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(img_path_gt))    #actual ground truth image loaded with SITK
        img_gt = img_gt.transpose() #CHECK THIS
        img_inp = sitk.GetArrayFromImage(sitk.ReadImage(img_path_inp))  #actual input image loaded with SITK
        img_inp = img_inp.transpose() #CHECK THIS

        ### Try ImageSeriesReader method (img paths need to be folders now) ###
        ### Also not working...won't recognize dicoms -_- ###

        # dcm_idx = 20 # hard-coded for now, a slice most likely to have tissue!
        
        # # First for ground truth 
        # reader_gt = sitk.ImageSeriesReader()
        # seriesID = reader_gt.GetGDCMSeriesIDs(img_path_gt)
        # print("series_names = ", seriesID)
        # dicom_names = reader_gt.GetGDCMSeriesFileNames(img_path_gt) #this apparently gets the file names in a sorted way...
        # print("dicon_names = ", dicom_names)
        # reader_gt.SetFileNames(dicom_names) # sets the file names from which reader will read
        # reader_gt.MetaDataDictionaryArrayUpdateOn() # turns on loading of meta data (header info)
        # reader_gt.LoadPrivateTagsOn()
        # img_gt = reader_gt.Execute()
        # img_gt = sitk.GetArrayFromImage(sitk.ReadImage(img_gt)) # extracts the image as a numpy array
        
        # # Next for input
        # reader_inp = sitk.ImageSeriesReader()
        # seriesID = reader_inp.GetGDCMSeriesIDs(img_path_gt)
        # dicom_names_inp = reader_inp.GetGDCMSeriesFileNames(img_path_inp)
        # reader_inp.SetFileNames(dicom_names_inp)
        # reader_inp.MetaDataDictionaryArrayUpdateOn()
        # reader_inp.LoadPrivateTagsOn()
        # img_inp = reader_inp.Execute()
        # img_inp = sitk.GetArrayFromImage(sitk.ReadImage(img_inp))
        
        # # Check the shapes...
        # print("------->img_gt.shape=",img_gt.shape)
        # print("------->img_inp.shape=",img_inp.shape)

        # img_gt = img_gt.transpose() #CHECK THIS
        # img_inp = img_inp.transpose() #CHECK THIS

        ### Resize the images to the target_size (shape should be [num_channel, X, Y]) ###
        if self.target_size is not None:
            img_gt = cv2.resize(img_gt, (self.target_size,self.target_size), interpolation=cv2.INTER_NEAREST)
            img_inp = cv2.resize(img_inp,(self.target_size,self.target_size),interpolation=cv2.INTER_NEAREST)

        ### Image standardization applied here ###
        
        # Z-score normalization of grey levels of the input image
        mean_inp = np.mean(img_inp)
        std_inp = np.std(img_inp)
        img_inp = (img_inp - mean_inp)/std_inp # ensures floating point operation

        # Z-score normalization of grey levels of the ground truth image
        mean_gt = np.mean(img_gt)
        std_gt = np.std(img_gt)
        img_gt = (img_gt - mean_gt)/std_gt

        # Limit the range of the images to [0, 1] if limit_range = True
        if self.limit_range == True:
            img_gt -= img_gt.min()
            img_gt /= img_gt.max()
            img_gt = img_gt*100
            img_inp -= img_inp.min()
            img_inp /= img_inp.max()
            img_inp = img_inp*100


        if self.transform is not None:
            #(Note that the ToTensor transformation -- as self.transform -- will yield image values in range 0 to 1 ONLY for images with range [0, 255])
            img_gt = self.transform(img_gt.astype("float32")) #apply any transformations during training to both GT and inp
            img_inp = self.transform(img_inp.astype("float32"))

        
        return img_gt, img_inp # each "item" is a pair of ground truth and input 2D images
