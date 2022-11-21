import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import scipy.ndimage as scin
from scipy import ndimage
import scipy.io as scio
import os
import random
import h5py
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)

class Event_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):

        self.TM = Training_mode

    def __len__(self):
        # if self.TM == 'Train':
        #     return self.train_len
        # elif self.TM == 'Test':
        #     return self.test_len
        return 18000
    def zoom_img(self,input_img,ratio_):
        output_shape = int(input_img.shape[-1]*ratio_)

        return np.concatenate([self.zoom_img_(img,output_shape = output_shape)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img_(self,input_img,output_shape):
        return input_img.reshape(input_img.shape[0],output_shape,-1).mean(-1).swapaxes(0,1).reshape(output_shape,output_shape,-1).mean(-1).swapaxes(0,1)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)

    def efame2event(self,Efame):
        [tt,twi,thi,tai] = np.where(Efame==True)
        event = np.concatenate([tt[:,np.newaxis]/100, twi[:,np.newaxis]/W, thi[:,np.newaxis]/H, (tai[:,np.newaxis]-0.5)*2],dim=1)

        return event
    def __getitem__(self, index):
        # file_path = '/data2/zzy/Lasot/output_event/rabbit-9.h5'
        file_path = '/home/zzy/rpg_vid2e/esim_py/tests/events2.h5'
        Bbox_path = '/data2/zzy/Lasot/LaSOTBenchmark/airplane/airplane-19/groundtruth.txt'
        evt_fam = h5py.File(file_path)['event'][:]
        gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
        temp_box = gt_bbox[0,:]
        temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
        temp_box[2:] = temp_box[2:] * 1.2 
        temp_box = temp_box.astype(np.int_)
        # evt_fam shape of [timestamps, w, h, act]
        print(evt_fam.shape)
        T, W, H, A = evt_fam.shape
        temp_box[0] = np.clip(temp_box[0], 0 ,W)
        temp_box[1] = np.clip(temp_box[1], 0 ,H)
        temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
        temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])

        template_bbox = np.ones([W,H],dtype=np.bool_)

        template_bbox[:temp_box[0],:] = False
        template_bbox[temp_box[0]+temp_box[2],:] = False
        template_bbox[:temp_box[1],:] = False
        template_bbox[temp_box[1]+temp_box[3],:] = False
        
        fam_idx = random.randint(100,int(T*0.7))
        fam_idx_01 = int(fam_idx / 33.3333)
        fam_idx = int(fam_idx_01*33.3333)
        
        template = evt_fam[:100]
        template = template* template_bbox[np.newaxis, :, :, np.newaxis]

        template = self.efame2event(template)

        curr = evt_fam[fam_idx-100:fam_idx]

        template = self.efame2event(curr)
        
        ground_truth = gt_bbox[fam_idx_01,:]
        ground_truth[0] = ground_truth[0]/ W
        ground_truth[1] = ground_truth[1]/ H
        ground_truth[2] = ground_truth[2]/ W
        ground_truth[3] = ground_truth[3]/ H

        return (template, curr,ground_truth)
        