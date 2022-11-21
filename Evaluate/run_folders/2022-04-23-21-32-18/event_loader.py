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
import time
# new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
class Event_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True):
        assert(Training_mode in ['Train','Test'])
        self.TM = Training_mode


        self.idx_list = ['1','2','3','4','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        self.idx_list_all = [str(i) for i in range(1,21)]
        self.idx_list_test = [ str(i) for i in range(1,21) if str(i) not in self.idx_list]
        if os.path.exists('/data/zzy/processed_template/'+'template_airplane-1.h5') == False:
            # print('generate template')
            self._generate_template()
        self.test_data_idx = 0
        

    def __len__(self):
        if self.TM == 'Train':
            return 10000
        else:
            name = 'airplane-' + self.idx_list_test[self.test_data_idx]
            # print(name)

            template = '/data/zzy/processed_template/'+'template_'+name+'.h5'
            template = h5py.File(template, 'r')['template'][:]

            output_event = '/data/zzy/processed_event/'+name +'.h5'
            output_file = h5py.File(output_event, 'r')
            temp_path = '/data/zzy/LaSOTBenchmark/airplane/'

            Bbox_path = temp_path + name + '/groundtruth.txt'

            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')

            return int(gt_bbox.shape[0] - 1)
    def _generate_template2(self):
        temp_path = '/data/zzy/LaSOTBenchmark/airplane/'
        names = os.listdir(temp_path)

        for name_i in self.idx_list_all:
            # print('generate template')
            name = 'airplane-'+name_i
            Bbox_path = temp_path + name + '/groundtruth.txt'
            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
            temp_box = gt_bbox[0,:]
            temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            temp_box[2:] = temp_box[2:] * 1.2 
            temp_box = temp_box.astype(np.int_)
            temp_img = temp_path + name + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape
            A = 2
            temp_box[0] = np.clip(temp_box[0], 0 ,W)
            temp_box[1] = np.clip(temp_box[1], 0 ,H)
            temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])


            file_path = '/data/zzy/processed_event/'+ name +'.h5'
            temp_evt_fam = h5py.File(file_path,'r')['event%08d'%1][:]
            eT = temp_evt_fam[:,2]
            eW = temp_evt_fam[:,0]
            eH = temp_evt_fam[:,1]
            eA = temp_evt_fam[:,3]

            template_index1 = eT<=100
            template_index2 = eW>=temp_box[0]
            template_index3 = eW<=temp_box[0]+temp_box[2]
            template_index4 = eH>=temp_box[1]
            template_index5 = eH<=temp_box[1]+temp_box[3]
            template_index = template_index1 *template_index2 *template_index3 *template_index4 *template_index5
            template = temp_evt_fam[template_index,:]

            # output_event = '/data/zzy/processed_template/'+'template_'+name+'.h5'
            # output_file = h5py.File(output_event, 'w')
            scio.savemat('/data/zzy/processed_template/'+name+'.mat',{'tempalte':template})

            # output_file.create_dataset('template',data = template)
            # output_file.close()
    def _generate_template3(self):
        temp_path = '/data/zzy/LaSOTBenchmark/airplane/'
        names = os.listdir(temp_path)

        for name_i in self.idx_list_all:
            # print('generate template')
            name = 'airplane-'+name_i
            Bbox_path = temp_path + name + '/groundtruth.txt'
            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
            temp_box = gt_bbox[0,:]
            temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            temp_box[2:] = temp_box[2:] * 1.2 
            temp_box = temp_box.astype(np.int_)
            temp_img = temp_path + name + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape
            A = 2
            temp_box[0] = np.clip(temp_box[0], 0 ,W)
            temp_box[1] = np.clip(temp_box[1], 0 ,H)
            temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])


            file_path = '/data/zzy/processed_event/'+ name +'.h5'
            temp_evt_fam = h5py.File(file_path,'r')['event%08d'%1][:]
            eT = temp_evt_fam[:,2]
            eW = temp_evt_fam[:,0]
            eH = temp_evt_fam[:,1]
            eA = temp_evt_fam[:,3]

            template_index1 = eT<=100
            template_index2 = eW>=temp_box[0]
            template_index3 = eW<=temp_box[0]+temp_box[2]
            template_index4 = eH>=temp_box[1]
            template_index5 = eH<=temp_box[1]+temp_box[3]
            template_index = template_index1 *template_index2 *template_index3 *template_index4 *template_index5
            template = temp_evt_fam[template_index,:]

            output_event = '/data/zzy/processed_template/'+'template_'+name+'.h5'
            output_file = h5py.File(output_event, 'w')

            output_file.create_dataset('template',data = template)
            output_file.close()

    def _generate_template(self):
        temp_path = '/data/zzy/LaSOTBenchmark/airplane/'
        names = os.listdir(temp_path)

        for name_i in self.idx_list_all:
            # print('generate template')
            name = 'airplane-'+name_i
            Bbox_path = temp_path + name + '/groundtruth.txt'
            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
            temp_box = gt_bbox[0,:]
            temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            temp_box[2:] = temp_box[2:] * 1.2 
            temp_box = temp_box.astype(np.int_)
            temp_img = temp_path + name + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape
            A = 2
            temp_box[0] = np.clip(temp_box[0], 0 ,W)
            temp_box[1] = np.clip(temp_box[1], 0 ,H)
            temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])


            file_path = '/data/zzy/processed_event/'+ name +'.h5'
            temp_evt_fam = h5py.File(file_path,'r')['event%08d'%1][:]
            eT = temp_evt_fam[:,2]
            eW = temp_evt_fam[:,0]
            eH = temp_evt_fam[:,1]
            eA = temp_evt_fam[:,3]

            template_index1 = eT<=100
            template_index2 = eW>=temp_box[0]
            template_index3 = eW<=temp_box[0]+temp_box[2]
            template_index4 = eH>=temp_box[1]
            template_index5 = eH<=temp_box[1]+temp_box[3]
            template_index = template_index1 *template_index2 *template_index3 *template_index4 *template_index5

            template = temp_evt_fam[template_index,:]

            template_index_F = template_index == False
            template_F = temp_evt_fam[template_index_F,:]
            template_F[:,3] = 0
            
            out_template = np.vstack([template, template_F])

            output_event = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
            output_file = h5py.File(output_event, 'w')

            output_file.create_dataset('template',data = out_template)
            output_file.close()

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
        # file_path = '/home/zzy/rpg_vid2e/esim_py/tests/events2.h5'
        # Bbox_path = '/home/zhu_19/airplane-19/groundtruth.txt'
        # gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
        evt_length = 200
        if self.TM == 'Train':
            data_idx = random.randint(0,18)
            name = 'airplane-' + self.idx_list[data_idx]

            try:
                template_name = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
                template = h5py.File(template_name, 'r')['template'][:]
                # output_event = '/data/zzy/processed_event/'+name +'.h5'
                output_event = '/data/zzy/output_event/'+name +'.h5'
                output_file = h5py.File(output_event, 'r')
                # Tkey = '/data/zzy/EvtTimeStamp/'+name  +'.h5'+'.npy'
                Tkey = np.load('/data/zzy/EvtTimeStamp/'+name +'.h5' +'.npy').item()

            except:
                time.sleep(0.2)
                template_name = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
                template = h5py.File(template_name, 'r')['template'][:]
                # output_event = '/data/zzy/processed_event/'+name +'.h5'
                output_event = '/data/zzy/output_event/'+name +'.h5'
                output_file = h5py.File(output_event, 'r')
                # Tkey = '/data/zzy/EvtTimeStamp/'+name  +'.h5'+'.npy'
                Tkey = np.load('/data/zzy/EvtTimeStamp/'+name  +'.h5'+'.npy').item()

                print('load error:{}'.format(template_name))

            temp_path = '/data/zzy/LaSOTBenchmark/airplane/'

            Bbox_path = temp_path + name + '/groundtruth.txt'

            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')

            # temp_img = temp_path + name + '/img/00000001.jpg'
            # img = cv2.imread(temp_img)

            H,W = h5py.File('/data/zzy/EvtTimeStamp/img_size.h5','r')[name][:]

            A = 2
            # T = int((len(output_file.keys()) - 1)*100)
            T = int(len(Tkey))
            
            fam_idx = random.randint(evt_length*2,T)
            fam_idx_01 = int(fam_idx / 33.3333)
            fam_idx = int(fam_idx_01*33.3333)
            # curr = [output_file['event%08d'%i][:] for i in range(int((fam_idx-evt_length)/100),int((fam_idx)/100+1))]
            # curr = np.vstack(curr)

            # curr = curr[int(fam_idx)%100:int(fam_idx)%100 + evt_length,:]

            # curr1 = [output_file['event%08d'%i][:] for i in range(max(1,int((fam_idx-evt_length)/100-1)),max(2,int((fam_idx)/100+2)))]
            # curr = np.vstack(curr1)
            # print('keys1:{},keys2:{}'.format(, Tkey[str(fam_idx)]))
            keys1 = int(Tkey[str(fam_idx-evt_length)])
            keys2 = int(Tkey[str(fam_idx)])
            curr = output_file['event'][keys1:keys2,:]

            # curr_flag001 = curr[:,2] <= fam_idx/1000+1
            # curr_flag002 = curr[:,2] >= (fam_idx-evt_length)/1000-1
            # curr_flag = curr_flag001 * curr_flag002
            # curr = curr[curr_flag,:]

            # curr_flag001 = curr[:,2] <= (fam_idx+1)/1000
            # curr_flag002 = curr[:,2] >= (fam_idx-evt_length-1)/1000
            # try:
            #     assert((np.max(curr[:,2])+0.005) >= (fam_idx+1)/1000)
            #     assert(np.min(curr[:,2]) <= (fam_idx-evt_length-1)/1000 or (fam_idx-evt_length-1)/1000 <= 0.01)
            # except:
            #     print('index:{},load data range:{}/{}, cut range:{}/{}, curr len:{} range of data:{}/ {}'.format(fam_idx_01, max(1,int((fam_idx-evt_length)/100-1)), max(2,int((fam_idx)/100+1)), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000, len(curr1), np.min(curr[:,2]), np.max(curr[:,2])))
            # curr_flag = curr_flag001 * curr_flag002
            # curr = curr[curr_flag,:]


            curr[:,2] = curr[:,2] - curr[0,2]
            
            n1,c = template.shape
            n2,c = curr.shape

            pn1 = np.random.randint(0,n1,[10000])
            pn2 = np.random.randint(0,n2,[20000])
            template = template[pn1,:].astype(np.float)
            curr = curr[pn2,:].astype(np.float)
            # template = self.efame2event(curr)
            
            ground_truth = gt_bbox[fam_idx_01,:].astype(np.float)
            # print(ground_truth)
            output_gt = np.zeros_like(ground_truth)
            output_gt[0] = (ground_truth[0]+ground_truth[2]/2)/ W
            output_gt[1] = (ground_truth[1]+ground_truth[3]/2)/ H
            output_gt[2] = ground_truth[2]/ W
            output_gt[3] = ground_truth[3]/ H

            ratio = np.array([W,H,evt_length,1])[:,np.newaxis]
            # ratio2 = np.array([W,W,H,H])

            curr = curr.transpose([1,0]) / ratio
            template = template.transpose([1,0]) / ratio
            return (template, curr, output_gt,ground_truth)
        else:
            # data_idx = random.randint(0,6)
            # self.test_data_idx = 0
            name = 'airplane-' + self.idx_list_test[self.test_data_idx]
            if index == 20:
                print(name)

            template = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
            template = h5py.File(template, 'r')['template'][:]

            output_event = '/data/zzy/processed_event/'+name +'.h5'
            output_file = h5py.File(output_event, 'r')
            temp_path = '/data/zzy/LaSOTBenchmark/airplane/'

            Bbox_path = temp_path + name + '/groundtruth.txt'

            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')

            temp_img = temp_path + name + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape

            A = 2
            T = int((len(output_file.keys()) - 1)*100)
            
            # fam_idx = random.randint(evt_length*2,T)
            # fam_idx_01 = int(fam_idx / 33.3333)
            fam_idx_01 = index+1
            fam_idx = int(fam_idx_01*33.3333)
            # curr = [output_file['event%08d'%i][:] for i in range(max(1,int((fam_idx-evt_length)/100)),int((fam_idx)/100+1))]
            # curr = np.vstack(curr)

            # # curr = curr[int(fam_idx)%100:int(fam_idx)%100 + evt_length,:]
            # curr_flag001 = curr[:,2] <= fam_idx
            # curr_flag002 = curr[:,2] >= fam_idx-evt_length
            # curr_flag = curr_flag001 * curr_flag002

            # curr = curr[curr_flag,:]

            # curr[:,2] = curr[:,2] - curr[0,2]
            # print('index:{},load data range:{}/{}, cut range:{}/{}'.format(index, max(1,int((fam_idx-evt_length)/100-1)), int((fam_idx)/100+1), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000))
            up = min(max(2,int((fam_idx)/100+2)),int(len(output_file.keys())+1))
            lower = int((fam_idx-evt_length)/100-1)
            curr1 = [output_file['event%08d'%i][:] for i in range(max(1,min(lower, up-1)),up)]
            curr = np.vstack(curr1)


            try:
                curr_flag001 = curr[:,2] <= (fam_idx+1)/1000
                curr_flag002 = curr[:,2] >= (fam_idx-evt_length-1)/1000
                assert((np.max(curr[:,2])+0.01) >= (fam_idx+1)/1000)
                assert(np.min(curr[:,2]) <= (fam_idx-evt_length-1)/1000 or (fam_idx-evt_length-1)/1000 <= 0.01)
            except:
                curr_flag001 = curr[:,2] <= curr[:,2].max()
                curr_flag002 = curr[:,2] >= curr[:,2].min()
                print('index:{}, load data range:{}/{}, cut range:{}/{}, curr len:{}, range of data:{}/{}'.format(index, max(1,int((fam_idx-evt_length)/100-1)), max(2,int((fam_idx)/100+1)), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000, len(curr1), np.min(curr[:,2]), np.max(curr[:,2])))

            curr_flag = curr_flag001 * curr_flag002
            curr = curr[curr_flag,:]
            # print('shape of curr:{}'.format(curr.shape))
            curr[:,2] = curr[:,2] - curr[0,2]
            
            n1,c = template.shape
            n2,c = curr.shape

            pn1 = np.random.randint(0,n1,[10000])
            pn2 = np.random.randint(0,n2,[40000])
            template = template[pn1,:].astype(np.float)
            curr = curr[pn2,:].astype(np.float)
            
            ground_truth = gt_bbox[fam_idx_01,:].astype(np.float)
            # print(ground_truth)
            output_gt = np.zeros_like(ground_truth)
            output_gt[0] = (ground_truth[0]+ground_truth[2]/2)/ W
            output_gt[1] = (ground_truth[1]+ground_truth[3]/2)/ H
            output_gt[2] = ground_truth[2]/ W
            output_gt[3] = ground_truth[3]/ H

            ratio = np.array([W,H,evt_length,1])[:,np.newaxis]
            ratio2 = np.array([W,W,H,H])

            curr = curr.transpose([1,0]) / ratio
            template = template.transpose([1,0]) / ratio
            return (template, curr, output_gt,ground_truth)

            # temp_box = gt_bbox[0,:]
            # temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            # temp_box[2:] = temp_box[2:] * 1.2 
            # temp_box = temp_box.astype(np.int_)

            # # evt_fam shape of [timestamps, w, h, act]
            # # print(evt_fam.shape)
            # # eT = evt_fam[:,2]
            # # eW = evt_fam[:,0]
            # # eH = evt_fam[:,1]
            # # eA = evt_fam[:,3]
            # # T, W, H, A = evt_fam.shape
            # W = 320
            # H = 240
            # A = 2
            # T = 63333

            # temp_box[0] = np.clip(temp_box[0], 0 ,W)
            # temp_box[1] = np.clip(temp_box[1], 0 ,H)
            # temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            # temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])

            # fam_idx_01 = index
            # fam_idx = int(fam_idx_01*33.3333)
            
            # template_index1 = eT<=100
            # template_index2 = eW>=temp_box[0]
            # template_index3 = eW<=temp_box[0]+temp_box[2]
            # template_index4 = eH>=temp_box[1]
            # template_index5 = eH<=temp_box[1]+temp_box[3]
            # template_index = template_index1 *template_index2 *template_index3 *template_index4 *template_index5
            # template = evt_fam[template_index,:]

            # curr_index1 = eT<=fam_idx
            # curr_index2 = eT>=fam_idx-100
            # curr_index = curr_index1 *curr_index2
            # curr = evt_fam[curr_index,:]

            # n1,c = template.shape
            # n2,c = curr.shape

            # pn1 = np.random.randint(0,n1,[10000])
            # pn2 = np.random.randint(0,n2,[20000])
            # template = template[pn1,:].astype(np.float)
            # curr = curr[pn2,:].astype(np.float)
            # # template = self.efame2event(curr)
            
            # ground_truth = gt_bbox[fam_idx_01,:].astype(np.float)
            # output_gt = np.zeros_like(ground_truth)
            # output_gt[0] = (ground_truth[0]+ground_truth[2]/2)/ W
            # output_gt[1] = (ground_truth[1]+ground_truth[3]/2)/ H
            # output_gt[2] = ground_truth[2]/ W
            # output_gt[3] = ground_truth[3]/ H
            # ratio = np.array([W,H,100,1])[:,np.newaxis]
            # ratio2 = np.array([W,W,H,H])

            # return (template.transpose([1,0]) / ratio, curr.transpose([1,0]) / ratio, output_gt)

        