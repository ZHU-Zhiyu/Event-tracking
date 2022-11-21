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
new_load =  lambda *a,**k: np.load(*a, allow_pickle=True, **k)
import random

from dv import AedatFile

class Event_dataset(Dataset):
    """
    
    get the Hyperspectral image and corrssponding RGB image  
    use all data : high resolution HSI, high resolution MSI, low resolution HSI
    """
    def __init__(self, output_shape=512,ratio = 1,Training_mode='Train',data_name = 'CAVE',use_generated_data = False, use_all_data = True,rank=0,generate_template=True):
        assert(Training_mode in ['Train','Test'])
        self.TM = Training_mode


        self.idx_list = ['1','2','3','4','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
        self.idx_list_all = [str(i) for i in range(1,21)]
        self.idx_list_test = [ str(i) for i in range(1,21) if str(i) not in self.idx_list]

        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab','crocodile','deer']
        # temp_total_names = ['basketball','bear','bicycle','bird']
        temp_total_names = ['bear']
        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab','crocodile','deer','electricfan']
        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab','crocodile']
        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab']
        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab','crocodile']
        # temp_total_names = ['basketball','bear','bicycle','bird','boat','bottle','bus','car','cat','cattle','coin','crab','crocodile']
        # temp_total_names = ['crocodile']
        total_names = [ ]
        for names in temp_total_names:
            [total_names.append(names+'-%d'%i) for i in range(1,21)]
        self.total_names = total_names

        self.test_names = np.loadtxt('/data/zzy/LaSOTBenchmark/test_names.txt',dtype='str')

        new_test = [ name for name in self.test_names if name in total_names]
        self.test_names = new_test
        self.train_name = [ name for name in total_names if name not in self.test_names ]
        self.test_names.sort(reverse=True)
        try:
            self.train_name.remove('basketball-9')
        except:
            print('error')
        try:
            self.train_name.remove('electricfan-7')
        except:
            print('error')
        try:
            self.test_names.remove('electricfan-20')
        except:
            print('error')
        try:
            self.test_names.remove('electricfan-18')
        except:
            print('error')
        try:
            self.test_names.remove('electricfan-10')
        except:
            print('error')
        try:
            self.test_names.remove('deer-8')
        except:
            print('error')
        try:
            self.test_names.remove('cattle-7')
        except:
            print('error')
        try:
            self.test_names.remove('cattle-2')
        except:
            print('error')
        try:
            self.test_names.remove('cat-3')
        except:
            print('error')
        try:
            self.test_names.remove('cat-20')
        except:
            print('error')
        try:
            self.test_names.remove('car-9')
        except:
            print('error')
        try:
            self.test_names.remove('car-6')
        except:
            print('error')
        try:
            self.test_names.remove('car-2')
        except:
            print('error')
        try:
            self.test_names.remove('bus-5')
        except:
            print('error')
        try:
            self.test_names.remove('bus-2')
        except:
            print('error')
        # try:
        #     self.test_names.remove('deer-4')
        # except:
        #     print('error')
        try:
            self.test_names.remove('crocodile-4')
        except:
            print('error')
        try:
            self.test_names.remove('crocodile-3')
        except:
            print('error')
        # try:
        #     self.train_name.remove('crocodile-4')
        # except:
        #     print('error')
        try:
            self.test_names.remove('bottle-18')
        except:
            print('error')
        try:
            self.test_names.remove('bottle-14')
        except:
            print('error')
        try:
            self.test_names.remove('bottle-12')
        except:
            print('error')
        try:
            self.test_names.remove('boat-4')
        except:
            print('error')
        try:
            self.test_names.remove('bird-2')
        except:
            print('error')
        try:
            self.test_names.remove('bird-3')
        except:
            print('error')
        try:
            self.test_names.remove('bicycle-9')
        except:
            print('error')
        # try:
        #     self.train_name.remove('bottle-14')
        # except:
        #     print('error')
        # self.train_name.remove('book-1')
        # self.train_name.remove('book-2')
        # print('training name :{}'.format(self.train_name))
#         self.test_names = ['airplane-1','airplane-9','airplane-13','airplane-15','basketball-1','basketball-6', 'basketball-7', 'basketball-11', 'bear-2', 'bear-4', 'bear-6', 'bear-17', 'bicycle-2',
#                             'bicycle-7', 'bicycle-9', 'bicycle-18', 'bird-2', 'bird-3', 'bird-15', 'bird-17', 'boat-3', 'boat-4', 'boat-12', 'boat-17', 'book-3', 'book-10','book-11]

        # if generate_template and rank == 0:
        #     # print('generate template')
        #     if self.TM == 'Train':
        #         self._generate_template()
        #         self._generate_template_test()
        #     elif self.TM == 'Test':
        #         # self._generate_template()
        #         self._generate_template_test()
        # self.test_len = []
        # temp_path00 = '/data/zzy/LaSOTBenchmark/'
        # for name_i in self.test_names:

        #     temp_path = temp_path00+ name_i.split('-')[0] + '/' + name_i
        #     # print(name)
        #     Bbox_path = temp_path + '/groundtruth.txt'
        #     gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
        #     self.test_len.append(int(gt_bbox.shape[0] - 7))
            
        # self.test_data_idx = 0
        # name = bike333

        # name = 'airplane_mul222' #275
        # name = 'box_hdr' #413
        # name = 'cow_mul222' #176
        # name = 'cow_mul222' #176
        # self.start_frame = 176
        name = 'bike_low' #176
        # self.test_names = ['box_low', 'dog', 'dove_mul', 'airplane_mul222', 'dove_mul222', 'elephant222', 'dove_motion', 'dog_motion', 'bike222', 'box_hdr', 'bike333', 'cup222', 'cow_mul222', 'bike_low', 'cup_low', 'bottle_mul222']
        # self.test_names = ['box_low', 'dog', 'dove_mul', 'airplane_mul222', 'dove_mul222', 'elephant222', 'dove_motion']
        self.test_names = ['fighter_mul','giraffe_low','giraffe_motion','giraffe222','ship','ship_motion','star','star_motion','star_mul','star_mul222',
                            'tank_low','tower','tower333','truck','truck_hdr','whale_mul222', 'dog_motion', 'bike222', 'box_hdr', 'bike333', 'cup222', 'cow_mul222',
                            'bike_low', 'cup_low', 'bottle_mul222','box_low', 'dog', 'dove_mul', 'airplane_mul222', 'dove_mul222', 'elephant222', 'dove_motion']
        # self.test_names = ['dog_motion', 'airplane_mul222', 'dove_mul22s2', 'elephant222', 'dove_motion']
        # self.test_names = ['giraffe_motion','giraffe222','ship','ship_motion','star','star_motion','star_mul','star_mul222','tank_low','tower','tower333','truck','truck_hdr','whale_mul222']
        # self.test_names = ['ball_motion','bear_motion','box_motion','cow_motion','star_motion','tank_motion','tower_motion','airplane_motion',]
        # self.test_names = ['airplane_motion']
        # self.test_names = ['airplane_motion','airplane_mul','airplane222','bear_motion','bear222','bike_mul']
        # self.test_names = ['cup_motion']
        # self.test_names = ['tower_motion', 'toy_car_motion','turtle_motion']
        # self.test_names = [ 'dog_motion', 'bike222', 'box_hdr', 'bike333', 'cup222', 'cow_mul222', 'bike_low', 'cup_low', 'bottle_mul222']
        # self.test_names = [ 'dog_motion', 'bike222', 'box_hdr', 'bike333', 'cup222', 'cow_mul222', 'bike_low', 'cup_low', 'bottle_mul222']
        # self.test_names = [ 'bike333']
        train_names = []
        # with open('eotb_train_split.txt', 'r')  as f:
        #     a = [i.strip() for i in f.readlines()]
        #     for line in a:
        #         train_names.append(line)

        self.test_crop = 40
        train_name_file = '/data1/zhu_19/saved_mat/FE108/train.txt'
        with open(train_name_file, 'r') as f:
            for line in f.readlines():
                # name = line.split()
                # start_frame_[file] = int(start_frame) + 1
                train_names.append(line)

        self.train_names = train_names
        self.template = None

        self.test_len = {}
    
        match_file = '/data1/zhu_19/saved_mat/FE108/pair.txt'
        start_frame_ = {}
        with open(match_file, 'r') as f:
            for line in f.readlines():
                file, start_frame = line.split()
                start_frame_[file] = int(start_frame) + 1
        self.start_frame_ = start_frame_

    def __len__(self):
        if self.TM == 'Train':
            return 10000
        else:
            temp_path00 = '/data/zzy/LaSOTBenchmark/'
            num = 0
            for name_i in self.test_names:
                # name = 'airplane-' + self.idx_list_test[self.test_data_idx]
                # temp_path = temp_path00+ name_i.split('-')[0] + '/' + name_i
                try:
                    gt_bbox = np.loadtxt('/data1/zhu_19/saved_mat/'+name_i+'/groundtruth_rect.txt',delimiter=',')
                except:
                    gt_bbox = np.loadtxt('/data/zzy/real_tracking_data/'+name_i+'/groundtruth_rect.txt',delimiter=',')

                print('RAW box max:{},min:{}'.format(gt_bbox.max(),gt_bbox.min()))
                fam_num, length = gt_bbox.shape
                # print(name)
                # Bbox_path = temp_path + '/groundtruth.txt'
                # gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
                num += int(gt_bbox.shape[0]) 
                self.test_len[name_i] = gt_bbox.shape[0]

            return num
    def _update_template(self,name):

        # self.start_frame = 328
        start_frame = self.start_frame_[name]
        try:
            f1 = AedatFile('/data1/zhu_19/saved_mat/'+name+'/events.aedat4')
            gt_bbox = np.loadtxt('/data1/zhu_19/saved_mat/'+name+'/groundtruth_rect.txt',delimiter=',')
        except:
            f1 = AedatFile('/data/zzy/real_tracking_data/'+name+'/events.aedat4')
            gt_bbox = np.loadtxt('/data/zzy/real_tracking_data/'+name+'/groundtruth_rect.txt',delimiter=',')

        print('RAW box max:{},min:{}'.format(gt_bbox.max(),gt_bbox.min()))
        fam_num, length = gt_bbox.shape
        # time_series = []
        events = np.hstack([packet for packet in f1['events'].numpy()])
        timestamps, x, y, polarities = events['timestamp'], events['x'], events['y'], events['polarity']
        events = np.vstack((  x, y, timestamps, polarities))
        # time_series.append(frame.timestamp_start_of_frame)
        events = np.swapaxes(events, 0, 1)
        frames = f1['frames']
        fam = [i for i in frames]

        fam_start = [ fam[i].timestamp_start_of_frame for i in range(len(fam))]
        events = events.astype(np.float_)
        H,W = fam[0].size
        self.H = H
        self.W = W
        gt_bbox[:,0] = gt_bbox[:,0] / H
        gt_bbox[:,1] = gt_bbox[:,1] / W
        gt_bbox[:,2] = gt_bbox[:,2] / H
        gt_bbox[:,3] = gt_bbox[:,3] / W
        print('Norm box max:{},min:{}'.format(gt_bbox.max(),gt_bbox.min()))
        gt_bbox = gt_bbox.clip(0,1)

        events[:,0] = events[:,0]/H
        events[:,1] = events[:,1]/W
        # self.start_frame = 710
        # self.fam_start = fam_start[start_frame: start_frame+fam_num]
        # --------------
        self.fam_start = fam_start[start_frame -1 : start_frame+fam_num -1]
        self.GT = gt_bbox

        fam_start_ = self.fam_start[0]
        image = fam[start_frame -1 : start_frame+fam_num -1 ]
        # --------------
        # self.GT = gt_bbox

        # fam_start_ = self.fam_start[0]
        # image = fam[start_frame: start_frame+fam_num]
        image = [i.image[np.newaxis,:,:] for i in image]

        image = np.concatenate(image,0)

        p_, n_ = events.shape
        self.events = [events[int((i) / self.test_crop*p_): int((i+1) / self.test_crop*p_) ,:]  for i in range(0,self.test_crop)]

        self.events_end = np.array([ temp_[-1,2] for temp_ in self.events])
        self.events_start = np.array([ temp_[0,2] for temp_ in self.events])


        eT = events[:,2]
        eW = events[:,0]
        eH = events[:,1]
        eA = events[:,3]
        # template 
        template_index1 = eT<= self.fam_start[1]
        # template_index1 = eT<= (fam_start_ + 1e6)
        template_index1_ = eT>= fam_start_
        print( (fam_start[6] - fam_start[0]) / 1e6 )
        print('size of gtbox:{}'.format(gt_bbox.shape))
        template_index2 = eW>=gt_bbox[0,0]
        template_index3 = eW<=gt_bbox[0,0]+gt_bbox[0,2]
        template_index4 = eH>=gt_bbox[0,1]
        template_index5 = eH<=gt_bbox[0,1]+gt_bbox[0,3]
        # template_index = template_index1*template_index1_*template_index2 *template_index3 *template_index4 *template_index5
        template_index = template_index1*template_index1_
        template_index02 = template_index1*template_index1_*template_index2 *template_index3 *template_index4 *template_index5
        print('sum of 1:{}, 2:{}, 3:{}, 4:{}, 5:{}, 6:{}'.format(template_index1.sum(), template_index1_.sum(), template_index2.sum(), template_index3.sum(), template_index4.sum(), template_index5.sum()))
        template = events[template_index02,:]
        print('size of template:{}'.format(template.shape))
        # template[:,2] = template[:,2] - template[0,2]

        # scio.savemat('/home/zhu_19/evt_tracking/evt_tracking_zomm_test_04/template.mat',{'tempalte':template, 'box':gt_bbox, 'frame_s': np.array(self.fam_start), 'image':image})

        n1,c = template.shape
        pn1 = np.random.randint(0,n1,[10000])
        template = template[pn1,:].astype(np.float)
        # assert(0==1)
        l1,c1 = template.shape
        Indicator1 = np.ones([l1,1])
        template = np.concatenate([template, Indicator1], axis = 1)


        self.template = template

    def _generate_template2(self):
        temp_path = '/data/zzy/LaSOTBenchmark/airplane/'
        names = os.listdir(temp_path)

        for name_i in self.total_names:
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


            file_path = '/data/zzy/output_event/'+ name +'.h5'
            temp_evt_fam = h5py.File(file_path,'r')['%08d'%1][:]
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


            file_path = '/data/zzy/output_event/'+ name +'.h5'
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
        temp_path00 = '/data/zzy/LaSOTBenchmark/'

        saved_file = h5py.File('/data/zzy/EvtTimeStamp/img_size.h5','w')

        for name_i in self.train_name:

            temp_path = temp_path00+ name_i.split('-')[0] + '/' + name_i

            # Bbox_path = temp_path + name + '/groundtruth.txt'
            Bbox_path = temp_path + '/groundtruth.txt'
            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
            temp_box = gt_bbox[0,:]
            temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            temp_box[2:] = temp_box[2:] * 1.2 
            temp_box = temp_box.astype(np.int_)
            temp_img = temp_path + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape
            A = 2
            temp_box[0] = np.clip(temp_box[0], 0 ,W)
            temp_box[1] = np.clip(temp_box[1], 0 ,H)
            temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])


            file_path = '/data/zzy/output_event/'+ name_i.split('-')[0] + '/' + name_i +'.h5'

            temp_evt_fam = h5py.File(file_path,'r')
            
            curr = []
            for i in range(0,20):
                try:
                    curr.append(temp_evt_fam['%08d'%i][:] )
                except:
                    continue
            temp_evt_fam = np.vstack(curr)
            temp_evt_fam = temp_evt_fam.astype(np.float_)

            eT = temp_evt_fam[:,2]
            eW = temp_evt_fam[:,0]
            eH = temp_evt_fam[:,1]
            eA = temp_evt_fam[:,3]

            # template_index1 = eT<=1e8
            template_index2 = eW>=temp_box[0]
            template_index3 = eW<=temp_box[0]+temp_box[2]
            template_index4 = eH>=temp_box[1]
            template_index5 = eH<=temp_box[1]+temp_box[3]
            template_index = template_index2 *template_index3 *template_index4 *template_index5
            
            template_index_T = np.where(template_index == True)[0]
            if template_index_T.shape[0]<= 10000:

                template = temp_evt_fam[template_index,:]
            else:
                template_index_T = list(template_index_T)[10000]
                template_index = template_index[:template_index_T]
                temp_evt_fam = temp_evt_fam[:template_index_T,:]
                template = temp_evt_fam[template_index,:]

            l1,c1 = template.shape
            Indicator1 = np.ones([l1,1])
            
            template_index_F = template_index == False
            template_F = temp_evt_fam[template_index_F,:]
            l2,c2 = template_F.shape
            Indicator2 = np.zeros([l2,1])
            # template_F[:,3] = 0

            Indicator = np.concatenate([Indicator1, Indicator2], axis = 0)

            
            out_template = np.vstack([template, template_F])
            out_template = np.concatenate([out_template, Indicator], axis = 1)
            # print('name of events:{}, num of output template:{}, minet:{}, max_et:{}, template1:{}, template2:{},template3:{},template4:{},template5:{}'.format(file_path, 
            #                                                                                                                                 np.sum(np.abs(out_template[:,3])),
            #                                                                                                                                 np.min(eT),
            #                                                                                                                                 np.max(eT),
            #                                                                                                                                 np.sum(np.abs(template_index1)),
            #                                                                                                                                 np.sum(np.abs(template_index2)),
            #                                                                                                                                 np.sum(np.abs(template_index3)),
            #                                                                                                                                 np.sum(np.abs(template_index4)),
            #                                                                                                                                 np.sum(np.abs(template_index5)),))
            os.makedirs('/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] , exist_ok = True)

            output_event = '/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] + '/' +'template_'+ name_i +'.h5'
            output_file = h5py.File(output_event, 'w')

            output_file.create_dataset('template',data = out_template)
            output_file.close()
            saved_file.create_dataset(name_i,data = np.array([H,W]))

        saved_file.close()

    # def _get_template(self):
    def resample_point(self, point, box, W, H,ratio=None):
        """
        point in N * 4
        Box in xywh mode

        """
        if np.sum(box)<1e-5:
            output = point
            output[:,0] = output[:,0] / W
            output[:,1] = output[:,1] / H
            GT_Box_changed = np.array([ box[0] / W, box[1] / H, box[2] / W, box[3] / H])

            return output , GT_Box_changed
        else:
            # p0x = random.uniform(0, box[0])
            # p0y = random.uniform(0, box[1])
            # p1x = random.uniform(box[0]+ box[2], 1)
            # p1y = random.uniform(box[1]+ box[3], 1)
            try:
                assert((box[0]+ box[2] -2)<=W)
                assert((box[1]+ box[3] -2)<=H)
            except:
                print('box:{},w:{},box:{},h:{}'.format(box[0]+ box[2], W, box[1]+ box[3], H))
            if ratio==None:
                # p0x = box[0]*random.uniform(0.8, 1)
                # p0y = box[1]*random.uniform(0.8, 1)
                # p1x = box[0]+ box[2] + (W - (box[0]+ box[2] -2))*random.uniform(0, 0.2)
                # p1y = box[1]+ box[3] + (H - (box[1]+ box[3] -2))*random.uniform(0, 0.2)
                p0x = box[0]-W*random.uniform(0.05, 0.2)
                p0y = box[1]-H*random.uniform(0.05, 0.2)
                p1x = box[0]+ box[2] + W*random.uniform(0.05, 0.2)
                p1y = box[1]+ box[3] + H*random.uniform(0.05, 0.2)
            
            else:
                p0x = box[0]*ratio
                p0y = box[1]*ratio
                p1x = box[0]+ box[2] + (W - (box[0]+ box[2] -2))*(1-ratio)
                p1y = box[1]+ box[3] + (H - (box[1]+ box[3] -2))*(1-ratio)

            eW = point[:,0]
            eH = point[:,1]

            template_index1 = eW>=p0x
            template_index2 = eH>=p0y
            template_index3 = eW<=p1x
            template_index4 = eH<=p1y
            template_index = template_index1 *template_index2 *template_index3 *template_index4
            output = point[template_index,:]
            output[:,0] = (output[:,0] - p0x) / (p1x - p0x + 1e-6)
            output[:,1] = (output[:,1] - p0y) / (p1y - p0y + 1e-6)

            GT_Box_changed = np.array([ (box[0] - p0x) / (p1x - p0x + 1e-6), (box[1] - p0y) / (p1y - p0y + 1e-6), box[2] / (p1x - p0x + 1e-6), box[3] / (p1y - p0y + 1e-6)])

            return output, GT_Box_changed
    def resample_point_normed(self, point, box, W, H,ratio=None):
        """
        point in N * 4
        Box in xywh mode

        """
        if np.sum(box)<1e-5:
            output = point
            output[:,0] = output[:,0] 
            output[:,1] = output[:,1] 
            GT_Box_changed = np.array([ box[0] , box[1] , box[2] , box[3] ])

            return output , GT_Box_changed
        else:
            # p0x = random.uniform(0, box[0])
            # p0y = random.uniform(0, box[1])
            # p1x = random.uniform(box[0]+ box[2], 1)
            # p1y = random.uniform(box[1]+ box[3], 1)
            try:
                assert((box[0]+ box[2] -0.01)<=1)
                assert((box[1]+ box[3] -0.01)<=1)
            except:
                print('box:{}'.format(box.max(),box.min()))
                print('box:{},w:{},box:{},h:{}'.format(box[0]+ box[2], W, box[1]+ box[3], H))
            if ratio==None:
                # p0x = box[0]*random.uniform(0.8, 1)
                # p0y = box[1]*random.uniform(0.8, 1)
                # p1x = box[0]+ box[2] + (W - (box[0]+ box[2] -2))*random.uniform(0, 0.2)
                # p1y = box[1]+ box[3] + (H - (box[1]+ box[3] -2))*random.uniform(0, 0.2)
                p0x = box[0]-random.uniform(0.05, 0.2)
                p0y = box[1]-random.uniform(0.05, 0.2)
                p1x = box[0]+ box[2] + random.uniform(0.05, 0.2)
                p1y = box[1]+ box[3] + random.uniform(0.05, 0.2)
            
            else:
                p0x = box[0]*ratio
                p0y = box[1]*ratio
                p1x = box[0]+ box[2] + (1 - (box[0]+ box[2] -2))*(1-ratio)
                p1y = box[1]+ box[3] + (1 - (box[1]+ box[3] -2))*(1-ratio)
            # print('shape of point:{}'.format(point.shape))
            eW = point[:,0]
            eH = point[:,1]

            template_index1 = eW>=p0x
            template_index2 = eH>=p0y
            template_index3 = eW<=p1x
            template_index4 = eH<=p1y
            template_index = template_index1 *template_index2 *template_index3 *template_index4
            output = point[template_index,:]
            output[:,0] = (output[:,0] - p0x) / (p1x - p0x + 1e-6)
            output[:,1] = (output[:,1] - p0y) / (p1y - p0y + 1e-6)

            GT_Box_changed = np.array([ (box[0] - p0x) / (p1x - p0x + 1e-6), (box[1] - p0y) / (p1y - p0y + 1e-6), box[2] / (p1x - p0x + 1e-6), box[3] / (p1y - p0y + 1e-6)])

            return output, GT_Box_changed

    def _generate_template_test(self):
        temp_path00 = '/data/zzy/LaSOTBenchmark/'
        # names = os.listdir(temp_path)

        saved_file = h5py.File('/data/zzy/EvtTimeStamp/img_size_test.h5','w')

        for name_i in self.test_names:

            temp_path = temp_path00+ name_i.split('-')[0] + '/' + name_i

            Bbox_path = temp_path + '/groundtruth.txt'
            gt_bbox = np.loadtxt(Bbox_path,delimiter=',')
            temp_box = gt_bbox[0,:]
            temp_box[:2] = temp_box[:2] - temp_box[2:] * 0.1 
            temp_box[2:] = temp_box[2:] * 1.2 
            temp_box = temp_box.astype(np.int_)
            temp_img = temp_path + '/img/00000001.jpg'
            img = cv2.imread(temp_img)

            H,W,_ = img.shape
            A = 2
            temp_box[0] = np.clip(temp_box[0], 0 ,W)
            temp_box[1] = np.clip(temp_box[1], 0 ,H)
            temp_box[2] = np.clip(temp_box[2], 0 ,W-temp_box[0])
            temp_box[3] = np.clip(temp_box[3], 0 ,H-temp_box[1])


            file_path = '/data/zzy/output_event/'+ name_i.split('-')[0] + '/' + name_i +'.h5'
            temp_evt_fam = h5py.File(file_path,'r')
            
            curr = []
            for i in range(0,3):
                try:
                    curr.append(temp_evt_fam['%08d'%i][:] )
                except:
                    print('error_1')
                    continue
            temp_evt_fam = np.vstack(curr)
            temp_evt_fam = temp_evt_fam.astype(np.float_)
            eT = temp_evt_fam[:,2]
            eW = temp_evt_fam[:,0]
            eH = temp_evt_fam[:,1]
            eA = temp_evt_fam[:,3]

            # template_index1 = eT<=1e8
            template_index2 = eW>=temp_box[0]
            template_index3 = eW<=temp_box[0]+temp_box[2]
            template_index4 = eH>=temp_box[1]
            template_index5 = eH<=temp_box[1]+temp_box[3]
            template_index = template_index2 *template_index3 *template_index4 *template_index5

            template_index_T = np.where(template_index == True)[0]
            if template_index_T.shape[0]<= 10000:

                template = temp_evt_fam[template_index,:]
            else:
                template_index_T = list(template_index_T)[10000]
                template_index = template_index[:template_index_T]
                temp_evt_fam = temp_evt_fam[:template_index_T,:]
                template = temp_evt_fam[template_index,:]
                
            l1,c1 = template.shape
            Indicator1 = np.ones([l1,1])

            template_index_F = template_index == False
            template_F = temp_evt_fam[template_index_F,:]
            # template_F = temp_evt_fam[template_index_F,:]
            l2,c2 = template_F.shape
            Indicator2 = np.zeros([l2,1])
            # template_F[:,3] = 0
            Indicator = np.concatenate([Indicator1, Indicator2], axis = 0)
            
            out_template = np.vstack([template, template_F])
            # out_template = np.vstack([template, template_F])
            out_template = np.concatenate([out_template, Indicator], axis = 1)
            # print('name of events:{}, num of output template:{}, minet:{}, max_et:{}, template1:{}, template2:{},template3:{},template4:{},template5:{}'.format(file_path, 
            #                                                                                                                                 np.sum(np.abs(out_template[:,3])),
            #                                                                                                                                 np.min(eT),
            #                                                                                                                                 np.max(eT),
            #                                                                                                                                 np.sum(np.abs(template_index1)),
            #                                                                                                                                 np.sum(np.abs(template_index2)),
            #                                                                                                                                 np.sum(np.abs(template_index3)),
            #                                                                                                                                 np.sum(np.abs(template_index4)),
            #                                                                                                                                 np.sum(np.abs(template_index5)),))
            os.makedirs('/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] , exist_ok = True)

            output_event = '/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] + '/' +'template_'+ name_i +'.h5'
            output_file = h5py.File(output_event, 'w')

            output_file.create_dataset('template',data = out_template)
            output_file.close()
            saved_file.create_dataset(name_i,data = np.array([H,W]))

        saved_file.close()


    def zoom_img(self,input_img,ratio_):
        output_shape = int(input_img.shape[-1]*ratio_)

        return np.concatenate([self.zoom_img_(img,output_shape = output_shape)[np.newaxis,:,:] for img in input_img],0)
    def zoom_img_(self,input_img,output_shape):
        return input_img.reshape(input_img.shape[0],output_shape,-1).mean(-1).swapaxes(0,1).reshape(output_shape,output_shape,-1).mean(-1).swapaxes(0,1)
    def recon_img(self, input_img):
        return cv2.resize(cv2.resize(input_img.transpose(1,2,0),dsize=(self.shape1,self.shape1)),dsize = (self.output_shape , self.output_shape)).transpose(2,0,1)

    def efame2event(self,Efame):
        [tt,twi,thi,tai] = np.where(Efame==True)
        event = np.concatenate([tt[:,np.newaxis]/100, twi[:,np.newaxis]/W, thi[:,np.newaxis]/H, (tai[:,np.newaxis]-0.5)*2],axis=1)

        return event
    def __getitem__(self, index):

        index_s = index
        temp = 0
        while True:
            temp_path00 = '/data/zzy/LaSOTBenchmark/'
            evt_length = 200
            if self.TM == 'Train':
                data_idx = random.randint(0,len(self.train_name)-1)
                # name = 'airplane-' + self.idx_list[data_idx]
                name_i = self.train_name[data_idx]
                # print('name :{}'.format(name_i))
                try:
                    # template_name = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
                    template_name = '/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] + '/' +'template_'+ name_i +'.h5'
                    template = h5py.File(template_name, 'r')['template'][:]
                    # output_event = '/data/zzy/processed_event/'+name +'.h5'
                    output_event = '/data/zzy/output_event/'+ name_i.split('-')[0] + '/' + name_i +'.h5'
                    output_file = h5py.File(output_event, 'r')
                except:
                    time.sleep(0.2)
                    # template_name = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
                    # template = h5py.File(template_name, 'r')['template'][:]
                    # output_event = '/data/zzy/processed_event/'+name +'.h5'
                    # output_file = h5py.File(output_event, 'r')
                    template_name = '/data/zzy/EvtTimeStamp/'+ name_i.split('-')[0] + '/' +'template_'+ name_i +'.h5'
                    template = h5py.File(template_name, 'r')['template'][:]
                    # output_event = '/data/zzy/processed_event/'+name +'.h5'
                    output_event = '/data/zzy/output_event/'+ name_i.split('-')[0] + '/' + name_i +'.h5'
                    output_file = h5py.File(output_event, 'r')


                    print('load error:{}'.format(template_name))

                temp_path = '/data/zzy/LaSOTBenchmark/airplane/'

                # Bbox_path = temp_path + name + '/groundtruth.txt'
                # Bbox_path = temp_path + name + '/groundtruth.txt'
                Bbox_path = temp_path00+ name_i.split('-')[0] + '/' + name_i+ '/groundtruth.txt'

                gt_bbox = np.loadtxt(Bbox_path,delimiter=',')

                # temp_img = temp_path + name + '/img/00000001.jpg'
                # img = cv2.imread(temp_img)

                # H,W,_ = img.shape
                H,W = h5py.File('/data/zzy/EvtTimeStamp/img_size.h5','r')[name_i][:]

                A = 2
                T = int((len(output_file.keys()) - 1)*100)
                
                fam_idx = random.randint(evt_length*2,T)
                fam_idx_01 = int(fam_idx / 33.3333)
                fam_idx = int(fam_idx_01*33.3333)
                
                ground_truth = gt_bbox[fam_idx_01,:].astype(np.float)
                ground_truth_temp1 = gt_bbox[0,:].astype(np.float)


                curr1 = []
                for i in range(max(1,int((fam_idx-evt_length)/100-1)),max(2,int((fam_idx)/100+3))):
                    try:
                        curr1.append(output_file['%08d'%i][:] )
                    except:
                        print('error_2')
                        continue
                # print('name:{}, L:{}, H:{}, len of list:{}'.format(name_i, max(1,int((fam_idx-evt_length)/100-1)), max(2,int((fam_idx)/100+3)), len(curr1)))
                curr = np.vstack(curr1)
                curr = curr.astype(np.float_)
                curr[:,2] = curr[:,2] / 1e9

                curr_flag001 = curr[:,2] <= (fam_idx+1)/1000
                curr_flag002 = curr[:,2] >= (fam_idx-evt_length-1)/1000
                try:
                    assert((np.max(curr[:,2])+0.005) >= (fam_idx+1)/1000)
                    assert(np.min(curr[:,2]) <= (fam_idx-evt_length-1)/1000 or (fam_idx-evt_length-1)/1000 <= 0.01)
                except:
                    print('error_3')
                    o = 1
                    # print('index:{},load data range:{}/{}, cut range:{}/{}, curr len:{} range of data:{}/ {}'.format(fam_idx_01, max(1,int((fam_idx-evt_length)/100-1)), max(2,int((fam_idx)/100+1)), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000, len(curr1), np.min(curr[:,2]), np.max(curr[:,2])))
                curr_flag = curr_flag001 * curr_flag002
                curr = curr[curr_flag,:]

                curr = self.resample_point(curr, ground_truth, W, H)
                # curr[:,2] = curr[:,2] - np.min(curr[:,2])
                try:
                    curr[:,2] = curr[:,2] - curr[0,2]
                except:
                    print('error_4')
                    continue
                num_idx = 0
                template = self.resample_point(template, ground_truth_temp1, W, H, ratio = 0.9)

                output_gt = np.zeros_like(ground_truth)
                output_gt[0] = (ground_truth[0]+ground_truth[2]/2)/ W
                output_gt[1] = (ground_truth[1]+ground_truth[3]/2)/ H
                output_gt[2] = ground_truth[2]/ W
                output_gt[3] = ground_truth[3]/ H

                ground_truth_temp = np.zeros_like(ground_truth_temp1)
                ground_truth_temp[0] = (ground_truth_temp1[0]+ground_truth_temp1[2]/2)/ W
                ground_truth_temp[1] = (ground_truth_temp1[1]+ground_truth_temp1[3]/2)/ H
                ground_truth_temp[2] = ground_truth_temp1[2]/ W
                ground_truth_temp[3] = ground_truth_temp1[3]/ H
                # output_gt = np.zeros_like(ground_truth)
                # output_gt[0] = (ground_truth[0]+ground_truth[2]/2)
                # output_gt[1] = (ground_truth[1]+ground_truth[3]/2)
                # output_gt[2] = ground_truth[2]
                # output_gt[3] = ground_truth[3]

                # ground_truth_temp = np.zeros_like(ground_truth_temp1)
                # ground_truth_temp[0] = (ground_truth_temp1[0]+ground_truth_temp1[2]/2)
                # ground_truth_temp[1] = (ground_truth_temp1[1]+ground_truth_temp1[3]/2)
                # ground_truth_temp[2] = ground_truth_temp1[2]
                # ground_truth_temp[3] = ground_truth_temp1[3]

                n1,c = template.shape
                template[:,2] = template[:,2] / 1e9
                temp = 0
                iteration = 0
                while num_idx <= 500:

                    if temp >0:
                        rand_num_ = np.abs(np.random.randn(n1))
                        Tmep_idx = 1- np.abs(template[:,3])*2
                        Tmep_idx = np.argsort(Tmep_idx*rand_num_)[:100]
                        out_template01 = template[Tmep_idx,:].astype(np.float)
                        out_template = np.concatenate([out_template01,out_template[:-100,:]],axis = 0)
                    else:
                        pn1 = np.random.randint(0,n1,[10000])
                        out_template = template[pn1,:].astype(np.float)
                    num_idx = np.sum(np.abs(out_template[:,3]))
                    temp+=1
                    if temp % 1000 == 0 :
                        print('name: {}, num_point: {}, total point:{}'.format(name_i, num_idx, np.sum(np.abs(template[:,3]))))
                template = out_template
                try:
                    assert(out_template.shape[0] == 10000)
                except:
                    print('template of {}, num_idx:{}， size:{}'.format(name, num_idx,out_template.shape))
                    
                n2,c = curr.shape
                pn2 = np.random.randint(0,n2,[10000])
                curr = curr[pn2,:].astype(np.float)

                ratio = np.array([W,H,1,1])[:,np.newaxis]
                ratio2 = np.array([W,W,H,H])
                ratio_t = np.array([W,H,1,1,1])[:,np.newaxis]

                curr = curr.transpose([1,0]) / ratio
                template = template.transpose([1,0]) / ratio_t
                # print('max data:{}, min data:{}'.format(np.max(curr[2,:]), np.min(curr[2,:])))
                return (template, curr, output_gt,ground_truth, ground_truth_temp)
            else:
                
                index = index_s
                temp = temp + 1
                name_idx = 0
                accum_num = 0
                
                while True:
                    if index >= self.test_len[self.test_names[name_idx]]:
                        index = index - self.test_len[self.test_names[name_idx]]
                        name_idx = name_idx + 1
                    else:
                        break

                name = self.test_names[name_idx]
                fam_idx_01 = index - 1

                if index == 0:
                    self._update_template(self.test_names[name_idx])

                    ground_truth_temp1 = self.GT[0,:].astype(np.float)
                    ground_truth_temp = np.zeros_like(ground_truth_temp1)
                    ground_truth_temp[0] = (ground_truth_temp1[0]+ground_truth_temp1[2]/2)
                    ground_truth_temp[1] = (ground_truth_temp1[1]+ground_truth_temp1[3]/2)
                    ground_truth_temp[2] = ground_truth_temp1[2]
                    ground_truth_temp[3] = ground_truth_temp1[3]

                    template = self.template
                    template, ground_truth_temp_c = self.resample_point_normed( template,self.GT[0,:] , 1, 1, ratio = 0.999)

                    # print('template min dim0:{}, max dim0:{}, min dim1:{}, max dim1:{}'.format(template[0,:].min(),template[0,:].max(), template[1,:].min(), template[1,:].max()))

                    n1,c = template.shape
                    template[:,2] = template[:,2] - template[0,2]
                    template[:,2] = template[:,2] / 1e6
                    template[:,3] = template[:,3]*2 - 1

                    ratio_t = np.array([1,1,1,1,1])[:,np.newaxis]
                    template = template.transpose([1,0]) / ratio_t

                    return (name, template, ground_truth_temp)


                    
                while True:
                    # try:
                    #     ground_truth = self.GT[fam_idx_01,:]
                    # except:
                    #     continue
                    ground_truth = self.GT[fam_idx_01,:]
                    output_gt = np.zeros_like(ground_truth)
                    output_gt[0] = (ground_truth[0]+ground_truth[2]/2)
                    output_gt[1] = (ground_truth[1]+ground_truth[3]/2)
                    output_gt[2] = ground_truth[2]
                    output_gt[3] = ground_truth[3]

                    select_range = 1

                    # start_frame = self.fam_start[max(1, fam_idx_01)]
                    # start_frame_pre = self.fam_start[max(0,fam_idx_01 - select_range)]
                    start_frame = self.fam_start[max(2, fam_idx_01 + 1)]
                    start_frame_pre = self.fam_start[max(1,fam_idx_01 + 1- select_range)]

                    flag_0 = self.events_start <= start_frame
                    flag_1 = self.events_end >= start_frame_pre

                    flag = flag_0 * flag_1

                    idx_flag = np.where(flag==1)[0]

                    evets = [self.events[i] for i in idx_flag]

                    evets = np.concatenate(evets, axis = 0)


                    flag_01 = evets[:,2] <= start_frame
                    flag_02 = evets[:,2] >= start_frame_pre

                    curr = evets[flag_01*flag_02,:]

                    # flag_01 = self.events[:,2] <= start_frame
                    # flag_02 = self.events[:,2] >= start_frame_pre

                    # curr = self.events[flag_01*flag_02,:]

                    try:
                        curr[:,2] = curr[:,2] - curr[0,2]
                    except:
                        print(curr)
                        continue
                    curr[:,2] = curr[:,2]/1e6

                    #-----------------------time crop-----------------------
                    # max_t = np.max(curr[:,2])
                    # min_t = np.min(curr[:,2])

                    # rand_range = 1.2
                    # assert(select_range >= rand_range)

                    # min_t = min_t + (max_t - min_t) * (select_range - rand_range)
                    # flag = curr[:,2] >= min_t

                    # curr = curr[flag,:]

                    #-----------------------time crop-----------------------

                    curr[:,2] = curr[:,2] - np.min(curr[:,2])
                    curr[:,3] = curr[:,3]*2 - 1

                    num_idx = 0

                    temp = 0
                    iteration = 0

                    n2,c = curr.shape

                    curr = curr.astype(np.float)

                    curr = curr.transpose([1,0])

                    return (name, curr,output_gt)
            # else:
            #     # data_idx = random.randint(0,6)
            #     # self.test_data_idx = 0
            #     name = 'airplane-' + self.idx_list_test[self.test_data_idx]
            #     if index == 20:
            #         print(name)

            #     template = '/data/zzy/EvtTimeStamp/'+'template_'+name+'.h5'
            #     template = h5py.File(template, 'r')['template'][:]

            #     output_event = '/data/zzy/processed_event/'+name +'.h5'
            #     output_file = h5py.File(output_event, 'r')
            #     temp_path = '/data/zzy/LaSOTBenchmark/airplane/'

            #     Bbox_path = temp_path + name + '/groundtruth.txt'

            #     gt_bbox = np.loadtxt(Bbox_path,delimiter=',')

            #     temp_img = temp_path + name + '/img/00000001.jpg'
            #     img = cv2.imread(temp_img)

            #     H,W,_ = img.shape
            #     # H,W = h5py.File('/data/zzy/EvtTimeStamp/img_size.h5','r')[name][:]

            #     A = 2
            #     T = int((len(output_file.keys()) - 1)*100)
                
            #     # fam_idx = random.randint(evt_length*2,T)
            #     # fam_idx_01 = int(fam_idx / 33.3333)
            #     fam_idx_01 = index+1
            #     fam_idx = int(fam_idx_01*33.3333)
            #     # curr = [output_file['event%08d'%i][:] for i in range(max(1,int((fam_idx-evt_length)/100)),int((fam_idx)/100+1))]
            #     # curr = np.vstack(curr)

            #     # # curr = curr[int(fam_idx)%100:int(fam_idx)%100 + evt_length,:]
            #     # curr_flag001 = curr[:,2] <= fam_idx
            #     # curr_flag002 = curr[:,2] >= fam_idx-evt_length
            #     # curr_flag = curr_flag001 * curr_flag002

            #     # curr = curr[curr_flag,:]

            #     # curr[:,2] = curr[:,2] - curr[0,2]
            #     # print('index:{},load data range:{}/{}, cut range:{}/{}'.format(index, max(1,int((fam_idx-evt_length)/100-1)), int((fam_idx)/100+1), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000))
            #     up = min(max(2,int((fam_idx)/100+2)),int(len(output_file.keys())+1))
            #     lower = int((fam_idx-evt_length)/100-1)
            #     curr1 = []
            #     try:
            #         for i in range(max(1,min(lower, up-1)),up):
            #             curr1.append(output_file['event%08d'%i][:] )
            #     curr = np.vstack(curr1)


            #     try:
            #         curr_flag001 = curr[:,2] <= (fam_idx+1)/1000
            #         curr_flag002 = curr[:,2] >= (fam_idx-evt_length-1)/1000
            #         assert((np.max(curr[:,2])+0.01) >= (fam_idx+1)/1000)
            #         assert(np.min(curr[:,2]) <= (fam_idx-evt_length-1)/1000 or (fam_idx-evt_length-1)/1000 <= 0.01)
            #     except:
            #         curr_flag001 = curr[:,2] <= curr[:,2].max()
            #         curr_flag002 = curr[:,2] >= curr[:,2].min()
            #         print('index:{}, load data range:{}/{}, cut range:{}/{}, curr len:{}, range of data:{}/{}'.format(index, max(1,int((fam_idx-evt_length)/100-1)), max(2,int((fam_idx)/100+1)), (fam_idx+1)/1000, (fam_idx-evt_length-1)/1000, len(curr1), np.min(curr[:,2]), np.max(curr[:,2])))

            #     curr_flag = curr_flag001 * curr_flag002
            #     curr = curr[curr_flag,:]
            #     # print('shape of curr:{}'.format(curr.shape))
            #     curr[:,2] = curr[:,2] - curr[0,2]
                
            #     n1,c = template.shape
            #     # pn1 = np.random.randint(0,n1,[10000])
            #     # template = template[pn1,:].astype(np.float)
            #     num_idx = 0
            #     temp = 0
            #     while num_idx <= 500:
            #         # time.sleep(0.2)
            #         if temp >0:
            #             rand_num_ = np.abs(np.random.randn(n1))
            #             Tmep_idx = 1- np.abs(template[:,3])*2
            #             Tmep_idx = np.argsort(Tmep_idx*rand_num_)[:100]
            #             out_template01 = template[Tmep_idx,:].astype(np.float)
            #             out_template = np.concatenate([out_template01,out_template[:-100,:]],axis = 0)
            #         else:
            #             pn1 = np.random.randint(0,n1,[10000])
            #             out_template = template[pn1,:].astype(np.float)
                        
            #         num_idx = np.sum(np.abs(out_template[:,3]))
            #         temp+=1
            #     template = out_template
            #     try:
            #         assert(out_template.shape[0] == 10000)
            #     except:
            #         print('template of {}, num_idx:{}， size:{}'.format(name, num_idx,out_template.shape))

            #     n2,c = curr.shape

            #     pn2 = np.random.randint(0,n2,[20000])
            #     curr = curr[pn2,:].astype(np.float)
                
            #     ground_truth = gt_bbox[fam_idx_01,:].astype(np.float)
            #     # print(ground_truth)
            #     output_gt = np.zeros_like(ground_truth)
            #     output_gt[0] = (ground_truth[0]+ground_truth[2]/2)/ W
            #     output_gt[1] = (ground_truth[1]+ground_truth[3]/2)/ H
            #     output_gt[2] = ground_truth[2]/ W
            #     output_gt[3] = ground_truth[3]/ H

            #     ratio = np.array([W,H,evt_length,1])[:,np.newaxis]
            #     ratio2 = np.array([W,W,H,H])

            #     curr = curr.transpose([1,0]) / ratio
            #     template = template.transpose([1,0]) / ratio
            #     return (template, curr, output_gt,ground_truth)

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

        