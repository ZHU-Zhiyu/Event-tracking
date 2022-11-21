from event_loader_02 import Event_dataset
# import sys

# sys.path.append('/home/zhu_19/evt_tracking/evt_tracking_v21/run_folders/2021-11-16-20-14-57/')
# sys.path.append('/home/zhu_19/evt_tracking/evt_tracking_v21/run_folders/2021-11-14-21-07-37/')
# from model import ETracking_Net
from TrackNet import ETracking_Net
from utils import load_checkpoint
from models import build_model
from config import get_config
from ConfigFunction import parse_option
import torch
import argparse
import torch.nn.functional as F
from torch.nn.init import xavier_normal_ as x_init
import itertools
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomTextLogger
import visdom
from torch.utils import data
from torchvision.utils import make_grid
import numpy as np
from loss import transt_loss
from util.box_ops import box_xywh_to_xyxy, box_iou1, box_cxcywh_to_xyxy
import os
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import scipy.io as scio

from Tracker import Event_tracker
port = 8100
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
def xavier_init_(model):
    for name,param in model.named_parameters():
        if 'weight' in name:
            x_init(param)

def loss_spa(hsi,hsi_t,Mse):
    loss = Mse(hsi[1],hsi_t[1]) + Mse(hsi[2],hsi_t[2])
    return loss
def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            module.eval()

def save_model(epoch,Renet):
    model_state_dict = {'Dict':Renet.state_dict()}
    os.makedirs('./save_model'+now,exist_ok=True)
    torch.save(model_state_dict,'./save_model'+now+'/state_dicr_{}.pkl'.format(epoch))

def load_model(model,mode_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    state_dict_m = model.state_dict()

    for k, v in mode_dict.items():
        name = k[7:] # remove `module.`
        # name = k # remove `module.`
        # if name in state_dict_m:
        if 'Iou_Net' not in name:
            new_state_dict[name] = v
    
    state_dict_m.update(new_state_dict)
    model.load_state_dict(state_dict_m)
    # model.load_state_dict(new_state_dict)
    model.eval()
    return model

def getbox(Conf, BBox):
    """ Conf : B, N, 1
        BBox : B, N, 4
    """
    B, N, C = BBox.shape
    Conf1 = Conf[:,:,0]
    indxBbox = Conf1.max(1)[-1]
    OutRange = torch.range(0,B-1)*N
    indxBbox = indxBbox + OutRange.to(indxBbox.device)
    bbox1 = BBox.reshape([B*N, C])
    # indexBbox = indxBbox.long()
    # print(indxBbox.int())
    bbox1 = bbox1[indxBbox.long(),:]
    assert(bbox1.shape[0] == B)

    return bbox1


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
conti = True
if __name__ == '__main__':
    _, config = parse_option()
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_epochs', type = int, default = 1, help = 'number of epochs of training')
    parse.add_argument('--batch_size',type = int, default = 1, help = 'size of the batches')
    parse.add_argument('--lr',type = float, default = 1e-4, help='learing rate of network')
    parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
    parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--envname',type = str, default = 'temp', help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--update_inter',type = int, default = 2, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--search_size',type = float, default = 0.25, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--local_rank', type=int, default=0)
    setup_seed(50)
    factor_lr = 1
    opt = parse.parse_args()
    torch.cuda.set_device(opt.local_rank)
    # Hyper_test = Hyper_dataset(output_shape=128,Training_mode='Test',data_name='CAVE')
    # Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4,pin_memory=True,
    #                                  drop_last=True)

    Edataset = Event_dataset(output_shape=128,Training_mode='Test',data_name='CAVE',rank = opt.local_rank, generate_template=False)
    datalen = Edataset.__len__()
    Edataset_loader = DataLoader(Edataset,batch_size=opt.batch_size,shuffle=False, num_workers=1,pin_memory=False,
                                     drop_last=True)

    device = torch.device('cuda:{}'.format(opt.local_rank))
    SwinTrans = build_model(config)
    SwinTrans.cuda().to(device)
    # max_accuracy = load_checkpoint(config, SwinTrans)
    
    ETracKNet = ETracking_Net(SwinT = SwinTrans).cuda().to(device)


    # device = torch.device('cuda:{}'.format(0))
    # device = torch.device('cpu')
    # ETracKNet = ETracking_Net().to(device)
    # spenet = SpeNet().cuda().to(device)
    # Tracking_loss = TrackingMatcher().cuda().to(device)
    Criterion = transt_loss(device)

    opt = parse.parse_args()
    # optimzier = torch.optim.Adam(itertools.chain(spenet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    optimzier = torch.optim.Adam(itertools.chain(ETracKNet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    # MaeLoss = torch.nn.L1Loss().to(device)   
    T_max = (datalen//(1*opt.batch_size))*1000
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5, last_epoch=-1) 

    env=opt.envname
    server='10.37.0.175'
    Loss_logger = VisdomPlotLogger('line',opts={'title':'loss2'},port=8100,env=env,server=server)
    Lossce_logger = VisdomPlotLogger('line',opts={'title':'loss_ce'},port=8100,env=env,server=server)
    Lossbbox_logger = VisdomPlotLogger('line',opts={'title':'loss_bbox'},port=8100,env=env,server=server)
    Lossgiou_logger = VisdomPlotLogger('line',opts={'title':'loss_giou'},port=8100,env=env,server=server)

    # b_r = 0
    # ssim_best = 0
    # psnr_best = 0
    if conti == True:
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v13/run_folders/2021-10-22-17-28-54/save_model2021-10-22-17-28-57/state_dicr_480.pkl')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_real_data/run_folders/2021-11-04-11-57-53/save_model2021-11-04-11-57-55/state_dicr_0.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v17/run_folders/2021-10-26-17-03-27/save_model2021-10-26-17-03-29/state_dicr_990.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v18/run_folders/2021-11-07-12-06-55/save_model2021-11-07-12-06-58/state_dicr_750.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v19/run_folders/2021-11-10-15-54-03/save_model2021-11-10-15-54-05/state_dicr_200.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v23/run_folders/2021-11-25-20-37-35/save_model2021-11-25-20-37-38/state_dicr_500.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_refine_real/run_folders/2021-12-02-15-47-34/save_model2021-12-02-15-47-36/state_dicr_180.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_refine_real/run_folders/2021-12-03-13-21-47/save_model2021-12-03-13-21-49/state_dicr_180.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_refine_real/run_folders/2021-12-04-11-23-14/save_model2021-12-04-11-23-16/state_dicr_400.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin/run_folders/2021-12-18-17-41-03/save_model2021-12-18-17-41-05/state_dicr_1680.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_real_refine/run_folders/2021-12-23-14-29-09/save_model2021-12-23-14-29-13/state_dicr_140.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_real_refine/run_folders/2021-12-25-15-11-33/save_model2021-12-25-15-11-36/state_dicr_140.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_real_refine/run_folders/2021-12-23-14-29-09/save_model2021-12-23-14-29-13/state_dicr_140.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_data_augment/run_folders/2022-01-04-10-19-32/save_model2022-01-04-10-19-34/state_dicr_200.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_data_augment/run_folders/2022-01-05-18-59-51/save_model2022-01-05-18-59-53/state_dicr_120.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_data_augment/run_folders/2022-01-15-17-15-42/save_model2022-01-15-17-15-45/state_dicr_100.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_Back_Flow/run_folders/2022-01-22-13-42-31/save_model2022-01-22-13-42-34/state_dicr_10.pkl' , map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_Back_Flow_02/run_folders/2022-01-23-21-19-39/save_model2022-01-23-21-19-41/state_dicr_38.pkl' , map_location='cpu')
        state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_Back_Flow_02/run_folders/2022-01-24-22-16-09/save_model2022-01-24-22-16-11/state_dicr_29.pkl' , map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/Evt_Swin_data_augment/run_folders/2022-01-12-17-30-18/save_model2022-01-12-17-30-20/state_dicr_100.pkl', map_location='cpu')
        
        
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_refine_real_dynamic/run_folders/2021-12-12-21-15-15/save_model2021-12-12-21-15-19/state_dicr_100.pkl', map_location='cpu')

        
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v21/run_folders/2021-11-16-20-14-57/save_model2021-11-16-20-14-59/state_dicr_250.pkl', map_location='cpu')
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_real_data/run_folders/2021-11-04-20-50-21/save_model2021-11-04-20-50-23/state_dicr_160.pkl', map_location='cpu')
        
        OutDict = state_dict['Dict']
        # print(OutDict)
        ETracKNet = load_model(ETracKNet, OutDict)
        print('loading the model')
        # ETracKNet_state_dict = ETracKNet.state_dict()
        # ETracKNet_state_dict.update(OutDict)
        # ETracKNet.load_state_dict(ETracKNet_state_dict)
        del state_dict
    Evtracker = Event_tracker(ETracKNet,opt)
    for epoch in range(opt.n_epochs):
        batch = 0
        loss_ = []
        loss_ce_ = []
        loss_bbox_ = []
        loss_giou_ = []
        iteration = 0
        saved_box = []
        saved_Gtbox = []
        all_iou = []
        noi_box = []
        saved_iou_all = []
        with torch.no_grad():
            for data_index in range(0,1):
                # Edataset.test_data_idx = 0
                # Edataset_loader = DataLoader(Edataset,batch_size=opt.batch_size,shuffle=False, num_workers=10,pin_memory=True,
                #                                 drop_last=True)
                # data_name = 'airplane-' + Edataset.idx_list_test[Edataset.test_data_idx]
                name_i_pre = None
                start = time.time()
                end = None
                preparation = None
                for name_i, Efame, GT in Edataset_loader:
                    # print('loop')
                    # name_i = 'name of data'

                    torch.cuda.synchronize()
                    start = time.time()
                    if end is not None:
                        preparation = start - end
                    iteration += 1
                    Evtracker.eval()
                    # template = template.to(device).float()
                    Efame = Efame.to(device).float()
                    GT = GT.to(device).float()
                    # GT_temp = GT_temp.to(device).float()
                    if name_i[0] != name_i_pre:
                        print('shape of initial event:{}'.format(Efame.shape))
                        Evtracker._init_box(GT[0,:])
                        Evtracker.encoding_template(Efame)
                        iteration = iteration - 1
                        if name_i_pre != None:
                            scio.savemat('/home/zhu_19/saved_mat/'+name_i_pre+'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0),'IOU':np.stack(all_iou, axis=0)})
                            print('------------------Saved_name:{}, iter:{}, IOU:{}, l1 box loss:{}, Mean total:{}.------------------'.format(name_i_pre, iteration, 
                                                                                iou.mean(), torch.abs(GT_box - Out_Box).sum(-1).mean(),np.mean(np.array(all_iou))))
                            saved_box = []
                            saved_Gtbox = []
                            saved_iou_all.append(np.mean(all_iou))
                            all_iou = []
                        name_i_pre = name_i[0]

                        continue
                    # scio.savemat('/home/zhu_19/saved_mat/'+ str(iteration) +'Efam.mat',{'Efam': Efame.detach().cpu().numpy(), 'Temp': template.detach().cpu().numpy()})
                    # print('min seq 1:{:.2}, 2：{:.2}, 3:{:.2}, 4:{:.2}, temp 1:{:.2}, 2：{:.2}, 3:{:.2}, 4:{:.2}, max seq 1:{:.2}, 2：{:.2}, 3:{:.2}, 4:{:.2}, temp 1:{:.2}, 2：{:.2}, 3:{:.2}, 4:{:.2}.'.format(
                    #                 Efame[:,0,:].min(), Efame[:,1,:].min(), Efame[:,2,:].min(), Efame[:,3,:].min(),
                    #                 template[:,0,:].min(), template[:,1,:].min(), template[:,2,:].min(), template[:,3,:].min(),
                    #                 Efame[:,0,:].max(), Efame[:,1,:].max(), Efame[:,2,:].max(), Efame[:,3,:].max(),
                    #                 template[:,0,:].max(),template[:,1,:].max(),template[:,2,:].max(),template[:,3,:].max()))
                    # print('GT of temp:{}'.format(GT_temp))
                    # if name_i[0] != name_i_pre:

                    GT_box_notation = [{'boxes':box[None,:], "labels": torch.zeros(1,device = device).long()} for box in GT]
                    if GT.sum()>1e-5:
                        # bbox, Pred_bboxs, Distance, _, _= Evtracker.eval_data(Efame,template, GT, GT_temp)
                        bbox, Pred_bboxs, _, _, _= Evtracker.eval_data(Efame, GT)
                        # OutBox = getbox(Class,bbox)
                        OutBox = bbox[:,0,:]
                        # OutBox = bbox[:,0,:]
                        # scio.savemat('/home/zhu_19/saved_mat/'+name_i[0]+'__'+ str(iteration) +'Transmision_matrix.mat',{'H_01':H_01.detach().cpu().numpy(),'H_02':H_02.detach().cpu().numpy()})
                        

                        GT_box = box_cxcywh_to_xyxy(GT.detach())
                        Out_Box = box_cxcywh_to_xyxy(OutBox.detach())
                        iou,_ = box_iou1(GT_box, Out_Box)
                        all_iou.append(iou.mean().detach().cpu().numpy())
                        noi_box.append(Pred_bboxs.detach().cpu().numpy())
                    # else:
                    #     continue
                    if iteration % 20 ==0:
                        print('name:{},iter:{}:IOU:{}, l1 box loss:{},Mean total:{}'.format(name_i[0], iteration, iou.mean(), 
                                                                                                        torch.abs(GT_box - Out_Box).sum(-1).mean(),np.mean(np.array(all_iou))))
                    saved_box.append(OutBox.detach().cpu().numpy())
                    saved_Gtbox.append(GT.detach().cpu().numpy())
                    name_i_pre = name_i[0]
                    # if iteration % 500 == 0:
                    #     print('Mean Iou:{}'.format(np.mean(np.array(all_iou))))
                    #     scio.savemat('/home/zhu_19/saved_mat/'+data_name+'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0)})
                    #     assert(0==1)
                    # if iteration % 500 == 0:
                    torch.cuda.synchronize()
                    end = time.time()
                    # if preparation is not None:
                    #     print('data running time:{} / preparation: {}'.format(end - start, preparation))

                saved_iou_all.append(np.mean(all_iou))
                print('Mean Iou:{}'.format(np.mean(np.array(all_iou))))
                print('Mean iou of different seqs:{}'.format(np.mean(saved_iou_all)))
                scio.savemat('/home/zhu_19/saved_mat/'+name_i[0]+'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0),'noi_box':np.stack(noi_box, axis=0)})
                        # assert(0==1)

                # print('Mean Iou:{}'.format(np.mean(np.array(all_iou))))
                # scio.savemat('/home/zhu_19/saved_mat/'+data_name+'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0)})
                
            #     optimized_loss = loss['loss_ce']  +  5*loss['loss_bbox'] +  2*loss['loss_giou']
            #     optimized_loss = optimized_loss.mean()
            #     optimzier.zero_grad()
            #     optimized_loss.backward()
            #     optimzier.step()
            #     loss_ce = loss['loss_ce'].detach().cpu().item()
            #     loss_bbox = loss['loss_bbox'].detach().cpu().item()
            #     loss_giou = loss['loss_giou'].detach().cpu().item()
            #     total = optimized_loss.detach().cpu().item()
            #     loss_.append(total)
            #     loss_ce_.append(loss_ce)
            #     loss_bbox_.append(loss_bbox)
            #     loss_giou_.append(loss_giou)
            #     print('[iteration:{}/{}]loss_ce:{}, loss_bbox:{},loss_giou:{}, total_loss:{}'.format(iteration, 
            #                                                                         len(Edataset_loader),
            #                                                                         loss_ce,
            #                                                                         loss_bbox,
            #                                                                         loss_giou,
            #                                                                         total))
            # Loss_logger.log(epoch,np.mean(np.array(loss_)))
            # Lossce_logger.log(epoch,np.mean(np.array(loss_ce_)))
            # Lossbbox_logger.log(epoch,np.mean(np.array(loss_bbox_)))
            # Lossgiou_logger.log(epoch,np.mean(np.array(loss_giou_)))
        # if epoch % 10 == 0:
        #     save_model(epoch,ETracKNet)

            # print(loss.keys())


        #     hsi_resize = hsi_resize.cuda().float()
        #     hsi_resize = torch.nn.functional.interpolate(hsi_resize,scale_factor=(8,8),mode='bilinear')
        #     hsi_g = hsi_g.cuda().float()
        #     hsi = hsi_resize
        #     scale_g = nn.AdaptiveAvgPool2d(1)(hsi_g)
        #     msi = [a.cuda().float().to(device) for a in msi]
        #     hsi_spe,scale,refined = spenet(hsi.to(device),msi[-1])
        #     scale_a = nn.AdaptiveAvgPool2d(1)(scale)
        #     hsi_2 = hsi_spe[-1][:,:31,:,:]
        #     optimzier.zero_grad()
        #     hsi_g_ = hsi_g - hsi
        #     hsi_g_ = hsi_g_ - torch.nn.AdaptiveAvgPool2d(1)(hsi_g_)

        #     loss = MaeLoss(fout.to(device),hsi_g_.to(device))+MaeLoss(refined.to(device),hsi_g.to(device))
        #     loss.backward()
        #     optimzier.step()
        #     schduler.step()
        #     loss_.append(loss.item())

    # if opt.local_rank == 0:
    #     save_model(epoch,spenet)


