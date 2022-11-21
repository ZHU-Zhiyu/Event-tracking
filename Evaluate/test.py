from event_loader_02 import Event_dataset
from TrackNet import ETracking_Net
from utils import load_checkpoint
from models import build_model
from config import get_config
from ConfigFunction import parse_option
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
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
        if 'Iou_Net' not in name:
            new_state_dict[name] = v
    
    state_dict_m.update(new_state_dict)
    model.load_state_dict(state_dict_m)
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
    setup_seed(200)
    factor_lr = 1
    opt = parse.parse_args()
    torch.cuda.set_device(opt.local_rank)

    Edataset = Event_dataset(output_shape=128,Training_mode='Test',data_name='CAVE',rank = opt.local_rank, generate_template=False)
    datalen = Edataset.__len__()
    Edataset_loader = DataLoader(Edataset,batch_size=opt.batch_size,shuffle=False, num_workers=1)

    device = torch.device('cuda:{}'.format(opt.local_rank))
    SwinTrans = build_model(config)
    SwinTrans.cuda().to(device)
    
    ETracKNet = ETracking_Net(SwinT = SwinTrans).cuda().to(device)

    Criterion = transt_loss(device)

    opt = parse.parse_args()
    optimzier = torch.optim.Adam(itertools.chain(ETracKNet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    T_max = (datalen//(1*opt.batch_size))*1000
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5, last_epoch=-1) 
    env=opt.envname
    os.mkdir('/data1/zhu_19/saved_mat/' + env)
    server='10.37.0.175'
    Loss_logger = VisdomPlotLogger('line',opts={'title':'loss2'},port=8100,env=env,server=server)
    Lossce_logger = VisdomPlotLogger('line',opts={'title':'loss_ce'},port=8100,env=env,server=server)
    Lossbbox_logger = VisdomPlotLogger('line',opts={'title':'loss_bbox'},port=8100,env=env,server=server)
    Lossgiou_logger = VisdomPlotLogger('line',opts={'title':'loss_giou'},port=8100,env=env,server=server)

    if conti == True:
        state_dict = torch.load('/home/zhu_19/evt_tracking/ablation/state_dict.pkl' , map_location='cpu')
        

        
        OutDict = state_dict['Dict']
        # print(OutDict)
        ETracKNet = load_model(ETracKNet, OutDict)
        print('loading the model')

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
        saved_rawGt = []
        saved_index = []
        Save_output = {}
        with torch.no_grad():
            for data_index in range(0,1):
                name_i_pre = None
                start = time.time()
                end = None
                preparation = None
                local_i = 0
                for name_i, index, Efame, GT, raw_gt in Edataset_loader:

                    torch.cuda.synchronize()
                    start = time.time()
                    if end is not None:
                        preparation = start - end
                    iteration += 1
                    Evtracker.eval()
                    # template = template.to(device).float()
                    Efame = Efame.to(device).float()
                    Efame2 = torch.clone(Efame).cpu().numpy()
                    GT = GT.to(device).float()
                    # GT_temp = GT_temp.to(device).float()
                    if name_i[0] != name_i_pre:
                        print('shape of initial event:{}'.format(Efame.shape))
                        Evtracker._init_box(GT[0,:])
                        Evtracker.encoding_template(Efame)
                        local_i = 0
                        iteration = iteration - 1
                        if name_i_pre != None:
                            scio.savemat('/data1/zhu_19/saved_mat/'+ env+'/' +name_i_pre +'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'raw_gt':np.stack(saved_rawGt,axis=0),
                            'GTBox':np.stack(saved_Gtbox, axis=0),'IOU':np.stack(all_iou, axis=0), 'index':np.stack(saved_index,axis=0)})
                            saved_box = []
                            saved_Gtbox = []
                            saved_rawGt = []
                            saved_index = []
                            saved_iou_all.append(np.mean(all_iou))
                            all_iou = []
                        name_i_pre = name_i[0]

                        continue

                    GT_box_notation = [{'boxes':box[None,:], "labels": torch.zeros(1,device = device).long()} for box in GT]
                    if GT.sum()>1e-5:

                        bbox, Pred_bboxs, prob, prob_phy, flow, Warpped_pos, X_pos= Evtracker.eval_data(Efame, GT)
                        OutBox = bbox[:,0,:]

                        GT_box = box_cxcywh_to_xyxy(GT.detach())
                        Out_Box = box_cxcywh_to_xyxy(OutBox.detach())
                        iou,_ = box_iou1(GT_box, Out_Box)
                        all_iou.append(iou.mean().detach().cpu().numpy())
                        noi_box.append(Pred_bboxs.detach().cpu().numpy())
                    # else:
                    #     continue
                    # if iteration % 20 ==0:
                    #     print('name:{},iter:{},index:{},IOU:{}, l1 box loss:{},Mean total:{}'.format(name_i[0], iteration, index.detach().cpu().numpy(), iou.mean(), 
                    #                                                                                     torch.abs(GT_box - Out_Box).sum(-1).mean(),np.mean(np.array(all_iou))))
                    saved_box.append(OutBox.detach().cpu().numpy())
                    saved_Gtbox.append(GT.detach().cpu().numpy())
                    saved_rawGt.append(raw_gt.detach().cpu().numpy())
                    saved_index.append(index.detach().cpu().numpy())
                    np.stack(saved_index,axis=0)
                    name_i_pre = name_i[0]
                    local_i += 1

                    torch.cuda.synchronize()
                    end = time.time()
                
                scio.savemat('/data1/zhu_19/saved_mat/'+ env+'/' +name_i[0] +'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0),'IOU':np.stack(all_iou, axis=0)})
                print('------------------Saved_name:{}, iter:{}, IOU:{}, l1 box loss:{}, Mean total:{}.------------------'.format(name_i_pre, iteration, 
                                                                                iou.mean(), torch.abs(GT_box - Out_Box).sum(-1).mean(),np.mean(np.array(all_iou))))
                saved_iou_all.append(np.mean(all_iou))
                print('Mean Iou:{}'.format(np.mean(np.array(all_iou))))
                print('Mean iou of different seqs:{}'.format(np.mean(saved_iou_all)))
                print('OP_05 of different seqs:{}'.format(np.mean(saved_iou_all>0.5)))
                print('OP_75 of different seqs:{}'.format(np.mean(saved_iou_all>0.75)))
                scio.savemat('/data1/zhu_19/saved_mat/'+name_i[0]+'PredBox.mat',{'PredBox':np.stack(saved_box,axis=0),'GTBox':np.stack(saved_Gtbox, axis=0),'noi_box':np.stack(noi_box, axis=0),'allbox':np.stack(saved_iou_all, axis=0)})


