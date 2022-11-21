from event_loader_02 import Event_dataset
from model import ETracking_Net

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
from util.box_ops import box_xywh_to_xyxy, box_iou1, box_cxcywh_to_xyxy, generalized_box_iou1

import os
import time
from torch.utils.data import DataLoader
import skimage.measure as skm
import skimage as ski
import random
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
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


def save_model(epoch,Renet,optimzier):
    model_state_dict = {'Dict':Renet.state_dict(),'Optim':optimzier.state_dict()}
    os.makedirs('./save_model'+now,exist_ok=True)
    torch.save(model_state_dict,'./save_model'+now+'/state_dicr_{}.pkl'.format(epoch))

# def load_model(model,mode_dict):
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     for k, v in mode_dict.items():
#         name = k[7:] # remove `module.`
#         # name = k # remove `module.`
#         new_state_dict[name] = v
#     model.load_state_dict(new_state_dict)
#     model.eval()
#     return model

def load_model(model,mode_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    pred_dict = model.state_dict()
    for k, v in mode_dict.items():
        name = k[7:] # remove `module.`
        # name = k # remove `module.`
        if name in pred_dict:
            new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    print('len of new_state_dict:{}'.format(len(new_state_dict)))
    pred_dict.update(new_state_dict)
    model.load_state_dict(pred_dict)
    model.eval()
    return model
def getbox(Conf, BBox):
    """ Conf : B, N, 1
        BBox : B, N, 4
    """
    B, N, C = BBox.shape
    Conf1 = Conf[:,:,0]
    indxBbox = Conf1.min(1)[-1]
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
    parse = argparse.ArgumentParser()
    parse.add_argument('--n_epochs', type = int, default = 10000, help = 'number of epochs of training')
    parse.add_argument('--batch_size',type = int, default = 22, help = 'size of the batches')
    parse.add_argument('--lr',type = float, default = 1e-4, help='learing rate of network')
    parse.add_argument('--b1',type = float, default = 0.9, help='adam:decay of first order momentum of gradient')
    parse.add_argument('--b2',type = float, default = 0.999, help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--envname',type = str, default = 'temp', help= 'adam: decay of first order momentum of gradient')
    parse.add_argument('--local_rank', type=int, default=0)
    factor_lr = 1
    opt = parse.parse_args()
    setup_seed(50+20*opt.local_rank)
    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )
    # Hyper_test = Hyper_dataset(output_shape=128,Training_mode='Test',data_name='CAVE')
    # Hyper_test = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4,pin_memory=True,
    #                                  drop_last=True)

    Edataset = Event_dataset(output_shape=128,Training_mode='Train',data_name='CAVE',rank = opt.local_rank, generate_template=False)
    datalen = Edataset.__len__()
    Hyper_train_sampler  = torch.utils.data.distributed.DistributedSampler(Edataset, shuffle=True)
    # Edataset_loader = DataLoader(Edataset,batch_size=opt.batch_size,shuffle=False, num_workers=4,pin_memory=True,
    #                                  drop_last=True)
    Edataset_loader = DataLoader(Edataset,batch_size=opt.batch_size, num_workers=5,pin_memory=True,
                                     drop_last=True,sampler=Hyper_train_sampler)
    # device = torch.device('cuda:{}'.format(0))
    device = torch.device('cuda:{}'.format(opt.local_rank))
    ETracKNet = ETracking_Net().cuda().to(device)
    # spenet = SpeNet().cuda().to(device)
    # Tracking_loss = TrackingMatcher().cuda().to(device)
    Criterion = transt_loss(device)

    opt = parse.parse_args()
    # optimzier = torch.optim.Adam(itertools.chain(spenet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    # optimzier = torch.optim.SGD(ETracKNet.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)
    # MaeLoss = torch.nn.L1Loss().to(device)   
    T_max = (datalen//(1*opt.batch_size))*1000

    env=opt.envname
    server='10.37.0.175'
    Loss_logger = VisdomPlotLogger('line',opts={'title':'loss2'},port=8100,env=env,server=server)
    IOU_logger = VisdomPlotLogger('line',opts={'title':'IOU'},port=8100,env=env,server=server)
    Lossbbox_logger = VisdomPlotLogger('line',opts={'title':'loss_ln'},port=8100,env=env,server=server)
    Lossgiou_logger = VisdomPlotLogger('line',opts={'title':'loss_giou'},port=8100,env=env,server=server)
    Loss_corr_pos_logger = VisdomPlotLogger('line',opts={'title':'loss_corr_pos'},port=8100,env=env,server=server)
    Loss_pred_iou_logger = VisdomPlotLogger('line',opts={'title':'loss_pred_iou'},port=8100,env=env,server=server)
    Loss_corr_neg_logger = VisdomPlotLogger('line',opts={'title':'loss_corr_neg'},port=8100,env=env,server=server)

    if conti == True:
        state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v21/save_model2021-11-07-12-06-58/state_dicr_750.pkl', map_location='cpu')
        
        # state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v21/run_folders/2021-11-13-13-15-03/save_model2021-11-13-13-15-05/state_dicr_250.pkl', map_location='cpu')
        OutDict = state_dict['Dict']
        # print(OutDict)
        # ETracKNet_state_dict = ETracKNet.state_dict()
        # ETracKNet_state_dict.update(OutDict)
        # ETracKNet.load_state_dict(ETracKNet_state_dict)
        ETracKNet = load_model(ETracKNet.cpu(), OutDict)
        del state_dict
        del OutDict
        ETracKNet = ETracKNet.cuda().to(device)
        # optimzier.load_state_dict(state_dict['Optim'])
        conti_epoch = 0
    else:
        conti_epoch = 0
    ETracKNet = DDP(ETracKNet, device_ids=[opt.local_rank], output_device=opt.local_rank,find_unused_parameters=True)
    optimzier = torch.optim.Adam(itertools.chain(ETracKNet.parameters()),lr = opt.lr,betas=(opt.b1,opt.b2),weight_decay=0)
    # optimzier = torch.optim.SGD(ETracKNet.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    if conti == True:
        state_dict = torch.load('/home/zhu_19/evt_tracking/evt_tracking_v17/run_folders/2021-10-26-17-03-27/save_model2021-10-26-17-03-29/state_dicr_990.pkl', map_location='cpu')
        # optimzier.load_state_dict(state_dict['Optim'])
        # optimzier.param_groups[0]['lr'] = 1e-5
        del state_dict
    schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimzier, T_max, eta_min=1e-5, last_epoch=-1) 
    torch.cuda.empty_cache()
    for epoch in range(conti_epoch, opt.n_epochs):
        batch = 0
        loss_ = []
        loss_ce_ = []
        loss_bbox_ = []
        loss_giou_ = []
        loss_corr_pos_ = []
        loss_corr_neg_ = []
        loss_iou_pred = []
        iteration = 0
        IOU = []
        for template, Efame, GT,ground_truth, GT_temp in Edataset_loader:
            # torch.cuda.empty_cache()
            iteration += 1
            ETracKNet.train()
            template = template.to(device).float()
            Efame = Efame.to(device).float()
            GT = GT.to(device).float()
            GT_temp = GT_temp.to(device).float()
            GT_box_notation = [{'boxes':box[None,:], "labels": torch.zeros(1,device = device).long()} for box in GT]
            # print('GT:{}, GT_template:{}'.format(GT[0,:], GT_temp[0,:]))
            # bbox, Class, Distance, corp_loss_pos, corp_loss_neg = ETracKNet(Efame, template, GT, GT_temp)
            bbox, Class, Distance, corp_loss_pos, corp_loss_neg, IOU_pred, IOU_GT = ETracKNet(Efame, template, GT, GT_temp)

            Iou_flag = IOU_GT >= 0.1
            Iou_flag = Iou_flag.float().detach()
            IOU_pred = IOU_pred[:,0,:]
            iou_pred_loss = torch.mean(torch.abs(IOU_pred - IOU_GT)*Iou_flag)/ (Iou_flag.mean() + 1e-6)
            # print('pred:{}'.format(IOU_pred.shape))
            # print('GT:{}'.format(IOU_GT.shape))
            OutBox = bbox[:,0,:]
            # print('max of Efame: dim1:{}, dim2:{},dim3:{},dim4:{}'.format(Efame[:,0,:].max(),Efame[:,1,:].max(),Efame[:,2,:].max(),Efame[:,3,:].max()))
            # print('max of template: dim1:{}, dim2:{},dim3:{},dim4:{}'.format(template[:,0,:].max(),template[:,1,:].max(),template[:,2,:].max(),template[:,3,:].max()))
            # optimized_loss = torch.mean(torch.abs(bbox[:,0,:] - GT).sum(-1))
            flag = GT.sum(-1)>=1e-5
            flag = flag.float().to(device)
            # optimized_loss = (bbox[:,0,:] - GT)**2
            optimized_loss = torch.abs(bbox[:,0,:] - GT)
            optimized_loss = optimized_loss.sum(-1)
            GT_box = box_cxcywh_to_xyxy(GT)
            OutBox = box_cxcywh_to_xyxy(OutBox)
            # print('GTBox:{},Outbox:{}'.format(GT_box,OutBox))
            # print('GT_box:{}, outbox:{}'.format(GT_box, OutBox))
            GIou_loss,iou_loss = generalized_box_iou1(GT_box, OutBox)
            iou_loss = -iou_loss +1
            iou_loss = iou_loss.mean()
            iou,_ = box_iou1(GT_box, OutBox)
            IOU.append(iou.mean().detach().cpu().numpy())
            # print(optimized_loss)
            # print(GIou_loss.shape)
            optimized_loss = torch.mean(optimized_loss*flag)
            GIou_loss = -(GIou_loss*flag).mean()
            # corp_loss_pos = torch.log(1-pos_01+1e-5) + torch.log(1-pos_02+1e-5)
            # corp_loss_pos = - torch.log(pos_01 +1e-5) - torch.log(pos_02 +1e-5) 
            # corp_loss_neg = - torch.log(1-neg_01 +1e-5) - torch.log(1-neg_02 +1e-5)
            # total = optimized_loss + GIou_loss*1 + (corp_loss_pos.mean() + corp_loss_neg.mean())*1
            total = optimized_loss + GIou_loss*1 + (corp_loss_pos.mean() + corp_loss_neg.mean())*1 + iou_pred_loss*1e2
            # total = optimized_loss + GIou_loss*5 
            # total = corp_loss_pos.mean() + corp_loss_neg.mean() + optimized_loss*0 + GIou_loss*0
            # print(total)
            optimzier.zero_grad()
            total.backward()
            optimzier.step()

            # total = optimized_loss.detach().cpu().item()
            loss_.append(total.detach().cpu().item())
            loss_giou_.append(GIou_loss.detach().cpu().item())
            # iou_pred_loss.detach().cpu().item()
            # iou_pred_loss = torch.nan_to_num(iou_pred_loss, nan=1.0)
            if iou_pred_loss == iou_pred_loss:
                loss_iou_pred.append(iou_pred_loss.detach().cpu().item())
            loss_bbox_.append(optimized_loss.detach().cpu().item())
            loss_corr_pos_.append(corp_loss_pos.mean().detach().cpu().item())
            loss_corr_neg_.append(corp_loss_neg.mean().detach().cpu().item())
            if opt.local_rank == 0:
                print('[iteration:{}/{}], total_loss:{:.2}, ln:{:.2}, Pos_m:{:.2}, Neg_m:{:.2}, Pos_x:{:.2}, Neg_x:{:.2}, GIOU:{:.2}, IOU:{:.2}, iou_net:{:.2}, ratio:{:.2}, e.g.pred/gt:{:.2}/{:.2}'.format(iteration, 
                                                                                    len(Edataset_loader),
                                                                                    total.detach().cpu().item(),
                                                                                    optimized_loss.detach().cpu().item(),
                                                                                    corp_loss_pos.mean(),
                                                                                    corp_loss_neg.mean(),
                                                                                    corp_loss_pos.min(),
                                                                                    corp_loss_neg.min(),
                                                                                    GIou_loss.detach().cpu().item(),
                                                                                    np.mean(np.array(IOU)),
                                                                                    iou_pred_loss.detach().cpu().item(),
                                                                                    Iou_flag.mean().detach().cpu().item(),
                                                                                    IOU_pred[0,0].detach().cpu().item(),
                                                                                    IOU_GT[0,0].detach().cpu().item(),
                                                                                    ))
                                                                                    # bbox[0,0,:].detach().cpu().numpy(),
                                                                                    # GT[0,:].detach().cpu().numpy()))
        if opt.local_rank == 0:

            Loss_logger.log(epoch,np.mean(np.array(loss_)))
            Lossbbox_logger.log(epoch,np.mean(np.array(loss_bbox_)))
            Lossgiou_logger.log(epoch,np.mean(np.array(loss_giou_)))
            IOU_logger.log(epoch,np.mean(np.array(IOU)))
            Loss_corr_pos_logger.log(epoch, np.mean(np.array(loss_corr_pos_)))
            Loss_pred_iou_logger.log(epoch,np.mean(np.array(loss_iou_pred)))
            Loss_corr_neg_logger.log(epoch, np.mean(np.array(loss_corr_neg_)))


            if epoch % 50 == 0:
                save_model(epoch,ETracKNet,optimzier)