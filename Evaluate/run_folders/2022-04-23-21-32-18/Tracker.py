import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.autograd import Variable, Function
# from knn_cuda import KNN
import scipy.io as scio
# from kmeans_pytorch import kmeans, kmeans_predict
import numpy as np

class Event_tracker(Module):
    def __init__(self,Tracking_net = None, opt=None):
        super(Event_tracker, self).__init__()

        self.tracking_net = Tracking_net
        self.indx = 1
        self.sample_idx = 1
        self.sample_range = 1
        
        self.Tempalate_fea = None
        self.Tempalate_fea_mid = None
        self.Tempalate_fea_stable = None
        self.Tempalate_fea_variad = None
        self.pred2curr_box = None
        self.update_inter = opt.update_inter
        self.search_size = opt.search_size
    def _apply_mask(self, Fea, Pnt, GT, Temp=False):

        Dist = (Pnt[:,:2,:] - GT[:,:2][:,:,None])**2
        Dist = Dist.sum(1, keepdim=True)
        if Temp == True:
            Dist = Dist + Pnt[:,2,:][:,None,:]**2
        else:
            Dist = Dist + (Pnt[:,2,:].max(1, keepdim = True)[0] - Pnt[:,2,:])[:,None,:]**2
        Dist = Dist.sqrt()*2
        
        Norm = (GT[:,2:][:,:,None]/2)**2
        Norm = Norm.sum(1, keepdim=True).sqrt()

        Dist = Dist/ (Norm + 1e-6)

        mask = torch.exp(Dist.detach())
        mask = 1/(mask+1e-6)
        Fea01 = Fea * mask
        return Fea01
        
    def _init_box(self,box):
        self._previous_pred = box
        self.sample_idx = 1

    def _Disappear(self):
        self._previous_pred = None

    def _Crop_input_resmaple01(self, Seq):
        if self._previous_pred == None:
            return Seq

        else:
            previous_pred = self._previous_pred
            previous_pred = previous_pred.to(Seq.device)

            search_size = self.search_size
            crop_range_low = previous_pred[:2] - previous_pred[2:]/2 - search_size / 2
            crop_range_low = torch.clamp(crop_range_low, 0, 1)
            
            crop_range_high= crop_range_low + previous_pred[2:] + search_size
            crop_range_high = torch.clamp(crop_range_high, 0, 1)

            crop_range_low = crop_range_high - previous_pred[2:] - search_size
            crop_range_low = torch.clamp(crop_range_low, 0, 1)

            flag1 = Seq[0,0,:] >= crop_range_low[0]
            flag2 = Seq[0,1,:] >= crop_range_low[1]
            flag3 = Seq[0,0,:] <= crop_range_high[0]
            flag4 = Seq[0,1,:] <= crop_range_high[1]

            flag = flag1*flag2*flag3*flag4

            output = Seq[:,:,flag]

            output[:,0,:] = (output[:,0,:] - crop_range_low[0]) / (crop_range_high[0] - crop_range_low[0] + 1e-6)
            output[:,1,:] = (output[:,1,:] - crop_range_low[1]) / (crop_range_high[1] - crop_range_low[1] + 1e-6)

            B, C, N = output.shape
            
            if self.indx < 5:
                t_min = Seq[0,2,:].min()
                t_max = Seq[0,2,:].max()

                t1 = t_min + 0.1*(t_max - t_min)

                flag00 = output[:,2,:] <= t1

                temp_point = output[:,:,flag00[0,:]]
                np.savetxt('/home/zhu_19/evt_tracking/points/target'+str(self.indx)+'.xyz', temp_point[0,:3,:].permute([1,0]).detach().cpu().numpy())


            pn_ = torch.randint(0,N,[10000]).to(output.device)

            output = output[:,:,pn_]

            return output , crop_range_low , crop_range_high

    def scale_transformation_l2g(self, Box, range_low, range_high):
        output = Box
        output[:,:,0] = output[:,:,0] *(range_high[0] - range_low[0]) + range_low[0]
        output[:,:,1] = output[:,:,1] *(range_high[1] - range_low[1]) + range_low[1]
        output[:,:,2] = output[:,:,2] *(range_high[0] - range_low[0])
        output[:,:,3] = output[:,:,3] *(range_high[1] - range_low[1])

        return output

    def scale_transformation_g2l(self, Box, range_low, range_high):
        output = Box
        output[0] = (output[0] - range_low[0]) / (range_high[0] - range_low[0] + 1e-6)
        output[1] = (output[1] - range_low[1]) / (range_high[1] - range_low[1]+ 1e-6)
        output[2] = output[2] / (range_high[0] - range_low[0]+ 1e-6)
        output[3] = output[3] / (range_high[1] - range_low[1]+ 1e-6)

        return output



    def eval_data(self,X, GT, Temp=None, GT_temp=None):
        X0 = X.clone()
        try:
            X, range_low, range_high = self._Crop_input_resmaple01(X)
        except:
            # print('Indx:{} Crop Error happen !!!'.format(self.indx))
            temp = self._previous_pred
            return temp[None,None,:].to(X.device), temp[None,None,:].to(X.device), torch.zeros(20).to(X.device), torch.zeros(20).to(X.device), torch.zeros(20).to(X.device)

        self.pred2curr_box = self.scale_transformation_g2l(self._previous_pred.clone(), range_low, range_high )

        if X.shape[2] >= 300:

            bbox, bboxs, prob, X_pos = self._forward_with_template(X)


            output = self.scale_transformation_l2g(bbox, range_low, range_high)
            output = output[0,0,:]
            bboxs = self.scale_transformation_l2g(bboxs[None,:,:].clone(), range_low, range_high)

            X_pos[:,0,:] = X_pos[:,0,:] *(range_high[0] - range_low[0]) + range_low[0]
            X_pos[:,1,:] = X_pos[:,1,:] *(range_high[1] - range_low[1]) + range_low[1]

            scio.savemat('/home/zhu_19/evt_tracking/saved_file02/'+str(self.indx)+'.mat',{'X':X0.detach().cpu().numpy(),'GT':GT.detach().cpu().numpy(),
                                                                                        'output':output.detach().cpu().numpy(),
                                                                                        'prob': prob.detach().cpu().numpy(), 'X_pos':X_pos.detach().cpu().numpy(),
                                                                                        'Noi':bboxs.detach().cpu().numpy(),
                                                                                        'croplow':range_low.detach().cpu().numpy(),
                                                                                        'crophigh':range_high.detach().cpu().numpy()})
            if self.indx < 5:
                np.savetxt('/home/zhu_19/evt_tracking/points/'+str(self.indx)+'.xyz', X0[0,:3,:].permute([1,0]).detach().cpu().numpy())
            self._previous_pred = output

            self.indx = self.indx + 1
            self.sample_idx = self.sample_idx + 1
            return output[None,None,:], output[None,None,:], 0, 0, 0
        else:
            temp = self._previous_pred
            return temp[None,None,:].to(X.device), temp[None,None,:].to(X.device), torch.zeros(20).to(X.device), torch.zeros(20).to(X.device), torch.zeros(20).to(X.device)

    def encoding_template(self, Temp):
        
        assert(Temp.shape[1] == 5)

        B,C,P = Temp.shape
        temp_point = Temp[:,4,:].abs().sum()
        ratio = temp_point / (B * P)
        Temp = self.tracking_net.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))

        Emb_Temp, _ = self.tracking_net.SwinTrans.track_embedding(Temp.permute([0,2,1]))

        Emb_Temp = self.tracking_net._ensemble_fea([Emb_Temp[-1].permute([0,2,1]) , Emb_Temp[-2].permute([0,2,1]), Emb_Temp[-3].permute([0,2,1]), Emb_Temp[-4].permute([0,2,1])])

        Temp_pos = Emb_Temp[:,:3,:]
        Emb_Temp = Emb_Temp[:,5:,:]


        B,C,P = Temp_pos.shape

        # Temp = self.tracking_net.SelfTransFormer1(Temp)
        # self.Tempalate_fea = {'Temp': Temp[:,:,:2*P], 'Temp_pos': Temp_pos[:,:,:2*P]}
        print('Encoding template !! shape of Emb_Temp:{}'.format(Emb_Temp.shape))
        self.Tempalate_fea = {'Temp': Emb_Temp, 'Temp_pos': Temp_pos}
        # self.Tempalate_fea = {'Temp': Temp, 'Temp_pos': Temp_pos}
        self.Tempalate_fea_mid = None
        self.Tempalate_fea_stable = None
        self.Tempalate_fea_variad = None

        return True
    
    def _update_template(self, X_fea, X_pos):

        assert(self.Tempalate_fea is not None)
        
        Temp = self.Tempalate_fea['Temp']
        Temp_pos = self.Tempalate_fea['Temp_pos']
        B,C,P = Temp.shape
        if self.Tempalate_fea_stable == None:
            template_inter = self.update_inter
            Temp_0 = Temp[:,:,::template_inter]
            Temp_pos_0 = Temp_pos[:,:,::template_inter]
            self.Tempalate_fea_stable = {'Temp':Temp_0, 'Temp_pos': Temp_pos_0}

            Temp_indx_ones = torch.ones([P]).bool()
            Temp_indx_ones[::template_inter] = False

            Temp_1 = Temp[:,:,Temp_indx_ones]
            Temp_pos_1 = Temp_pos[:,:,Temp_indx_ones]


            self.Tempalate_fea_variad = {'Temp': Temp_1,'Temp_pos': Temp_pos_1}
        
        inter = 3
        fea_stable_0 = self.Tempalate_fea_stable['Temp']
        fea_stable_pos_0 = self.Tempalate_fea_stable['Temp_pos']
        fea_variad_0 = self.Tempalate_fea_variad['Temp']
        fea_variad_pos_0 = self.Tempalate_fea_variad['Temp_pos']
        B,C,P = fea_variad_0.shape
        Temp_indx_ones = torch.ones([P]).bool()
        Temp_indx_ones[::inter] = False
        fea_variad_1 = fea_variad_0[:,:,Temp_indx_ones]
        fea_variad_pos_1 = fea_variad_pos_0[:,:,Temp_indx_ones]

        B,C,PX = X_fea.shape

        X_fea_0 = X_fea[:,:,::inter]
        X_pos_0 = X_pos[:,:,::inter]

        fea_variad_1 = torch.cat([fea_variad_1, X_fea_0],dim=2)
        fea_variad_pos_1 = torch.cat([fea_variad_pos_1, X_pos_0],dim=2)
        self.Tempalate_fea_variad = {'Temp':fea_variad_1, 'Temp_pos':fea_variad_pos_1}

        self.Tempalate_fea_mid = {'Temp': torch.cat([fea_stable_0, fea_variad_1], dim=2), 'Temp_pos': torch.cat([fea_stable_pos_0, fea_variad_pos_1], dim=2)}

        return True
    def _resample_point(self,fea,fea_pos,clus_idx,num):
        fea_ = fea[:,:,clus_idx]
        fea_pos_ = fea_pos[:,:,clus_idx]

        B,C,P = fea_.shape
        
        idx = torch.randint(0,P,[num])

        fea_out = fea_[:,:,idx]
        pos_out = fea_pos_[:,:,idx]

        assert(pos_out.shape[1] == 3)

        out = torch.cat([pos_out, fea_out], dim = 1)

        return out

    def _physical_probability(self, pre_box, X_pos):
        '''
        pre_box in shape of [B, 4]
        X_pos in shpae of [B, 2, P]
        Out:
        prob_out = [B,1,P]
        '''

        dist_p2b = X_pos - pre_box[:,:2,None]

        dist_p2b = dist_p2b**2
        dist_p2b = dist_p2b / (pre_box[:,2:,None] / 2 + 1e-6)**2
        dist_p2b = dist_p2b.sum(dim=1) / 2
        dist_p2b = dist_p2b.sqrt()

        prob_out = 1 - 1.1 * dist_p2b
        prob_out = torch.clamp(prob_out, min = 0, max = 1)
        return prob_out[:,None,:]

    def _cluster_resample(self, X, X_pos, prob, clus_num = 3):
        
        clsuter_fea = torch.cat([X_pos[:,:2,:], prob], dim = 1)
        kmeans_cpu = kmeans(X=clsuter_fea[0,:,:].permute([1,0]), num_clusters=clus_num, device=torch.device('cpu'),tqdm_flag=False)
        kmeans_idx = kmeans_cpu[0].cuda()

        B,C,P = X.shape

        clus_idx = [ kmeans_idx == i for i in range(0, clus_num)]

        clus_fea =  [ self._resample_point(fea = X*prob, fea_pos = X_pos, clus_idx = idx, num = P) for idx in clus_idx]

        clus_fea = torch.cat(clus_fea, dim = 0)
            
        return clus_fea[:,3:,:], clus_fea[:,:3,:]

    def _learning_box(self, X1, X_pos):
        B,C,P = X1.shape
        if self.sample_idx >= self.sample_range:

            cat_fea = X1
            cat_pos = X_pos
            total = 1
            weighted_fea =self.tracking_net._weight_point(cat_fea,cat_pos)
            bbox = self.tracking_net.Regression_layer(weighted_fea)
            Pred_bbox = bbox[:,:,0].reshape([total, B, 4]).permute([1,2,0])

        else:
            Pred_bbox = self.pred2curr_box[None,:,None]

        return Pred_bbox

    def forward(self,X, Temp, GT, GT_temp, Efame_pre, Box_pre):

        assert(torch.isnan(X).sum()==0)
        assert(torch.isnan(Temp).sum()==0)
        assert(X.shape[1] == 4)
        assert(Temp.shape[1] == 5)

        B,C,P = X.shape
        X0 = torch.zeros([B,1,P],device = X.device).float()
        X = torch.cat([X,X0],dim = 1)
        X = self.extractor(X,out_num=1024,Is_temp=False)

        B,C,P = Temp.shape
        temp_point = Temp[:,4,:].abs().sum()
        ratio = temp_point / (B * P)
        Temp = self.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        self.SwinTrans.eval()
        Emb_X_, _ = self.SwinTrans.track_embedding(X.permute([0,2,1]))
        Emb_Temp, _ = self.SwinTrans.track_embedding(Temp.permute([0,2,1]))

        Emb_X = self._ensemble_fea([Emb_X_[-1].permute([0,2,1]).clone() , Emb_X_[-2].permute([0,2,1]).clone() , Emb_X_[-3].permute([0,2,1]).clone() , Emb_X_[-4].permute([0,2,1]).clone() ])
        Emb_Temp = self._ensemble_fea([Emb_Temp[-1].permute([0,2,1]) , Emb_Temp[-2].permute([0,2,1]), Emb_Temp[-3].permute([0,2,1]), Emb_Temp[-4].permute([0,2,1])])
        
        # Back warp warpping the event cloud and compute the CD loss
        Ensemble_X = self._ensemble_low_fea([X, Emb_X_[-4].permute([0,2,1]), Emb_X_[-3].permute([0,2,1])])
        Ensemble_X = self._Backward_warp(Ensemble_X)
        Warpped_pos = Ensemble_X[:,:2,:]

        prob_phy = self._regress_phy_pro(Warpped_pos, Box_pre)

        Cd_loss1, Cd_loss2 = self._point_cd_loss(torch.cat([Ensemble_X[:,:2,:],Ensemble_X[:,3:4,:],Ensemble_X[:,2:3,:]], dim=1), torch.cat([Efame_pre[:,:2,:],Efame_pre[:,3:4,:]], dim=1))

        X_pos = Emb_X[:,:3,:]
        Temp_pos = Emb_Temp[:,:3,:]
        Emb_X = Emb_X[:,5:,:]
        Emb_Temp = Emb_Temp[:,5:,:]

        X_pos_emb = self.pos_embedding(X_pos)
        Temp_pos_emb = self.pos_embedding(Temp_pos)
        Emb_X = self.SelfTransFormer1(Emb_X + X_pos_emb)
        Emb_Temp = self.SelfTransFormer1(Emb_Temp + Temp_pos_emb)

        Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(Emb_X, Emb_Temp)
        Distance = torch.zeros(20)

        IOU_pred, IOU_GT = self.Iou_Net.train_forward(X = Emb_X, Temp = Emb_Temp, X_pos = X_pos, Box = GT, sample_num = 100)
        prob_sem = self.Regress_relation(torch.cat([Emb_X, Temp_output],dim=1))

        assert(prob_sem.shape[1] == 1)

        Flag = self._Point_in_box(X_pos,GT)
        GT_prob = Flag.float()
        GT_prob_neg = -GT_prob + 1.0

        corp_loss_pos_sem, corp_loss_neg_sem = self._cal_class_loss(prob_sem, GT_prob, GT_prob_neg)
        corp_loss_pos_phy, corp_loss_neg_phy = self._cal_class_loss(prob_phy, GT_prob, GT_prob_neg)
        Bbox_output, bbox_all = self._weight_cal_box(Emb_X, prob_sem, prob_phy, X_pos)
        
        return Bbox_output.permute([0,2,1]), bbox_all.permute([0,2,1]), Distance, corp_loss_pos_sem, corp_loss_neg_sem, corp_loss_pos_phy, corp_loss_neg_phy, IOU_pred, IOU_GT , Cd_loss1, Cd_loss2

    def _forward_with_template(self,X,GT=None):

        assert(self.Tempalate_fea is not None)
        assert(torch.isnan(X).sum()==0)
        assert(X.shape[1] == 4)

        B,C,P = X.shape
        X0 = torch.zeros([B,1,P],device = X.device).float()
        X = torch.cat([X,X0],dim = 1)
        X = self.tracking_net.extractor(X,out_num=1024,Is_temp=False)

        X_pos_01 = X[:,:3,:]
        Emb_X_00, _ = self.tracking_net.SwinTrans.track_embedding(X.permute([0,2,1]))

        Emb_X = self.tracking_net._ensemble_fea([Emb_X_00[-1].permute([0,2,1]) , Emb_X_00[-2].permute([0,2,1]), Emb_X_00[-3].permute([0,2,1]), Emb_X_00[-4].permute([0,2,1])])

        Ensemble_X_01 = self.tracking_net._ensemble_low_fea([X, Emb_X_00[-4].permute([0,2,1]), Emb_X_00[-3].permute([0,2,1])])
        Ensemble_X_02 = self.tracking_net._Backward_warp(Ensemble_X_01)
        Warpped_pos = Ensemble_X_02[:,:2,:]

        prob_phy = self.tracking_net._regress_phy_pro_test(Warpped_pos, self.pred2curr_box[None,:].clone())

        if self.indx < 5:
            np.savetxt('/home/zhu_19/evt_tracking/points/after_project'+str(self.indx)+'.xyz', torch.cat([Warpped_pos[0,:,:], Ensemble_X_01[0,2:3,:]],dim=0).permute([1,0]).detach().cpu().numpy())
            np.savetxt('/home/zhu_19/evt_tracking/points/before_project'+str(self.indx)+'.xyz', Ensemble_X_01[0,:3,:].permute([1,0]).detach().cpu().numpy())

        X_pos = Emb_X[:,:3,:].clone()

        Emb_X = Emb_X[:,5:,:]

        X_pos_emb = self.tracking_net.pos_embedding(X_pos)

        X = self.tracking_net.SelfTransFormer1(Emb_X.clone() + X_pos_emb)

        if self.Tempalate_fea_mid is None:
            Temp = self.Tempalate_fea['Temp']
            Temp_pos = self.Tempalate_fea['Temp_pos']
        else:
            Temp = self.Tempalate_fea_mid['Temp']
            Temp_pos = self.Tempalate_fea_mid['Temp_pos']
        Temp_pos_emb = self.tracking_net.pos_embedding(Temp_pos)

        Temp = self.tracking_net.SelfTransFormer1(Temp + Temp_pos_emb)



        Temp_output, sparse_loss_01, H_01 = self.tracking_net.CrossTransFormer(X,Temp)

        Distance = torch.zeros(20)


        prob_sem = self.tracking_net.Regress_relation(torch.cat([X, Temp_output],dim=1))

        assert(prob_sem.shape[1] == 1)

        Bbox_output, bbox_all = self.tracking_net._weight_cal_box(X, prob_sem, prob_phy, X_pos)

        # X1 = X * prob
        # X2 = X * prob_phy

        # Pred_bbox1 = self._learning_box(X1.clone(), X_pos.clone())
        # Pred_bbox2 = self._learning_box(X2.clone(), X_pos.clone())

        ratio = 0.95

        Pred_bbox = Bbox_output * ratio + self.pred2curr_box[None,:,None] * (1-ratio)
        

        Pred_bbox_final = Pred_bbox[:,:,0]

        Pred_bboxs = bbox_all    
        
        flag_prob = prob_sem >= 0.8

        flag_prob = flag_prob[0,0,:]

        X_ = Emb_X[:,:,flag_prob]

        X_pos_ = X_pos[:,:,flag_prob]

        prob_ = prob_sem[:,:,flag_prob]

        idx_prob = prob_.sort(-1,descending = False)[-1]
        B,C,P = X_.shape

        idx_prob_ = idx_prob.repeat([1,C,1])
        idx_prob_pos = idx_prob.repeat([1,3,1])

        X_ = torch.gather(X_,-1,idx_prob_)

        X_pos_ = torch.gather(X_pos_, -1, idx_prob_pos)

        Flag_template_pred = self.tracking_net._Point_in_box(X_pos_, Pred_bbox_final)
        Flag_template_pred = Flag_template_pred.bool()[0,0,:]

        X_ = X_[:,:,Flag_template_pred]
        X_pos_ = X_pos_[:,:,Flag_template_pred]

        self._update_template(X_, X_pos_)

        return Pred_bbox_final[:,None,:], Pred_bboxs[0,:,:].permute([1,0]), prob_phy, X_pos

    def forward(self,X, Temp, GT, GT_temp):
        
        assert(torch.isnan(X).sum()==0)
        assert(torch.isnan(Temp).sum()==0)
        assert(X.shape[1] == 4)
        assert(Temp.shape[1] == 5)

        B,C,P = X.shape
        X0 = torch.zeros([B,1,P],device = X.device).float()
        X = torch.cat([X,X0],dim = 1)
        X = self.extractor(X,out_num=1024,Is_temp=False)
        X_pos = X[:,:3,:]
        B,C,P = Temp.shape
        temp_point = Temp[:,4,:].abs().sum()
        ratio = temp_point / (B * P)

        Temp = self.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        Temp_pos = Temp[:,:3,:]
        X = X[:,5:,:]
        temp_point = Temp[:,4,:].abs().detach()
        sum_point = temp_point.sum(-1,keepdim=True)

        Temp = Temp[:,5:,:]

        X = self.SelfTransFormer1(X)
        Temp = self.SelfTransFormer1(Temp)

        Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(X,Temp)

        Distance = torch.zeros(20)


        Temp_output, sparse_loss_02, H_02 = self.CrossTransFormer2(X,Temp_output)

        temp_point = temp_point[:,None,:]

        Temp_output = Temp_output*temp_point
        B0, C0, P0 = X.shape
        weighted_Temp =self._weight_point(Temp_output, Temp_pos)
        
        Temp_output = weighted_Temp.repeat([1,1,P0])
        prob = self.Regress_relation(torch.cat([X, Temp_output],dim=1))
        assert(prob.shape[1] == 1)

        Flag = self._Point_in_box(X_pos,GT)
        GT_prob = Flag.float()
        GT_prob_neg = -GT_prob + 1.0

        corp_loss_pos = - torch.log(prob+1e-6) * GT_prob
        corp_loss_neg = - torch.log(-prob + 1.0 +1e-6) * GT_prob_neg
        corp_loss_pos = corp_loss_pos.sum([1,2]) / (GT_prob.sum([1,2]) + 1e-6)
        corp_loss_neg = corp_loss_neg.sum([1,2]) / (GT_prob_neg.sum([1,2]) + 1e-6)
        X1 = X * prob
        

        weighted_fea =self._weight_point(X,prob, Temp_output, X_pos)

        bbox = self.Regression_layer(weighted_fea)
        return bbox.permute([0,2,1]), bbox.permute([0,2,1]), Distance, corp_loss_pos, corp_loss_neg, prob, X_pos
