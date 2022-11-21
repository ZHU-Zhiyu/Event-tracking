import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.autograd import Variable, Function
from util.box_ops import box_xywh_to_xyxy, box_iou1, box_cxcywh_to_xyxy, generalized_box_iou1

class CrossTransFormer(Module):
    def __init__(self, group =32, bias = True):
        super(CrossTransFormer, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features = 1024)
        self.bn2 = nn.BatchNorm1d(num_features = 1024)
        self.bn3 = nn.BatchNorm1d(num_features = 1024)
        self.h = 1
        cin = 773
        cout = 773
        self.k_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=1, bias=bias),)
        self.v_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=1, bias=bias),)
        self.q_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=1, bias=bias),)

        self.dropout_layer01 = nn.Dropout2d(p=0.4, inplace=True)
        self.dropout_layer02 = nn.Dropout2d(p=0.4, inplace=True)
    def forward(self, Xq, Xk):
        assert(Xq.shape[1] == Xk.shape[1])

        Q = Xk
        K = self.k_gen(Xk)
        V = self.v_gen(Xq)
        V = self.dropout_layer01(V)
        K = self.dropout_layer02(K)
        b,c,pk = K.shape
        b,c,pk = K.shape
        _,_,pq = V.shape
        Q = Q.reshape(b,self.h,-1,pk)
        K = K.reshape(b,self.h,-1,pk)
        V = V.reshape(b,self.h,-1,pq)

        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-2)

        # sparse_loss = sparse_constrain(H)

        out = torch.einsum('bhij,bhki->bhkj',H,Q)

        out = out.reshape([b,c,pq])
        return out

class IOU_Net(Module):
    def __init__(self, bias = False, group = 4):
        super(IOU_Net, self).__init__()
        self.bn1 = nn.GroupNorm(group, 256)
        self.bn2 = nn.GroupNorm(group, 128)
        self.Temp_adaptive_pool = nn.Sequential(nn.Conv1d(773, 128, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(128, 1, kernel_size=1, bias=bias))

        self.Cross = CrossTransFormer()
        self.Searched_adaptive_pool = nn.Sequential(nn.Conv1d(773, 128, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(128, 1, kernel_size=1, bias=bias))

        self.Regression_layer = nn.Sequential(nn.Conv1d(773*2, 256, kernel_size=1, bias=bias),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(256, 128, kernel_size=1, bias=bias),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(128, 1, kernel_size=1, bias=bias))
        # self.Regression_layer = nn.Sequential(nn.Conv1d(2048, 1, kernel_size=1, bias=bias))
    def _pool(self, fea_in, Flag = None):
        """
        Flag == None : pooling for the template
        Flag == True : pooling for search feature
        fea_in B,C,P
        Flag B,nb,P
        """
        if Flag == None:
            wht = self.Temp_adaptive_pool(fea_in)

            wht = torch.softmax(wht, dim = -1)

            fea = fea_in * wht

            fea = fea.sum(dim= -1, keepdim = True)

            return fea

        else:
            wht = self.Searched_adaptive_pool(fea_in)
            # wht B 1 P
            wht = torch.exp(wht)
            B,N,C,P = Flag.shape
            Flag_r = Flag == False

            wht_ = wht[:,None,:,:].repeat([1,N,1,1]) * Flag
            # wht B N P

            wht_ = wht_ / (wht_.sum(dim = -1, keepdim = True) + 1e-6)
            

            wht_r = wht[:,None,:,:].repeat([1,N,1,1]) * Flag_r
            # wht B N P

            wht_r = wht_r / (wht_r.sum(dim = -1, keepdim = True) + 1e-6)

            # print('shape of fea_in:{}, shape of wht:{}'.format(fea_in.shape, wht.shape))
            fea = torch.einsum('bcp,bnp->bcn', fea_in, wht_[:,:,0,:])
            fea_r = torch.einsum('bcp,bnp->bcn', fea_in, wht_r[:,:,0,:])

            return torch.cat([fea.detach(), fea_r.detach()], dim = 1)
            # return torch.cat([fea, fea_r], dim = 1)

            # fea = fea.sum(dim= -1, keepdim = True)




    def train_forward(self, X, Temp, X_pos, Box, sample_num = 20):
        """
        X in [B,C,P]
        Temp in [B,C,P_s]
        X_pos in [B,C,P_s]
        Box in [B,4]
        """
        [B,C,P] = X.shape
        Box_ = Box[:,:,None]

        Distrub = torch.randn([B,4,sample_num],device = X.device)*0.3
        # Distrub_02 = torch.randn([B,sample_num],device = X.device)*2 - 1
        ratio = torch.cat([Box_[:,2:], Box_[:,2:]], dim = 1)
        # print('Shape of distrub:{}, ratio:{}'.format(Distrub.shape, ratio.shape))
        Distrub = Distrub * ratio

        Noi_Box = Box_ + Distrub

        Noi_Box = torch.clamp(Noi_Box, 0, 1)
        
        Noi_Box = Noi_Box.permute([0,2,1]).reshape([B*sample_num,4])

        Box_ = Box_.repeat([1,1,sample_num])
        
        Box_ = Box_.permute([0,2,1]).reshape([B*sample_num,4])

        # Noi_Box = 

        Noi_Box_ = box_cxcywh_to_xyxy(Noi_Box)
        Box_0 = box_cxcywh_to_xyxy(Box_)
        Noi_Box_ = torch.clamp(Noi_Box_, 0, 1)

        # print('noisy_box:{}, box:{}'.format(Noi_Box_, Box_0))
        IOU_gt,_ = box_iou1(Noi_Box_, Box_0)

        IOU_gt = IOU_gt.reshape([B,sample_num])
        X_pos_ = X_pos[:,None,:,:].repeat([1, sample_num, 1, 1])
        Flag = self._Point_in_box(X_pos_, Noi_Box.reshape([B,sample_num,4]))

        fea_temp = self.Cross(X, Temp)

        # print('X:{}, fea_temp:{}'.format(X.shape, fea_temp.shape))
        # fea_temp  = self._pool(Temp)
        print('shape of X:{}, fea_temp:{}, Flag:{}'.format(X.shape, fea_temp.shape, Flag.shape))
        fea_X  = self._pool(X * fea_temp, Flag)

        IOU_pred = self.Regression_layer(fea_X)


        return IOU_pred, IOU_gt

    def _Point_in_box(self,Pos,Box):
        """
        Input: Pos: B*3*P;
            Box: B*4*i;
        output:
            Flag: B*i*P;
        """
        # r1 = (Pos[:,0,:] - Box[:,0][:,None])/ (Box[:,2][:,None] / 2 + 1e-6)
        # r2 = (Pos[:,1,:] - Box[:,1][:,None])/ (Box[:,3][:,None] / 2 + 1e-6)
        # r = r1**2 + r2**2
        # r = r.sqrt()

        # Flag = r <= 1
        # print('shape of Pos:{}, Box:{}'.format(Pos.shape, Box.shape))
        flag1 = Pos[:,:,0,:] >= Box[:,:,0][:,:,None] - Box[:,:,2][:,:,None] / 2
        flag2 = Pos[:,:,1,:] >= Box[:,:,1][:,:,None] - Box[:,:,3][:,:,None] / 2
        flag3 = Pos[:,:,0,:] <= Box[:,:,0][:,:,None] + Box[:,:,2][:,:,None] / 2
        flag4 = Pos[:,:,1,:] <= Box[:,:,1][:,:,None] + Box[:,:,3][:,:,None] / 2

        Flag = flag1 *flag2 *flag3 *flag4


        return Flag[:,:,None,:].float()
        
    
    def Track(self, X, Temp, X_pos, Box):
        """
        X in [B,C,P]
        Temp in [B,C,P_s]
        X_pos in [B,C,P_s]
        Box in [B,4,nb]
        """
        [B,C,P] = X.shape
        # Box_ = Box[:,:,None]

        # Distrub = torch.randn([B,4,sample_num],device = X.device)*2 - 1
        # # Distrub_02 = torch.randn([B,sample_num],device = X.device)*2 - 1
        # ratio = torch.cat([Box_[:,2:], Box_[:,2:]], dim = 1)[:,:,None]

        # Distrub = Distrub * ratio

        # Noi_Box = Box_ + Distrub

        # Noi_Box = torch.clamp(Noi_Box, 0, 1)
        
        # Noi_Box = Noi_Box.permute([0,2,1]).reshape([B*sample_num,4])

        # Box_ = Box_.repeat([1,1,sample_num])
        
        # Box_ = Box_.permute([0,2,1]).reshape([B*sample_num,4])

        # # Noi_Box = 

        # IOU_gt = box_iou1(Noi_Box, Box_)

        # IOU_gt = IOU_gt.reshape[B,sample_num]

        Flag = self._Point_in_box(X_pos, Box)

        fea_temp  = self._pool(Temp)
        fea_X  = self._pool(X * fea_temp, Flag)

        IOU_pred = self.Regression_layer(fea_X)


        return IOU_pred, IOU_gt



    def forward(self, X, Temp, X_pos, Box):
        """
        X in [B,C,P]
        Temp in [B,C,P_s]
        X_pos in [B,C,P_s]
        Box in [B,4,nb]
        """


        return 0