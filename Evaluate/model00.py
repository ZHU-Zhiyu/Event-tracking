import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
import numpy as np
# from knn_cuda import KNN
def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)

# class ChamferLoss(nn.Module):
#     def __init__(self):
#         super(ChamferLoss, self).__init__()
#         self.use_cuda = torch.cuda.is_available()

def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_points_x,device = x.device)
    diag_ind_y = torch.arange(0, num_points_y,device = x.device)
    # if x.get_device() != -1:
    #     diag_ind_x = diag_ind_x.to(x.device)
    #     diag_ind_y = diag_ind_y.to(x.device)
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P

def ChamferLoss(preds, gts):
    P = batch_pairwise_dist(gts, preds)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.sum(mins)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.sum(mins)
    return loss_1 + loss_2


def knn(x,x2, k):
    with torch.no_grad():
        inner = -2*torch.matmul(x.transpose(2, 1), x2)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        xx2 = torch.sum(x2**2, dim=1, keepdim=True)
        pairwise_distance = -xx.transpose(2, 1) - inner - xx2
    
        idx = pairwise_distance.topk(k=k, dim=-2)[1]   # (batch_size, num_points, k)
        return idx

class Dynamic_sampling(Module):
    def __init__(self,k):
        super(Dynamic_sampling, self).__init__()
        self.k = k
        # self.Knn = KNN(k=self.k,transpose_mode=False)

    def forward(self, x, s_num, Is_temp=False, Temp_ratio=0):
        # x input image
        # num sampled number of point

        B,C,P = x.shape
        if s_num <=P:
            rand_num = torch.rand(B,P).to(x.device)
            rand_num = rand_num.abs()
            if Is_temp:
                assert(Temp_ratio>0)
                Temp_point = int(s_num*Temp_ratio)
                Temp_point = max(1,Temp_point)

                act = x[:,3,:]
                act = 1 - act.abs()*2
                rand_num = rand_num*act.detach()

                batch_rand_perm = rand_num.sort(dim=1)[1]

                batch_rand_perm0 = batch_rand_perm[:,:Temp_point]

                batch_rand_perm1 = batch_rand_perm[:,-(s_num - Temp_point):]

                batch_rand_perm = torch.cat([batch_rand_perm0, batch_rand_perm1], dim = 1)
            else:
                batch_rand_perm = rand_num.sort(dim=1)[1]
                batch_rand_perm = batch_rand_perm[:,:s_num]


            batch_rand_perm = batch_rand_perm[:,None,:].repeat(1,C,1)

            select_point = torch.gather(x,2,batch_rand_perm)
            
        else:
            rand_num = torch.rand(B,P).to(x.device)

            batch_rand_perm = rand_num.sort(dim=1)[1]

            batch_rand_perm = batch_rand_perm[:,:s_num-P]

            batch_rand_perm = batch_rand_perm[:,None,:].repeat(1,C,1)

            select_point = torch.gather(x,2,batch_rand_perm)
            select_point = torch.cat([x,select_point],dim=2)

        if C == 4:
            # _, indx = self.Knn(x[:,:3,:],select_point[:,:3,:])
            indx = knn(x[:,:3,:],select_point[:,:3,:], self.k )
        else:
            indx = knn(x, select_point, self.k)
        assert(select_point.shape[2] == s_num)
        # print('Is template:{}, shape batch_rand_perm:{}, select_point:{}, x:{}'.format(Is_temp, batch_rand_perm.shape, select_point.shape, x.shape))
        # print('shape of s_num:{}, indx1:{}'.format(s_num, indx.shape))

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1)*P

        indx = indx + idx_base

        indx = indx.view(-1)
        x = x.transpose(2, 1).contiguous() 
        feature = x.view(B*P, -1)[indx, :]
        # print('shape of indx2:{}, feature:{}'.format(indx.shape,feature.shape))
        feature = feature.view(B, s_num, self.k, C) 
        # select_point = select_point.view(B, s_num, 1, C).repeat(1, 1, self.k, 1)
        # feature = torch.cat((feature-select_point, select_point), dim=3).permute(0, 3, 1, 2).contiguous()
        # feature = select_point.permute(0, 3, 1, 2).contiguous()

        return feature.permute(0, 3, 1, 2).contiguous()

class Feature_extraction_layer(Module):
    def __init__(self, neighb=8, group = 32, inc = 32, outc = 64):
        super(Feature_extraction_layer,self).__init__()
        self.sample_layer = Dynamic_sampling(k=neighb)
        self.bn = nn.GroupNorm(group, int(outc/2))
        self.bn2 = nn.GroupNorm(group, int(outc/2))

        self.PosEncoding = nn.Sequential(nn.Conv2d(10, int(outc/2), kernel_size=1, bias=False),
                                self.bn2,
                                nn.LeakyReLU(negative_slope=0.2))
        # if inc == 4:
        #     self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=False),
        #                             self.bn,
        #                             nn.LeakyReLU(negative_slope=0.2))
        # else:

        self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=False),
                                self.bn,
                                nn.LeakyReLU(negative_slope=0.2))


        self.attention_mode = nn.Sequential(nn.Conv2d(outc, 32, kernel_size=1, bias=False),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(32, 1, kernel_size=1, bias=False))

        self.Branch1 = nn.Conv1d(outc, outc, kernel_size=1, bias=False)
        
        self.Branch2 = nn.Conv1d(inc, outc, kernel_size=1, bias=False)
        # print('inc:{}'.format(inc))
    def forward(self,x, num_points,Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        
        assert(torch.isnan(x).sum()==0)

        x1 = self.sample_layer(x, num_points, Is_temp = Is_temp, Temp_ratio = Temp_ratio)

        assert(torch.isnan(x1).sum()==0)
        # x1 shape in B C P K
        B, C, P, K = x1.shape
        PC = x1[:,:3,:,0]
        PC = PC[:,:,:,None].repeat([1,1,1,K])
        PA = x1[:,:3,:,:]
        temp = ((PA-PC)**2).sum(1,keepdim=True).sqrt()
        assert(torch.isnan(temp).sum()==0)

        PEmb = torch.cat([PA,PC,PA-PC,temp],dim=1)

        assert(torch.isnan(PEmb).sum()==0)
        # print('max:{},min:{}'.format(PEmb.max(),PEmb.min()))
        PEmb = self.PosEncoding(PEmb)

        assert(torch.isnan(PEmb).sum()==0)
        # print('x1 shape:{}'.format(x1.shape))
        if C == 4:
            # print('X1 shape:{}'.format(x1.shape))
            x_ = self.conv(x1)
        else:
            x_ = self.conv(x1[:,4:,:,:])

        feature = torch.cat([x_,PEmb],dim=1)
        attention = self.attention_mode(feature)
        attention = F.softmax(attention,dim = -1)

        assert(torch.isnan(attention).sum()==0)

        feature = feature * attention
        feature = feature.sum(-1,keepdim=False)

        assert(torch.isnan(feature).sum()==0)

        Fea1 = self.Branch1(feature)
        
        if C == 4:
            Fea2 = self.Branch2(x1[:,:,:,0])
        else:
            Fea2 = self.Branch2(x1[:,4:,:,0])

        assert(torch.isnan(Fea1).sum()==0)
        assert(torch.isnan(Fea2).sum()==0)

        output = F.leaky_relu(Fea1 + Fea2, negative_slope=0.1)

        assert(torch.isnan(output).sum()==0)

        return torch.cat([x1[:,:4,:,0], output],dim=1)

class Feature_extraction(Module):
    def __init__(self, group=32):
        super(Feature_extraction, self).__init__()

        self.Embedding_layer01 = Feature_extraction_layer(neighb=12, group = 4, inc = 4, outc = 128)
        self.Embedding_layer02 = Feature_extraction_layer(neighb=12, group = 16, inc = 128, outc = 256)
        self.Embedding_layer03 = Feature_extraction_layer(neighb=12, group = 16, inc = 256, outc = 512)
        self.Embedding_layer04 = Feature_extraction_layer(neighb=4, group = 16, inc = 512, outc = 1024)
        # self.sample_layer1 = Dynamic_sampling(k=8)
        # self.sample_layer2 = Dynamic_sampling(k=16)
        # self.sample_layer3 = Dynamic_sampling(k=16)
        # self.sample_layer4 = Dynamic_sampling(k=4)

        # self.bn1 = nn.GroupNorm(group, 64)
        # self.bn2 = nn.GroupNorm(group, 128)
        # self.bn3 = nn.GroupNorm(group, 256)
        # self.bn4 = nn.GroupNorm(group, 512)

        # self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))

        # self.conv4 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))

    def forward(self,x, out_num=256, ratio_list = [0.1, 0.05, 0.025], Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer01(x, max(int(P * ratio_list[0]), out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x1:{}'.format(x))
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer02(x, max(int(P * ratio_list[1]), out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x2:{}'.format(x))
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer03(x, max(int(P * ratio_list[2]), out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x3:{}'.format(x))
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer04(x, int(out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x4:{}'.format(x))

        return x
        # B, C, P = x.shape
        
        # x1 = self.sample_layer1(x, max(int(P * ratio_list[0]), out_num))

        # x1 = self.conv1(x1)
        # x1 = x1.max(dim=-1, keepdim=False)[0]

        # # x1 shape B C P*ratio_list[0]

        # x2 = self.sample_layer2(x1, max(int(P * ratio_list[1]), out_num))

        # x2 = self.conv2(x2)
        # x2 = x2.max(dim=-1, keepdim=False)[0]

        # # x2 shape B C P*ratio_list[1]

        # x3 = self.sample_layer3(x2, max(int(P * ratio_list[2]), out_num))

        # x3 = self.conv3(x3)
        # x3 = x3.max(dim=-1, keepdim=False)[0]

        # # x3 shape B C P*ratio_list[2]

        # x4 = self.sample_layer4(x3, int(out_num))

        # x4 = self.conv4(x4)
        # x4 = x4.max(dim=-1, keepdim=False)[0]
        # # print('shape of x1:{}, x2:{}, x3:{}, x4:{}'.format(x1.shape,x2.shape,x3.shape,x4.shape))
        # # x4 shape B C out_num
        # return x4

class SelfTransFormer(Module):
    def __init__(self,cin = 512, cout = 1024, group=32):
        super(SelfTransFormer, self).__init__()
        self.bn1 = nn.GroupNorm(group, cout)
        self.bn2 = nn.GroupNorm(group, cout)
        self.bn3 = nn.GroupNorm(group, cout)
        self.h = 8
        self.q_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.1))
    def forward(self,x):
        Q = self.q_gen(x)
        K = self.k_gen(x)
        V = self.v_gen(x)
        b,c,p = K.shape
        Q = Q.reshape(b,self.h,-1,p)
        K = K.reshape(b,self.h,-1,p)
        V = V.reshape(b,self.h,-1,p)
        # print('shape of Q:{}, K:{}, V:{}'.format(Q.shape, K.shape, V.shape))
        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-1)

        out = torch.einsum('bhij,bhcj->bhci',H,Q)
        out = out.reshape([b,c,p])
        return out

class CrossTransFormer(Module):
    def __init__(self, group =32):
        super(CrossTransFormer, self).__init__()
        self.bn1 = nn.GroupNorm(group, 1024)
        self.bn2 = nn.GroupNorm(group, 1024)
        self.bn3 = nn.GroupNorm(group, 1024)
        self.h = 8
        self.q_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.1))
    def forward(self, Xq, Xk):
        assert(Xq.shape[1] == Xk.shape[1])
        Q = self.q_gen(Xq)
        K = self.k_gen(Xk)
        V = self.v_gen(Xq)
        b,c,pk = K.shape
        _,_,pq = V.shape
        Q = Q.reshape(b,self.h,-1,pq)
        K = K.reshape(b,self.h,-1,pk)
        V = V.reshape(b,self.h,-1,pq)

        # print('shape of Q:{}, K:{}, V:{}'.format(Q.shape, K.shape, V.shape))
        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-1)

        out = torch.einsum('bhij,bhcj->bhci',H,Q)
        out = out.reshape([b,c,pk])
        return out

class ETracking_Net(Module):
    def __init__(self,):
        super(ETracking_Net, self).__init__()
        self.extractor = Feature_extraction()
        self.CrossTransFormer = CrossTransFormer()
        self.SelfTransFormer1 = SelfTransFormer(cin = 1024, cout = 1024)
        self.SelfTransFormer2 = SelfTransFormer(cin = 1024, cout = 1024)

        self.bnc = nn.GroupNorm(32, 256)
        self.bnr = nn.GroupNorm(32, 512)


        self.Regress = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=False),
                                   self.bnr,
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(512, 4, kernel_size=1, bias=False),
                                   nn.Sigmoid())

        self.Classifier = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=False),
                                   self.bnc,
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(256, 1, kernel_size=1, bias=False),
                                   nn.Sigmoid())


    def forward(self,X, Temp, GT=None):
        # print('max:{},min:{}'.format(X.max(),X.min()))
        """
        X in shape of B, 4, N1; (W H T Act)
        temp in shape of B, 4, N2; (W H T Act)
        GT in shape of B, 4. (cxcyxy) format
        """
        assert(torch.isnan(X).sum()==0)
        assert(torch.isnan(Temp).sum()==0)

        X = self.extractor(X,out_num=1024,Is_temp=False)
        B,C,P = Temp.shape
        temp_point = Temp[:,3,:].abs().sum()
        ratio = temp_point / (B * P)
        Temp = self.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        X_pos = X[:,:3,:]
        X = X[:,4:,:]
        temp_point = Temp[:,3,:].abs().detach()
        sum_point = temp_point.sum(-1,keepdim=True)
        # print(temp_point.detach().cpu().numpy())
        temp_point01 = temp_point.detach().cpu().numpy()
        # print(sum_point.float().sum().cpu().detach().item())

        Temp = Temp[:,4:,:]
        # print('Before Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        X = self.SelfTransFormer1(X)
        Temp = self.SelfTransFormer1(Temp)
        # print('After Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        Adapted_tem = self.CrossTransFormer(X,Temp)

        Adapted_tem01 = torch.cat([Adapted_tem,Adapted_tem.detach(),Adapted_tem.detach(),Adapted_tem.detach()],dim = 2)

        EMDst = earth_mover_distance(X, Adapted_tem01)

        # if GT == None:
        #     # Testind Mode
        #     X = self.CrossTransFormer(X,Adapted_tem)

        #     X = self.SelfTransFormer2(X)

        #     temp_point = temp_point[:,None,:]
        #     # print('shape of X:{}, temp_point:{},sum_point:{}'.format(X.shape,temp_point.shape,sum_point.shape))
        #     X = X*temp_point
        #     X = X.sum(-1,keepdim=True) / sum_point[:,:,None]
        #     bbox = self.Regress(X)

        #     return bbox.permute([0,2,1]), bbox.permute([0,2,1]), EMDst
        # else:
        # Training mode
        # (X, Temp)
        X_ = self.CrossTransFormer(X,Adapted_tem)
        # X = X.clone()
        X_pos_lim00 = X_pos[:,0,:]<=GT[:,0][:,None]+GT[:,2][:,None]/2
        X_pos_lim01 = X_pos[:,0,:]>=GT[:,0][:,None]-GT[:,2][:,None]/2
        X_pos_lim02 = X_pos[:,1,:]<=GT[:,1][:,None]+GT[:,3][:,None]/2
        X_pos_lim03 = X_pos[:,1,:]>=GT[:,1][:,None]-GT[:,3][:,None]/2
        X_pos_lim = X_pos_lim00 * X_pos_lim01 * X_pos_lim02 * X_pos_lim03

        B,C,P = X.shape
        Avg_point = max(1,int(X_pos_lim.float().sum().cpu().detach().item()/B))

        CDst =  (X_pos[:,0,:] - GT[:,0][:,None])**2 + (X_pos[:,1,:]-GT[:,1][:,None])**2 + 1e-6
        CDst_idx = CDst.sqrt().sort(-1)[1][:,:Avg_point][:,None,:]
        CDst_idx = CDst_idx.repeat(1,C,1)
        X_restrict = torch.gather(X,-1,CDst_idx)

        # print(torch.from_numpy(temp_point01).sort(-1)[1])
        temp_idx = temp_point.sort(-1)[1]
        temp_idx = temp_idx[:,None,:]
        a = int(np.sum(temp_point01)/B)
        a = max(1,a)
        temp_idx = temp_idx[:,:,-a:]
        temp_idx = temp_idx.repeat(1,C,1)
        temp_idx = temp_idx.to(Adapted_tem.device)
        Adapted_tem_restrict = torch.gather(Adapted_tem,-1,temp_idx)

        # print('device raw:{}, device temp:{}'.format(X_restrict.shape, Adapted_tem_restrict.shape))
        CDloss = ChamferLoss(X_restrict.transpose(2,1),Adapted_tem_restrict.transpose(2,1) )
        
        X_ = self.SelfTransFormer2(X_)
        
        temp_point = temp_point[:,None,:]
        # print('shape of X:{}, temp_point:{},sum_point:{}'.format(X.shape,temp_point.shape,sum_point.shape))
        X_ = X_*temp_point
        X_ = X_.sum(-1,keepdim=True) / sum_point[:,:,None]
        bbox = self.Regress(X_)

        return bbox.permute([0,2,1]), bbox.permute([0,2,1]), EMDst, CDloss

# class DeformConv(Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride=1,
#                  padding=0,
#                  dilation=1,
#                  num_deformable_groups=1):
#         super(DeformConv, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.num_deformable_groups = num_deformable_groups

#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, *self.kernel_size))

#         self.reset_parameters()

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)

#     def forward(self, input, offset):
#         return deform_conv_function(input, offset, self.weight, self.stride,
#                              self.padding, self.dilation,
#                              self.num_deformable_groups)
