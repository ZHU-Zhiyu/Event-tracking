import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
# from knn_cuda import KNN


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

    def forward(self, x, s_num):
        # x input image
        # num sampled number of point

        B,C,P = x.shape
        if s_num <=P:
            rand_num = torch.rand(B,P).to(x.device)

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
        # print('shape batch_rand_perm:{}, select_point:{}, x:{}'.format(batch_rand_perm.shape, select_point.shape, x.shape))
        # print('shape of s_num:{}, indx1:{}'.format(s_num, indx.shape))

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1)*P

        indx = indx + idx_base

        indx = indx.view(-1)
        x = x.transpose(2, 1).contiguous() 
        feature = x.view(B*P, -1)[indx, :]
        # print('shape of indx2:{}, feature:{}'.format(indx.shape,feature.shape))
        feature = feature.view(B, s_num, self.k, C) 
        select_point = select_point.view(B, s_num, 1, C).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature-select_point, select_point), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

class Feature_extraction(Module):
    def __init__(self):
        super(Feature_extraction, self).__init__()

        self.sample_layer1 = Dynamic_sampling(k=8)
        self.sample_layer2 = Dynamic_sampling(k=16)
        self.sample_layer3 = Dynamic_sampling(k=16)
        self.sample_layer4 = Dynamic_sampling(k=4)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv4 = nn.Sequential(nn.Conv2d(256*2, 512, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self,x, out_num=256, ratio_list = [0.1, 0.05, 0.025]):
        B, C, P = x.shape
        
        x1 = self.sample_layer1(x, max(int(P * ratio_list[0]), out_num))

        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]

        # x1 shape B C P*ratio_list[0]

        x2 = self.sample_layer2(x1, max(int(P * ratio_list[1]), out_num))

        x2 = self.conv2(x2)
        x2 = x2.max(dim=-1, keepdim=False)[0]

        # x2 shape B C P*ratio_list[1]

        x3 = self.sample_layer3(x2, max(int(P * ratio_list[2]), out_num))

        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]

        # x3 shape B C P*ratio_list[2]

        x4 = self.sample_layer4(x3, int(out_num))

        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]
        # print('shape of x1:{}, x2:{}, x3:{}, x4:{}'.format(x1.shape,x2.shape,x3.shape,x4.shape))
        # x4 shape B C out_num
        return x4

class SelfTransFormer(Module):
    def __init__(self,cin = 512, cout = 1024):
        super(SelfTransFormer, self).__init__()
        self.bn1 = nn.BatchNorm1d(cout)
        self.bn2 = nn.BatchNorm1d(cout)
        self.bn3 = nn.BatchNorm1d(cout)
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
    def __init__(self):
        super(CrossTransFormer, self).__init__()
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.bn3 = nn.BatchNorm1d(1024)
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
        self.SelfTransFormer1 = SelfTransFormer(cin = 512, cout = 1024)
        self.SelfTransFormer2 = SelfTransFormer(cin = 1024, cout = 1024)

        self.bnc = nn.BatchNorm1d(256)
        self.bnr = nn.BatchNorm1d(512)


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


    def forward(self,X, Temp):
        
        X = self.extractor(X,out_num=1024)

        Temp = self.extractor(Temp,out_num=256)
        # print('Before Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        X = self.SelfTransFormer1(X)
        Temp = self.SelfTransFormer1(Temp)
        # print('After Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        Adapted_tem = self.CrossTransFormer(X,Temp)

        X = self.CrossTransFormer(X,Adapted_tem)

        X = self.SelfTransFormer2(X)

        Class = self.Classifier(X)
        X = torch.mean(X,dim=-1,keepdim=True)
        bbox = self.Regress(X)

        return bbox.permute([0,2,1]), Class.permute([0,2,1])
