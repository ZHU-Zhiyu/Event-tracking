import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.autograd import Variable, Function
from IOUNet import IOU_Net
# from knn_cuda import KNN
def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)
def sparse_constrain(X):
    assert(len(X.shape) == 4)
    B, C, P1, P2 = X.shape
    Iden = torch.eye(P1,device= X.device)
    Iden = Iden[None, None, :, :]
    Iden = Iden.repeat(B, C, 1, 1)

    mid = torch.einsum('bhci,bhki->bhck', X,X)

    loss = mid - Iden

    loss = loss**2

    return loss.mean()

def gether_neibour(X, indx, sample_n, neigh):
    '''
    Output shape B, P1, n1, C
    '''
    B, C, P = X.shape
    B1, n1, p1 = indx.shape
    assert(B == B1)
    assert(n1 == neigh)
    indx = indx.reshape([B1,1, n1*p1]).repeat([1,C,1])
    feature = torch.gather(X, 2, indx)
    feature = feature.reshape([B, C, n1, p1]).permute([0,3,2,1])
    return feature
class GradNorm(Function):
    @staticmethod
    def forward(ctx, input_x):
        a = torch.clone(input_x.detach())
        b = torch.clone(input_x.detach())

        return a, b
    @staticmethod
    def backward(ctx, grad_a, grad_b):
        # [B,C,H,W] = grad_output.shape
        # scale, = ctx.saved_tensors
        grad_a_nrom = (grad_a**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        grad_b_nrom = (grad_b**2).sum(dim=[1,2,3],keepdim=True).sqrt()
        grad_a = grad_a / (grad_a_nrom + 1e-6)
        grad_b = grad_b / (grad_b_nrom + 1e-6)
        grad_min = torch.cat([grad_a_nrom, grad_b_nrom],dim=1)
        grad_min = grad_min.min(dim=1,keepdim = True)[0]
        grad_a = grad_a* grad_min
        grad_b = grad_b* grad_min

        return grad_a+grad_b

def knn(x,x2, k, dilation = 1):
    with torch.no_grad():
        inner = -2*torch.matmul(x.transpose(2, 1), x2)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        xx2 = torch.sum(x2**2, dim=1, keepdim=True)
        pairwise_distance = -xx.transpose(2, 1) - inner - xx2
    
        idx = pairwise_distance.topk(k=k*dilation, dim=-2)[1]   # (batch_size, num_points, k)
        return idx[:,:,::dilation]

class Dynamic_sampling_M(Module):
    def __init__(self,k, dilation = 1):
        super(Dynamic_sampling_M, self).__init__()
        self.k = k

    def forward(self, x, s_num, Is_temp=False, Temp_ratio=0):

        B,C,P = x.shape
        if s_num <=P:

            get_neighbour = s_num
            length = math.ceil(math.sqrt(get_neighbour))
            mesh = torch.arange(0,length).float()/length
            mesh = mesh.to(x.device)
            mesh01 = mesh[None,:].repeat([length, 1])
            mesh02 = mesh[:,None].repeat([1, length])
            mesh01 = mesh01.reshape([1,1,length*length])
            mesh02 = mesh02.reshape([1,1,length*length])
            mesh = torch.cat([mesh01, mesh02],dim=1)
            mesh = mesh.repeat([B,1,1])
            PointSample = mesh

            # PointSample = torch.rand(B, 2, get_neighbour).to(x.device)
            Point_x = x[:,:2,:]
            min_x = Point_x.min(dim=2,keepdim = True)[0]
            max_x = Point_x.max(dim=2,keepdim = True)[0]
            PointSample = PointSample*(max_x - min_x) + min_x
            indx = knn(Point_x,PointSample, 1 )
            select_point = gether_neibour(x, indx, get_neighbour, 1)
            select_point = select_point[:,:,0,:].permute([0,2,1])

            
        else:
            rand_num = torch.rand(B,P).to(x.device)

            batch_rand_perm = rand_num.sort(dim=1)[1]

            batch_rand_perm = batch_rand_perm[:,:s_num-P]

            batch_rand_perm = batch_rand_perm[:,None,:].repeat(1,C,1)

            select_point = torch.gather(x,2,batch_rand_perm)
            select_point = torch.cat([x,select_point],dim=2)

        if C == 5:
            indx = knn(x[:,:3,:],select_point[:,:3,:], self.k )

        else:
            indx = knn(x[:,:3,:],select_point[:,:3,:], self.k )

        try:
            assert(select_point.shape[2] == s_num)
        except:
            print('Error')

        feature = gether_neibour(x, indx, s_num, self.k)

        return feature.permute(0, 3, 1, 2).contiguous()

class Dynamic_sampling(Module):
    def __init__(self,k, dilation = 1):
        super(Dynamic_sampling, self).__init__()
        self.k = k
        self.dilation = dilation
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

                act = x[:,4,:]
                act = 1 - act.abs()*2
                rand_num = rand_num*act.detach()

                batch_rand_perm = rand_num.sort(dim=1)[1]

                batch_rand_perm0 = batch_rand_perm[:,:Temp_point]
                if s_num - Temp_point == 0:
                    batch_rand_perm = batch_rand_perm0
                else:
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

        if C == 5:
            # _, indx = self.Knn(x[:,:3,:],select_point[:,:3,:])
            indx = knn(x[:,:3,:],select_point[:,:3,:], self.k, self.dilation)
        else:
            indx = knn(x[:,5:,:], select_point[:,5:,:], self.k, self.dilation)

        try:
            assert(select_point.shape[2] == s_num)
        except:
            print('Error')
            print('shape of Point:{}, s_num:{}, Temp_point:{}, batch_rand_perm:{},batch_rand_perm0:{}, batch_rand_perm1:{}'.format(select_point.shape, s_num, Temp_point, batch_rand_perm.shape,batch_rand_perm0.shape, batch_rand_perm1.shape))

        idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1)*P

        indx = indx + idx_base

        indx = indx.view(-1)
        x = x.transpose(2, 1).contiguous() 
        feature = x.view(B*P, -1)[indx, :]
        feature = feature.view(B, s_num, self.k, C) 

        return feature.permute(0, 3, 1, 2).contiguous()


class Feature_extraction_layer_basic(Module):
    def __init__(self, neighb=8, group = 32, inc = 32, outc = 64,bias = True, dilation = 4, normalized_sampling = True):
        super(Feature_extraction_layer_basic,self).__init__()
        if normalized_sampling:
            self.sample_layer = Dynamic_sampling_M(k=neighb, dilation = dilation)
        else:
            self.sample_layer = Dynamic_sampling(k=neighb, dilation = dilation)

        self.bn = nn.BatchNorm2d(num_features = int(outc/2))
        self.bn2 =nn.BatchNorm2d(num_features = int(outc/2))

        self.PosEncoding = nn.Sequential(nn.Conv2d(10, int(outc/2), kernel_size=1, bias=bias))

        self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=bias))


        self.attention_mode = nn.Sequential(nn.Conv2d(outc, 32, kernel_size=1, bias=bias),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(32, 1, kernel_size=1, bias=bias))

        self.Branch1 = nn.Conv1d(outc, outc, kernel_size=1, bias=bias)
        
        self.Branch2 = nn.Conv1d(inc, outc, kernel_size=1, bias=bias)
        # print('inc:{}'.format(inc))
    def forward(self,x, num_points,Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        
        assert(torch.isnan(x).sum()==0)

        x1 = self.sample_layer(x, num_points, Is_temp = Is_temp, Temp_ratio = Temp_ratio)

        assert(torch.isnan(x1).sum()==0)
        # x1 shape in B C P K
        B, C, P, K = x1.shape
        pos = x1[:,:5,:,0]
        
        PC = x1[:,:3,:,0]
        PC = PC[:,:,:,None].repeat([1,1,1,K])
        PA = x1[:,:3,:,:]
        temp = ((PA-PC)**2).sum(1,keepdim=True).sqrt()
        assert(torch.isnan(temp).sum()==0)

        PEmb = torch.cat([PA,PC,PA-PC,temp],dim=1)

        assert(torch.isnan(PEmb).sum()==0)

        PEmb = self.PosEncoding(PEmb)

        assert(torch.isnan(PEmb).sum()==0)
        x_ = self.conv(x1[:,:4,:,:])

        feature = torch.cat([x_,PEmb],dim=1)
        attention = self.attention_mode(feature)
        attention = F.softmax(attention,dim = -1)

        assert(torch.isnan(attention).sum()==0)

        feature = feature * attention
        feature = feature.sum(-1,keepdim=False)

        assert(torch.isnan(feature).sum()==0)

        output = self.Branch1(feature)
        
        return torch.cat([x1[:,:5,:,0], output],dim=1)

class Feature_extraction(Module):
    def __init__(self, group=32):
        super(Feature_extraction, self).__init__()

        self.Embedding_layer01 = Feature_extraction_layer_basic(neighb=8, group = 4, inc = 4, outc = 96, dilation=1, normalized_sampling = True)

    def forward(self,x, out_num=256, ratio_list = [0.5, 0.25, 0.16, 0.1], Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer01(x, max(3136, out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        return x


class SelfTransFormer(Module):
    def __init__(self,cin = 512, cout = 1024, group=32,bias=True):
        super(SelfTransFormer, self).__init__()
        self.bn1 = nn.GroupNorm(group, cout)
        self.bn2 = nn.GroupNorm(group, cout)
        self.bn3 = nn.GroupNorm(group, cout)

        self.h = 8
        self.q_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=8, bias=bias),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=8, bias=bias),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=8, bias=bias),
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

        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-1)

        out = torch.einsum('bhij,bhcj->bhci',H,Q)
        out = out.reshape([b,c,p])
        return out

class CrossTransFormer(Module):
    def __init__(self, group =32, bias = True):
        super(CrossTransFormer, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features = 1024)
        self.bn2 = nn.BatchNorm1d(num_features = 1024)
        self.bn3 = nn.BatchNorm1d(num_features = 1024)
        self.h = 1
        cin = 1024
        cout = 1024
        self.k_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),)
        self.v_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),)
        self.q_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),)

        self.dropout_layer01 = nn.Dropout2d(p=0.4, inplace=True)
        self.dropout_layer02 = nn.Dropout2d(p=0.4, inplace=True)
    def forward(self, Xq, Xk):
        assert(Xq.shape[1] == Xk.shape[1])

        Q = self.q_gen(Xk)
        K = self.k_gen(Xk)
        V = self.v_gen(Xq)
        # V = self.dropout_layer01(V)
        # K = self.dropout_layer02(K)
        b,c,pk = K.shape
        b,c,pk = K.shape
        _,_,pq = V.shape
        Q = Q.reshape(b,self.h,-1,pk)
        K = K.reshape(b,self.h,-1,pk)
        V = V.reshape(b,self.h,-1,pq)

        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-2)

        sparse_loss = sparse_constrain(H)

        out = torch.einsum('bhij,bhki->bhkj',H,Q)

        out = out.reshape([b,c,pq])
        return out, sparse_loss, H

class CrossTransFormer02(Module):
    def __init__(self, group =32, bias = True):
        super(CrossTransFormer02, self).__init__()
        self.bn1 = nn.GroupNorm(group, 1024)
        self.bn2 = nn.GroupNorm(group, 1024)
        self.bn3 = nn.GroupNorm(group, 1024)
        # self.bn1 = nn.BatchNorm1d(num_features = 1024)
        # self.bn2 = nn.BatchNorm1d(num_features = 1024)
        # self.bn3 = nn.BatchNorm1d(num_features = 1024)
        self.h = 1
        self.q_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=bias),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=bias),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, bias=bias),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.1))
        # self.norm_layer = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),)
        # self.GradNorm_layer = GradNorm().apply
        self.dropout_layer01 = nn.Dropout2d(p=0.4, inplace=True)
        self.dropout_layer02 = nn.Dropout2d(p=0.4, inplace=True)
    def forward(self, Xq, Xk):
        assert(Xq.shape[1] == Xk.shape[1])
        # Q = self.q_gen(Xq)
        Q = Xk
        K = self.q_gen(Xk)
        V = self.v_gen(Xq)
        V = self.dropout_layer01(V)
        K = self.dropout_layer02(K)
        b,c,pk = K.shape
        _,_,pq = V.shape
        Q = Q.reshape(b,self.h,-1,pq)
        K = K.reshape(b,self.h,-1,pk)
        V = V.reshape(b,self.h,-1,pq)

        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        H = nn.functional.softmax(H, dim=-2)
        # H_01, H_02 = self.GradNorm_layer(H)

        sparse_loss = sparse_constrain(H)

        out = torch.einsum('bhij,bhik->bhjk',H,Q)
        out = out.reshape([b,c,pk])
        return out, sparse_loss, H

class ETracking_Net(Module):
    def __init__(self,SwinT = None, bias = False):
        super(ETracking_Net, self).__init__()
        self.extractor = Feature_extraction()
        self.CrossTransFormer = CrossTransFormer()

        self.SelfTransFormer1 = SelfTransFormer(cin = 2112, cout = 1024)

        self.Iou_Net = IOU_Net()
        self.SwinTrans = SwinT

        C_swin = 1024
        self.Regression_layer = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(256, 4, kernel_size=1, bias=bias),
                                   nn.Sigmoid())

        self.Regression_ipc = nn.Sequential(nn.Conv1d(1024, 128, kernel_size=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(128, 32, kernel_size=1, bias=True),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(32, 1, kernel_size=1, bias=True))

        self.bnc = nn.GroupNorm(32, 256)
        self.bnr = nn.GroupNorm(32, 512)
        self.fusion_bias = nn.parameter.Parameter(torch.tensor([5e-1,5e-1]).float(), requires_grad=True).float()
        self.weight_point_01 = nn.Sequential(nn.Conv1d(C_swin, 512, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),)
        self.weight_point_02 = nn.Sequential(nn.Conv1d(3, 512, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),)
        self.weight_point_03 = nn.Sequential(nn.Conv1d(1024, 1, kernel_size=1, bias=bias))

        self.Regress_relation = nn.Sequential(nn.Conv1d(1024+C_swin, 512, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(512, 1, kernel_size=1, bias=bias),
                                   nn.Sigmoid())
        self.pos_embedding = nn.Conv1d(3, 2112, kernel_size=1, bias=bias)

        self.flow_estimate = nn.Sequential(nn.Conv1d(480, 32, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(32, 2, kernel_size=1, bias=bias))
        
        self.classifier_physical = nn.Sequential(nn.Conv1d(7, 512, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(512, 1, kernel_size=1, bias=bias),
                                   nn.Sigmoid())



    def _ensemble_fea(self,X):

        x_01, x_02, x_03, x_04 = X
        target_pos = x_04[:,:3,:]
        pos_01 = x_01[:,:3,:]
        pos_02 = x_02[:,:3,:]
        pos_03 = x_03[:,:3,:]
        s_num = target_pos.shape[2]

        indx_01 = knn(pos_01,target_pos, 1 )
        fea_01 = gether_neibour(x_01, indx_01, s_num, 1)

        indx_02 = knn(pos_02,target_pos, 1 )
        fea_02 = gether_neibour(x_02, indx_02, s_num, 1)

        indx_03 = knn(pos_03,target_pos, 1 )
        fea_03 = gether_neibour(x_03, indx_03, s_num, 1)

        fea_01 = fea_01.mean(2,keepdim=False).permute([0,2,1])[:,5:,:]
        fea_02 = fea_02.mean(2,keepdim=False).permute([0,2,1])[:,5:,:]
        fea_03 = fea_03.mean(2,keepdim=False).permute([0,2,1])[:,5:,:]

        fea_out = torch.cat([x_04,fea_01,fea_02,fea_03],dim = 1)
        return fea_out

    def _ensemble_low_fea(self,X):
        
        x_01, x_02, x_03 = X
        target_pos = x_02[:,:3,:]
        pos_01 = x_01[:,:3,:]
        pos_02 = x_02[:,:3,:]
        s_num = target_pos.shape[2]

        indx_01 = knn(pos_01,target_pos, 3 )
        fea_01 = gether_neibour(x_01, indx_01, s_num, 3)

        fea_01 = fea_01[:, :, :, 5:]
        fea_02 = fea_01 - fea_01[:,:,0:1,:]
        fea_02 = fea_02[:,:,1:,:]


        [B, p1, n1, C] = fea_02.shape

        fea_02 = fea_02.reshape([B, p1, n1*C]).permute([0,2,1])
        fea_out = torch.cat([x_02,fea_01[:,:,0,:].permute([0,2,1]),fea_02],dim = 1)
        return fea_out

    def _weight_point(self, fea_in, pos):
        fea = self.weight_point_01(fea_in)

        pos = self.weight_point_02(pos)

        fea_out = torch.cat([fea, pos], dim = 1)

        weight_point = self.weight_point_03(fea_out)

        weight_point = nn.functional.softmax(weight_point, dim = 2)

        fea_out = weight_point*fea_out

        fea_out = torch.sum(fea_out, dim = 2, keepdim = True)
        return fea_out

    def _Point_in_box(self,Pos,Box):
        """
        Input: Pos: B*3*P;
            Box: B*4;
        output:
            Flag: B*1*P;
        """
        r1 = (Pos[:,0,:] - Box[:,0][:,None])/ (Box[:,2][:,None] / 2 + 1e-6)
        r2 = (Pos[:,1,:] - Box[:,1][:,None])/ (Box[:,3][:,None] / 2 + 1e-6)
        r = r1**2 + r2**2
        r = r.sqrt()

        Flag = r <= 1


        return Flag[:,None,:].float()
    def _cal_Transmission(self, H_i, Seq_Pos, Temp_pos, Seq_box, Temp_box):
        """
        H: B*C*Ns*Nt
        Seq_Pos: B*3*Ns;
        Temp_pos: B*3*Nt;
        Seq_Box: B*4;
        Temp_Box: B*4;
        """
        H = H_i.permute([0,1,3,2])
        B,C,Ns,Nt = H.shape
        B0,_,Ns0 = Seq_Pos.shape
        B1,_,Nt1 = Temp_pos.shape
        assert(Ns == Ns0)
        assert(Nt == Nt1)
        Flag_Seq = self._Point_in_box(Seq_Pos, Seq_box)
        Flag_Temp = self._Point_in_box(Temp_pos, Temp_box)
        Flag_Temp_neg = -Flag_Temp + 1

        # Mask_H in B*1 * Ns*Nt
        Mask_H_pos = Flag_Seq[:,:,:,None] * Flag_Temp[:,:,None,:]

        # Pos_sample in B*C * Ns*Nt
        Pos_sample = H * Mask_H_pos
        Pos_sample = Pos_sample.sum([1,2,3])/C
        Pos_sample = Pos_sample/(Flag_Temp.sum([1,2]).detach()+1e-6)

        # Mask_H in B*1 * Ns*Nt
        Mask_H_neg = Flag_Seq[:,:,:,None] * Flag_Temp_neg[:,:,None,:]

        # Neg_sample in B*1 * Ns*Nt
        Neg_sample = H * Mask_H_neg
        Neg_sample = Neg_sample.sum([1,2,3])/C
        Neg_sample = Neg_sample/(Flag_Temp_neg.sum([1,2]).detach()+1e-6)
        return Pos_sample, Neg_sample
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

    def eval_(self,X, Temp, GT, GT_temp):
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

        Temp = self.extractor(Temp, out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        Temp_pos = Temp[:,:3,:]
        X = X[:,5:,:]
        temp_point = Temp[:,4,:].abs().detach()
        sum_point = temp_point.sum(-1,keepdim=True)

        Temp = Temp[:,5:,:]
        X_pos_emb = self.pos_embedding(X_pos)
        Temp_pos_emb = self.pos_embedding(Temp_pos)

        X = self.SelfTransFormer1(X + X_pos_emb)
        Temp = self.SelfTransFormer1(Temp + Temp_pos_emb)

        Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(X,Temp)

        Pos_01, Neg_01 = self._cal_Transmission(H_01, X_pos, Temp_pos, GT, GT_temp)

        Distance = torch.zeros(20)


        X, sparse_loss_02, H_02 = self.CrossTransFormer2(X,Temp_output)

        Pos_02, Neg_02 = self._cal_Transmission(H_02, X_pos, Temp_pos, GT, GT_temp)

        temp_point = temp_point[:,None,:]
        X = X*temp_point
        X = X.sum(-1,keepdim=True) / (sum_point[:,:,None] + 1e-6)
        bbox = self.Regress(X)

        return bbox.permute([0,2,1]), bbox.permute([0,2,1]), Distance, sparse_loss_02 + sparse_loss_01, Pos_01, Pos_02, Neg_01, Neg_02, H_01, H_02
    def _sample_init(self, X0):

        et0 = X0[:,3,:]
        et0_min = X0.min(dim=1,keepdim=False)[0]
        et0_max = X0.max(dim=1,keepdim=False)[0]

        flag0 = et0 >= (et0_min + (et0_max - et0_min)*0.1)
        num = flag0.float().sum(dim=1).mean()
        num = num.int()
        
        flag_idx = flag0.float().sort(1,descending=True)[1]
        flag_idx = flag_idx[:,:num]
        flag_idx = flag_idx[:,None,:].repeat([1,4,1])

        X = torch.gather(X0,2,flag_idx)

        return X
        
    def _Backward_warp(self, X0):

        X_pos = X0[:,:5,:]

        flow = self.flow_estimate(X0[:,5:,:])
        et = X_pos[:,3,:]
        et =  (et - et.min(dim=1,keepdim=True)[0]) / (et.max(dim=1,keepdim=True)[0] - et.min(dim=1,keepdim=True)[0] + 1e-6)

        move = flow * et[:,None,:]
        # move = flow

        project_pos = X0[:,:2,:] + move
        return torch.cat((project_pos,X0[:,2:4,:]), dim=1), flow

    def _regress_phy_pro_test(self, X_pos, Bbox):
        B, C, N = X_pos.shape
        assert(len(Bbox.shape) == 2)
        # Bbox_info = Bbox[:,:,None].repeat([1,1,N])
        Bbox_info = Bbox[:,:,None]

        dist = X_pos[:,:2,:] - Bbox_info[:,:2,:]

        dist  = dist / (Bbox_info[:,2:,:]/2 + 1e-6)
        dist = dist**2
        dist = dist.sum(dim = 1, keepdim=True).sqrt()

        dist = -dist + 1

        Fomulate_inform = torch.cat([Bbox_info.repeat([1,1,784]), X_pos[:,:2,:], dist], dim = 1)

        pro_phy = self.classifier_physical(Fomulate_inform)

        # pro_phy = torch.sigmoid(dist)

        # X_info = torch.cat([X_pos, Bbox_info], dim = 1)

        # pro_phy = self.classifier_physical(X_info)

        return pro_phy


    def _point_cd_loss(self, X_warp, X0):
        '''
        All X_warp and X0 has only one 
        '''
        assert(X_warp.shape[1] == 4)
        assert(X0.shape[1] == 3)
        loss_thresh = 0.1
        indx_01 = knn(X_warp[:,:3,:], X0, 1 )
        fea_01 = gether_neibour(X_warp, indx_01, 0, 1)
        et01 = fea_01[:,:,0,:].permute([0,2,1])[:,3,:]

        loss_1 = fea_01[:,:,0,:].permute([0,2,1])[:,:3,:] - X0
        loss_1 = loss_1[:,:2,:].abs().sum(dim=1)
        a = loss_1 < loss_thresh
        loss_1_ = loss_1 / (loss_thresh) * a.float()

        loss_1 = loss_1_.mean()

        # ------------------------------------
        indx_02 = knn(X0, X_warp[:,:3,:], 1 )
        fea_02 = gether_neibour(X0, indx_02, 0, 1)
        et02 = X_warp[:,3,:]

        loss_2 = fea_02[:,:,0,:].permute([0,2,1]) - X_warp[:,:3,:]
        loss_2 = loss_2[:,:2,:].abs().sum(dim=1)
        a = loss_2 < loss_thresh
        loss_2_ = loss_2 / (loss_thresh) * a.float()
        
        loss_2 = loss_2_.mean()

        return loss_1, loss_2
    
    def _cal_class_loss(self, prob, GT_prob, GT_prob_neg):

        
        corp_loss_pos = - torch.log(prob+1e-6) * GT_prob
        corp_loss_neg = - torch.log(-prob + 1.0 +1e-6) * GT_prob_neg
        corp_loss_pos = corp_loss_pos.sum([1,2]) / (GT_prob.sum([1,2]) + 1e-6)
        corp_loss_neg = corp_loss_neg.sum([1,2]) / (GT_prob_neg.sum([1,2]) + 1e-6)

        return corp_loss_pos, corp_loss_neg

    def _weight_cal_box(self, Emb_X, prob_sem, prob_phy, X_pos):

        X_sem = Emb_X * prob_sem
        weighted_fea_sem =self._weight_point(X_sem, X_pos)

        X_phy = Emb_X * prob_phy
        weighted_fea_phy =self._weight_point(X_phy, X_pos)

        fea_fused = torch.cat([weighted_fea_sem, weighted_fea_phy], dim=2)
        confidence = self.Regression_ipc(fea_fused)
        
        confidence1 = torch.softmax(confidence+self.fusion_bias[None,None,:], dim = 2)
        # confidence in shape of B, 1, 2

        bbox_fused = self.Regression_layer(weighted_fea_phy)


        return bbox_fused, bbox_fused, torch.softmax(confidence, dim = 2)

    def _regress_phy_pro(self, X_pos, Bbox):
        B, C, N = X_pos.shape
        assert(len(Bbox.shape) == 2)
        Bbox_info = Bbox[:,:,None].repeat([1,1,N])

        X_info = torch.cat([X_pos, Bbox_info], dim = 1)

        pro_phy = self.classifier_physical(X_info)

        return pro_phy

    def forward(self,X, Temp, GT, GT_temp, Efame_pre, Box_pre_noi, Box_pre_noi_free):

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

        prob_phy = self._regress_phy_pro_test(Warpped_pos, Box_pre_noi)

        # prob_phy_noi_free = self._regress_phy_pro_test(Warpped_pos, Box_pre_noi_free)

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