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
    B, C, P = X.shape

    idx_base = torch.arange(0, B, device=X.device).view(-1, 1, 1)*P

    indx = indx + idx_base

    indx = indx.view(-1)

    X = X.transpose(2, 1).contiguous() 
    feature = X.view(B*P, -1)[indx, :]
    feature = feature.view(B, sample_n, neigh, C) 

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
        # stdnrom = 1/math.sqrt(C*H*W)
        # gradnrom = stdnrom / gradnrom

        # gradnrom = torch.clamp(gradnrom,min=0,max=1)

        # grad_output = gradnrom * grad_output
        # grad_output = scale * grad_output
        # avg = (torch.mean(grad_GAN**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
        # grad_GAN = torch.sign(grad_GAN)*avg
        # avg2 = (torch.mean(grad_line**2,dim=[1,2,3],keepdim=True)/(H*C*W)).sqrt()
        # grad_line = torch.sign(grad_line)*avg2

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

            # if Is_temp:
            #     rand_num = torch.rand(B,P).to(x.device)
            #     rand_num = rand_num.abs()
            #     assert(Temp_ratio>0)
            #     Temp_point = int(s_num*Temp_ratio)
            #     Temp_point = max(1,Temp_point)

            #     act = x[:,4,:]
            #     act = 1 - act.abs()*2
            #     rand_num = rand_num*act.detach()

            #     batch_rand_perm = rand_num.sort(dim=1)[1]

            #     batch_rand_perm0 = batch_rand_perm[:,:Temp_point]
            #     batch_rand_perm = batch_rand_perm0[:,None,:].repeat(1,C,1)

            #     select_point_temp = torch.gather(x,2,batch_rand_perm)
            #     get_neighbour = s_num - Temp_point
            #     # batch_rand_perm1 = batch_rand_perm[:,-(s_num - Temp_point):]

            #     # batch_rand_perm = torch.cat([batch_rand_perm0, batch_rand_perm1], dim = 1)

            # else:
            #     # batch_rand_perm = rand_num.sort(dim=1)[1]
            #     # batch_rand_perm = batch_rand_perm[:,:s_num]
            #     get_neighbour = s_num
            #     select_point_temp = None
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
            # if Is_temp:
            #     select_point = torch.cat([select_point_temp, select_point], dim = 2)


            # batch_rand_perm = batch_rand_perm[:,None,:].repeat(1,C,1)

            # select_point = torch.gather(x,2,batch_rand_perm)
            
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
            indx = knn(x[:,5:,:], select_point[:,5:,:], self.k)
            # indx = knn(x[:,:3,:],select_point[:,:3,:], self.k )

        try:
            assert(select_point.shape[2] == s_num)
        except:
            print('Error')
            print('shape of Point:{}, s_num:{}, Temp_point:{}, batch_rand_perm:{},batch_rand_perm0:{}, batch_rand_perm1:{}'.format(select_point.shape, s_num, Temp_point, batch_rand_perm.shape,batch_rand_perm0.shape, batch_rand_perm1.shape))

        # idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1)*P

        # indx = indx + idx_base

        # indx = indx.view(-1)
        # x = x.transpose(2, 1).contiguous() 
        # feature = x.view(B*P, -1)[indx, :]
        # feature = feature.view(B, s_num, self.k, C) 

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
    def __init__(self, neighb=8, group = 32, inc = 32, outc = 64,bias = True, dilation = 4, normalized_sampling = True):
        super(Feature_extraction_layer,self).__init__()
        if normalized_sampling:
            self.sample_layer = Dynamic_sampling_M(k=neighb, dilation = dilation)
        else:
            self.sample_layer = Dynamic_sampling(k=neighb, dilation = dilation)

        # self.bn = nn.GroupNorm(group, int(outc/2))
        # self.bn2 = nn.GroupNorm(group, int(outc/2))
        self.bn = nn.BatchNorm2d(num_features = int(outc/2))
        self.bn2 =nn.BatchNorm2d(num_features = int(outc/2))

        self.PosEncoding = nn.Sequential(nn.Conv2d(10, int(outc/2), kernel_size=1, bias=bias),
                                self.bn2,
                                nn.LeakyReLU(negative_slope=0.2))
        # if inc == 4:
        #     self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=False),
        #                             self.bn,
        #                             nn.LeakyReLU(negative_slope=0.2))
        # else:

        self.conv = nn.Sequential(nn.Conv2d(inc, int(outc/2), kernel_size=1, bias=bias),
                                self.bn,
                                nn.LeakyReLU(negative_slope=0.2))


        self.attention_mode = nn.Sequential(nn.Conv2d(outc, 32, kernel_size=1, bias=bias),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv2d(32, 1, kernel_size=1, bias=bias))

        self.Branch1 = nn.Conv1d(outc, outc, kernel_size=1, bias=bias)
        
        self.Branch2 = nn.Conv1d(inc, outc, kernel_size=1, bias=bias)
        # print('inc:{}'.format(inc))
    def forward(self,x, num_points,Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        
        # assert(torch.isnan(x).sum()==0)

        x1 = self.sample_layer(x, num_points, Is_temp = Is_temp, Temp_ratio = Temp_ratio)

        # assert(torch.isnan(x1).sum()==0)
        # x1 shape in B C P K
        B, C, P, K = x1.shape
        PC = x1[:,:3,:,0]
        PC = PC[:,:,:,None].repeat([1,1,1,K])
        PA = x1[:,:3,:,:]
        temp = ((PA-PC)**2).sum(1,keepdim=True).sqrt()
        # assert(torch.isnan(temp).sum()==0)

        PEmb = torch.cat([PA,PC,PA-PC,temp],dim=1)

        # assert(torch.isnan(PEmb).sum()==0)
        # print('max:{},min:{}'.format(PEmb.max(),PEmb.min()))
        PEmb = self.PosEncoding(PEmb)

        # assert(torch.isnan(PEmb).sum()==0)
        # print('x1 shape:{}'.format(x1.shape))
        if C == 5:
            # print('X1 shape:{}'.format(x1.shape))
            x_ = self.conv(x1[:,:4,:,:])
        else:
            x_ = self.conv(x1[:,5:,:,:])

        feature = torch.cat([x_,PEmb],dim=1)
        attention = self.attention_mode(feature)
        attention = F.softmax(attention,dim = -1)

        # assert(torch.isnan(attention).sum()==0)

        feature = feature * attention
        feature = feature.sum(-1,keepdim=False)

        # assert(torch.isnan(feature).sum()==0)

        Fea1 = self.Branch1(feature)
        
        if C == 5:
            Fea2 = self.Branch2(x1[:,:4,:,0])
        else:
            Fea2 = self.Branch2(x1[:,5:,:,0])

        # assert(torch.isnan(Fea1).sum()==0)
        # assert(torch.isnan(Fea2).sum()==0)

        output = F.leaky_relu(Fea1 + Fea2, negative_slope=0.01)

        # assert(torch.isnan(output).sum()==0)

        return torch.cat([x1[:,:5,:,0], output],dim=1)

class Feature_extraction(Module):
    def __init__(self, group=32):
        super(Feature_extraction, self).__init__()

        self.Embedding_layer01 = Feature_extraction_layer(neighb=8, group = 4, inc = 4, outc = 128, dilation=1, normalized_sampling = True)
        self.Embedding_layer02 = Feature_extraction_layer(neighb=6, group = 16, inc = 128, outc = 256, dilation=1, normalized_sampling = True)
        self.Embedding_layer03 = Feature_extraction_layer(neighb=6, group = 16, inc = 256, outc = 512, dilation=1, normalized_sampling = True)
        self.Embedding_layer04 = Feature_extraction_layer(neighb=4, group = 16, inc = 512, outc = 1024, dilation=1, normalized_sampling = True)
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

    def forward(self,x, out_num=256, ratio_list = [0.5, 0.25, 0.16, 0.1], Is_temp = False, Temp_ratio = 0):
        B, C, P = x.shape
        # assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer01(x, max(math.ceil(math.sqrt(int(P * ratio_list[0])))**2, out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x1:{}'.format(x))
        # assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer02(x, max(math.ceil(math.sqrt(int(P * ratio_list[1])))**2, out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x2:{}'.format(x))
        # assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer03(x, max(math.ceil(math.sqrt(int(P * ratio_list[2])))**2, out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
        # print('x3:{}'.format(x))
        # assert(torch.isnan(x).sum()==0)
        x = self.Embedding_layer04(x, max(math.ceil(math.sqrt(int(P * ratio_list[3])))**2, out_num), Is_temp = Is_temp, Temp_ratio = Temp_ratio)
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
    def __init__(self,cin = 512, cout = 1024, group=32,bias=True):
        super(SelfTransFormer, self).__init__()
        self.bn1 = nn.GroupNorm(group, cout)
        self.bn2 = nn.GroupNorm(group, cout)
        self.bn3 = nn.GroupNorm(group, cout)
        # self.bn1 = nn.BatchNorm1d(num_features = cout)
        # self.bn2 = nn.BatchNorm1d(num_features = cout)
        # self.bn3 = nn.BatchNorm1d(num_features = cout)
        self.h = 8
        self.q_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(cin, cout, kernel_size=1, groups=4, bias=bias),
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
    def __init__(self, group =32, bias = True):
        super(CrossTransFormer, self).__init__()
        self.bn1 = nn.BatchNorm1d(num_features = 1024)
        self.bn2 = nn.BatchNorm1d(num_features = 1024)
        self.bn3 = nn.BatchNorm1d(num_features = 1024)
        self.h = 1
        # self.q_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, groups=4, bias=bias),)
                                #    self.bn1,
                                #    nn.LeakyReLU(negative_slope=0.1))
        
        self.k_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, groups=4, bias=bias),)
                                #    self.bn2,
                                #    nn.LeakyReLU(negative_slope=0.1))
        self.v_gen = nn.Sequential(nn.Conv1d(1024, 1024, kernel_size=1, groups=4, bias=bias),)
                                #    self.bn3,
                                #    nn.LeakyReLU(negative_slope=0.1))
        # self.norm_layer = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),)
        # self.GradNorm_layer = GradNorm().apply
        self.dropout_layer01 = nn.Dropout2d(p=0.4, inplace=True)
        self.dropout_layer02 = nn.Dropout2d(p=0.4, inplace=True)
    def forward(self, Xq, Xk):
        # assert(Xq.shape[1] == Xk.shape[1])
        # Q = self.q_gen(Xq)
        # Q = Xq
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

        # print('shape of Q:{}, K:{}, V:{}'.format(Q.shape, K.shape, V.shape))
        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)

        # H_std = H.std(2,keepdim=True)
        # H_mean= H.mean(2,keepdim=True)

        # H = (H-H_mean)/(H_std + 1e-6)
        # H1 = H.reshape([b*self.h , -1 , pq])
        # H1 = self.norm_layer(H1)
        # H = H1.reshape([b, self.h, -1 , pq])

        H = nn.functional.softmax(H, dim=-2)
        # H_01, H_02 = self.GradNorm_layer(H)

        sparse_loss = sparse_constrain(H)

        # out = torch.einsum('bhij,bhcj->bhci',H,Q)
        out = torch.einsum('bhij,bhki->bhkj',H,Q)
        # print('shape of feq H:{}, Q:{} ')
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
        # assert(Xq.shape[1] == Xk.shape[1])
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

        # print('shape of Q:{}, K:{}, V:{}'.format(Q.shape, K.shape, V.shape))
        H = torch.einsum('bhci,bhcj->bhij', K,V) / math.sqrt(c / self.h)
        # H_std = H.std(2,keepdim=True)
        # H_mean= H.mean(2,keepdim=True)

        # H = (H-H_mean)/(H_std + 1e-6)
        # H1 = H.reshape([b*self.h , -1 , pq])
        # H1 = self.norm_layer(H1)
        # H = H1.reshape([b, self.h, -1 , pq])

        H = nn.functional.softmax(H, dim=-2)
        # H_01, H_02 = self.GradNorm_layer(H)

        sparse_loss = sparse_constrain(H)

        out = torch.einsum('bhij,bhik->bhjk',H,Q)
        out = out.reshape([b,c,pk])
        return out, sparse_loss, H

class ETracking_Net(Module):
    def __init__(self,bias=False):
        super(ETracking_Net, self).__init__()
        self.extractor = Feature_extraction()
        self.CrossTransFormer = CrossTransFormer()
        # self.CrossTransFormer2 = CrossTransFormer02()
        self.SelfTransFormer1 = SelfTransFormer(cin = 1024, cout = 1024)
        # self.SelfTransFormer2 = SelfTransFormer(cin = 1024, cout = 1024)

        # self.Regression_layer = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=bias),
        #                            nn.LeakyReLU(negative_slope=0.1),
        #                            nn.Conv1d(512, 1, kernel_size=1, bias=bias),
        #                            nn.Sigmoid())


        self.Iou_Net = IOU_Net()
        self.Regression_layer = nn.Sequential(nn.Conv1d(1024, 512, kernel_size=1, bias=bias),
                                #    self.bnr,
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(512, 4, kernel_size=1, bias=bias),
                                   nn.Sigmoid())
                                   
        self.bnc = nn.GroupNorm(32, 256)
        self.bnr = nn.GroupNorm(32, 512)

        self.weight_point_01 = nn.Sequential(nn.Conv1d(1024, 128, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),)
        self.weight_point_02 = nn.Sequential(nn.Conv1d(3, 128, kernel_size=1, bias=bias),
                                   nn.LeakyReLU(negative_slope=0.1),)
        self.weight_point_03 = nn.Sequential(nn.Conv1d(256, 1, kernel_size=1, bias=bias))

        self.Regress_relation = nn.Sequential(nn.Conv1d(1024*2, 512, kernel_size=1, bias=bias),
                                #    self.bnr,
                                   nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv1d(512, 1, kernel_size=1, bias=bias),
                                   nn.Sigmoid())

        self.Classifier = nn.Sequential(nn.Conv1d(1024, 256, kernel_size=1, bias=bias),
                                   self.bnc,
                                   nn.ReLU(inplace=True),
                                   nn.Conv1d(256, 1, kernel_size=1, bias=bias),
                                   nn.Sigmoid())


    def _weight_point(self, fea_in, pos):
        fea = self.weight_point_01(fea_in)

        pos = self.weight_point_02(pos)

        fea_out = torch.cat([fea, pos], dim = 1)

        weight_point = self.weight_point_03(fea_out)

        weight_point = nn.functional.softmax(weight_point, dim = 2)

        fea_out = weight_point*fea_in

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
        # flag1 = Pos[:,0,:] >= Box[:,0][:,None] - Box[:,2][:,None] / 2
        # flag2 = Pos[:,1,:] >= Box[:,1][:,None] - Box[:,3][:,None] / 2
        # flag3 = Pos[:,0,:] <= Box[:,0][:,None] + Box[:,2][:,None] / 2
        # flag4 = Pos[:,1,:] <= Box[:,1][:,None] + Box[:,3][:,None] / 2

        # Flag = flag1*flag2*flag3 * flag4


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
        # assert(Ns == Ns0)
        # assert(Nt == Nt1)
        Flag_Seq = self._Point_in_box(Seq_Pos, Seq_box)
        Flag_Temp = self._Point_in_box(Temp_pos, Temp_box)
        Flag_Temp_neg = -Flag_Temp + 1


        # print('point of Seq:{}, Temp:{}'.format(Flag_Seq.sum(-1).mean(),Flag_Temp.sum(-1).mean()))
        # print('Flag_Temp:{}, Flag_Seq:{}'.format(Flag_Temp.sum(-1),Flag_Seq.sum(-1)))
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
        # print('shape of Dist0:{}'.format(Dist.shape))
        if Temp == True:
            Dist = Dist + Pnt[:,2,:][:,None,:]**2
        else:
            Dist = Dist + (Pnt[:,2,:].max(1, keepdim = True)[0] - Pnt[:,2,:])[:,None,:]**2
        Dist = Dist.sqrt()*2
        # print('shape of Dist1:{}'.format(Dist.shape))
        
        Norm = (GT[:,2:][:,:,None]/2)**2
        Norm = Norm.sum(1, keepdim=True).sqrt()

        Dist = Dist/ (Norm + 1e-6)

        # mask = torch.pow(10, - Dist)
        # mask = - torch.Sigmoid(-20*mask + 10) + 1
        # f(x)=-((1)/(1+ℯ^(-50 x+50)))+1
        # mask = -50*Dist + 50
        mask = torch.exp(Dist.detach())
        mask = 1/(mask+1e-6)
        # mask = 1 - mask
        # print('shape of Fea2:{}, mask:{}'.format(Fea.shape, mask.shape))
        Fea01 = Fea * mask
        return Fea01

    def eval_(self,X, Temp, GT, GT_temp):
        # print('max:{},min:{}'.format(X.max(),X.min()))
        assert(torch.isnan(X).sum()==0)
        assert(torch.isnan(Temp).sum()==0)
        assert(X.shape[1] == 4)
        assert(Temp.shape[1] == 5)
        # B,C,P = Temp.shape
        B,C,P = X.shape
        X0 = torch.zeros([B,1,P],device = X.device).float()
        X = torch.cat([X,X0],dim = 1)
        X = self.extractor(X,out_num=1024,Is_temp=False)
        X_pos = X[:,:3,:]
        B,C,P = Temp.shape
        temp_point = Temp[:,4,:].abs().sum()
        ratio = temp_point / (B * P)
        # print('---------------------------temp_point:{}, ratio:{}----------------'.format(temp_point, ratio))
        Temp = self.extractor(Temp, out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        Temp_pos = Temp[:,:3,:]
        X = X[:,5:,:]
        temp_point = Temp[:,4,:].abs().detach()
        sum_point = temp_point.sum(-1,keepdim=True)

        Temp = Temp[:,5:,:]
        # print('Before Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        X = self.SelfTransFormer1(X)
        Temp = self.SelfTransFormer1(Temp)
        # print('After Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(X,Temp)

        Pos_01, Neg_01 = self._cal_Transmission(H_01, X_pos, Temp_pos, GT, GT_temp)

        # Adapted_tem01 = self._apply_mask(Temp_output, Temp_pos, GT_temp,Temp=True)
        # X01 = self._apply_mask(X, X_pos, GT,Temp=False)

        # Adapted_tem01 = torch.cat([Adapted_tem01,Adapted_tem01.detach(),Adapted_tem01.detach(),Adapted_tem01.detach()],dim = 2)

        # Distance = earth_mover_distance(X01, Adapted_tem01)
        Distance = torch.zeros(20)


        X, sparse_loss_02, H_02 = self.CrossTransFormer2(X,Temp_output)

        Pos_02, Neg_02 = self._cal_Transmission(H_02, X_pos, Temp_pos, GT, GT_temp)
        # print('sparse loss_01:{}, sparse_loss_02:{}, distance:{}'.format(sparse_loss_01.mean(), sparse_loss_02.mean(), Distance.mean()))
        #  = Temp_output
        # X = self.SelfTransFormer2(X)

        # Class = self.Classifier(X)
        # X = torch.mean(X,dim=-1,keepdim=True)
        # bbox = self.Regress(X)

        # return bbox.permute([0,2,1]), Class.permute([0,2,1])

        # Class = self.Classifier(X)
        # X = torch.mean(X,dim=-1,keepdim=True)
        temp_point = temp_point[:,None,:]
        # print('shape of X:{}, temp_point:{},sum_point:{}'.format(X.shape,temp_point.shape,sum_point.shape))
        X = X*temp_point
        X = X.sum(-1,keepdim=True) / (sum_point[:,:,None] + 1e-6)
        bbox = self.Regress(X)

        return bbox.permute([0,2,1]), bbox.permute([0,2,1]), Distance, sparse_loss_02 + sparse_loss_01, Pos_01, Pos_02, Neg_01, Neg_02, H_01, H_02

    def forward(self,X, Temp, GT, GT_temp):
        # print('max:{},min:{}'.format(X.max(),X.min()))
        assert(torch.isnan(X).sum()==0)
        assert(torch.isnan(Temp).sum()==0)
        assert(X.shape[1] == 4)
        assert(Temp.shape[1] == 5)
        # B,C,P = Temp.shape
        B,C,P = X.shape
        X0 = torch.zeros([B,1,P],device = X.device).float()
        X = torch.cat([X,X0],dim = 1)
        X = self.extractor(X,out_num=1024,Is_temp=False)
        X_pos = X[:,:3,:]
        B,C,P = Temp.shape
        temp_point = Temp[:,4,:].abs().sum()
        ratio = temp_point / (B * P)
        # print('---------------------------temp_point:{}, ratio:{}----------------'.format(temp_point, ratio))
        Temp = self.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
        Temp_pos = Temp[:,:3,:]
        X = X[:,5:,:]
        temp_point = Temp[:,4,:].abs().detach()
        sum_point = temp_point.sum(-1,keepdim=True)

        Temp = Temp[:,5:,:]
        # print('Before Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        X = self.SelfTransFormer1(X)
        Temp = self.SelfTransFormer1(Temp)
        # print('After Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
        # Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(X,Temp)
        Temp_output, sparse_loss_01, H_01 = self.CrossTransFormer(X,Temp)
        # Pos_01, Neg_01 = self._cal_Transmission(H_01, X_pos, Temp_pos, GT, GT_temp)

        # Adapted_tem01 = self._apply_mask(Temp_output, Temp_pos, GT_temp,Temp=True)
        # X01 = self._apply_mask(X, X_pos, GT,Temp=False)

        # Adapted_tem01 = torch.cat([Adapted_tem01,Adapted_tem01.detach(),Adapted_tem01.detach(),Adapted_tem01.detach()],dim = 2)

        # Distance = earth_mover_distance(X01, Adapted_tem01)
        Distance = torch.zeros(20)


        # Temp_output, sparse_loss_02, H_02 = self.CrossTransFormer2(X,Temp_output)

        # Pos_02, Neg_02 = self._cal_Transmission(H_02, X_pos, Temp_pos, GT, GT_temp)
        # print('sparse loss_01:{}, sparse_loss_02:{}, distance:{}'.format(sparse_loss_01.mean(), sparse_loss_02.mean(), Distance.mean()))
        #  = Temp_output
        # X = self.SelfTransFormer2(X)

        # Class = self.Classifier(X)
        # X = torch.mean(X,dim=-1,keepdim=True)
        # bbox = self.Regress(X)

        # return bbox.permute([0,2,1]), Class.permute([0,2,1])

        # Class = self.Classifier(X)
        # X = torch.mean(X,dim=-1,keepdim=True)

        # temp_point = temp_point[:,None,:]
        # # print('shape of X:{}, temp_point:{},sum_point:{}'.format(X.shape,temp_point.shape,sum_point.shape))
        # Temp_output = Temp_output*temp_point
        # # X = X.max(-1,keepdim=True) / (sum_point[:,:,None] + 1e-6)
        # B0, C0, P0 = X.shape
        # # Temp_output = Temp_output.sum(-1,keepdim=True)/ (sum_point[:,:,None] + 1e-6)
        # weighted_Temp =self._weight_point(Temp_output, Temp_pos)
        
        # Temp_output = weighted_Temp.repeat([1,1,P0])

        IOU_pred, IOU_GT = self.Iou_Net.train_forward(X = X, Temp = Temp, X_pos = X_pos, Box = GT, sample_num = 100)
        prob = self.Regress_relation(torch.cat([X, Temp_output],dim=1))
        assert(prob.shape[1] == 1)

        Flag = self._Point_in_box(X_pos,GT)
        GT_prob = Flag.float()
        GT_prob_neg = -GT_prob + 1.0
        # prob_loss = prob[:,0,:]
        # corp_loss_pos = - torch.log(pos_01 +1e-5) - torch.log(pos_02 +1e-5) 
        # corp_loss_neg = - torch.log(1-neg_01 +1e-5) - torch.log(1-neg_02 +1e-5)
        corp_loss_pos = - torch.log(prob+1e-6) * GT_prob
        corp_loss_neg = - torch.log(-prob + 1.0 +1e-6) * GT_prob_neg
        corp_loss_pos = corp_loss_pos.sum([1,2]) / (GT_prob.sum([1,2]) + 1e-6)
        corp_loss_neg = corp_loss_neg.sum([1,2]) / (GT_prob_neg.sum([1,2]) + 1e-6)
        X1 = X * prob

        # X1 = X1.max(dim=-1,keepdim = True)[0]
        weighted_fea =self._weight_point(X1, X_pos)

        bbox = self.Regression_layer(weighted_fea)
        # print('shape of bbox:{}'.format(bbox.shape))
        return bbox.permute([0,2,1]), bbox.permute([0,2,1]), Distance, corp_loss_pos, corp_loss_neg, IOU_pred, IOU_GT

    # def forward(self,X, Temp):
    #     # print('max:{},min:{}'.format(X.max(),X.min()))
    #     assert(torch.isnan(X).sum()==0)
    #     assert(torch.isnan(Temp).sum()==0)
    #     X = self.extractor(X,out_num=1024,Is_temp=False)
    #     B,C,P = Temp.shape
    #     temp_point = Temp[:,3,:].abs().sum()
    #     ratio = temp_point / (B * P)
    #     Temp = self.extractor(Temp,out_num=256,Is_temp=True,Temp_ratio=max(0.02,ratio))
    #     X = X[:,4:,:]
    #     temp_point = Temp[:,3,:].abs().detach()
    #     sum_point = temp_point.sum(-1,keepdim=True)

    #     Temp = Temp[:,4:,:]
    #     # print('Before Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
    #     X = self.SelfTransFormer1(X)
    #     Temp = self.SelfTransFormer1(Temp)
    #     # print('After Tranformer ---- shape of X:{}, shape of Temp:{}'.format(X.shape,Temp.shape))
    #     Adapted_tem = self.CrossTransFormer(X,Temp)

    #     Adapted_tem01 = torch.cat([Adapted_tem,Adapted_tem.detach(),Adapted_tem.detach(),Adapted_tem.detach()],dim = 2)

    #     Distance = earth_mover_distance(X, Adapted_tem01)

    #     X = self.CrossTransFormer(X,Adapted_tem)

    #     X = self.SelfTransFormer2(X)

    #     # Class = self.Classifier(X)
    #     # X = torch.mean(X,dim=-1,keepdim=True)
    #     # bbox = self.Regress(X)

    #     # return bbox.permute([0,2,1]), Class.permute([0,2,1])

    #     # Class = self.Classifier(X)
    #     # X = torch.mean(X,dim=-1,keepdim=True)
    #     temp_point = temp_point[:,None,:]
    #     # print('shape of X:{}, temp_point:{},sum_point:{}'.format(X.shape,temp_point.shape,sum_point.shape))
    #     X = X*temp_point
    #     X = X.sum(-1,keepdim=True) / (sum_point[:,:,None] + 1e-6)
    #     bbox = self.Regress(X)

    #     return bbox.permute([0,2,1]), bbox.permute([0,2,1]), Distance

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
