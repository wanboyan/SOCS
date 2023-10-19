# @Time    : 06/05/2021
# @Author  : Wei Chen
# @Project : Pycharm
import torch.nn as nn
import nfmodel.gcn3d as gcn3d
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as functional
from EQNet.eqnet.models.query_producer.qnet import *
import absl.flags as flags
FLAGS = flags.FLAGS



class Pose_Ts(nn.Module):
    def __init__(self,k=6,F=1283):
        super(Pose_Ts, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs





class GCN3D_segR(nn.Module):
    def __init__(self, support_num, neighbor_num):
        super(GCN3D_segR, self).__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.vecnum = 3
        dim_fuse = sum([128, 128, 256, 256, 512])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, self.vecnum, 1),
        )

    def forward(self,vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """

        bs, vertice_num, _ = vertices.size()

        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace= True)


        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1,2)).transpose(1,2), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1,2)).transpose(1,2), inplace= True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                     v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0] #(bs, f)
        feature_dict={'conv0':{'pos':vertices,'fea':fm_0},
                      'conv1':{'pos':vertices,'fea':fm_1},
                      'conv2':{'pos':v_pool_1,'fea':fm_2},
                      'conv3':{'pos':v_pool_1,'fea':fm_3},
                      'conv4':{'pos':v_pool_2,'fea':fm_4},

                      }
        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim= 2)

        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch) ## B N 50?
        recon=pred
        point_fea=fm_fuse
        global_fea=f_global
        return recon,point_fea,global_fea,feature_dict


class Point_center(nn.Module):
    def __init__(self):
        super(Point_center, self).__init__()

        # self.conv1 = torch.nn.Conv2d(12, 64, 1) ##c
        self.conv1 = torch.nn.Conv1d(3, 128, 1) ## no c
        self.conv2 = torch.nn.Conv1d(128, 256, 1)

        ##here
        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        # self.conv4 = torch.nn.Conv1d(1024,1024,1)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)

        # self.bn4 = nn.BatchNorm1d(1024)
        # self.global_feat = global_feat

    def forward(self, x,obj):## 5 6 30 1000
        batchsize = x.size()[0]
        n_pts = x.size()[2]

        x = F.relu(self.bn1(self.conv1(x))) ## 5 64 30 1000
        x = F.relu(self.bn2(self.conv2(x))) ## 5 64 1 1000
        x = (self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x2 = torch.max(x, -1, keepdim=True)[0]#5 512 1
        # x2=torch.mean(x, -1, keepdim=True)
        obj = obj.view(-1, 1)
        one_hot = torch.zeros(batchsize, 16).scatter_(1, obj.cpu().long(), 1)
        # print(one_hot[1,:])
        if torch.cuda.is_available():
            one_hot = one_hot.cuda()
        one_hot2 = one_hot.unsqueeze(2)
        return torch.cat([x2, one_hot2],1)
        #
        # return x2
    # return pointfeat2

class Point_center_res_cate(nn.Module):
    def __init__(self):
        super(Point_center_res_cate, self).__init__()

        # self.feat = Point_vec_edge()
        self.feat = Point_center()
        self.conv1 = torch.nn.Conv1d(512+16, 256,1)
        self.conv2 = torch.nn.Conv1d(256, 128,1)
        # self.drop1 = nn.Dropout(0.1)
        self.conv3 = torch.nn.Conv1d(128, 6,1 )


        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x, obj):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # print(x.size())
        # tes
        x = self.feat(x, obj) ## Bx1024x1xN
        T_feat = x
        # x=x.squeeze(2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = (self.bn2(self.conv2(x)))

        x=self.drop1(x)
        x = self.conv3(x)



        x = x.squeeze(2)
        x=x.contiguous()##Bx6
        xt = x[:,0:3]
        xs = x[:,3:6]
        if torch.isnan(xs).any():
            print('asd')
        return xt,xs


class Rot_green(nn.Module):
    def __init__(self, k=3,F=1280):
        super(Rot_green, self).__init__()
        self.f=F
        self.k = k


        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x


class Rot_red(nn.Module):
    def __init__(self, k=24,F=1036):
        super(Rot_red, self).__init__()
        self.f=F
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.f , 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256,256,1)
        self.conv4 = torch.nn.Conv1d(256,self.k,1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x=self.drop1(x)
        x = self.conv4(x)

        x=x.squeeze(2)
        x = x.contiguous()


        return x

def square_distance(src, dst):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py

    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Code from: https://github.com/qq456cvb/Point-Transformers/blob/master/pointnet_util.py
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim_global,dim_inp, dim, nneigh=7, reduce_dim=True, separate_delta=True):
        super().__init__()

        # dim_inp = dim
        # dim = dim  # // 2
        self.dim = dim

        self.nneigh = nneigh
        self.separate_delta = separate_delta

        self.fc_delta = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        #if self.separate_delta:
        #    self.fc_delta2 = nn.Sequential(
        #        nn.Linear(3, dim),
        #        nn.ReLU(),
        #        nn.Linear(dim, dim)
        #
        #    )

        self.fc_gamma = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.w_k_global = nn.Linear(dim_global, dim, bias=False)
        self.w_v_global = nn.Linear(dim_global, dim, bias=False)

        self.w_qs = nn.Linear(dim_global, dim, bias=False)

        self.w_ks = nn.Linear(dim_inp, dim, bias=False)
        self.w_vs = nn.Linear(dim_inp, dim, bias=False)

        if not reduce_dim:
            self.fc = nn.Linear(dim, dim_inp)
        self.reduce_dim = reduce_dim

    # xyz_q: B x n_queries x 3
    # lat_rep: B x dim
    # xyz: B x n_anchors x 3,
    # points: B x n_anchors x dim
    def forward(self, xyz_q, lat_rep, xyz, points):
        with torch.no_grad():
            dists = square_distance(xyz_q, xyz)
            ## knn group
            knn_idx = dists.argsort()[:, :, :self.nneigh]  # b x nQ x k
            #print(knn_idx.shape)

            #knn = KNN(k=self.nneigh, transpose_mode=True)
            #_, knn_idx = knn(xyz, xyz_q)  # B x npoint x K
            ##
            #print(knn_idx.shape)

        b, nQ, _ = xyz_q.shape
        # b, nK, dim = points.shape

        if len(lat_rep.shape) == 2:
            q_attn = self.w_qs(lat_rep).unsqueeze(1).repeat(1, nQ, 1)
            k_global = self.w_k_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(1).repeat(1, nQ, 1).unsqueeze(2)
        else:
            q_attn = self.w_qs(lat_rep)
            k_global = self.w_k_global(lat_rep).unsqueeze(2)
            v_global = self.w_v_global(lat_rep).unsqueeze(2)

        k_attn = index_points(self.w_ks(points),
                              knn_idx)  # b, nQ, k, dim  # self.w_ks(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        k_attn = torch.cat([k_attn, k_global], dim=2)
        v_attn = index_points(self.w_vs(points), knn_idx)  # #self.w_vs(points).unsqueeze(1).repeat(1, nQ, 1, 1)
        v_attn = torch.cat([v_attn, v_global], dim=2)
        xyz = index_points(xyz, knn_idx)  # xyz = xyz.unsqueeze(1).repeat(1, nQ, 1, 1)
        pos_encode = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
        pos_encode = torch.cat([pos_encode, torch.zeros([b, nQ, 1, self.dim], device=pos_encode.device)],
                               dim=2)  # b, nQ, k+1, dim
        if self.separate_delta:
            pos_encode2 = self.fc_delta(xyz_q[:, :, None] - xyz)  # b x nQ x k x dim
            pos_encode2 = torch.cat([pos_encode2, torch.zeros([b, nQ, 1, self.dim], device=pos_encode2.device)],
                                    dim=2)  # b, nQ, k+1, dim
        else:
            pos_encode2 = pos_encode

        attn = self.fc_gamma(q_attn[:, :, None] - k_attn + pos_encode)
        attn = functional.softmax(attn, dim=-2)  # b x nQ x k+1 x dim

        res = torch.einsum('bmnf,bmnf->bmf', attn, v_attn + pos_encode2)  # b x nQ x dim

        if not self.reduce_dim:
            res = self.fc(res)
        return res

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Copied from https://github.com/autonomousvision/convolutional_occupancy_networks

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class PointTransformerDecoderOcc(nn.Module):
    """
    AIR-Net decoder

    Attributes:
        dim_inp int: dimensionality of encoding (global and local latent vectors)
        dim int: internal dimensionality
        nneigh int: number of nearest anchor points to draw information from
        hidden_dim int: hidden dimensionality of final feed-forward network
        n_blocks int: number of blocks in feed forward network
    """
    def __init__(self, dim_global,dim_inp=1280, dim=200, nneigh=7, hidden_dim=64, n_blocks=5):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks


        self.ct1 = CrossTransformerBlock(dim_global,dim_inp, dim, nneigh=nneigh)
        #self.fc_glob = nn.Linear(dim_inp, dim)

        self.init_enc = nn.Linear(dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_dim) for i in range(n_blocks)
        ])

        self.fc_c = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_dim, 7)

        self.actvn = F.relu
        self.scale_branch = nn.Linear(512, 3)
    def forward(self, xyz_q, encoding):
        """
        TODO update commont to include encoding dict
        :param xyz_q [B x n_queries x 3]: queried 3D coordinates
        :param lat_rep [B x dim_inp]: global latent vectors
        :param xyz [B x n_anchors x 3]: anchor positions
        :param feats [B x n_anchros x dim_inp]: local latent vectors
        :return: occ [B x n_queries]: occupancy probability for each queried 3D coordinate
        """

        lat_rep = encoding['z']
        global_feat=lat_rep
        xyz = encoding['anchors']
        feats = encoding['anchor_feats']

        lat_rep = self.ct1(xyz_q, lat_rep, xyz, feats)  # + self.fc_glob(lat_rep).unsqueeze(1).repeat(1, xyz_q.shape[1], 1) +
        net = self.init_enc(lat_rep)

        for i in range(self.n_blocks):
            net = net + self.fc_c[i](lat_rep)
            net = self.blocks[i](net)
        out= self.fc_out(self.actvn(net))
        nocs =out[:,:,:3]
        occ=nn.functional.sigmoid(out[:,:,3:4])
        w3d=out[:,:,4:7]
        scale = self.scale_branch(global_feat).exp()
        return nocs,occ,lat_rep,w3d,scale


class DampingNet(nn.Module):
    def __init__(self, num_params=9,min=-6,max=5):
        super().__init__()
        const = torch.zeros(num_params)
        self.min_=min
        self.max_=max
        self.register_parameter('const', torch.nn.Parameter(const))


    def forward(self):
        lambda_ = 10.**(self.min_ + self.const.sigmoid()*(self.max_ - self.min_))
        return lambda_



class MyQNet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.qnet = QNet(model_cfg)
        self.q_target_chn = self.qnet.model_cfg.get('Q_TARGET_CHANNEL')
        self.fc_out_second = nn.Linear(self.q_target_chn, FLAGS.bin_size*3)
        self.fc_out_first = nn.Linear(self.q_target_chn, FLAGS.bin_size*3//10)
        self.dampingnet=DampingNet()
    def forward(self, query_points, feature_dicts,stage='first',debug=False):
        batch_size=query_points.shape[0]
        multi_scale_support_sets={}
        data_dict={}
        for source,set in feature_dicts.items():
            pos=set['pos']
            point_num=pos.shape[1]
            batch_info=torch.arange(0,batch_size).view(-1,1,1).repeat(1,point_num,1).float().cuda()
            pos=torch.cat([batch_info,pos],dim=-1)
            pos=pos.reshape(-1,4)
            fea=set['fea']
            fea_dim=fea.shape[-1]
            fea=fea.reshape(-1,fea_dim)
            if FLAGS.cut_backbone:
                fea=fea.detach()
            multi_scale_support_sets[source]={'support_features':fea,'support_points':pos}
        data_dict['multi_scale_support_sets']=multi_scale_support_sets
        data_dict['batch_size']=batch_size
        data_dict['query_positions']=query_points
        query_features=self.qnet(data_dict)['query_features']
        if stage=='first':
            out=self.fc_out_first(query_features)
        elif stage=='second':
            out=self.fc_out_second(query_features)
        if debug:
            return out,query_features
        else:
            return out