import torch
import torch.nn as nn
import absl.flags as flags
from absl import app
from tools.rot_utils import get_vertical_rot_vec, get_rot_mat_y_first

FLAGS = flags.FLAGS  # can control the weight of each term here
import torch.nn.functional as F

class consistency_loss(nn.Module):
    def __init__(self):
        super(consistency_loss, self).__init__()
        self.loss_func = nn.L1Loss()

    def forward(self, bin_list,pred_fsnet_list,mean_shape,sym):
        query_points=bin_list['query_points']
        pred_nocs_bin=bin_list['pred_nocs_bin']
        x_bin_resolution=bin_list['x_bin_resolution']
        y_bin_resolution=bin_list['y_bin_resolution']
        z_bin_resolution=bin_list['z_bin_resolution']
        x_start=bin_list['x_start']
        y_start=bin_list['y_start']
        z_start=bin_list['z_start']
        bs=query_points.shape[0]
        p_green_R=pred_fsnet_list['Rot1']
        p_red_R=pred_fsnet_list['Rot2']
        Pred_T=pred_fsnet_list['Tran'].reshape(bs,1,3)
        Pred_s=(pred_fsnet_list['Size']+mean_shape).reshape(bs,1,3)
        pred_nocs_bin=pred_nocs_bin.permute(0,2,3,1)
        index=torch.arange(FLAGS.bin_size,device=pred_nocs_bin.device).reshape(1,1,1,-1)
        pred_nocs_bin_max=torch.sum(F.gumbel_softmax(pred_nocs_bin,hard=True,dim=-1)*index,dim=-1)
        pred_nocs_y=(pred_nocs_bin_max[:,:,1]*y_bin_resolution+y_start)*Pred_s[:,:,1]

        if sym[0][0]==0:
            p_R = get_rot_mat_y_first(p_green_R.view(bs, -1), p_red_R.view(bs, -1))
            fs_nocs=(query_points-Pred_T) @ p_R
            pred_nocs_x=(pred_nocs_bin_max[:,:,0]*x_bin_resolution+x_start)*Pred_s[:,:,0]
            pred_nocs_z=(pred_nocs_bin_max[:,:,2]*z_bin_resolution+z_start)*Pred_s[:,:,2]
            pred_nocs=torch.stack([pred_nocs_x,pred_nocs_y,pred_nocs_z],dim=-1)
            loss=self.loss_func(pred_nocs,fs_nocs)
        else:
            fs_re_y=torch.sum((query_points-Pred_T)*(p_green_R.view(bs, 1,-1)),dim=-1)
            pred_nocs_x=(pred_nocs_bin_max[:,:,0]*x_bin_resolution+x_start) *Pred_s[:,:,0]# bx3
            fs_re_x = torch.norm(torch.cross(query_points-Pred_T, p_green_R.view(bs, 1,-1), dim=-1),dim=-1)
            fs_nocs=torch.stack([fs_re_x,fs_re_y],dim=-1)
            pred_nocs=torch.stack([pred_nocs_x,pred_nocs_y],dim=-1)
            loss=self.loss_func(pred_nocs,fs_nocs)
        return FLAGS.consistency_w*loss

class inter_loss(nn.Module):
    def __init__(self):
        super(inter_loss, self).__init__()
        self.loss_func = nn.L1Loss()
        # self.kl=nn.KLDivLoss(log_target=True,reduction='batchmean')

    def forward(self,pred_fsnet_list_1,pred_fsnet_list_2,bin_list_1,bin_list_2,sym):
        pred_nocs_bin_1=bin_list_1['pred_nocs_bin']
        pred_nocs_bin_2=bin_list_2['pred_nocs_bin']
        x_bin_resolution=bin_list_1['x_bin_resolution']
        y_bin_resolution=bin_list_1['y_bin_resolution']
        z_bin_resolution=bin_list_1['z_bin_resolution']
        x_start=bin_list_1['x_start']
        y_start=bin_list_1['y_start']
        z_start=bin_list_1['z_start']

        pred_nocs_bin_1=pred_nocs_bin_1.permute(0,2,3,1)
        index=torch.arange(FLAGS.bin_size,device=pred_nocs_bin_1.device).reshape(1,1,1,-1)
        pred_nocs_bin_max_1=torch.sum(F.gumbel_softmax(pred_nocs_bin_1,hard=True,dim=-1)*index,dim=-1)
        pred_nocs_x_1=(pred_nocs_bin_max_1[:,:,0]*x_bin_resolution+x_start)
        pred_nocs_y_1=(pred_nocs_bin_max_1[:,:,1]*y_bin_resolution+y_start)
        pred_nocs_z_1=(pred_nocs_bin_max_1[:,:,2]*z_bin_resolution+z_start)


        pred_nocs_bin_2=pred_nocs_bin_2.permute(0,2,3,1)
        index=torch.arange(FLAGS.bin_size,device=pred_nocs_bin_2.device).reshape(1,1,1,-1)
        pred_nocs_bin_max_2=torch.sum(F.gumbel_softmax(pred_nocs_bin_2,hard=True,dim=-1)*index,dim=-1)
        pred_nocs_x_2=(pred_nocs_bin_max_2[:,:,0]*x_bin_resolution+x_start)
        pred_nocs_y_2=(pred_nocs_bin_max_2[:,:,1]*y_bin_resolution+y_start)
        pred_nocs_z_2=(pred_nocs_bin_max_2[:,:,2]*z_bin_resolution+z_start)

        p_green_R_1=pred_fsnet_list_1['Rot1']
        p_red_R_1=pred_fsnet_list_1['Rot2']
        p_green_R_2=pred_fsnet_list_2['Rot1']
        p_red_R_2=pred_fsnet_list_2['Rot2']
        bs=p_green_R_1.shape[0]
        Pred_T_1=pred_fsnet_list_1['Tran'].reshape(bs,1,3)
        Pred_T_2=pred_fsnet_list_2['Tran'].reshape(bs,1,3)

        if sym[0][0]==0:
            r1=torch.cat([p_green_R_1,p_red_R_1],dim=-1)
            r2=torch.cat([p_green_R_2,p_red_R_2],dim=-1)
            t1=Pred_T_1
            t2=Pred_T_2
            nocs_1=torch.stack([pred_nocs_x_1,pred_nocs_y_1,pred_nocs_z_1],dim=-1)
            nocs_2=torch.stack([pred_nocs_x_2,pred_nocs_y_2,pred_nocs_z_2],dim=-1)
        else:
            r1=torch.cat([p_green_R_1],dim=-1)
            r2=torch.cat([p_green_R_2],dim=-1)
            t1=Pred_T_1
            t2=Pred_T_2
            nocs_1=torch.stack([pred_nocs_x_1,pred_nocs_y_1],dim=-1)
            nocs_2=torch.stack([pred_nocs_x_2,pred_nocs_y_2],dim=-1)
        loss_inter_r=torch.mean(torch.norm(r1 - r2, dim=-1))
        loss_inter_t=torch.mean(torch.norm(t1 - t2, dim=-1))
        loss_inter_nocs=torch.mean(torch.norm(nocs_1 - nocs_2, dim=-1))
        return {
            'inter_r':loss_inter_r,
            'inter_t':loss_inter_t,
            'inter_nocs':loss_inter_nocs,
        }



class inter_loss_v2(nn.Module):
    def __init__(self):
        super(inter_loss_v2, self).__init__()
        self.loss_func = nn.L1Loss()
        # self.kl=nn.KLDivLoss(log_target=True,reduction='batchmean')

    def forward(self,pred_fsnet_list_1,pred_fsnet_list_2,bin_list_1,bin_list_2,sym):
        pred_nocs_bin_1=bin_list_1['pred_nocs_bin']
        pred_nocs_bin_2=bin_list_2['pred_nocs_bin']


        pred_nocs_bin_1=pred_nocs_bin_1.permute(0,2,3,1)
        pred_nocs_bin_2=pred_nocs_bin_2.permute(0,2,3,1)


        p_green_R_1=pred_fsnet_list_1['Rot1']
        p_red_R_1=pred_fsnet_list_1['Rot2']
        p_green_R_2=pred_fsnet_list_2['Rot1']
        p_red_R_2=pred_fsnet_list_2['Rot2']
        bs=p_green_R_1.shape[0]
        Pred_T_1=pred_fsnet_list_1['Tran'].reshape(bs,1,3)
        Pred_T_2=pred_fsnet_list_2['Tran'].reshape(bs,1,3)
        Pred_s_1=pred_fsnet_list_1['Size'].reshape(bs,1,3)
        Pred_s_2=pred_fsnet_list_2['Size'].reshape(bs,1,3)
        if sym[0][0]==0:
            r1=torch.cat([p_green_R_1,p_red_R_1],dim=-1)
            r2=torch.cat([p_green_R_2,p_red_R_2],dim=-1)
            t1=Pred_T_1
            t2=Pred_T_2
            s1=Pred_s_1
            s2=Pred_s_2
            dis1=pred_nocs_bin_1.reshape(-1,FLAGS.bin_size)
            dis2=pred_nocs_bin_2.reshape(-1,FLAGS.bin_size)
        else:
            r1=torch.cat([p_green_R_1],dim=-1)
            r2=torch.cat([p_green_R_2],dim=-1)
            t1=Pred_T_1
            t2=Pred_T_2
            s1=Pred_s_1
            s2=Pred_s_2
            dis1=pred_nocs_bin_1[:,:,:2].reshape(-1,FLAGS.bin_size)
            dis2=pred_nocs_bin_2[:,:,:2].reshape(-1,FLAGS.bin_size)
        loss_inter_r=torch.mean(torch.norm(r1 - r2, dim=-1))
        loss_inter_t=torch.mean(torch.norm(t1 - t2, dim=-1))
        loss_inter_s=torch.mean(torch.norm(s1 - s2, dim=-1))

        # loss_inter_nocs=F.kl_div(dis1.log(),dis2,reduction='batchmean')
        loss_inter_nocs=F.softmax(dis2,dim=1)*(F.log_softmax(dis2,dim=1)-F.log_softmax(dis1,dim=1))
        loss_inter_nocs=loss_inter_nocs.sum(dim=1).mean()
        # print(1)
        return {
            'inter_r':loss_inter_r,
            'inter_t':loss_inter_t,
            'inter_nocs':loss_inter_nocs,
            'inter_s':loss_inter_s
        }




class nocs_loss(nn.Module):
    def __init__(self):
        super(nocs_loss, self).__init__()
        self.loss_func = nn.L1Loss()
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)

    def forward(self,bin_list,sym):
        pred_nocs_bin=bin_list['pred_nocs_bin']
        query_nocs_bin=bin_list['query_nocs_bin']
        if sym[0][0]==0:
            return self.loss_bin_fun(pred_nocs_bin,query_nocs_bin).mean()*FLAGS.interpo_w
        else:
            return self.loss_bin_fun(pred_nocs_bin[:,:,:,:2],query_nocs_bin[:,:,:2]).mean()*FLAGS.interpo_w