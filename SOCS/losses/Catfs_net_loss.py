import torch
import torch.nn as nn
import torch.nn.functional as F
import absl.flags as flags
from absl import app
import mmcv
FLAGS = flags.FLAGS  # can control the weight of each term here

import math


class fs_net_loss(nn.Module):
    def __init__(self):
        super(fs_net_loss, self).__init__()
        if FLAGS.fsnet_loss_type == 'l1':
            self.loss_func_t = nn.L1Loss()
            self.loss_func_s = nn.L1Loss()
            self.loss_func_Rot1 = nn.L1Loss(reduce=False)
            self.loss_func_Rot2 = nn.L1Loss(reduce=False)
            self.loss_func_Recon = nn.L1Loss()
        elif FLAGS.fsnet_loss_type == 'smoothl1':   # same as MSE
            self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot1 = nn.SmoothL1Loss(beta=0.5,reduce=False)
            self.loss_func_Rot2 = nn.SmoothL1Loss(beta=0.5,reduce=False)
            self.loss_func_Recon = nn.SmoothL1Loss(beta=0.3)
        else:
            raise NotImplementedError

    def forward(self, name_list, pred_list, gt_list, sym):
        loss_list = {}
        gt_R=gt_list['Rotation']
        loss_list["Rot1"] = FLAGS.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_R, sym)

        loss_list["Rot2"] = FLAGS.rot_2_w * self.cal_loss_Rot2(pred_list["Rot2"], gt_R, sym)

        loss_list["Recon"] = FLAGS.recon_w * self.cal_loss_Recon(pred_list["Recon"], gt_list["Recon"])

        loss_list["Tran"] = FLAGS.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])

        loss_list["Size"] = FLAGS.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])

        return loss_list

    def cal_loss_R_con(self, p_rot_g, p_rot_r, g_rot_g, g_rot_r, p_g_con, p_r_con, sym):
        dis_g = p_rot_g - g_rot_g    # bs x 3
        dis_g_norm = torch.norm(dis_g, dim=-1)   # bs
        p_g_con_gt = torch.exp(-13.7 * dis_g_norm * dis_g_norm)  # bs
        res_g = self.loss_func_r_con(p_g_con_gt, p_g_con)
        res_r = 0.0
        bs = p_rot_g.shape[0]
        for i in range(bs):
            if sym[i, 0] == 0:
                dis_r = p_rot_r[i, ...] - g_rot_r[i, ...]
                dis_r_norm = torch.norm(dis_r)   # 1
                p_r_con_gt = torch.exp(-13.7 * dis_r_norm * dis_r_norm)
                res_r += self.loss_func_r_con(p_r_con_gt, p_r_con[i])
        res_r = res_r / bs
        return res_g + res_r


    def cal_loss_Rot1(self, pred_v, gt_R,sym):
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
        sym_Rs=torch.zeros([2,3,3],dtype=torch.float32, device=pred_v.device)
        sym_Rs[0]=torch.tensor([[1, 0, 0], [0, 1, 0],[0, 0, 1]])
        sym_Rs[1]=torch.tensor([[1, 0, 0], [0, -1, 0],[0, 0, -1]])
        corner=torch.tensor([0, 1, 0],dtype=torch.float).to(pred_v.device)
        for i in range(bs):
            sym_now=sym[i]
            pred_v_now = pred_v[i, ...]
            gt_R_now=gt_R[i]
            if sym_now[2]==1:
                can_Rs=torch.matmul(gt_R_now,sym_Rs)
                can_gts=torch.matmul(can_Rs,corner)
                can_losses=self.loss_func_Rot1(pred_v_now,can_gts).mean(-1)
                res[i] = can_losses.min()
            else:
                gt_v=torch.matmul(gt_R_now,corner)
                res[i] = self.loss_func_Rot1(pred_v_now, gt_v).mean(-1)
        res = torch.mean(res)
        return res

    def cal_loss_Rot2(self, pred_v, gt_R,sym):
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        sym_Rs=None
        base=int(sym[0][5].item())
        corner=torch.tensor([1, 0, 0],dtype=torch.float).to(pred_v.device)
        if base>0:
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=pred_v.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                              [0, 1, 0],
                              [-math.sin(theta), 0, math.cos(theta)]])


        for i in range(bs):
            sym_now = sym[i]
            if sym_now[0] == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_R_now=gt_R[i]
                if base>0:

                    can_Rs=torch.matmul(gt_R_now,sym_Rs)
                    can_gts=torch.matmul(can_Rs,corner)
                    can_losses=self.loss_func_Rot2(pred_v_now,can_gts).mean(-1)
                    res_now = can_losses.min()
                else:
                    gt_v=torch.matmul(gt_R_now,corner)
                    res_now = self.loss_func_Rot2(pred_v_now, gt_v).mean(-1)
                res += res_now
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_cosine_dis(self, pred_v, gt_v):
        # pred_v  bs x 6, gt_v bs x 6
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32).to(pred_v.device)
        for i in range(bs):
            pred_v_now = pred_v[i, ...]
            gt_v_now = gt_v[i, ...]
            res[i] = (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
        res = torch.mean(res)
        return res

    def cal_cosine_dis_sym(self, pred_v, gt_v, sym):
        # pred_v  bs x 6, gt_v bs x 6
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += (1.0 - torch.sum(pred_v_now * gt_v_now)) * 2.0
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res


    def cal_rot_regular_angle(self, pred_v1, pred_v2, sym):
        bs = pred_v1.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            if sym[i, 0] == 1:
                continue
            y_direction = pred_v1[i, ...]
            z_direction = pred_v2[i, ...]
            residual = torch.dot(y_direction, z_direction)
            res += torch.abs(residual)
            valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_Recon(self, pred_recon, gt_recon):
        return self.loss_func_Recon(pred_recon, gt_recon)

    def cal_loss_Tran(self, pred_trans, gt_trans):
        return self.loss_func_t(pred_trans, gt_trans)

    def cal_loss_Size(self, pred_size, gt_size):
        return self.loss_func_s(pred_size, gt_size)
