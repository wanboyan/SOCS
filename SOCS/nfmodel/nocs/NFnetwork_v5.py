import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v4 import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts,Point_center_res_cate, \
    VADLogVar,Decoder
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from nnutils.torch_pso import *

def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD



class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)

        self.backbone1=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.two_back:
            self.backbone2=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.feat_for_ts:
            self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        else:
            self.ts=Point_center_res_cate()
        self.loss_fs_net = fs_net_loss()
        self.loss_consistency=consistency_loss()
        self.loss_inter=inter_loss()
        self.loss_coord=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        # self.loss_coord=nn.MSELoss(reduction='mean')
        self.loss_coord_sym=nn.SmoothL1Loss(beta=0.5,reduction='none')
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)



    def forward_fsnet(self,point_fea,pc_center,center):
        feat_for_ts = pc_center
        objs=torch.zeros_like(feat_for_ts[:,0,0])
        T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))
        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)
        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Tran': T + center.squeeze(1),
            'Size': s,
        }
        return pred_fsnet_list

    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                coefficient=None,control_points=None,re_control_points=None,re_coefficient=None,
                model_idx=None,deform_sdf_points=None,
                std_model=None,
                do_refine=False):


        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None




        PC = PC_sample(def_mask, depth, camK, gt_2D)
        real_scale=mean_shape+gt_s
        if FLAGS.aug_deform:
            prop_aug = torch.rand(1)
            if prop_aug < FLAGS.aug_bc_pro:
                canonical_PC=torch.bmm(PC-gt_t.reshape(-1,1,3),gt_R)/real_scale.reshape(-1,1,3)
                if FLAGS.regular_tps:
                    PC2std=torch_tps_regular_transform(canonical_PC,coefficient,control_points)
                    std2target=torch_tps_regular_transform(PC2std,re_coefficient,re_control_points)
                else:
                    PC2std=torch_tps_transform(canonical_PC,coefficient,control_points)
                    std2target=torch_tps_transform(PC2std,re_coefficient,re_control_points)
                new_PC=torch.bmm(std2target*real_scale.reshape(-1,1,3),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
                # show_open3d(new_PC[0].detach().cpu().numpy(),PC[0].detach().cpu().numpy(),)
                PC=new_PC
        PC = PC.detach()
        PC, gt_R, gt_t, gt_s = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                     aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)

        real_scale=mean_shape+gt_s

        pad_nocs=pad_points
        max_real_scale=torch.max(real_scale,dim=-1)[0]
        pad_nocs=pad_nocs*(max_real_scale.reshape(-1,1,1))/(real_scale.reshape(-1,1,3))

        # min_model_point=torch.min(model_point,dim=1,keepdim=True)[0]
        # max_model_point=torch.max(model_point,dim=1,keepdim=True)[0]
        # model_size=max_model_point-min_model_point
        # norm_model_point=model_point/model_size
        #
        # min_std_model_point=torch.min(std_model,dim=1,keepdim=True)[0]
        # max_std_model_point=torch.max(std_model,dim=1,keepdim=True)[0]
        # std_model_size=max_std_model_point-min_std_model_point
        #
        #
        # norm_deform_model=torch_tps_transform(norm_model_point,coefficient,control_points)
        # deform_model=norm_deform_model*std_model_size.cuda()
        # show_open3d(model_point[10].detach().cpu().numpy(),std_model[0].numpy())
        # show_open3d(deform_model[10].detach().cpu().numpy(),std_model[0].numpy())

        query_nocs=sdf_points
        query_num=query_nocs.shape[1]

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        pc_num=PC.shape[1]
        model_idx=model_idx.long()
        vae_loss={}
        vae_loss['KLD']=0
        vae_loss['CD']=0
        bs = PC.shape[0]

        recon,point_fea,global_fea,feature_dict,feature_dict_detach= self.backbone1(pc_center)


        if FLAGS.two_back:
            recon,point_fea,global_fea,_,_= self.backbone2(pc_center)

        a=10
        if FLAGS.debug==1:
            delta_t1=torch.tensor([[[0.00,0.05,0.0]]]).cuda()
            delta_s1=torch.tensor([[[1,1,1]]]).cuda()
            delta_r1=torch.zeros(bs,3,3).cuda()
            for i in range(bs):
                x=torch.Tensor(1).cuda()
                x[0]=40
                y=torch.Tensor(1).cuda()
                y[0]=0
                z=torch.Tensor(1).cuda()
                z[0]=0
                delta_r1[i] = get_rotation_torch(x, y, z)
            init_R=torch.bmm(gt_R,delta_r1)
        else:
            delta_t1 = torch.rand(bs, 1, 3).cuda()
            delta_t1 = delta_t1.uniform_(-0.05, 0.05)
            delta_s1 = torch.rand(bs, 1, 3).cuda()
            delta_s1 = delta_s1.uniform_(0.8, 1.2)
            delta_r1=torch.zeros(bs,3,3).cuda()
            for i in range(bs):
                x=torch.Tensor(1).cuda()
                x.uniform_(-a,a)
                y=torch.Tensor(1).cuda()
                y.uniform_(-a,a)
                z=torch.Tensor(1).cuda()
                z.uniform_(-a,a)
                delta_r1[i] = get_rotation_torch(x, y, z)
            init_R=torch.bmm(delta_r1,gt_R)
        init_t=gt_t.reshape(-1,1,3)+delta_t1
        init_t=init_t.reshape(-1,3)-center.reshape(-1,3)
        init_shape=real_scale.reshape(-1,1,3)*delta_s1
        if sym[0][0]==1:
            init_shape=init_shape[:,:,:2]
        def scale2to3(new_scale):
            if new_scale.shape[2]==2:
                tmp_new_scale=torch.zeros_like(new_scale)
                tmp_new_scale[:,:,:2]=new_scale[:,:,:2]
                tmp_new_scale=torch.cat([tmp_new_scale,new_scale[:,:,0:1]],dim=-1)
            else:
                tmp_new_scale=new_scale
            return tmp_new_scale







        m=torch.nn.LogSoftmax(dim=-1)
        rvecs=[]
        for i in range(bs):
            rvecs.append(cv2.Rodrigues(init_R[i].cpu().numpy())[0][:,0])
        rvecs=np.stack(rvecs)
        new_Rvec=torch.from_numpy(rvecs).float().cuda().requires_grad_()
        new_t=init_t.clone().requires_grad_()
        new_scale=init_shape.clone().requires_grad_()

        if FLAGS.use_deform:
            if FLAGS.regular_tps:
                pad_nocs_deform=torch_tps_regular_transform(pad_nocs,coefficient,control_points)
            else:
                pad_nocs_deform=torch_tps_transform(pad_nocs,coefficient,control_points)
            query_nocs_deform=deform_sdf_points
        else:
            query_nocs_deform=query_nocs
            pad_nocs_deform=pad_nocs


        gt_pad_camera=torch.bmm((pad_nocs.detach()*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()
        gt_query_deform=torch.bmm((query_nocs_deform.detach()*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)

        gt_query_camera=torch.bmm((query_nocs*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3)
        cur_query_camera=torch.bmm((query_nocs*scale2to3(init_shape)),init_R.permute(0,2,1))+init_t.reshape(-1,1,3)
        # show_open3d(gt_pad_camera[4].detach().cpu().numpy(),pc_center[4].detach().cpu().numpy())

        pad_bin_first,pad_bin_value_first=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size//10,pad_nocs)
        pad_bin_first_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size//10,pad_nocs_deform)
        pad_bin_second,pad_bin_value_second=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,pad_nocs)
        pad_bin_second_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,pad_nocs_deform)
        if FLAGS.min_loss:
            base=6
            pad_num=pad_nocs.shape[1]
            batch_size=pad_nocs.shape[0]
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=pad_nocs.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                                        [0, 1, 0],
                                        [-math.sin(theta), 0, math.cos(theta)]])
            sym_pad_nocs_deform=torch.matmul(pad_nocs_deform.unsqueeze(1),sym_Rs.permute(0,2,1).unsqueeze(0))

            sym_pad_bin_second_deform,_=self.to_bin(cat_name,0,FLAGS.bin_size,sym_pad_nocs_deform.reshape(-1,pad_num,3))
            sym_pad_bin_second_deform=sym_pad_bin_second_deform.reshape(batch_size,base,pad_num,3)

        query_bin_second_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,query_nocs_deform)




        if FLAGS.use_nocs_loss:
            pred_pad_bin_dict=self.qnet(gt_pad_camera,feature_dict)



        def do_nocs_loss(stage,pad_bin,bin_size):
            batch_size=pad_bin.shape[0]
            pad_num=pad_bin.shape[1]
            pred_bin=pred_pad_bin_dict[stage].reshape(batch_size,pad_num,3,bin_size).permute(0,-1,1,2).contiguous()
            if sym[0][0]==0:
                return self.loss_bin_fun(pred_bin,pad_bin).mean()*FLAGS.interpo_w
            else:
                return self.loss_bin_fun(pred_bin[:,:,:,:2],pad_bin[:,:,:2]).mean()*FLAGS.interpo_w

        def do_sym_nocs_loss(stage,sym_pad_bin,bin_size):
            batch_size=sym_pad_bin.shape[0]
            base=sym_pad_bin.shape[1]
            pad_num=sym_pad_bin.shape[2]
            sym_pred_bin=pred_pad_bin_dict[stage].reshape(batch_size,1,pad_num,3,bin_size).repeat(1,base,1,1,1).permute(0,-1,1,2,3).contiguous()
            loss=self.loss_bin_fun(sym_pred_bin,sym_pad_bin).mean(-1).mean(-1).min(-1)[0].mean()
            return loss*FLAGS.interpo_w

        if FLAGS.use_nocs_loss:
            # nocs_loss1=do_nocs_loss('first',pad_bin_first_deform,FLAGS.bin_size//10)
            if FLAGS.min_loss:
                nocs_loss2=do_sym_nocs_loss('second',sym_pad_bin_second_deform,FLAGS.bin_size)
            else:
                nocs_loss2=do_nocs_loss('second',pad_bin_second_deform,FLAGS.bin_size)
            nocs_loss=nocs_loss2
        else:
            nocs_loss=0
        def cal_sym_coord(cur_query,query_nocs):
            base=6
            sym_Rs=torch.zeros([base,3,3],dtype=torch.float32, device=query_nocs.device)
            for i in range(base):
                theta=float(i/base)*2*math.pi
                sym_Rs[i]=torch.tensor([[math.cos(theta), 0, math.sin(theta)],
                                        [0, 1, 0],
                                        [-math.sin(theta), 0, math.cos(theta)]])
            sym_query_nocs=torch.matmul(query_nocs.unsqueeze(1),sym_Rs.permute(0,2,1).unsqueeze(0))
            sym_gt_query=torch.matmul(sym_query_nocs*real_scale.reshape(-1,1,1,3),gt_R.permute(0,2,1).unsqueeze(1))+gt_t.reshape(-1,1,1,3)-center.reshape(-1,1,1,3)
            loss_coord_sym=self.loss_coord_sym(cur_query.unsqueeze(1),sym_gt_query)
            loss_coord_sym=loss_coord_sym.mean(-1).mean(-1)
            loss_coord_sym=loss_coord_sym.min(dim=1)[0].mean()
            return loss_coord_sym

        def gt_refine():

            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            cur_s=scale2to3(new_scale)
            cur_query_deform=torch.bmm((query_nocs_deform.detach()*cur_s.reshape(-1,1,3)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy())
            if sym[0][0]==1:
                loss=cal_sym_coord(cur_query_deform,query_nocs_deform.detach())
            else:
                loss=self.loss_coord(cur_query_deform,gt_query_deform)
            # print(loss)
            return loss




        def refine(query_bin,stage='first',bin_size=FLAGS.bin_size//10):

            R_list=[]
            for i in range(bs):
                R_list.append(Rodrigues.apply(new_Rvec[i]))
            cur_R=torch.stack(R_list,dim=0)
            cur_t=new_t
            cur_s=scale2to3(new_scale)

            batch_size=query_nocs.shape[0]




            cur_query_deform=torch.bmm((query_nocs_deform.detach()*cur_s.reshape(-1,1,3)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
            # show_open3d(pc_center[0].detach().cpu().numpy(),cur_query_deform[0].detach().cpu().numpy())
            pred_nocs_bin=self.qnet(cur_query_deform,feature_dict)[stage].reshape(batch_size,query_num,3,bin_size)

            pred_nocs_log_dis=m(pred_nocs_bin)
            query_log_prob=torch.gather(pred_nocs_log_dis,-1,query_bin.unsqueeze(-1)).squeeze(-1)
            if sym[0][0]==1:
                log_prob=-query_log_prob[:,:,:2].sum(-1).mean()
            else:
                log_prob=-query_log_prob.sum(-1).mean()
            return log_prob


        track_grads=[]
        track_Rvecs=[]
        track_ts=[]
        track_scales=[]
        track_steps=15
        pred_grad_1=None
        pred_grad_2=None

        opt0=torch.optim.Adam([new_Rvec,new_t,new_scale], lr=0.01)
        def gen_track():
            for i in range(track_steps):
                opt0.zero_grad()
                gt_loss=gt_refine()
                gt_loss.backward()
                track_Rvecs.append(new_Rvec.clone().detach().cpu())
                track_ts.append(new_t.clone().detach().cpu())
                track_scales.append(new_scale.clone().detach().cpu())
                track_grads.append(torch.cat([new_Rvec.grad.clone().reshape(bs,-1),
                                          new_t.grad.clone().reshape(bs,-1),
                                          new_scale.grad.clone().reshape(bs,-1)],dim=-1))
                opt0.step()



        if FLAGS.use_refine_loss:
            gt_grad=None
            gen_track()
            track_Rvecs=torch.stack(track_Rvecs,dim=0).transpose(0,1)
            track_ts=torch.stack(track_ts,dim=0).transpose(0,1)
            track_scales=torch.stack(track_scales,dim=0).transpose(0,1)
            track_grads=torch.stack(track_grads,dim=0).transpose(0,1)
            track_choose=torch.randint(low=0,high=track_steps,size=[bs])
            batch_choose=torch.arange(bs)

            new_Rvec=track_Rvecs[batch_choose,track_choose].cuda().requires_grad_()
            new_t=track_ts[batch_choose,track_choose].cuda().requires_grad_()
            new_scale=track_scales[batch_choose,track_choose].cuda().requires_grad_()
            gt_grad=track_grads[batch_choose,track_choose].cuda().requires_grad_()
            # show_open3d(gt_query_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())



            opt2=torch.optim.SGD([new_Rvec,new_t,new_scale], lr=0.01)






            def closure2():
                opt2.zero_grad()
                cur_loss=refine( query_bin_second_deform,'second',FLAGS.bin_size)
                cur_loss.backward()
                # print(new_Rvec.grad)
                return cur_loss


                # print('finish1')
            for i in range(1):
                closure2()
                opt2.step()
                pred_grad_2=torch.cat([new_Rvec.grad.reshape(bs,-1).requires_grad_(),
                                       new_t.grad.reshape(bs,-1).requires_grad_(),
                                       new_scale.grad.reshape(bs,-1).requires_grad_()],dim=-1)


            gt_grad=gt_grad/torch.norm(gt_grad,dim=-1,keepdim=True).detach()
            # pred_grad_1=pred_grad_1/torch.norm(pred_grad_1,dim=-1,keepdim=True)
            pred_grad_2=pred_grad_2/torch.norm(pred_grad_2,dim=-1,keepdim=True)
            # loss_grad=(self.loss_coord(pred_grad_1,gt_grad)+self.loss_coord(pred_grad_2,gt_grad))*FLAGS.grad_w
            loss_grad=self.loss_coord(pred_grad_2,gt_grad)*FLAGS.grad_w
        else:
            loss_grad=0




        compare2init={
            'Rot1': 0,
            'Rot2': 0,
            'Recon': 0,
            'Tran': 0,
            'Size':0,
        }
        if FLAGS.use_fsnet:
            name_fs_list=['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size']
            pred_fsnet_list=self.forward_fsnet(point_fea,pc_center,center)
            gt_green_v, gt_red_v = get_gt_v(gt_R)
            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Tran': gt_t,
                'Size': gt_s,
            }
            fsnet_loss=self.loss_fs_net(name_fs_list,pred_fsnet_list,gt_fsnet_list,sym)
        else:
            fsnet_loss={
                'Rot1': 0,
                'Rot2': 0,
                'Rot1_cos':0,
                'Rot2_cos':0,
                'Rot_r_a':0,
                'Tran': 0,
                'Size': 0
            }
        loss_dict={}
        loss_dict['consistency_loss']={'consistency':0}
        loss_dict['inter_loss'] = {
            'inter_r':0,
            'inter_t':0,
            'inter_nocs':0,
        }
        loss_dict['interpo_loss']={'Nocs':nocs_loss,'grad':loss_grad,'deform':0}
        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['compare2init']=compare2init
        loss_dict['vae']=vae_loss
        return loss_dict




    def to_bin(self,cat_name,sym,bin_size,pad_nocs):
        ratio_x=ratio_dict[cat_name][0]
        ratio_y=ratio_dict[cat_name][1]
        ratio_z=ratio_dict[cat_name][2]
        pad_nocs_r=pad_nocs.clone()
        if sym==1:
            x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            x_start=0
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=0
            z_bin_resolution=0
            pad_nocs_r[:,:,0]=torch.norm(pad_nocs_r[:,:,(0,2)],dim=-1)
            pad_nocs_r[:,:,2]=0
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,1]=torch.clamp(((pad_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,1]=torch.clamp(((pad_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,2]=torch.clamp(((pad_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        pad_bin_value = torch.zeros((3,bin_size)).to(pad_nocs_r.device)
        pad_bin_value[0]=x_start+torch.arange(bin_size)*x_bin_resolution
        pad_bin_value[1]=y_start+torch.arange(bin_size)*y_bin_resolution
        pad_bin_value[2]=z_start+torch.arange(bin_size)*z_bin_resolution
        return pad_nocs_bin,pad_bin_value


    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = PC.shape[0]
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_bc = torch.rand(1)
            # only do bc for mug and bowl


            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new


            prop_bb = torch.rand(1)
            model_point_new=model_point[i,...]
            if prop_bb < FLAGS.aug_bb_pro:
                #  R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new,model_point_new = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                               gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb[i, ...],
                                                               model_points=model_point[i,...],nocs_scale=nocs_scale[i, ...])
                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new

            #  augmentation finish
        return PC, gt_R, gt_t, gt_s


    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.backbone1.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
            }
        )
        if FLAGS.two_back:
            params_lr_list.append(
                {
                    "params": self.backbone2.parameters(),
                    "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
                }
            )
        params_lr_list.append(
            {
                "params": self.rot_red.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )
        params_lr_list.append(
            {
                "params": self.rot_green.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )

        params_lr_list.append(
            {
                "params": self.qnet.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_interpo,
                "betas":(0.9, 0.99)
            }
        )
        params_lr_list.append(
            {
                "params": self.ts.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_ts,

            }
        )



        return params_lr_list

