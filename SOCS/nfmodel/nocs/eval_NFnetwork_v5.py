import gc

import cv2
import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from network.point_sample.pc_sample import PC_sample
from nfmodel.uti_tool import *
from nfmodel.nocs.NFnetwork_v5 import NFPose
from nfmodel.nocs.NFnetwork import NFPose as basemodel
from EQNet.eqnet.ops.knn.knn_utils import knn_query
from EQNet.eqnet.ops.grouping.grouping_utils import grouping_operation
from nnutils.torch_pso import  *
from network.point_sample.pc_sample import farthest_sample
from easydict import EasyDict
import torch.optim as optim
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch

class multi_NFPose(nn.Module):
    def __init__(self,cat_names):
        super(multi_NFPose, self).__init__()
        self.per_networks=nn.ModuleDict()
        self.per_bases=nn.ModuleDict()
        self.cat_names=cat_names
        self.base_cats=['bottle','can','bowl','laptop']
        # self.base_cats=[]
        for cat_name in cat_names:
            cat_model_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            cat_model=NFPose()
            cat_model.load_state_dict(torch.load(cat_model_path))
            cat_base=basemodel()
            cat_base_path=os.path.join(FLAGS.base_dir,cat_name,FLAGS.resume_base_model_name)
            cat_base.load_state_dict(torch.load(cat_base_path))
            self.per_networks[cat_name]=cat_model
            self.per_bases[cat_name]=cat_base
    def forward(self, depth, cat_id_0base, camK, gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None,def_mask=None,
                model_points=None, nocs_scale=None,rgb=None,gt_mask=None,
                coefficients=None,control_points=None, std_models=None,
                picture_index=None,rgb_whole=None):
        if FLAGS.pic_save:
            if not os.path.exists(FLAGS.pic_save_dir):
                os.mkdir(FLAGS.pic_save_dir)
            FLAGS.cur_eval_index=picture_index
            rgb_path=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+'_rgb.png')
            plt.imshow(rgb_whole[:,:,(2,1,0)])
            cv2.imwrite(rgb_path,rgb_whole )

        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        PC = PC_sample(def_mask, depth, camK, gt_2D)

        PC = PC.detach()
        obj_num=PC.shape[0]
        res_list=[]
        scale_list=[]
        for i in range(obj_num):
            if FLAGS.per_obj is not '':
                cat_name=FLAGS.per_obj
            else:
                cat_name=self.cat_names[cat_id_0base[i]]
            res,scale=self.per_infer(cat_name,sym[i],PC[i],mean_shape[i],gt_R[i],gt_t[i],gt_s[i],
                                     coefficients[i],control_points[i],std_models[i],rgb_whole,model_points[i])
            res_list.append(res)
            scale_list.append(scale)
        return torch.stack(res_list,dim=0),torch.stack(scale_list,dim=0)

    def per_infer(self,cat_name,sym,pc,mean_shape,gt_R=None,gt_T=None,gt_s=None,
                  coefficients=None,control_points=None,std_model=None,rgb_whole=None,model_point=None):
        # show_open3d(pc.detach().cpu().numpy(),pc.detach().cpu().numpy())
        choose=farthest_sample(model_point.cuda(),1024)
        model_point=model_point[choose]
        cat_model=self.per_networks[cat_name]
        cat_base=self.per_bases[cat_name]
        PC=pc.unsqueeze(0)
        real_shape=gt_s
        mean_shape=mean_shape.detach().cpu().numpy()

        center=PC.mean(dim=1,keepdim=True)

        gt_T=gt_T-center.squeeze()
        pc_num=PC.shape[1]
        pc_center=PC-center

        # pc_center=pc_center@ gt_R.T+gt_T

        recon,point_fea,global_fea,feature_dict,_= cat_model.backbone1(pc_center)
        recon,point_fea,global_fea,feature_dict_base= cat_base.backbone(pc_center)
        if FLAGS.two_back:
            recon,point_fea,global_fea,_,_= cat_model.backbone2(pc_center)
        if FLAGS.use_base:
            if cat_name in self.base_cats:
                recon,point_fea,global_fea,feature_dict,= cat_base.backbone(pc_center)
            else:
                recon,point_fea,global_fea,_,= cat_base.backbone(pc_center)
            p_green_R=cat_base.rot_green(point_fea.permute(0,2,1))
            p_red_R=cat_base.rot_red(point_fea.permute(0,2,1))
            if FLAGS.feat_for_ts:
                feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
                T, s = cat_base.ts(feat_for_ts.permute(0, 2, 1))
            else:
                feat_for_ts = pc_center
                objs=torch.zeros_like(feat_for_ts[:,0,0])
                T, s = cat_base.ts(feat_for_ts.permute(0, 2, 1),objs)
        else:
            p_green_R=cat_model.rot_green(point_fea.permute(0,2,1))
            p_red_R=cat_model.rot_red(point_fea.permute(0,2,1))
            if FLAGS.feat_for_ts:
                feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
                T, s = cat_model.ts(feat_for_ts.permute(0, 2, 1))
            else:
                feat_for_ts = pc_center
                objs=torch.zeros_like(feat_for_ts[:,0,0])
                T, s = cat_model.ts(feat_for_ts.permute(0, 2, 1),objs)


        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)
        Pred_T = T  # bs x 3
        Pred_s = s  # this s is


        p_green_R =p_green_R[0].detach().cpu().numpy()
        p_red_R=p_red_R[0].detach().cpu().numpy()
        Pred_T=Pred_T[0].detach().cpu().numpy()
        Pred_s=Pred_s[0].detach().cpu().numpy()+mean_shape


        if FLAGS.use_mean_init:
            Pred_T = np.zeros_like(Pred_T)
            Pred_s=mean_shape

        if sym[0] < 1 :
            num_cor=3
            cor0 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        else:
            num_cor=2
            cor0 = np.array([[0, 0, 0], [0, 1, 0]])
        cor0= cor0/np.linalg.norm(cor0)
        pred_axis = np.zeros((num_cor,3))
        pred_axis[1,:]=p_green_R
        if num_cor==3:
            pred_axis[2,:]=p_red_R
        pose=gettrans(cor0.reshape((num_cor, 3)), pred_axis.reshape((num_cor, 1, 3)))
        fake_rotation = pose[0][0:3, 0:3]


        grid_rotation=fake_rotation
        grid_T=Pred_T
        grid_s=Pred_s

        fake_grid = grids['fake_grid'].numpy()
        boxsize = FLAGS.fake_radius*2
        fake_grid = boxsize * (fake_grid)
        fake_grid_scaled=fake_grid*grid_s.max()
        fake_query_np=fake_grid_scaled @ grid_rotation.T + grid_T
        # fake_grid_scaled=fake_grid*(pre.cpu().numpy()).max()
        # fake_query_np=fake_grid_scaled @ (gt_R.cpu().numpy()).T + (gt_T.cpu().numpy())
        if FLAGS.verbose:
            show_open3d(pc_center[0].detach().cpu().numpy(),fake_query_np)
        fake_query=torch.from_numpy(fake_query_np).unsqueeze(0).float().cuda()
        fake_nocs=torch.from_numpy(fake_grid_scaled).unsqueeze(0).float().cuda()
        fake_num=fake_query.shape[1]


        new_query_num=FLAGS.new_query_num
        box_size = FLAGS.query_radius*2
        points_uniform = np.random.rand(new_query_num, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        new_query_nocs=torch.from_numpy(points_uniform).float().unsqueeze(0).cuda().detach()


        def get_neighbor(query_point,support_points,num_neighber):
            query_cnt=query_point.new_zeros(1).int()
            query_cnt[0]=query_point.shape[0]
            support_cnt=support_points.new_zeros(1).int()
            support_cnt[0]=support_points.shape[0]

            index_pair = knn_query(
                num_neighber,
                support_points, support_cnt,
                query_point, query_cnt).int()
            neighbor_pos=grouping_operation(
                support_points, support_cnt, index_pair, query_cnt).permute(0,2,1).squeeze(1)

            return neighbor_pos







            std_min_point=torch.min(std_model ,dim=0,)[0]
            std_max_point=torch.max(std_model ,dim=0,)[0]
            std_size=std_max_point-std_min_point
            std_diag=torch.norm(std_size)
            std_model_nocs=std_model/std_diag
            std_model_nocs_nu=std_model/std_size

            std_model_nocs_color=(std_model_nocs+0.6)*0.9

            show_nocs(std_model_nocs.cpu().numpy(),nocs=std_model_nocs_color.cpu().numpy())

            support0_nocs_nu_deform=torch_tps_transform(support0_nocs_nu.unsqueeze(0),
                                                        coefficients.unsqueeze(0),control_points.unsqueeze(0))[0]

            nocs_norm=np.array([0.1,1,1])
            nocs_start=np.array([0.16,-0.5,-0.5])
            support0_nocs_std=support0_nocs_nu_deform*std_size/std_diag
            support0_nocs_std_color=(support0_nocs_std.cpu().numpy()-nocs_start)/nocs_norm
            support0_nocs_std_color=np.clip(support0_nocs_std_color,0,1)

            support0_nocs_color=(support0_nocs.cpu().numpy()-nocs_start)/nocs_norm
            support0_nocs_color=np.clip(support0_nocs_color,0,1)

            std_model_noc_color=(std_model_nocs.cpu().numpy()-nocs_start)/nocs_norm
            std_model_noc_color=np.clip(std_model_noc_color,0,1)




        fake_nocs_bin_first,bin_value_first=to_bin(cat_name,sym[0],FLAGS.bin_size//10,fake_nocs)
        new_query_nocs_bin_first,_=to_bin(cat_name,sym[0],FLAGS.bin_size//10,new_query_nocs)
        fake_nocs_bin_second,bin_value_second=to_bin(cat_name,sym[0],FLAGS.bin_size,fake_nocs)
        new_query_nocs_bin_second,_=to_bin(cat_name,sym[0],FLAGS.bin_size,new_query_nocs)

        m=torch.nn.LogSoftmax(dim=-1)
        s=torch.nn.Softmax(dim=-1)
        if cat_name in self.base_cats:
            pred_fake_nocs_bin_dict={}
            pred_fake_nocs_bin_dict['second']=cat_base.qnet(fake_query,feature_dict)
        else:
            pred_fake_nocs_bin_dict=cat_model.qnet(fake_query,feature_dict)

        pred_fake_nocs_dict={}
        pred_fake_dis_log_dict={}
        pred_fake_dis_dict={}
        for stage ,bin_size in zip(['second'],[FLAGS.bin_size]):
            pred_fake_bin=pred_fake_nocs_bin_dict[stage].reshape(1,fake_num,3,bin_size)
            pred_fake_bin_max=pred_fake_bin.max(-1)[1]
            if FLAGS.min_loss:
                pred_fake_nocs=to_value(cat_name,0,bin_size,pred_fake_bin_max)
            else:
                pred_fake_nocs=to_value(cat_name,sym[0],bin_size,pred_fake_bin_max)
            pred_fake_dis_log=m(pred_fake_bin.clone()).detach()
            pred_fake_dis=s(pred_fake_bin.clone()).detach()
            pred_fake_nocs_dict[stage]=pred_fake_nocs
            pred_fake_dis_log_dict[stage]=pred_fake_dis_log
            pred_fake_dis_dict[stage]=pred_fake_dis






        if sym[0]==1:
            sym_s=Pred_s[:2]
        else:
            sym_s=Pred_s
        new_Rvec=torch.from_numpy(cv2.Rodrigues(fake_rotation)[0][:,0]).float().cuda().requires_grad_()
        new_T=torch.from_numpy(Pred_T).float().cuda().requires_grad_()
        new_scale=torch.from_numpy(sym_s).float().cuda().requires_grad_()
        if FLAGS.align_origin:
            new_key_coeff=coefficients[:FLAGS.keypoint_num+1].clone().float().cuda().requires_grad_()
            new_coeff=torch.zeros((FLAGS.keypoint_num+1+1+3,3)).float().cuda()
            new_coeff[FLAGS.keypoint_num+1+1:FLAGS.keypoint_num+1+1+3]=torch.eye(3).float().cuda()
            new_coeff[:FLAGS.keypoint_num+1]=new_key_coeff
        else:
            new_key_coeff=coefficients[:FLAGS.keypoint_num].clone().float().cuda().requires_grad_()
            new_coeff=torch.zeros((FLAGS.keypoint_num+1+3,3)).float().cuda()
            new_coeff[FLAGS.keypoint_num+1:FLAGS.keypoint_num+1+3]=torch.eye(3).float().cuda()
            new_coeff[:FLAGS.keypoint_num]=new_key_coeff
        # new_coeff=coefficients.clone().requires_grad_()
        def interpo_dis(new_query,fake_query,fake_dis,stage,bin_size):
            if FLAGS.use_interpo:
                assert fake_query.shape[0]==1
                fake_num=fake_dis.shape[1]
                query_batch=new_query.shape[0]
                query_num=new_query.shape[1]
                fake_dis=fake_dis.reshape(fake_num,-1)
                new_query=new_query.reshape(-1,3)
                fake_query=fake_query.reshape(-1,3)
                query_cnt=new_query.new_zeros(1).int()
                query_cnt[0]=new_query.shape[0]
                fake_cnt=fake_query.new_zeros(1).int()
                fake_cnt[0]=fake_query.shape[0]
                index_pair =knn_query(
                    8,
                    fake_query,fake_cnt,
                    new_query,query_cnt
                ).int()
                neighbor_pos=grouping_operation(
                    fake_query,fake_cnt,index_pair,query_cnt).permute(0,2,1)
                neighbor_dis=grouping_operation(
                    fake_dis,fake_cnt,index_pair,query_cnt).permute(0,2,1)
                weight=1/(torch.norm((new_query.unsqueeze(1)-neighbor_pos),dim=-1)+1e-10)
                denomin=torch.sum(weight,dim=-1,keepdim=True)
                weight=weight/(denomin)
                new_query_dis=torch.sum(neighbor_dis*weight.unsqueeze(-1),dim=1).reshape(query_batch,query_num,-1)
            else:
                pred_query_bin=cat_model.qnet(new_query,feature_dict,stage).reshape(1,new_query_num,3,bin_size)
                new_query_dis=m(pred_query_bin)
            return new_query_dis

        def objective(bin_value,new_query_bin,stage='first',bin_size=FLAGS.bin_size//10):
            cur_R=Rodrigues.apply(new_Rvec)
            cur_t=new_T
            cur_s=scale2to3(new_scale)

            cal_fake_nocs=torch.bmm(fake_query-cur_t.reshape(-1,1,3),cur_R.reshape(1,3,3))/cur_s.reshape(-1,1,3)

            pred_bin=pred_fake_nocs_bin_dict[stage].reshape(-1,fake_num,3,bin_size).detach()
            l2=0
            if FLAGS.use_prob:


                if FLAGS.eval_use_deform:
                    keypoint_num=new_key_coeff.shape[0]
                    new_coeff[:keypoint_num]=new_key_coeff
                    new_query_nocs_deform=torch_tps_transform(new_query_nocs,new_coeff.unsqueeze(0),control_points.unsqueeze(0))
                    l2=(torch.sum(new_coeff[:keypoint_num]**2)+torch.sum((control_points.T @ new_coeff[:keypoint_num])**2))
                else:
                    new_query_nocs_deform=new_query_nocs



                new_query=(new_query_nocs_deform*cur_s) @ cur_R.T + cur_t

                query_dis_log=interpo_dis(new_query,fake_query,pred_fake_dis_log_dict[stage],stage,bin_size).reshape(1,new_query_num,3,bin_size)

                log_prob=torch.gather(query_dis_log,-1,new_query_bin.unsqueeze(-1))[0,:,:,0]
                if sym[0]==1:
                    log_prob=(log_prob[:,0]+log_prob[:,1]).mean()
                else:
                    log_prob=(log_prob[:,0]+log_prob[:,1]+log_prob[:,2]).mean()

            else:
                log_prob=0
            if FLAGS.use_distance:
                pred_fake_nocs=pred_fake_nocs_dict[stage]
                if FLAGS.eval_use_deform:
                    keypoint_num=new_key_coeff.shape[0]
                    new_coeff[:keypoint_num]=new_key_coeff
                    pred_fake_nocs_deform=torch_tps_transform(pred_fake_nocs,new_coeff.unsqueeze(0),control_points.unsqueeze(0))
                    l2=(torch.sum(new_coeff[:keypoint_num]**2)+torch.sum((control_points.T @ new_coeff[:keypoint_num])**2))
                else:
                    pred_fake_nocs_deform=pred_fake_nocs

                if sym[0]==1 and FLAGS.min_loss==0:

                    cal_fake_nocs_r=torch.norm(cal_fake_nocs[:,:,(0,2)],dim=-1)
                    cal_fake_nocs[:,:,0]=cal_fake_nocs_r
                    distance=torch.norm(cal_fake_nocs[:,:,:2] - pred_fake_nocs[:,:,:2], dim=-1)
                    scene_score=distance.mean()
                else:
                    pred_fake=(pred_fake_nocs_deform*cur_s) @ cur_R.T + cur_t
                    distance=torch.norm(pred_fake - fake_query, dim=-1)
                    scene_score=-torch.exp(-distance**2).mean()
                    scene_score=distance.mean()
            else:
                scene_score=0
            if FLAGS.use_adaptive_distance:
                new_query=(new_query_nocs*cur_s) @ cur_R.T + cur_t
                new_query_detach=new_query.detach()

                pred_query_nocs=interpo_dis(new_query_detach,fake_query,pred_fake_nocs_dict[stage],stage,bin_size).reshape(1,new_query_num,3)

                pred_qury=(pred_query_nocs*cur_s) @ cur_R.T + cur_t
                pred_qury_detach=pred_qury.detach()
                if sym[0]==0:
                    distance=torch.norm(pred_qury - new_query_detach, dim=-1)
                else:
                    distance=torch.norm(pred_qury[:,:,:2] - new_query_detach[:,:,:2], dim=-1)
                scene_score=-torch.exp(-distance**2).mean()+l2*0.01
                # scene_score=distance.mean()
            score=-log_prob+scene_score

            if torch.isnan(score):
                print('nan')

            return score

        cur_R=Rodrigues.apply(new_Rvec)
        cur_t=new_T
        cur_s=scale2to3(new_scale)
        cur_query=(new_query_nocs.detach()*cur_s)@cur_R.T+cur_t
        if FLAGS.verbose==1:
            show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=new_query_nocs[0].detach().cpu().numpy()+0.5)

        if FLAGS.use_pso:
            def wrapper(pso_size=30,fake_dis=None,query_bin=None,stage='first',bin_size=FLAGS.bin_size//10):
                re_query_nocs=new_query_nocs.repeat(pso_size,1,1)
                re_query_bin=query_bin.repeat(pso_size,1,1)
                def eval_fun(cur_R,cur_t,cur_s):
                    cur_re_query=torch.bmm((re_query_nocs*cur_s.reshape(-1,1,3)),cur_R.permute(0,2,1))+cur_t.reshape(-1,1,3)
                    # show_open3d(cur_query[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
                    with torch.no_grad():
                        re_query_dis=interpo_dis(cur_re_query,fake_query,fake_dis,stage,bin_size).reshape(pso_size,new_query_num,3,bin_size)
                        query_prob=torch.gather(re_query_dis,-1,re_query_bin.unsqueeze(-1)).squeeze(-1)
                        if sym[0]==1:
                            prob=(query_prob[:,:,0]*query_prob[:,:,1]).mean(-1)
                        else:
                            prob=(query_prob[:,:,0]*query_prob[:,:,1]*query_prob[:,:,2]).mean(-1)
                        return prob
                return eval_fun

            assert pc_center.shape[0]==1
            pso_size=216
            iter_num=20
            # eval_fun=wrapper(pso_size,fake_dis_first,new_query_nocs_bin_first,'first',FLAGS.bin_size//10)
            eval_fun=wrapper(pso_size,pred_fake_dis_log_dict['second'],new_query_nocs_bin_second,'second',FLAGS.bin_size)
            ini_R=Rodrigues.apply(new_Rvec).detach()
            pso=PSO(surface_points=pc_center[0].cpu().numpy(),size=pso_size,iter_num=iter_num,
                    init_R=ini_R.cpu().numpy(),eval_fun=eval_fun)
            pso_vec,pso_t,pso_s,_=pso.update_pso()
            new_Rvec=torch.from_numpy(pso_vec).float().cuda().requires_grad_()
            new_T=torch.from_numpy(pso_t).float().cuda().requires_grad_()
            new_scale=torch.from_numpy(pso_s).float().cuda().requires_grad_()

        cur_R=Rodrigues.apply(new_Rvec)
        cur_t=new_T
        cur_s=scale2to3(new_scale)
        cur_query=(new_query_nocs.detach()*cur_s)@cur_R.T+cur_t
        if FLAGS.verbose==1:
            show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=new_query_nocs[0].detach().cpu().numpy()+0.5)

        if FLAGS.use_refine:
            lr=FLAGS.refine_lr
            step1=0
            step2=FLAGS.refine_step
            params=[{'params':new_scale},{'params':new_Rvec},{'params':new_T},
                    # {'params':new_key_coeff},
                    # {'params':new_key_coeff,'lr':0.01}
            ]
            if FLAGS.eval_optimizer=='adam':
                opt1 = torch.optim.Adam(params, lr=lr)
                opt2 = torch.optim.Adam(params, lr=lr)
            else:
                opt2=optim.LBFGS([new_Rvec,new_T,new_scale],lr=1,max_iter=100,line_search_fn="strong_wolfe")



            def closure1():
                opt1.zero_grad()
                cur_loss=objective(bin_value_first,new_query_nocs_bin_first,'first',FLAGS.bin_size//10)
                cur_loss.backward(retain_graph=True)
                return cur_loss

            def closure2():
                opt2.zero_grad()
                cur_loss=objective(bin_value_second,new_query_nocs_bin_second,'second',FLAGS.bin_size)
                cur_loss.backward(retain_graph=True)
                # print(new_key_coeff.grad)
                return cur_loss




            if FLAGS.eval_optimizer=='adam':
                with torch.enable_grad():
                    for i in range(step1):
                        closure1()
                        opt1.step()
                    for i in range(step2):
                        closure2()
                        opt2.step()
            else:
                opt2.step(closure2)
        cur_R=Rodrigues.apply(new_Rvec)
        cur_t=new_T
        cur_s=scale2to3(new_scale)
        if FLAGS.eval_use_deform:
            keypoint_num=new_key_coeff.shape[0]
            new_coeff[:keypoint_num]=new_key_coeff
            new_query_nocs_deform=torch_tps_transform(new_query_nocs,new_coeff.unsqueeze(0),control_points.unsqueeze(0))
        else:
            new_query_nocs_deform=new_query_nocs
        cur_query=(new_query_nocs_deform.detach()*cur_s)@cur_R.T+cur_t
        if FLAGS.verbose==1:
            show_open3d(pc_center[0].detach().cpu().numpy(),cur_query[0].detach().cpu().numpy(),color_2=new_query_nocs[0].detach().cpu().numpy()+0.5)

        res = torch.eye(4, dtype=torch.float).to(new_scale.device)
        res[:3,:3]=cur_R
        res[:3,3]=new_T+center.reshape(3)



        return res,cur_s

def to_value(cat_name,sym,bin_size,fake_bin):
    ratio_x=ratio_dict[cat_name][0]
    ratio_y=ratio_dict[cat_name][1]
    ratio_z=ratio_dict[cat_name][2]
    fake_nocs=fake_bin.clone().float()
    if sym==1:
        x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
        y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
        x_start=0
        y_start=(-FLAGS.pad_radius)*ratio_y
        z_start=0
        z_bin_resolution=0
    else:
        x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
        y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
        z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
        x_start=(-FLAGS.pad_radius)*ratio_x
        y_start=(-FLAGS.pad_radius)*ratio_y
        z_start=(-FLAGS.pad_radius)*ratio_z
    fake_nocs[:,:,0]=fake_bin[:,:,0]*x_bin_resolution+x_start
    fake_nocs[:,:,1]=fake_bin[:,:,1]*y_bin_resolution+y_start
    fake_nocs[:,:,2]=fake_bin[:,:,2]*z_bin_resolution+z_start
    return fake_nocs

def to_bin(cat_name,sym,bin_size,pad_nocs):
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


def scale2to3(new_scale):
    if new_scale.shape[-1]==2:
        tmp_new_scale=torch.zeros_like(new_scale)
        tmp_new_scale[:2]=new_scale[:2]
        tmp_new_scale=torch.cat([tmp_new_scale,new_scale[0:1]],dim=-1)
    else:
        tmp_new_scale=new_scale
    return tmp_new_scale
