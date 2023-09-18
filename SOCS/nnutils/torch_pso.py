import numpy as np
import torch
from nfmodel.uti_tool import *
from nnutils.torch_util import *
import cv2
import numpy
import math
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from scipy.spatial import ConvexHull, Delaunay
import open3d as o3d
class Particle:

    def __init__(
            self,
            init_angle_min = -1,
            init_angle_max = 1,
            angle_min=-math.pi,
            angle_max=math.pi,
            tran_max = 0.1,
            tran_min =-0.1,
            scale_min = 0.0,
            scale_max = 0.6,
            max_tran_speed = 0.01,
            max_scale_speed= 0.01,
            surface_points=None,
            new_surface_points=None,
            init_R=np.eye(3)
    ):
        self.init_angle_min=init_angle_min
        self.init_angle_max=init_angle_max
        self.angle_min=angle_min
        self.angle_max=angle_max
        self.tran_max=tran_max
        self.tran_min=tran_min
        self.scale_min=scale_min
        self.scale_max=scale_max
        self.max_tran_speed=max_tran_speed
        self.max_scale_speed=max_scale_speed
        self.surface_points=surface_points
        self.new_surface_points=new_surface_points
        self.init_R=init_R
        or1=np.array([0.5,-0.5,-0.5])
        or2=np.array([0.5,0.5,-0.5])
        or3=np.array([0.5,0.5,0.5])
        or4=np.array([0.5,-0.5,0.5])
        or5=np.array([-0.5,-0.5,-0.5])
        or6=np.array([-0.5,0.5,-0.5])
        or7=np.array([-0.5,0.5,0.5])
        or8=np.array([-0.5,-0.5,0.5])
        self.corner=np.array([or1,or2,or3,or4,or5,or6,or7,or8])
        total_try=0
        while(1):
            total_try+=1
            delta_euler= np.random.uniform(self.init_angle_min, self.init_angle_max,size=(3))
            delta_R=eulerAnglesToRotationMatrix(delta_euler)
            cur_R=delta_R@self.init_R
            corners,init_scale,init_t=self.cal_aligned_bbox(cur_R)
            if self.check_pose(delta_euler,init_t,init_scale) or total_try>5:
                self.pose_ts=np.concatenate([init_t,init_scale],axis=0)
                self.pose_SO3=delta_R
                break
        # show_open3d(surface_points,corners)
        # show_open3d(new_surface_points,corners)
        speed_euler= np.random.uniform(self.init_angle_min, self.init_angle_max,size=(3))
        speed_R=eulerAnglesToRotationMatrix(delta_euler)
        speed_t=np.random.uniform(-max_tran_speed, max_tran_speed,size=(3))
        speed_s=np.random.uniform(-max_scale_speed, max_scale_speed,size=(3))
        self.speed_SO3=speed_R
        self.speed_ts=np.concatenate([speed_t,speed_s],axis=0)
        self.best_pose_ts = np.zeros(6)
        self.best_pose_SO3 = np.eye(3)
        self.fitness_value = -float('inf')

    def check_pose(self,delta_euler,trans,scale):
        # angle_flag=np.all(delta_euler<self.angle_max) and np.all(delta_euler>self.angle_min)
        angle_flag=True
        flag= angle_flag and \
        np.all(trans<self.tran_max) and np.all(trans>self.tran_min) and \
        np.all(scale<self.scale_max) and np.all(scale>self.scale_min)
        return flag
        # return True

    def clamp_speed(self,speed_ts):
        clamped_speed_ts=speed_ts
        clamped_speed_ts[:3]=speed_ts[:3].clip(-self.max_tran_speed,self.max_tran_speed)
        clamped_speed_ts[3:6]=speed_ts[3:6].clip(-self.max_scale_speed,self.max_scale_speed)
        return clamped_speed_ts

    def cal_aligned_bbox(self,cur_R):
        aligned_surface_points=self.new_surface_points@cur_R
        aligned_corners,scale,aligned_center=get_3D_corner_aligned(aligned_surface_points)
        corners=aligned_corners @ cur_R.T
        center=aligned_center @ cur_R.T
        return corners,scale,center
    def show(self):
        cur_corner=(self.corner*self.pose_ts[3:6]) @ self.pose_SO3.T + self.pose_ts[:3]
        show_open3d(cur_corner,self.surface_points)


class PSO:

    def __init__(
            self,
            size= 30,
            iter_num= 30,
            surface_points=None,
            eval_fun=None,
            init_R=np.eye(3),
    ):
        self.w = 1           # Inertia weight of PSO.
        self.c1 = self.c2 = 2
        self.vol_w=0
        self.contain_w=0
        self.net_w=10
        self.size = size
        self.iter_num = iter_num
        self.surface_points=surface_points
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.surface_points)
        pcd_new=o3d.geometry.PointCloud.remove_radius_outlier(pcd,100,0.05)[0]
        self.new_surface_points=np.asarray(pcd_new.points)
        self.init_R=init_R
        self.eval_fun=eval_fun
        or1=np.array([0.5,-0.5,-0.5])
        or2=np.array([0.5,0.5,-0.5])
        or3=np.array([0.5,0.5,0.5])
        or4=np.array([0.5,-0.5,0.5])
        or5=np.array([-0.5,-0.5,-0.5])
        or6=np.array([-0.5,0.5,-0.5])
        or7=np.array([-0.5,0.5,0.5])
        or8=np.array([-0.5,-0.5,0.5])
        self.corner=np.array([or1,or2,or3,or4,or5,or6,or7,or8])
        self.best_pose_SO3 = np.eye(3)
        self.best_pose_ts = np.zeros(6)
        self.fitness_value=-float('inf')
        # Population_initialization
        self.Particle_list = [
            Particle(surface_points=self.surface_points,new_surface_points=self.new_surface_points,init_R=self.init_R) for _ in range(size)]


    def batch_fit_fun(self):
        fitness_list=[]
        R_list=[]
        t_list=[]
        s_list=[]
        for part in self.Particle_list:
            cur_R=part.pose_SO3@self.init_R
            cur_t=part.pose_ts[:3]
            cur_s=part.pose_ts[3:6]
            fitness=self.fit_fun(cur_R,cur_t,cur_s,False)
            fitness_list.append(fitness)
            R_list.append(cur_R)
            t_list.append(cur_t)
            s_list.append(cur_s)
        tensor_R=torch.from_numpy(np.stack(R_list,axis=0)).float().cuda()
        tensor_t=torch.from_numpy(np.stack(t_list,axis=0)[:,None,:]).float().cuda()
        tensor_s=torch.from_numpy(np.stack(s_list,axis=0)[:,None,:]).float().cuda()
        if self.eval_fun is not None:
            net_fitness_list=self.eval_fun(tensor_R,tensor_t,tensor_s).tolist()
        else:
            net_fitness_list=[0 for i in range(len(fitness_list))]
        for i in range(len(fitness_list)):
            fitness_list[i]=fitness_list[i]+net_fitness_list[i]*self.net_w
        return fitness_list

    def fit_fun(self,cur_R,cur_t,cur_s,vis):
        # cur_corner=(self.corner*cur_s) @ cur_R.T + cur_t
        # flags=Delaunay(cur_corner).find_simplex(self.surface_points)
        # if FLAGS.verbose and vis:
        #     show_open3d(cur_corner,self.surface_points)
        # pos=(flags>0).sum()
        # total=len(flags)
        # ratio=pos/total
        # vol=cur_s[0]*cur_s[1]*cur_s[2]
        # fitness=self.contain_w*ratio-vol*self.vol_wt
        # return fitness
        return 0
    def update_pso(self):
        for iter in range(self.iter_num):
            # self.w=self.w*0.99
            fitness_list=self.batch_fit_fun()
            for fitness,part in zip(fitness_list,self.Particle_list):
                if fitness>part.fitness_value:
                    part.fitness_value=fitness
                    part.best_pose_SO3=part.pose_SO3
                    part.best_pose_ts=part.pose_ts
                if fitness>self.fitness_value:
                    if FLAGS.verbose:
                        print('surpass',fitness)
                    # self.fit_fun(part.pose,True)
                    self.fitness_value=fitness
                    self.best_pose_SO3=part.pose_SO3
                    self.best_pose_ts=part.pose_ts

            for part in self.Particle_list:
                self.update_speed(part)
            for i in range(len(self.Particle_list)):
                part=self.Particle_list[i]
                part.pose_SO3=part.speed_SO3 @ part.pose_SO3
                tran=part.pose_ts[:3]+part.speed_ts[:3]
                scale=part.pose_ts[3:6]+part.speed_ts[3:6]
                part.pose_ts=np.concatenate([tran,scale],axis=0)
                if not part.check_pose(None,tran,scale):
                    if FLAGS.verbose:
                        part.show()
                        print('failure')
                    self.Particle_list[i]=Particle(surface_points=self.surface_points,new_surface_points=self.new_surface_points,init_R=self.init_R)
            if FLAGS.verbose:
                print(iter)

        delta_R=self.best_pose_SO3
        cur_R=delta_R@self.init_R
        cur_t=self.best_pose_ts[:3]
        cur_s=self.best_pose_ts[3:6]
        self.fit_fun(cur_R,cur_t,cur_s,True)
        cur_vec=cv2.Rodrigues(cur_R)[0][:,0]
        return cur_vec,cur_t,cur_s,self.fitness_value



    def update_speed(self,part):
        cur_speed_so3=cv2.Rodrigues(part.speed_SO3)[0][:,0]
        dir_globel_so3=cv2.Rodrigues(self.best_pose_SO3 @ part.pose_SO3.T)[0][:,0]
        dir_local_so3=cv2.Rodrigues(part.best_pose_SO3 @ part.pose_SO3.T)[0][:,0]

        speed_so3=self.w*cur_speed_so3 + self.c1 * np.random.uniform(size=1)*dir_local_so3 \
        +self.c2 * np.random.uniform(size=1)*dir_globel_so3
        speed_SO3,_=cv2.Rodrigues(speed_so3)
        part.speed_SO3=speed_SO3
        speed_t = self.w * part.speed_ts[:3] \
                      + self.c1 * np.random.uniform(size=1) * (part.best_pose_ts - part.pose_ts)[:3] \
                      + self.c2 * np.random.uniform(size=1) * (self.best_pose_ts - part.pose_ts)[:3]
        speed_s = self.w * part.speed_ts[3:6] \
                  + self.c1 * np.random.uniform(size=1) * (part.best_pose_ts - part.pose_ts)[3:6] \
                  + self.c2 * np.random.uniform(size=1) * (self.best_pose_ts - part.pose_ts)[3:6]
        speed_ts=np.concatenate([speed_t,speed_s],axis=0)

        part.speed_ts = part.clamp_speed(speed_ts)






if __name__ == '__main__':
    surface_points=np.random.rand(1000,3)*np.array([1.2,0.6,0.8])
    surface_points=0.4 * (surface_points)
    delta_euler= np.array([0.5,0,0])
    delta_R=eulerAnglesToRotationMatrix(delta_euler)
    surface_points=surface_points @ delta_R.T
    center=surface_points.mean(0)
    surface_points=surface_points-center+np.array([0.05,0,0])
    # Particle(surface_points=surface_points)
    pso=PSO(surface_points=surface_points)
    pso.update_pso()
