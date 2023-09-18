import numpy as np
import torch
from nfmodel.uti_tool import *
from nnutils.torch_util import *
import cv2
import numpy
import math
from scipy.spatial import ConvexHull, Delaunay
class Particle:

    def __init__(
            self,
            delta_R,
            surface_points=None,
            init_R=np.eye(3)
    ):
        self.surface_points=surface_points
        self.init_R=init_R
        cur_R=delta_R@self.init_R
        corners,init_scale,init_t=self.cal_aligned_bbox(cur_R)
        self.pose_ts=np.concatenate([init_t,init_scale],axis=0)
        self.pose_SO3=delta_R




    def cal_aligned_bbox(self,cur_R):
        aligned_surface_points=self.surface_points@cur_R
        aligned_corners,scale,aligned_center=get_3D_corner_aligned(aligned_surface_points)
        corners=aligned_corners @ cur_R.T
        center=aligned_center @ cur_R.T
        return corners,scale,center


class PSO:

    def __init__(
            self,
            size= 30,
            iter_num= 30,
            surface_points=None,
            eval_fun=None,
            init_R=np.eye(3),
    ):
        self.vol_w=0
        self.contain_w=0
        self.net_w=1
        self.size = size
        self.iter_num = iter_num
        self.surface_points=surface_points
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

        # Population_initialization
        start=-60
        end=60
        interval=int(pow(self.size,1/3))+1
        resolution=(end-start)/interval
        self.Particle_list=[]
        for i in range(interval):
            for j in range(interval):
                for k in range(interval):
                    x_angle=start+i*resolution
                    y_angle=start+j*resolution
                    z_angle=start+k*resolution
                    delta_euler= np.array([x_angle,y_angle,z_angle])
                    delta_R=eulerAnglesToRotationMatrix(delta_euler)
                    self.Particle_list.append(Particle(surface_points=self.surface_points,delta_R=delta_R,init_R=self.init_R))


    def batch_fit_fun(self):
        fitness_list=[]
        R_list=[]
        t_list=[]
        s_list=[]
        for part in self.Particle_list:
            cur_R=part.pose_SO3@self.init_R
            cur_t=part.pose_ts[:3]
            cur_s=part.pose_ts[3:6]
            fitness=self.fit_fun(cur_R,cur_t,cur_s)
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

    def fit_fun(self,cur_R,cur_t,cur_s,vis=False):
        cur_corner=(self.corner*cur_s) @ cur_R.T + cur_t
        flags=Delaunay(cur_corner).find_simplex(self.surface_points)
        if vis:
            show_open3d(cur_corner,self.surface_points)
        pos=(flags>0).sum()
        total=len(flags)
        ratio=pos/total
        vol=cur_s[0]*cur_s[1]*cur_s[2]
        fitness=self.contain_w*ratio-vol*self.vol_w
        return fitness

    def update_pso(self):

        fitness_list=self.batch_fit_fun()
        max_fitness=-float('inf')
        best_part=None
        for fitness,part in zip(fitness_list,self.Particle_list):
            if fitness>max_fitness:
                max_fitness=fitness
                best_part=part
        return best_part.pose_SO3,best_part.pose_ts,max_fitness



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
