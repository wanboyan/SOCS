import os
import cv2
import math
import random

import mmcv
import numpy as np
import _pickle as cPickle
# from config.config import *
import absl.flags as flags
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS
import json
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox
from tools.dataset_utils import *
from nfmodel.uti_tool import *
from pathlib import Path


class PoseDataset(data.Dataset):
    def __init__(self, source=None, mode='train', data_dir=None,
                 n_pts=1024, img_size=256, per_obj=''):
        '''

        :param source: 'CAMERA' or 'Real' or 'CAMERA+Real'
        :param mode: 'train' or 'test'
        :param data_dir: 'path to dataset'
        :param n_pts: 'number of selected sketch point', no use here
        :param img_size: cropped image size
        '''
        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.img_size = img_size

        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        big_img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        small_img_list_path = ['CAMERA/train_small_list.txt', 'Real/train_small_list.txt',
                         'CAMERA/val_small_list.txt', 'Real/test_list.txt']
        camera_model_file_path = ['obj_models/camera_train.pkl', 'obj_models/camera_val.pkl']
        real_model_file_path = [ 'obj_models/real_train.pkl', 'obj_models/real_test.pkl']
        if FLAGS.dataset_size=='small':
            img_list_path=small_img_list_path
        else:
            img_list_path=big_img_list_path
        if mode == 'train':
            del img_list_path[2:]
        else:
            del img_list_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            model_file_path=camera_model_file_path
        elif source == 'Real':
            del img_list_path[0]
            model_file_path=real_model_file_path
        else:
            model_file_path=camera_model_file_path+real_model_file_path
            if FLAGS.big_in_whole:
                img_list_path=big_img_list_path[0:1]+big_img_list_path[1:2]
            else:
                img_list_path=small_img_list_path[0:1]+big_img_list_path[1:2]

        img_list = []
        subset_len = []
        #  aggregate all availabel datasets
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1] - subset_len[0]]
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.cat_name2id = {'bottle': 1, 'bowl': 2, 'camera': 3, 'can': 4, 'laptop': 5, 'mug': 6}
        self.id2cat_name = {'1': 'bottle', '2': 'bowl', '3': 'camera', '4': 'can', '5': 'laptop', '6': 'mug'}
        self.id2cat_name_CAMERA = {'1': '02876657',
                                   '2': '02880940',
                                   '3': '02942699',
                                   '4': '02946921',
                                   '5': '03642806',
                                   '6': '03797390'}
        # if source == 'CAMERA':
        #     self.id2cat_name = self.id2cat_name_CAMERA
        self.per_obj = per_obj
        self.per_obj_id = None
        # only train one object
        if self.per_obj in self.cat_names:
            self.per_obj_id = self.cat_name2id[self.per_obj]
            img_list_cache_dir = os.path.join(self.data_dir, 'img_list')
            if not os.path.exists(img_list_cache_dir):
                os.makedirs(img_list_cache_dir)
            if source=='CAMERA+Real' and FLAGS.big_in_whole:
                img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_big_in_whole_{FLAGS.dataset_size}_img_list.txt')
            else:
                img_list_cache_filename = os.path.join(img_list_cache_dir, f'{per_obj}_{source}_{mode}_{FLAGS.dataset_size}_img_list.txt')
            if os.path.exists(img_list_cache_filename):
                print(f'read image list cache from {img_list_cache_filename}')
                img_list_obj = [line.rstrip('\n') for line in open(os.path.join(data_dir, img_list_cache_filename))]
            else:
                # needs to reorganize img_list
                s_obj_id = self.cat_name2id[self.per_obj]
                img_list_obj = []
                from tqdm import tqdm
                for i in tqdm(range(len(img_list))):
                    gt_path = os.path.join(self.data_dir, img_list[i] + '_label.pkl')
                    try:
                        with open(gt_path, 'rb') as f:
                            gts = cPickle.load(f)
                        id_list = gts['class_ids']
                        if s_obj_id in id_list:
                            img_list_obj.append(img_list[i])
                    except:
                        print(f'WARNING {gt_path} is empty')
                        continue
                with open(img_list_cache_filename, 'w') as f:
                    for img_path in img_list_obj:
                        f.write("%s\n" % img_path)
                print(f'save image list cache to {img_list_cache_filename}')
                # iter over  all img_list, cal sublen

            if len(subset_len) == 2:
                camera_len  = 0
                real_len = 0
                for i in range(len(img_list_obj)):
                    if 'CAMERA' in img_list_obj[i].split('/'):
                        camera_len += 1
                    else:
                        real_len += 1
                self.subset_len = [camera_len, real_len]
            #  if use only one dataset
            #  directly load all data
            img_list = img_list_obj

        self.img_list = img_list
        self.length = len(self.img_list)



        models = {}
        camera_models={}
        real_models={}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                tmp_models=cPickle.load(f)
                if 'camera' in path:
                    models.update(tmp_models)
                    camera_models.update(tmp_models)
                elif 'real' in path:
                    models.update(tmp_models)
                    real_models.update(tmp_models)
        self.models = models


        if FLAGS.ratio_source!='':
            source=FLAGS.ratio_source
        ratio_cache_filename = os.path.join(self.data_dir, f'{source}_ratio.json')

        model_name2cat_name={}
        if source=='CAMERA':
            parent_paths=['obj_models/train','obj_models/val']
            for parent_path in parent_paths:
                for cat_id,cat_cat_name in self.id2cat_name_CAMERA.items():
                    model_names=sorted([p.name for p in Path(os.path.join(data_dir,parent_path,str(cat_cat_name))).glob('*')])
                    for model_name in model_names:
                        model_name2cat_name[model_name]=self.id2cat_name[cat_id]
        elif source=='Real':
            for model_name in models.keys():
                model_name_cat= model_name.split('_')[0]

                flag=False
                for cat_name in self.cat_names:
                    if cat_name in model_name_cat:
                        flag=True
                        model_name2cat_name[model_name]=cat_name
                assert flag
        else:
            parent_paths=['obj_models/train','obj_models/val']
            for parent_path in parent_paths:
                for cat_id,cat_cat_name in self.id2cat_name_CAMERA.items():
                    model_names=sorted([p.name for p in Path(os.path.join(data_dir,parent_path,str(cat_cat_name))).glob('*')])
                    for model_name in model_names:
                        model_name2cat_name[model_name]=self.id2cat_name[cat_id]
            for model_name in real_models.keys():
                model_name_cat= model_name.split('_')[0]
                for cat_name in self.cat_names:
                    if cat_name in model_name_cat:
                        model_name2cat_name[model_name]=cat_name

        modelname_cache_file = os.path.join(self.data_dir, f'modelname_cache_file.json')
        with open(modelname_cache_file,'w') as f:
                json.dump(model_name2cat_name,f)

        if os.path.exists(ratio_cache_filename):
            with open(ratio_cache_filename) as f:
                ratio_dict.update(json.load(f))
        else:
            if FLAGS.use_prior:
                assert os.path.exists(ratio_cache_filename)
            get_ratio(models,self.cat_names,model_name2cat_name)
            with open(ratio_cache_filename,'w') as f:
                json.dump(ratio_dict,f)

        cur_cat_models=[]
        for k,v in models.items():
            if model_name2cat_name[k]==self.per_obj:
                cur_cat_models.append(k)
        self.cur_cat_models=cur_cat_models
        FLAGS.cur_cat_model_num=len(cur_cat_models)
        print(len(cur_cat_models),' models in cur category')
        # move the center to the body of the mug
        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.camera_intrinsics = np.array([[577.5, 0, 319.5], [0, 577.5, 239.5], [0, 0, 1]],
                                          dtype=np.float)  # [fx, fy, cx, cy]
        self.real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)

        self.invaild_list = []
        self.mug_sym = mmcv.load(os.path.join(self.data_dir, 'Real/train/mug_handle.pkl'))



        self.sdf_points=make_scatter(200000,FLAGS.query_radius*2)
        self.pad_points=make_scatter(200000,FLAGS.pad_radius*2)
        pad_points_r=np.linalg.norm(self.pad_points,axis=-1)
        self.sphere_points=self.pad_points[pad_points_r<=0.5]
        # uniform_scale=np.array([1,1,1])
        # self.fake_grid=make_grid(8000,uniform_scale)
        # self.label_pos=make_grid(500,uniform_scale)

        with open(FLAGS.keypoint_path, 'rb') as f:
            self.model2keypoint = cPickle.load(f)
        self.cat2std_idx={'camera':2,'bottle':5}

        cur_cat_transform=[]
        std_idx=self.cat2std_idx[per_obj]
        std_model_name=cur_cat_models[std_idx]
        self.std_model=self.models[std_model_name]
        std_keypoint=self.model2keypoint[std_model_name]
        for model_name in cur_cat_models:
            cur_keypoint=self.model2keypoint[model_name]
            tps=TPS(std_keypoint,cur_keypoint)
            cur_cat_transform.append(tps)
        self.cur_cat_transform=cur_cat_transform




        print('{} images found.'.format(self.length))
        print('{} models loaded.'.format(len(self.models)))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if FLAGS.debug:
            index=13
        #   load ground truth
        #  if per_obj is specified, then we only select the target object
        # index = index % self.length  # here something wrong
        img_path = os.path.join(self.data_dir, self.img_list[index])
        if img_path in self.invaild_list:
            return self.__getitem__((index + 1) % self.__len__())
        try:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        except:
            return self.__getitem__((index + 1) % self.__len__())
        if 'CAMERA' in img_path.split('/'):
            out_camK = self.camera_intrinsics
            img_type = 'syn'
        else:
            out_camK = self.real_intrinsics
            img_type = 'real'

        # select one foreground object,
        # if specified, then select the object
        if self.per_obj != '':
            if FLAGS.debug==1:
                idx=gts['class_ids'].tolist().index(self.per_obj_id)
            else:
                idx = gts['class_ids'].index(self.per_obj_id)
        else:
            idx = random.randint(0, len(gts['instance_ids']) - 1)

        mug_handle = 1

        rgb = cv2.imread(img_path + '_color.png')
        if rgb is not None:
            rgb = rgb[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        im_H, im_W = rgb.shape[0], rgb.shape[1]
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)

        depth_path = img_path + '_depth.png'
        if os.path.exists(depth_path):
            depth = load_depth(depth_path)
        else:
            return self.__getitem__((index + 1) % self.__len__())
        if FLAGS.fill_depth:
            depth = fill_missing(depth, 1000.0, 1)
        mask_path = img_path + '_mask.png'
        mask = cv2.imread(mask_path)
        if mask is not None:
            mask = mask[:, :, 2]
        else:
            return self.__getitem__((index + 1) % self.__len__())

        coord = cv2.imread(img_path + '_coord.png')
        if coord is not None:
            coord = coord[:, :, :3]
        else:
            return self.__getitem__((index + 1) % self.__len__())
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]

        # aggragate information about the selected object
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # here resize and crop to a fixed size 256 x 256
        bbox_xyxy = np.array([cmin, rmin, cmax, rmax])
        bbox_center, scale = aug_bbox_DZI(FLAGS, bbox_xyxy, im_H, im_W)
        bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
        bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)
        ## roi_image ------------------------------------
        roi_img = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        # roi_coord_2d ----------------------------------------------------
        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        # import matplotlib.pyplot as plt
        # plt.imshow(roi_img.transpose(1, 2, 0))
        # plt.show()
        mask_target = mask.copy().astype(np.float)
        mask_target[mask != inst_id] = 0.0
        mask_target[mask == inst_id] = 1.0
        # depth[mask_target == 0.0] = 0.0
        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )

        roi_depth = np.expand_dims(roi_depth, axis=0)
        # normalize depth
        depth_valid = roi_depth > 0
        if np.sum(depth_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())
        roi_m_d_valid = roi_mask.astype(np.bool) * depth_valid
        if np.sum(roi_m_d_valid) <= 1.0:
            return self.__getitem__((index + 1) % self.__len__())

        depth_v_value = roi_depth[roi_m_d_valid]
        depth_normalize = (roi_depth - np.min(depth_v_value)) / (np.max(depth_v_value) - np.min(depth_v_value))
        depth_normalize[~roi_m_d_valid] = 0.0
        # cat_id, rotation translation and scale
        cat_id = gts['class_ids'][idx] - 1  # convert to 0-indexed
        # note that this is nocs model, normalized along diagonal axis
        model_name = gts['model_list'][idx]
        model = self.models[gts['model_list'][idx]].copy().astype(np.float32) # 20000 points
        model_indices = np.random.randint(len(model), size=FLAGS.random_points)
        try:
            model_idx=self.cur_cat_models.index(gts['model_list'][idx])
        except:
            print('invalid id',gts['model_list'][idx])
            return self.__getitem__((index + 1) % self.__len__())





        if FLAGS.debug:
            model_indices=np.arange(FLAGS.random_points)
        model=model[model_indices]

        nocs_scale = gts['scales'][idx]  # nocs_scale = image file / model file
        # fsnet scale (from model) scale residual
        real_scale,fsnet_scale, mean_shape = self.get_fs_net_scale(self.id2cat_name[str(cat_id + 1)], model, nocs_scale)



        real_scale = real_scale /1000.0
        fsnet_scale = fsnet_scale / 1000.0
        mean_shape = mean_shape / 1000.0
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]

        # dense depth map
        dense_depth = depth_normalize
        # sym
        sym_info = self.get_sym_info(self.id2cat_name[str(cat_id + 1)], mug_handle=mug_handle)
        if FLAGS.decoder_only or img_type=='syn' :
            if sym_info[0]==1 and np.abs(real_scale[0]-real_scale[2])>0.01:
                # show_open3d(model,model)
                return self.__getitem__((index + 1) % self.__len__())


        # add nnoise to roi_mask
        roi_mask_def = defor_2D(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        # generate augmentation parameters
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()


        sdf_indices = np.random.randint(len(self.sdf_points), size=FLAGS.query_num)
        pad_indeces = np.random.randint(len(self.sdf_points), size=FLAGS.pad_num)
        sphere_indeces = np.random.randint(len(self.sphere_points), size=FLAGS.pad_num)

        if FLAGS.debug:
            sdf_indices=np.arange(FLAGS.query_num)
            pad_indeces=np.arange(FLAGS.pad_num)
        pad_points = self.pad_points[pad_indeces]
        sdf_points=self.sdf_points[sdf_indices]
        # show_open3d(sdf_points,sdf_points)
        sphere_points=self.sphere_points[sphere_indeces]


        deform_sdf_points=self.cur_cat_transform[model_idx](sdf_points)
        deform_std_model=self.cur_cat_transform[model_idx](self.std_model)
        # show_open3d(self.std_model,model)
        # show_open3d(deform_std_model,model)

        # max_real_scale=real_scale.max()
        # pad_points=pad_points*max_real_scale/real_scale
        # nocs_sdf_points=np.concatenate([sdf_points,pad_points],axis=0)
        # query_points=(nocs_sdf_points*real_scale) @ (rotation.T) + translation

        # show_open3d(nocs_sdf_points*real_scale,model*nocs_scale)




        data_dict = {}
        data_dict['roi_img'] = torch.as_tensor(roi_img.astype(np.float32)).contiguous()
        data_dict['roi_depth'] = torch.as_tensor(roi_depth.astype(np.float32)).contiguous()
        data_dict['dense_depth'] = torch.as_tensor(dense_depth.astype(np.float32)).contiguous()
        data_dict['depth_normalize'] = torch.as_tensor(depth_normalize.astype(np.float32)).contiguous()
        data_dict['cam_K'] = torch.as_tensor(out_camK.astype(np.float32)).contiguous()
        data_dict['roi_mask'] = torch.as_tensor(roi_mask.astype(np.float32)).contiguous()
        data_dict['cat_id'] = torch.as_tensor(cat_id, dtype=torch.float32).contiguous()
        data_dict['rotation'] = torch.as_tensor(rotation, dtype=torch.float32).contiguous()
        data_dict['translation'] = torch.as_tensor(translation, dtype=torch.float32).contiguous()
        data_dict['fsnet_scale'] = torch.as_tensor(fsnet_scale, dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.as_tensor(sym_info.astype(np.float32)).contiguous()
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2d, dtype=torch.float32).contiguous()
        data_dict['mean_shape'] = torch.as_tensor(mean_shape, dtype=torch.float32).contiguous()
        data_dict['aug_bb'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['roi_mask_deform'] = torch.as_tensor(roi_mask_def, dtype=torch.float32).contiguous()
        data_dict['model_point'] = torch.as_tensor(model, dtype=torch.float32).contiguous()
        data_dict['nocs_scale'] = torch.as_tensor(nocs_scale, dtype=torch.float32).contiguous()
        data_dict['pad_points'] = torch.as_tensor(pad_points,dtype=torch.float32).contiguous()
        data_dict['sdf_points'] = torch.as_tensor(sdf_points,dtype=torch.float32).contiguous()
        data_dict['deform_sdf_points'] = torch.as_tensor(deform_sdf_points,dtype=torch.float32).contiguous()
        data_dict['sphere_points'] = torch.as_tensor(sphere_points,dtype=torch.float32).contiguous()
        data_dict['model_idx'] = torch.as_tensor(model_idx,dtype=torch.float32).contiguous()
        return data_dict


    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm


    def get_fs_net_scale(self, c, model, nocs_scale):
        # model pc x 3
        if c=='mug':
            lx = 2 * np.amax(np.abs(model[:, 0]))
            ly = 2 * np.amax(np.abs(model[:, 1]))
            lz = 2 * np.amax(np.abs(model[:, 2]))
        else:
            lx = max(model[:, 0]) - min(model[:, 0])
            ly = max(model[:, 1]) - min(model[:, 1])
            lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000
        if self.source=='Real' or self.source=='CAMERA+Real' :
            if c == 'bottle':
                unitx = 87
                unity = 220
                unitz = 89
            elif c == 'bowl':
                unitx = 165
                unity = 80
                unitz = 165
            elif c == 'camera':
                unitx = 88
                unity = 128
                unitz = 156
            elif c == 'can':
                unitx = 68
                unity = 146
                unitz = 72
            elif c == 'laptop':
                unitx = 346
                unity = 200
                unitz = 335
            elif c == 'mug':
                unitx = 146
                unity = 83
                unitz = 114
        elif self.source=='CAMERA':
            if c == 'bottle':
                unitx = 324 / 4
                unity = 874 / 4
                unitz = 321 / 4
            elif c == 'bowl':
                unitx = 675 / 4
                unity = 271 / 4
                unitz = 675 / 4
            elif c == 'camera':
                unitx = 464 / 4
                unity = 487 / 4
                unitz = 702 / 4
            elif c == 'can':
                unitx = 450 / 4
                unity = 753 / 4
                unitz = 460 / 4
            elif c == 'laptop':
                unitx = 581 / 4
                unity = 445 / 4
                unitz = 672 / 4
            elif c == 'mug':
                unitx = 670 / 4
                unity = 540 / 4
                unitz = 497 / 4
        return np.array([lx_t, ly_t, lz_t]), np.array([lx_t - unitx, ly_t - unity, lz_t - unitz]), np.array([unitx, unity, unitz])

    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 'bottle':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'bowl':
            sym = np.array([1, 1, 0, 1], dtype=np.int)
        elif c == 'camera':
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        elif c == 'can':
            sym = np.array([1, 1, 1, 1], dtype=np.int)
        elif c == 'laptop':
            sym = np.array([0, 1, 0, 0], dtype=np.int)
        elif c == 'mug' and mug_handle == 1:
            sym = np.array([0, 1, 0, 0], dtype=np.int)  # for mug, we currently mark it as no symmetry
        elif c == 'mug' and mug_handle == 0:
            sym = np.array([1, 0, 0, 0], dtype=np.int)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int)
        return sym
