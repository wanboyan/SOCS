from __future__ import print_function

import absl.flags as flags

# eval
flags.DEFINE_string('eval_out', 'output/v5_pad8', '')
flags.DEFINE_integer('fake_grid_num', 2000, '')
flags.DEFINE_integer('new_query_num', 500, '')
flags.DEFINE_integer('use_mean_init',0, '')
flags.DEFINE_string('per_obj', 'camera', '')
flags.DEFINE_float('fake_radius',0.6 , '')
flags.DEFINE_float('refine_lr', 0.0011, '')
flags.DEFINE_integer('refine_step', 100, '')
flags.DEFINE_integer('use_pso',0,'')
flags.DEFINE_integer('verbose',0,'')
flags.DEFINE_integer('use_interpo',1,'')
flags.DEFINE_integer('use_refine',1,'')
flags.DEFINE_integer('use_base',1, '')
flags.DEFINE_integer('use_distance',1 , '')
flags.DEFINE_integer('use_adaptive_distance',0, '')
flags.DEFINE_integer('use_prob',1, '')
flags.DEFINE_integer('eval_use_deform',0, '')
flags.DEFINE_integer('shuffle',0, '')
flags.DEFINE_string('eval_optimizer','adam', '')

flags.DEFINE_string('pic_save_dir', 'pic_save_dir', '')
flags.DEFINE_integer('pic_save', 1, '')
flags.DEFINE_integer('cur_eval_index', -1, '')


# train
flags.DEFINE_integer('aug_deform', 0, '')
flags.DEFINE_integer('aug_bg', 0, '')
flags.DEFINE_integer('debug', 0, '')
flags.DEFINE_float('ratio', 1, '')
flags.DEFINE_string('ratio_source', 'Real', '')
flags.DEFINE_integer('two_back', 0, '')
flags.DEFINE_integer('use_fsnet',0,'')
flags.DEFINE_integer('use_deform_emb',1,'')
flags.DEFINE_integer('use_deform',0,'')
flags.DEFINE_integer('cur_cat_model_num',0,'')
flags.DEFINE_integer('keypoint_num',36,'')
flags.DEFINE_integer('use_refine_loss',0,'')
flags.DEFINE_integer('use_nocs_loss',1,'')
flags.DEFINE_integer('align_origin',0,'')
flags.DEFINE_integer('regular_tps',0,'')
flags.DEFINE_integer('feat_for_ts', 0, '')
flags.DEFINE_integer('force_sym', 0, '')
flags.DEFINE_integer('min_loss', 0, '')

flags.DEFINE_float('inter_nocs_lr', 1.0, '')
flags.DEFINE_float('inter_R_lr', 1.0, '')
flags.DEFINE_float('inter_t_lr', 1.0, '')



flags.DEFINE_integer('use_consistency', 0, '')
flags.DEFINE_integer('use_inter', 0, '')

flags.DEFINE_integer('cut_backbone', 0, '')

flags.DEFINE_float('consistency_w', 1.0, '')


flags.DEFINE_integer('use_global', 1, '')
flags.DEFINE_integer('use_dir_rpe', 0, '')
flags.DEFINE_integer('bin_size', 100, '')
flags.DEFINE_integer('query_num', 1000, '')
flags.DEFINE_integer('pad_num', 1000, '')
flags.DEFINE_float('query_radius', 0.6, '')
flags.DEFINE_float('pad_radius', 0.6, '')



flags.DEFINE_float('lr_embedding', 1e3, '')
flags.DEFINE_float('lr_decoder', 1.0, '')
flags.DEFINE_float('lr_backbone', 1.0, '')
flags.DEFINE_float('lr_rot', 1.0, '')
flags.DEFINE_float('lr_interpo', 1.0, '')
flags.DEFINE_float('lr_ts', 1.0, '')


flags.DEFINE_float('deform_w', 10, '')
flags.DEFINE_float('kld_w', 0, '')
flags.DEFINE_float('cd_w', 0, '')
flags.DEFINE_float('grad_w', 1.0, '')
flags.DEFINE_float('interpo_w', 10, '')
flags.DEFINE_float('tran_w', 8.0, '')
flags.DEFINE_float('size_w', 8.0, '')
flags.DEFINE_float('recon_w', 8.0, '')
flags.DEFINE_float('rot_1_w', 8.0, '')
flags.DEFINE_float('rot_2_w', 8.0, '')
flags.DEFINE_float('rot_regular', 4.0, '')
flags.DEFINE_float('r_con_w', 1.0, '')

flags.DEFINE_integer('feat_c_R', 1280, 'input channel of rotation')
# flags.DEFINE_integer('feat_c_R', 1792, 'input channel of rotation')
flags.DEFINE_integer('R_c', 3, 'output channel of rotation, here confidence(1)+ rot(3)')
flags.DEFINE_integer('feat_c_ts', 1283, 'input channel of translation and size')
# flags.DEFINE_integer('feat_c_ts', 1795, 'input channel of translation and size')
flags.DEFINE_integer('Ts_c', 6,  'output channel of translation (3) + size (3)')

flags.DEFINE_integer('feat_face',768, 'input channel of the face recon')
flags.DEFINE_integer('face_recon_c', 6 * 5, 'for every point, we predict its diFtance and normal to each face')
#  the storage form is 6*3 normal, then the following 6 parametes distance, the last 6 parameters confidence
flags.DEFINE_integer('gcn_sup_num', 7, 'support number for gcn')
flags.DEFINE_integer('gcn_n_num', 10, 'neighbor number for gcn')



# datasets
flags.DEFINE_integer('obj_c', 6, 'nnumber of categories')
flags.DEFINE_string('eval_dataset', 'Real', 'CAMERA or CAMERA+Real')
flags.DEFINE_string('dataset', 'Real', 'CAMERA or CAMERA+Real')
flags.DEFINE_string('dataset_dir', '/data_nvme/nocs_data', 'path to the dataset')
flags.DEFINE_string('detection_dir', '/data_nvme/nocs_data/segmentation_results', 'path to detection results')
flags.DEFINE_string('dataset_size','small','big for whole dataset')
flags.DEFINE_integer('big_in_whole',1,'')

# dynamic zoom in
flags.DEFINE_float('DZI_PAD_SCALE', 1.5, '')
flags.DEFINE_string('DZI_TYPE', 'uniform', '')
flags.DEFINE_float('DZI_SCALE_RATIO', 0.25, '')
flags.DEFINE_float('DZI_SHIFT_RATIO', 0.25, '')

# input parameters
flags.DEFINE_integer("img_size", 256, 'size of the cropped image')

# data aug parameters
flags.DEFINE_integer('roi_mask_r', 3, 'radius for mask aug')
flags.DEFINE_float('roi_mask_pro', 0.5, 'probability to augment mask')
flags.DEFINE_float('aug_pc_pro', 1, 'probability to augment pc')
flags.DEFINE_float('aug_pc_r', 0.002, 'change 2mm')
flags.DEFINE_float('aug_rt_pro', 0.3, 'probability to augment rt')
flags.DEFINE_float('aug_bb_pro', 0.3, 'probability to augment size')
flags.DEFINE_float('aug_bc_pro', 0.3, 'box cage based augmentation, only valid for bowl, mug')

# pose network


# point selection
flags.DEFINE_integer('random_points', 1028, 'number of points selected randomly')
flags.DEFINE_string('sample_method', 'farthest', 'basic')

# train parameters
# train##################################################
flags.DEFINE_integer("train", 1, "1 for train mode")
# flags.DEFINE_integer('eval', 0, '1 for eval mode')
flags.DEFINE_string('device', 'cuda:0', '')
# flags.DEFINE_string("train_gpu", '0', "gpu no. for training")
flags.DEFINE_integer("num_workers", 8, "cpu cores for loading dataset")
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('total_epoch', 70, 'total epoches in training')
flags.DEFINE_integer('train_steps', 1500, 'number of batches in each epoch')  # batchsize is 8, then 3000
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate

# test parameters

# for different losses
flags.DEFINE_string('fsnet_loss_type', 'l1', 'l1 or smoothl1')



# training parameters
# learning rate scheduler
flags.DEFINE_float('lr', 30e-4, '')
 # initial learning rate w.r.t basic lr
flags.DEFINE_float('lr_pose', 1.0, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')  # some parameter for the scheduler### optimizer  ####
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'linear/warm_flat_anneal/')
flags.DEFINE_string('anneal_method', 'cosine', '')
flags.DEFINE_float('anneal_point', 0.5, '')
flags.DEFINE_string('optimizer_type', 'Ranger', '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_float('warmup_factor', 0.001, '')
flags.DEFINE_integer('warmup_iters', 1000, '')
flags.DEFINE_string('warmup_method', 'linear', '')
flags.DEFINE_float('gamma', 0.1, '')
flags.DEFINE_float('poly_power', 0.9, '')

# save parameters
#model to save
flags.DEFINE_integer('save_every', 5, '')  # save models every 'save_every' epoch
flags.DEFINE_integer('log_every', 10, '')  # save log file every 100 iterations
flags.DEFINE_string('model_save', 'output/v5_pad7', 'path to save checkpoint')


# resume
flags.DEFINE_integer('resume',0, '1 for resume, 0 for training from the start')
# flags.DEFINE_string('resume_dir', '/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/engine/output/v5_pad7','')
flags.DEFINE_string('resume_dir', '/data_nvme/data_output/output/whole_lr0.003','')
flags.DEFINE_string('resume_model_name', 'model_69.pth','')
flags.DEFINE_string('resume_base_model_name', 'model_69.pth','')
flags.DEFINE_integer('resume_point', 0, 'the epoch to continue the training')


flags.DEFINE_string('base_dir','/home/wanboyan/Documents/cmr-master/extern/GPV_Pose-master/output/NFwhole_prior_7', '')

flags.DEFINE_string('keypoint_path', '/data_nvme/nocs_data/keypoint.pkl','')

###################for evaluation#################
flags.DEFINE_integer('eval_visualize_pcl', 0, 'save pcl when evaluation')
flags.DEFINE_integer('eval_inference_only', 0, 'inference without evaluation')

