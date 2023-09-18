import os
import random


import torch
from absl import app

from config.nocs.NFconfig_v5 import *
from tools.training_utils import build_lr_rate, get_gt_v, build_optimizer
from nfmodel.nocs.NFnetwork_v5 import NFPose

FLAGS = flags.FLAGS
from datasets.nocs.load_nf_data_v5 import PoseDataset
import numpy as np
import time

# from creating log
import tensorflow as tf
from tools.eval_utils import setup_logger, compute_sRT_errors


def seed_everything(seed=20):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

device = 'cuda'
def train(argv):
    if FLAGS.debug==1:
        seed_everything(20)
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    cat_list=['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    # cat_list=[ 'bottle','camera', 'can', 'laptop']


    if FLAGS.per_obj is not '':
        cat_list=[FLAGS.per_obj]
    for cat_name in cat_list:
        FLAGS.append_flags_into_file(os.path.join(FLAGS.model_save, 'config.txt'))
        cat_model_save=os.path.join(FLAGS.model_save,cat_name)
        if not os.path.exists(cat_model_save):
            os.makedirs(cat_model_save)
        tf.compat.v1.disable_eager_execution()
        tb_writter = tf.compat.v1.summary.FileWriter(cat_model_save)
        logger = setup_logger('train_log', os.path.join(cat_model_save, 'log.txt'))
        for key, value in FLAGS.flag_values_dict().items():
            logger.info(key + ':' + str(value))

        # resume or not

        # build dataset annd dataloader
        if FLAGS.debug==1:
            FLAGS.batch_size=1
            train_dataset = PoseDataset(source=FLAGS.dataset, mode='test',
                                        data_dir=FLAGS.dataset_dir, per_obj=cat_name)
        else:
            train_dataset = PoseDataset(source=FLAGS.dataset, mode='train',
                                        data_dir=FLAGS.dataset_dir, per_obj=cat_name)


        network = NFPose()
        network = network.to(device)
        if FLAGS.debug==1:
            network.eval()
        else:
            network.train()

        if FLAGS.resume:
            resume_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            network.load_state_dict(torch.load(resume_path))
            # network.load_prior(cat_name,FLAGS.prior_dir,FLAGS.prior_name)
            s_epoch = FLAGS.resume_point
        else:
            s_epoch = 0


        st_time = time.time()
        train_steps = FLAGS.train_steps
        global_step = train_steps * s_epoch  # record the number iteration
        train_size = train_steps * FLAGS.batch_size
        indices = []
        page_start = - train_size

        #  build optimizer
        param_list = network.build_params()
        optimizer = build_optimizer(param_list)
        optimizer.zero_grad()   # first clear the grad
        scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
        #  training iteration, this code is develop based on object deform net
        for epoch in range(s_epoch, FLAGS.total_epoch):
            # train one epoch
            logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                          ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))
            # create optimizer and adjust learning rate accordingly
            # sample train subset
            page_start += train_size
            len_last = len(indices) - page_start
            if len_last < train_size:
                indices = indices[page_start:]
                if FLAGS.dataset == 'CAMERA+Real':
                    camera_len = train_dataset.subset_len[0]
                    real_len = train_dataset.subset_len[1]
                    real_indices = list(range(camera_len, camera_len + real_len))
                    camera_indices = list(range(camera_len))
                    n_repeat = (train_size - len_last) // (4 * real_len) + 1
                    data_list = random.sample(camera_indices, 3 * n_repeat * real_len) + real_indices * n_repeat
                    random.shuffle(data_list)
                    indices += data_list
                else:
                    data_list = list(range(train_dataset.length))
                    for i in range((train_size - len_last) // train_dataset.length + 1):
                        random.shuffle(data_list)
                        indices += data_list
                page_start = 0
            train_idx = indices[page_start:(page_start + train_size)]
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                           sampler=train_sampler,
                                                           num_workers=FLAGS.num_workers, pin_memory=True)
            network.train()
            batch_start=time.time()
            batch_loss_end=time.time()
            #################################
            for i, data in enumerate(train_dataloader, 1):
                batch_start=time.time()
                do_refine=True
                loss_dict= network(rgb=data['roi_img'].to(device), depth=data['roi_depth'].to(device),
                              obj_id=data['cat_id'].to(device), camK=data['cam_K'].to(device), gt_mask=data['roi_mask'].to(device),
                              gt_R=data['rotation'].to(device), gt_t=data['translation'].to(device),
                              gt_s=data['fsnet_scale'].to(device), mean_shape=data['mean_shape'].to(device),
                              gt_2D=data['roi_coord_2d'].to(device), sym=data['sym_info'].to(device),
                              aug_bb=data['aug_bb'].to(device), aug_rt_t=data['aug_rt_t'].to(device), aug_rt_r=data['aug_rt_R'].to(device),
                              def_mask=data['roi_mask_deform'].to(device),
                              pad_points=data['pad_points'].to(device),sdf_points=data['sdf_points'].to(device),
                              model_point=data['model_point'].to(device), nocs_scale=data['nocs_scale'].to(device),
                              model_idx=data['model_idx'].to(device),
                              deform_sdf_points=data['deform_sdf_points'].to(device),
                              coefficient=data['tps_coefficient'].to(device),
                              control_points=data['tps_control_points'].to(device),
                              re_coefficient=data['re_tps_coefficient'].to(device),
                              re_control_points=data['re_tps_control_points'].to(device),
                              std_model=data['std_model'],
                              do_aug=True,
                              cat_name=cat_name,do_refine=do_refine)
                fsnet_loss = loss_dict['fsnet_loss']
                inter_loss=loss_dict['inter_loss']
                interpo_loss= loss_dict['interpo_loss']
                consistency_loss=loss_dict['consistency_loss']
                compare2init=loss_dict['compare2init']
                vae_loss=loss_dict['vae']
                total_loss = sum(fsnet_loss.values()) + sum(interpo_loss.values())\
                             +sum(consistency_loss.values())+sum(inter_loss.values())+sum(vae_loss.values())

                # backward
                if global_step % FLAGS.accumulate == 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
                    optimizer.step()
                    scheduler.step()
                    # for name,params in network.decoder.named_parameters():
                    #     print(name,params.grad)
                    optimizer.zero_grad()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)

                batch_dur=time.time()-batch_loss_end
                batch_loss_end=time.time()
                batch_loss_dur=batch_loss_end-batch_start
                global_step += 1
                summary = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='lr',
                                                                                 simple_value=optimizer.param_groups[0]["lr"]),
                                                      tf.compat.v1.Summary.Value(tag='train_loss', simple_value=total_loss),
                                                      tf.compat.v1.Summary.Value(tag='rot_loss_1',
                                                                                 simple_value=fsnet_loss['Rot1']),
                                                      tf.compat.v1.Summary.Value(tag='rot_loss_2',
                                                                                 simple_value=fsnet_loss['Rot2']),
                                                      tf.compat.v1.Summary.Value(tag='rot_1_cos',
                                                                                 simple_value=fsnet_loss['Rot1_cos']),
                                                      tf.compat.v1.Summary.Value(tag='rot_2_cos',
                                                                                 simple_value=fsnet_loss['Rot2_cos']),
                                                      tf.compat.v1.Summary.Value(tag='rot_regular',
                                                                                 simple_value=fsnet_loss['Rot_r_a']),
                                                      tf.compat.v1.Summary.Value(tag='T_loss',
                                                                                 simple_value=fsnet_loss['Tran']),
                                                      tf.compat.v1.Summary.Value(tag='size_loss',
                                                                                 simple_value=fsnet_loss['Size']),
                                                      tf.compat.v1.Summary.Value(tag='nocs',
                                                                                 simple_value=interpo_loss['Nocs']),
                                                      tf.compat.v1.Summary.Value(tag='grad',
                                                                                 simple_value=interpo_loss['grad']),
                                                      tf.compat.v1.Summary.Value(tag='deform',
                                                                                 simple_value=interpo_loss['deform']),
                                                      tf.compat.v1.Summary.Value(tag='consistency',
                                                                                 simple_value=consistency_loss['consistency']),
                                                      tf.compat.v1.Summary.Value(tag='inter_r',
                                                                                 simple_value=inter_loss['inter_r']),
                                                      tf.compat.v1.Summary.Value(tag='inter_t',
                                                                                 simple_value=inter_loss['inter_t']),
                                                      tf.compat.v1.Summary.Value(tag='inter_nocs',
                                                                                 simple_value=inter_loss['inter_nocs']),
                                                      tf.compat.v1.Summary.Value(tag='CD',
                                                                                 simple_value=vae_loss['CD'])

                                                      ])
                tb_writter.add_summary(summary, global_step)
                # tb_writter.add_mesh('box',vertices=state['pc'],colors=state['color'])
                if i % FLAGS.log_every == 0:
                    logger.info('Batch {0} Loss:{1:f}, rot_loss:{2:f}, size_loss:{3:f}, trans_loss:{4:f},'
                                'Nocs_loss:{5:f} grad:{6:f} inter:{7:f}'.format(
                            i, total_loss, (fsnet_loss['Rot1']+fsnet_loss['Rot2']),
                        fsnet_loss['Size'], fsnet_loss['Tran'],
                    interpo_loss['Nocs'],interpo_loss['grad'],inter_loss['inter_nocs']))
                    logger.info('batch_dur {0:f} , bach_loss_dur {1:f}'.format(batch_dur,batch_loss_dur))

            logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<'.format(epoch))

            # save model
            if (epoch + 1) % FLAGS.save_every == 0 or (epoch + 1) == FLAGS.total_epoch:
                torch.save(network.state_dict(), '{0}/model_{1:02d}.pth'.format(cat_model_save, epoch))


if __name__ == "__main__":
    app.run(train)
