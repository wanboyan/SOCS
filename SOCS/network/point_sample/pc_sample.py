import torch
import torch.nn.functional as F
import numpy as np
import absl.flags as flags

FLAGS = flags.FLAGS

def PC_sample(obj_mask, Depth, camK, coor2d):
    '''
    :param Depth: bs x 1 x h x w
    :param camK:
    :param coor2d:
    :return:
    '''
    # handle obj_mask
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    '''
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num

    PC = torch.zeros([bs, samplenum, 3], dtype=torch.float32, device=Depth.device)

    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        # basic sampling
        if FLAGS.sample_method == 'basic':
            l_all = p_n_now.shape[0]
            if l_all <= 1.0:
                return None, None
            if l_all >= samplenum:
                replace_rnd = False
            else:
                replace_rnd = True

            choose = np.random.choice(l_all, samplenum, replace=replace_rnd)  # can selected more than one times
            p_select = p_n_now[choose, :]
        elif FLAGS.sample_method == 'farthest':
            choose=farthest_sample(p_n_now,samplenum)
            l_all = p_n_now.shape[0]
            p_select = p_n_now[choose, :]
            # if l_all <= 500:
            #     continue
        # reprojection
        if p_select.shape[0] > samplenum:
            p_select = p_select[p_select.shape[0]-samplenum:p_select.shape[0], :]

        PC[i, ...] = p_select[:, :3]

    return PC / 1000.0


def proj(obj_mask, Depth, camK, coor2d):
    if obj_mask.shape[1] == 2:   # predicted mask
        obj_mask = F.softmax(obj_mask, dim=1)
        _, obj_mask = torch.max(obj_mask, dim=1)
    '''
    import matplotlib.pyplot as plt
    plt.imshow(obj_mask[0, ...].detach().cpu().numpy())
    plt.show()
    '''
    bs, H, W = Depth.shape[0], Depth.shape[2], Depth.shape[3]
    x_label = coor2d[:, 0, :, :]
    y_label = coor2d[:, 1, :, :]

    rand_num = FLAGS.random_points
    samplenum = rand_num
    PCs=[]

    for i in range(bs):
        dp_now = Depth[i, ...].squeeze()   # 256 x 256
        x_now = x_label[i, ...]   # 256 x 256
        y_now = y_label[i, ...]
        obj_mask_now = obj_mask[i, ...].squeeze()  # 256 x 256
        dp_mask = (dp_now > 0.0)
        fuse_mask = obj_mask_now.float() * dp_mask.float()

        camK_now = camK[i, ...]

        # analyze camK
        fx = camK_now[0, 0]
        fy = camK_now[1, 1]
        ux = camK_now[0, 2]
        uy = camK_now[1, 2]

        x_now = (x_now - ux) * dp_now / fx
        y_now = (y_now - uy) * dp_now / fy

        p_n_now = torch.cat([x_now[fuse_mask > 0].view(-1, 1),
                             y_now[fuse_mask > 0].view(-1, 1),
                             dp_now[fuse_mask > 0].view(-1, 1)], dim=1)

        PCs.append(p_n_now/1000.0)
    return PCs



def farthest_sample(points,sample_num):
    from OpenPCDet.pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
    point_cnt=points.new_zeros(1).int()
    point_cnt[0]=points.shape[0]
    sample_cnt=points.new_zeros(1).int()
    sample_cnt[0]=sample_num
    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        points.contiguous(), point_cnt, sample_cnt
    ).long()
    return sampled_pt_idxs