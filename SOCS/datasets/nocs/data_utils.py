import os
import numpy as np
from pathlib import Path


def get_nocs_models(nocs_data_dir, only_real=False):
    camera_model_path=['obj_models/camera_train.pkl', 'obj_models/camera_val.pkl']
    real_model_path=['obj_models/real_test.pkl','obj_models/real_train.pkl']
    mean_shape_path='obj_models/mean_shapes.npy'
    cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
    cat_shapenet_names={1: '02876657',
                        2: '02880940',
                        3: '02942699',
                        4: '02946921',
                        5: '03642806',
                        6: '03797390'}

    model_name2cat_name={}
    real_models = {}
    camera_models={}
    models={}
    import _pickle as cPickle
    for path in real_model_path:
        with open(os.path.join(nocs_data_dir,path), 'rb') as f:
            tmp_models=cPickle.load(f)
            real_models.update(tmp_models)

    for path in camera_model_path:
        with open(os.path.join(nocs_data_dir,path), 'rb') as f:
            tmp_models=cPickle.load(f)
            camera_models.update(tmp_models)

    for model_name in real_models.keys():
        model_name_cat= model_name.split('_')[0]
        flag=False
        for cat_name in cat_names:
            if cat_name in model_name_cat:
                flag=True
                model_name2cat_name[model_name]=cat_name
        assert flag



    parent_paths=['obj_models/train','obj_models/val']
    for parent_path in parent_paths:
        for index,cat_shapenet_name in cat_shapenet_names.items():
            model_names=sorted([p.name for p in Path(os.path.join(nocs_data_dir,parent_path,cat_shapenet_name)).glob('*')])
            for model_name in model_names:
                model_name2cat_name[model_name]=cat_names[index-1]




    mean_shape_np=np.load(os.path.join(nocs_data_dir,mean_shape_path))

    mean_models={}
    for i,mean_shape in enumerate(mean_shape_np):
        cur_name=cat_names[i]+'mean'
        mean_models[cur_name]=mean_shape
        model_name2cat_name[cur_name]=cat_names[i]
    models.update(mean_models)
    models.update(real_models)
    if not only_real:
        models.update(camera_models)
    return model_name2cat_name,models