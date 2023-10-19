#!/bin/bash
cd /data/home/scv2060/run/wby/GPV_Pose-master
module load CUDA/11.3
module load anaconda/2020.11
source activate pytorch1.11.0
python -m engine.NFtrain_v5 --per_obj bottle --dataset CAMERA+Real --dataset_dir /data/home/scv2060/run/wby/nocs_data \
--keypoint_path /data/run01/scv2060/wby/nocs_data/keypoint.pkl \
--lr 0.003 \
--aug_bg 0 \
--train_steps 1500 \
--use_deform 1 \
--model_save output/whole_lr0.003_1500_deform
