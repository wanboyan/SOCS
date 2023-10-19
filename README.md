# SOCS
Pytorch implementation of SOCS: Semantically-aware Object Coordinate Space for Category-Level 6D Object Pose Estimation under Large Shape Variations
([link](https://arxiv.org/abs/2303.10346))

visualization of coordinate in NOCS:
![teaser1](pic/71b9002ada8a67abbec5b68dc28a2333.gif)
visualization of coordinate in SOCS:
![teaser2](pic/d2171df993a2789e431c73115cda8b06.gif)
pose optimization by SOCS
![teaser3](pic/animation.gif)

## Training

```shell
python -m engine.NFtrain_v5 --per_obj bottle --dataset Real --dataset_dir /nocs_data \
--keypoint_path /nocs_data/keypoint.pkl \
--lr 0.003 \
--aug_bg 0 \
--train_steps 1500 \
--use_deform 1 \
--model_save output/whole_lr0.003_1500_deform
```