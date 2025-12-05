#!/bin/bash
DATA_DIR="/home/yinxiaoran/data/modified_libero_rlds"

datasets=(
  "libero_10_no_noops"
  "libero_goal_no_noops"
  "libero_object_no_noops"
  "libero_spatial_no_noops"
)

for ds in "${datasets[@]}"; do
  echo "转换数据集 $ds"
  python convert_libero_data_to_lerobot.py --data_dir $DATA_DIR --raw_dataset_name $ds
done
