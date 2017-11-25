#!/bin/bash
###data_dir###
path="/home_local/kenin815/data/"
dir_audio=$path'500query_data/audio_map_fast.txt'
dir_query=$path'500query_data/query_map_fast.txt'
dir_vallabel=$path'test_small_iv.label'
dir_val_audio=$path'500query_data/val_audio_fast.txt'
dir_val_query=$path'500query_data/val_query_fast.txt'
###model_para###
train_mode=1 # 1 for training , 0 for testing

dir_model="model_save/"
gpu_num=0
audio_maxframe=15
query_maxframe=15
batch_size=150
test_batch=400
epoch_num=10
vec_dim=128
lr=0.001
layer=2
hop_num=3
saved_model_name="model_save/weight/model_7_0.5650"
echo "Data "
echo "    Audio_dir     : $dir_audio        "
echo "    Query_dir     : $dir_query        "
echo "    Query_val_dir : $dir_val_query    "
echo "    Audio_val_dir : $dir_val_audio    "
echo "    Label_val_dir : $dir_vallabel     "
echo " "
echo "Model_dir: $dir_model "

mkdir -p $dir_model"/weight"

python main.py $dir_model $gpu_num $dir_audio $dir_query no_label $dir_val_audio $dir_val_query $dir_vallabel  $train_mode\
       $audio_maxframe $query_maxframe $batch_size $test_batch $epoch_num $vec_dim $lr  $layer $saved_model_name $hop_num
