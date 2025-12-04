#!/bin/bash
# ROOT_DIR="/home/yzq/mnt/code/MLLM/LLaVA-NeXT"

# ========================================LLaVA-NeXT-Video-7B-DPO================================== 
idx=-5
frames_thre=10
cfg='stvg/experiments/hcstvg2.yaml'
split='val'
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 stvg/hc_llava_next_video.py \
    --cfg_file ${cfg} \
    --dataset_name 'hcstvg2' \
    --split ${split} \
    --model-path "/home/zaiquyang2/scratch/mllm/model_zoo/LLaVA-NeXT-Video-7B-DPO" \
    --load_8bit True \
    --model_name 'LLaVA_NeXT_Video_7B' \
    --overwrite True \
    --mm_spatial_pool_stride 4 \
    --conv-mode vicuna_v1 \
    --mm_spatial_pool_mode "average" \
    --mm_newline_position "no_token" \
    --yes_id 3582 \
    --no_id 1217 \
    --s_lr 8.0 \
    --t_lr 8.0 \
    --ttt_epoch 2 \
    --temporal_thre 0.5 \
    --frames_thre ${frames_thre} \
    --llm_infer_mode 'base' \
    --img_size 336 \
    --token_idx ${idx} \
    --exp_comment "_token_${idx}" \
    INPUT.SAMPLE_FRAMES 20 \
    INPUT.RESOLUTION 336

# bash custom_infer/STVG/run_hcstvg2.sh 

# idx=-3
# frames_thre=10
# cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
# split='val'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_one_vision.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg2' \
#     --split ${split} \
#     --model-path "model_zoo/llava-onevision-qwen2-7b-ov" \
#     --load_8bit False \
#     --model_name 'llava_onevision_qwen2_7b_ov' \
#     --overwrite True \
#     --mm_spatial_pool_stride 5 \
#     --conv-mode qwen_1_5 \
#     --mm_spatial_pool_mode "bilinear" \
#     --mm_newline_position "one_token" \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 8.0 \
#     --t_lr 8.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'base' \
#     --img_size 336 \
#     --token_idx ${idx} \
#     --exp_comment "_token_track_oracle" \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 336







# bash custom_infer/STVG/run_hcstvg2.sh model_zoo/LLaVA-NeXT-Video-7B-DPO vicuna_v1 30 4 average no_token True "hhh.mp4"

# srun --partition=stingy --qos=normal --nodes=1 --cpus-per-task=8 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=80G -t 00-01:00:00 --pty bash -i
# scancel 781131
# squeue --partition=special_cs
# squeue -u zaiquyang2