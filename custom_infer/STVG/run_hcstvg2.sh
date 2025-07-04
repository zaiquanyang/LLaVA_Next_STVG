#!/bin/bash
# ROOT_DIR="/home/yzq/mnt/code/MLLM/LLaVA-NeXT"

# ========================================LLaVA-NeXT-Video-7B-DPO================================== 
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_next_video_7b.py \
#     --cfg_file 'custom_infer/STVG/experiments/hcstvg2.yaml' \
#     --split 'val' \
#     --model-path "model_zoo/LLaVA-NeXT-Video-7B-DPO" \
#     --load_8bit True \
#     --load_4bit False \
#     --model_name 'LLaVA_NeXT_Video_7B' \
#     --overwrite True \
#     --mm_spatial_pool_stride 4 \
#     --conv-mode vicuna_v1 \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --img_size 336 \
#     --yes_id 3582 \
#     --no_id 1217 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre 10 \
#     --llm_infer_mode 'base' \
#     --token_idx -5 \
#     --exp_comment '_debug' \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 336
# bash custom_infer/STVG/run_hcstvg2.sh 

idx=-3
frames_thre=10
cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
split='val'
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_one_vision.py \
    --cfg_file ${cfg} \
    --dataset_name 'hcstvg2' \
    --split ${split} \
    --model-path "model_zoo/llava-onevision-qwen2-7b-ov" \
    --load_8bit False \
    --model_name 'llava_onevision_qwen2_7b_ov' \
    --overwrite True \
    --mm_spatial_pool_stride 5 \
    --conv-mode qwen_1_5 \
    --mm_spatial_pool_mode "bilinear" \
    --mm_newline_position "one_token" \
    --yes_id 9693 \
    --no_id 2152 \
    --s_lr 8.0 \
    --t_lr 8.0 \
    --ttt_epoch 0 \
    --temporal_thre 0.5 \
    --frames_thre ${frames_thre} \
    --llm_infer_mode 'base' \
    --img_size 336 \
    --token_idx ${idx} \
    --exp_comment "_token_${idx}" \
    INPUT.SAMPLE_FRAMES 20 \
    INPUT.RESOLUTION 336



# https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2/discussions/1
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_video_qwen2_7b.py \
#     --model-path "model_zoo/LLaVA-Video-7B-Qwen2" \
#     --load_8bit False \
#     --model_name 'llava_video_7B_Qwen2' \
#     --overwrite True \
#     --mm_spatial_pool_stride 4 \
#     --conv-mode qwen_1_5 \
#     --mm_spatial_pool_mode "bilinear" \
#     --mm_newline_position "no_token" \
#     --layers_N 28 \
#     --heads_N 28 \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 1 \
#     --temporal_thre 0.5 \
#     --frames_thre 10 \
#     --llm_infer_mode 'focus_attr_w_reverse' \
#     --token_idx -3 \
#     --exp_comment 'ret_reverse_loss' \
#     INPUT.SAMPLE_FRAMES 20

# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_qwen2_vl.py \
#     --cfg_file 'custom_infer/STVG/experiments/hcstvg2.yaml' \
#     --split 'val' \
#     --model_name 'Qwen2_VL_7B' \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --img_size 224 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre 10 \
#     --llm_infer_mode 'base' \
#     --token_idx -3 \
#     --exp_comment _debug \
#     INPUT.SAMPLE_FRAMES 10 \
#     INPUT.RESOLUTION 224

# bash custom_infer/STVG/run_hcstvg2.sh model_zoo/LLaVA-NeXT-Video-7B-DPO vicuna_v1 30 4 average no_token True "hhh.mp4"

# srun --partition=stingy --qos=normal --nodes=1 --cpus-per-task=8 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=80G -t 00-01:00:00 --pty bash -i
# scancel 781131
# squeue --partition=special_cs
# squeue -u zaiquyang2