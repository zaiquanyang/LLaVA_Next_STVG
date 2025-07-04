#!/bin/bash
ROOT_DIR="/home/zaiquyang2/scratch/mllm/LLaVA-NeXT"

# ================================================ LLaVA_NeXT_Video_7B
# idx=-5
# frames_thre=7
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_next_video.py \
#     --cfg_file 'custom_infer/STVG/experiments/hcstvg1.yaml' \
#     --dataset_name 'hcstvg1' \
#     --split 'test' \
#     --model-path "model_zoo/LLaVA-NeXT-Video-7B-DPO" \
#     --load_8bit True \
#     --model_name 'LLaVA_NeXT_Video_7B' \
#     --overwrite True \
#     --mm_spatial_pool_stride 4 \
#     --conv-mode vicuna_v1 \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --yes_id 3582 \
#     --no_id 1217 \
#     --s_lr 8.0 \
#     --t_lr 8.0 \
#     --ttt_epoch 2 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'focus_attr' \
#     --img_size 336 \
#     --token_idx ${idx} \
#     --exp_comment "_debug" \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 336

# bash custom_infer/STVG/run_hcstvg1.sh
# ================================================ LLaVA_OV_7B
# idx=-3
# frames_thre=7
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_one_vision.py \
#     --cfg_file 'custom_infer/STVG/experiments/hcstvg1.yaml' \
#     --dataset_name 'hcstvg1' \
#     --split 'test' \
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
#     --ttt_epoch 1 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'base' \
#     --img_size 336 \
#     --token_idx ${idx} \
#     --exp_comment "_token_${idx}" \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 336

# bash custom_infer/STVG/run_hcstvg1.sh
# ===================================== Qwen2_VL ===================================== 
# idx=-3
# frames_thre=7
# cfg='custom_infer/STVG/experiments/hcstvg1.yaml'
# split='test'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_qwen2_vl.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg1' \
#     --split ${split} \
#     --model_name 'Qwen2_VL_7B' \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 2 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'base' \
#     --img_size 224 \
#     --token_idx ${idx} \
#     --exp_comment _token_${idx}_debug \
#     INPUT.SAMPLE_FRAMES 10 \
#     INPUT.RESOLUTION 224

# bash custom_infer/STVG/run_hcstvg1.sh
# idx=-3
# frames_thre=4
# cfg='custom_infer/STVG/experiments/hcstvg1.yaml'
# split='test'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_internvl2_5.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg1' \
#     --split ${split} \
#     --model_name 'InetrnVL-2-5-4B' \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 2 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'focus_attr' \
#     --img_size 448 \
#     --token_idx ${idx} \
#     --exp_comment "_token_${idx}_debug" \
#     INPUT.SAMPLE_FRAMES 10 \
#     INPUT.RESOLUTION 448

# bash custom_infer/STVG/run_hcstvg1.sh
# ===================================== ShareGPT4Video ===================================== 
idx=-1
frames_thre=6
cfg='custom_infer/STVG/experiments/hcstvg1.yaml'
split='test'
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_sharegpt4video.py \
    --cfg_file ${cfg} \
    --dataset_name 'hcstvg1' \
    --split ${split} \
    --model_name 'ShareGPT4Video' \
    --mm_spatial_pool_mode "average" \
    --mm_newline_position "no_token" \
    --yes_id 9891 \
    --no_id 2201 \
    --s_lr 0.0 \
    --t_lr 0.0 \
    --ttt_epoch 1 \
    --temporal_thre 0.5 \
    --frames_thre ${frames_thre} \
    --llm_infer_mode 'base' \
    --img_size 336 \
    --token_idx ${idx} \
    --exp_comment "_token_${idx}" \
    INPUT.SAMPLE_FRAMES 16 \
    INPUT.RESOLUTION 336