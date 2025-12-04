#!/bin/bash
#SBATCH --account pa_cs_department
#SBATCH --partition=special_cs         # special_cs
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=8        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:a100:1             # Using 1 GPU card
#SBATCH --mem=100GB               # Request 50GB memory
#SBATCH --time=24:00:00        # Time limit day-hrs:min:sec
#SBATCH --job-name=VP_Tuning_HCSTVG2_debug
#SBATCH --output=hpc_logs/p1_%x.log  # Standard output
#SBATCH --error=hpc_logs/p1_%x.err   # Standard error log
#SBATCH --nodelist=hpc-gpu015
module load cuda/12.1.0
module load gcc

# ------attention the pytorch version------
source /home/zaiquyang2/scratch/anaconda3/etc/profile.d/conda.sh
conda init
conda activate llava

# cd  project dir #SBATCH --exclude=hpc-gpu014,hpc-gpu016

cd /home/zaiquyang2/scratch/mllm/LLaVA-NeXT

echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Process ID: $$"

# ============================================= LLAVA_Next_Video ======================================
idx=-5
frames_thre=10
cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
split='val'
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_next_video.py \
    --cfg_file ${cfg} \
    --dataset_name 'hcstvg2' \
    --split ${split} \
    --model-path "model_zoo/LLaVA-NeXT-Video-7B-DPO" \
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
    --ttt_epoch 0 \
    --temporal_thre 0.5 \
    --frames_thre ${frames_thre} \
    --llm_infer_mode 'focus_attr' \
    --img_size 336 \
    --token_idx ${idx} \
    --exp_comment "_token_${idx}_mid_20_layers" \
    INPUT.SAMPLE_FRAMES 20 \
    INPUT.RESOLUTION 336
# bash custom_infer/STVG/run_hc.sh 

# ===================================== LLAVA_OneVision ===================================== 
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
#     --exp_comment "_token_oracle_track" \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 336

# https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2/discussions/1
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_llava_video_qwen2_7b.py \
#     --model-path "model_zoo/LLaVA-Video-7B-Qwen2" \
#     --load_8bit False \
#     --model_name 'llava_video_7B_Qwen2' \
#     --overwrite True \
#     --mm_spatial_pool_stride 4 \
#     --conv-mode qwen_1_5 \
#     --mm_spatial_pool_mode "bilinear" \
#     --mm_newline_position "one_token" \
#     --layers_N 28 \
#     --heads_N 28 \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 8.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 2 \
#     --temporal_thre 0.5 \
#     --frames_thre 10 \
#     --llm_infer_mode 'focus_attr' \
#     --token_idx -3 \
#     --exp_comment '(attrQ_cap)' \
#     INPUT.SAMPLE_FRAMES 20

# ===================================== Qwen2_VL ===================================== 
# idx=-3
# frames_thre=10
# cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
# split='val'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_qwen2_vl.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg2' \
#     --split ${split} \
#     --model_name 'Qwen2_VL_7B' \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'base' \
#     --img_size 224 \
#     --token_idx ${idx} \
#     --exp_comment "_token_${idx}" \
#     INPUT.SAMPLE_FRAMES 20 \
#     INPUT.RESOLUTION 224

# ===================================== Intern_VL2-5 ===================================== 
# idx=-3
# frames_thre=5
# cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
# split='val'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_internvl2_5.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg2' \
#     --split ${split} \
#     --model_name 'InternVL3-2B' \
#     --yes_id 9693 \
#     --no_id 2152 \
#     --s_lr 0.0 \
#     --t_lr 0.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'base' \
#     --img_size 448 \
#     --token_idx ${idx} \
#     --exp_comment "_token_mean_last_14_layers" \
#     INPUT.SAMPLE_FRAMES 10 \
#     INPUT.RESOLUTION 448

# ===================================== sharegpt4video ===================================== 
# idx=-4
# frames_thre=8
# cfg='custom_infer/STVG/experiments/hcstvg2.yaml'
# split='val'
# PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 python3 custom_infer/STVG/hc_sharegpt4video.py \
#     --cfg_file ${cfg} \
#     --dataset_name 'hcstvg2' \
#     --split ${split} \
#     --model_name 'ShareGPT4Video' \
#     --mm_spatial_pool_mode "average" \
#     --mm_newline_position "no_token" \
#     --yes_id 9891 \
#     --no_id 2201 \
#     --s_lr 8.0 \
#     --t_lr 8.0 \
#     --ttt_epoch 0 \
#     --temporal_thre 0.5 \
#     --frames_thre ${frames_thre} \
#     --llm_infer_mode 'focus_whole' \
#     --img_size 336 \
#     --token_idx ${idx} \
#     --exp_comment "_token_${idx}" \
#     INPUT.SAMPLE_FRAMES 16 \
#     INPUT.RESOLUTION 336

# bash custom_infer/STVG/run_hc_stvgv2.sh model_zoo/LLaVA-NeXT-Video-7B-DPO vicuna_v1 30 4 average no_token True "hhh.mp4"

# srun --partition=gpu_a100 --qos=normal --nodes=1 --cpus-per-task=8 --nodelist=gpu-a100-05 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=80G  -t 00-01:00:00 --pty bash -i
# srun --partition=stingy --qos=normal --nodes=1 --cpus-per-task=8 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=80G  -t 00-01:00:00 --pty bash -i
# srun --partition=stingy --qos=normal --nodes=1 --cpus-per-task=4 --ntasks-per-node=1 --gres=gpu:v100s:1 --mem=32G  -t 00-01:00:00 --pty bash -i
# srun --partition=stingy --qos=normal --nodes=1 --cpus-per-task=8 --ntasks-per-node=1 --gres=gpu:a100:1 --mem=24G -t 00-01:00:00 --nodelist=hpc-gpu015 --pty bash -i
# scancel 781131
# squeue --partition=special_cs
# squeue -u zaiquyang2
# tensorboard --logdir="" --port=6006
# squeue --nodelist=hpc-gpu015
# scontrol show node=hpc-gpu015