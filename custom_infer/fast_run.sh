bash scripts/video/demo/video_demo.sh model_zoo/LLaVA-NeXT-Video-7B-DPO vicuna_v1 30 4 average no_token True playground/demo/17_KWoSGtglCms.mp4

# Example model: lmms-lab/LLaVA-NeXT-Video-7B-DPO

# Prompt mode: vicuna_v1 (use mistral_direct for lmms-lab/LLaVA-NeXT-Video-34B-DPO)

# Sampled frames: 32 (Defines how many frames to sample from the video.)

# Spatial pooling stride: 2 (With original tokens for one frame at 24x24, if stride=2, then the tokens for one frame are 12x12.)

# Spatial pooling mode: average (Options: average, max.)

# Local video path: ./data/llava_video/video-chatgpt/evaluation/Test_Videos/v_Lf_7RurLgp0.mp4