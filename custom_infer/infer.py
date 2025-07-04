import argparse
import torch
from visualizer import get_local
get_local.activate()

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai

from PIL import Image
import numpy as np

from custom_infer.utils import load_video, load_video_base64, Q_to_prompt, torch_infer
from custom_infer.vis import show_image_relevance, show_video_relevance

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)
    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # ================================ Initialize the model
    
    model_name = get_model_name_from_path(args.model_path)
    # Set model configuration parameters if they exist
    if args.overwrite == True:      # True
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
        overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
        overwrite_config["mm_newline_position"] = args.mm_newline_position

        cfg_pretrained = AutoConfig.from_pretrained(args.model_path)
        
        # import pdb;pdb.set_trace()
        if "qwen" not in args.model_path.lower():
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            if scaling_factor >= 2:
                if "vicuna" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor
        
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
    else:
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)


    # import pdb;pdb.set_trace()
    if getattr(model.config, "force_sample", None) is not None:
        args.force_sample = model.config.force_sample
    else:
        args.force_sample = False

    if getattr(model.config, "add_time_instruction", None) is not None:
        args.add_time_instruction = model.config.add_time_instruction
    else:
        args.add_time_instruction = False

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_name = args.output_name
    answers_file = os.path.join(args.output_dir, f"{output_name}.json")
    ans_file = open(answers_file, "w")

    video_path = args.video_path

    all_video_pathes = []

    # Check if the video_path is a directory or a file
    if os.path.isdir(video_path):
        # If it's a directory, loop over all files in the directory
        for filename in os.listdir(video_path):
                    # Load the video file
            cur_video_path = os.path.join(video_path, f"{filename}")
            all_video_pathes.append(os.path.join(video_path, cur_video_path))
    else:
        # If it's a file, just process the video
        all_video_pathes.append(video_path) 

    for video_path in all_video_pathes:

        question = "Is there a woman pushed her way through the door in this image/video?"
        cap = 'please describe this video/image'
        # ============= Check if the video exists
        if os.path.exists(video_path):
            video_, frame_time, video_time, pil_images = load_video(video_path, args)
            video = image_processor.preprocess(video_, return_tensors="pt")["pixel_values"].half().cuda()
            video = [video]

        # ===================================== load prompt
        prompt_, input_ids_, keywords_, stopping_criteria_, attention_masks_, tokenizer_ = Q_to_prompt(cap, args, model, tokenizer)
        prompt, input_ids, keywords, stopping_criteria, attention_masks, tokenizer = Q_to_prompt(question, args, model, tokenizer)

        grid_size = 6
        with torch.inference_mode():
            output_ids_, question_to_img_att_ = torch_infer(cfg_pretrained, model, video, input_ids_, attention_masks_, stopping_criteria_, grid_size)
            output_ids, question_to_img_att = torch_infer(cfg_pretrained, model, video, input_ids, attention_masks, stopping_criteria, grid_size)
            
        
        question_to_img_att_ = question_to_img_att_.reshape(-1, video[0].size(0), grid_size*grid_size)        # Q_l x f_n x 12*12
        question_to_img_att = question_to_img_att.reshape(-1, video[0].size(0), grid_size*grid_size)
        
        rela_att = question_to_img_att[:] / (question_to_img_att_[-5] + 0.0001)   # Q_l x f_n x 12*12
        
        vid_name = video_path.split('/')[-1][:-4]
        os.makedirs(f'work_dirs/{vid_name}', exist_ok=True)


        vid_rela = show_video_relevance(
                pil_images[:], 
                preprocess, 
                att_maps=rela_att[-5, :], 
                norm_all_attention=True, 
                att_hw=(24,24),
                )
        vid_rela.save(f'work_dirs/{vid_name}/all_frame_rela_att_norm_vid_narrow.png')
        breakpoint()
        for v_idx in range(rela_att.shape[1]):

            fig_rela = show_image_relevance(
                rela_att[:, v_idx], 
                pil_images[v_idx], 
                preprocess, 
                norm_all_attention=False, 
                att_hw=(24,24), 
                bbox=None, 
                tokens=None, 
                prompt=""
                )

            fig_rela.save(f'work_dirs/{vid_name}/vid_{v_idx}_th_frame_q0_rela_att.png')

        outputs_ = tokenizer.batch_decode(output_ids_, skip_special_tokens=True)[0].strip()
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        print(f"Question: {prompt}\n")
        print(f"Response: {outputs_}\n")
        print(f"Response: {outputs}\n")
        # breakpoint()
        # tokenizer.tokenize(prompt)


    ans_file.close()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)