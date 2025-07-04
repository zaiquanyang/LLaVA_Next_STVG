import json
import os

import torch
from PIL import Image
import numpy as np
import random 
import torch.nn.functional as F

from custom_infer.STVG.InternVL_Tuning.conversation import get_conv_template
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval
from custom_infer.STVG.Sharegpt4video.spatial_tuning import spatial_llm_att, spatial_llm_logit

import sys
sys.path.append("/home/yzq/mnt/code/MLLM/LLaVA-NeXT/custom_infer/STVG/Sharegpt4video")
from ShareGPT4Video.llava.mm_utils import (get_model_name_from_path, process_images, tokenizer_image_token)
from ShareGPT4Video.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ShareGPT4Video.llava.conversation import conv_templates

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def create_frame_grid(img_array, interval_width=50):
    n, h, w, c = img_array.shape
    grid_size = int(np.ceil(np.sqrt(n)))

    horizontal_band = np.ones((h, interval_width, c),
                              dtype=img_array.dtype) * 255
    vertical_band = np.ones((interval_width, w + (grid_size - 1)
                            * (w + interval_width), c), dtype=img_array.dtype) * 255

    rows = []
    for i in range(grid_size):
        row_frames = []
        for j in range(grid_size):
            idx = i * grid_size + j
            if idx < n:
                frame = img_array[idx]
            else:
                frame = np.ones_like(img_array[0]) * 255
            # if j > 0:
            #     row_frames.append(horizontal_band)
            row_frames.append(frame)
        combined_row = np.concatenate(row_frames, axis=1)
        # if i > 0:
        #     rows.append(vertical_band)
        rows.append(combined_row)

    final_grid = np.concatenate(rows, axis=0)
    return final_grid


def resize_image_grid(image, max_length=1920):
    width, height = image.size
    if max(width, height) > max_length:
        if width > height:
            scale = max_length / width
        else:
            scale = max_length / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        img_resized = image.resize((new_width, new_height), Image.BILINEAR)
    else:
        img_resized = image
    return img_resized


def prepare_prompt(qs, conv_mode):

    pre_query_prompt = ""
    conv = conv_templates[conv_mode].copy()
    if pre_query_prompt is not None:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + pre_query_prompt + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    return prompt


def llava_embedd_tuning(cfg_pretrained, tokenizer, model, processor, video, questions, grid_size, targets, args, vid_i, logger):
    
    llm_reasoning = True
    if llm_reasoning:
        # =============================== prepare input_embedds for ShareGPT4Video
        img_grid = create_frame_grid(video, 50)
        img_grid = Image.fromarray(img_grid).convert("RGB")
        img_grid = resize_image_grid(img_grid)
        image_tensor = process_images([img_grid], processor, model.config)[0]
        general_prompt = f'Describe this video.'
        general_prompt = prepare_prompt(general_prompt, "llava_llama_3")
        
        input_ids = tokenizer_image_token(general_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(device=model.device, non_blocking=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
        
        inputs, position_ids, attention_mask, _, inputs_embeds_q, _ = model.prepare_inputs_labels_for_multimodal(
                        input_ids=input_ids, 
                        position_ids=None, 
                        past_key_values=None,
                        labels=None,
                        attention_mask=None,
                        images=image_tensor.to(dtype=torch.float16, device=model.device, non_blocking=True),
                        image_sizes=[img_grid.size]
                    )
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token is not None else tokenizer.eos_token_id
        
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][-10:], skip_special_tokens=True)
        print('Prompt tokens: ', tokens)

        # v_n  = 576
        img_token_idx = int(torch.where(input_ids == -200)[1])
        img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + 576, :]
        pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
        
        tokens_format = [img_token_idx, img_token_idx+576, 576]
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        spatial_prompt_history = spatial_prompt
        
        if args.ttt_epoch > 0:
            tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_tuning')
            logger.info(f'focus_attribute_tuning:  {tmp_questions}')
            # tmp_questions = ['please describe this video', f"Is there {questions.strip('.').strip()} in this video."]
            spatial_prompt, spatial_prompt_history, spatial_tuning_loss_list = spatial_llm_logit(
                                                    model, 
                                                    tmp_questions.copy(), 
                                                    pre_img_token_embedds, 
                                                    img_token_embedds, 
                                                    None,
                                                    spatial_prompt, spatial_prompt_history, 
                                                    6, 
                                                    video, 
                                                    tokens_format, 
                                                    args.ttt_epoch,
                                                    None, 
                                                    tokenizer, args,
                                                    args.s_lr
                                                    )
            
        tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_infer')
        logger.info(f'focus_attribute_infer:  {tmp_questions}')
        # tmp_questions = [f"Is there {questions.strip('.').strip()} in this video.", 'please describe this video']
        llava_att_pred_track_id, llava_rela_att, _ = spatial_llm_att(
                                                model,
                                                tmp_questions.copy(),
                                                pre_img_token_embedds,
                                                img_token_embedds,
                                                None,
                                                spatial_prompt,
                                                spatial_prompt_history,
                                                6,
                                                video,
                                                tokens_format, 
                                                targets,
                                                None, 
                                                tokenizer, args, cfg_pretrained
                                            )
        spatial_pred_track_id = llava_att_pred_track_id.item()
        
        contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=False)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')
        
        return bbox_pred_tmp, temporal_pred_tmp, None
    else:
        pass




def prepare_instructions(ref_sent, targets, mode='tuning_spatial_loc'):
    
    ref_Q = f"Is there {ref_sent.strip('.').strip()} in this video?"
    gpt_4_proc_attr = targets['gpt_4_proc_attr']            # 默认解耦出一个
    gpt_4_proc_action = targets['gpt_4_proc_action']        # 默认解耦出一个，当 考虑sub-actions的时候考虑多个
    # gpt_4_proc_sub_actions = targets['gpt_4_proc_sub_actions']
    
    attr_Q, action_Q = ["please describe this video"], ["please describe this video"]
    decompose_fail = False
    for sent in gpt_4_proc_attr:
        if "please describe this video" in sent:
            attr_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]#["please describe this video"]
            decompose_fail = True
        else:
            attr_Q = [f"Is there {sent.strip('.').strip()} in this video?"]

    for sent in gpt_4_proc_action:
        if "please describe this video" in sent:
            action_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]#["please describe this video"]
            decompose_fail = True
        else:
            action_Q = [f"Is there {sent.strip('.').strip()} in this video?"]
    # assert decompose_fail == False, print('please setting decompose_fail right !')
    
            
    tmp_questions = []
    
    if mode == "focus_attribute_tuning":
        if "please describe this video" in attr_Q:          # 当前这个解耦无效-fail
            tmp_questions = ["please describe this video", ref_Q]
        else:
            # tmp_questions.extend(action_Q)
            # tmp_questions.append(ref_Q)                                                   # ref_Q - action_
            tmp_questions = ["please describe this video"]                              # attr_Q - ""
            tmp_questions.extend(attr_Q)
        return tmp_questions
    elif mode == "focus_attribute_infer":
        if "please describe this video" in attr_Q:          # 当前这个解耦无效-fail
            tmp_questions = [ref_Q, "please describe this video"]
        else:
            # Method 1.     attr_Q - " "
            # tmp_questions = attr_Q
            # tmp_questions.append("please describe this video")
            # Method 2.   attr_Q - action_Q
            tmp_questions = [ref_Q]
            tmp_questions.append("please describe this video")
        return tmp_questions
    else:
        breakpoint()