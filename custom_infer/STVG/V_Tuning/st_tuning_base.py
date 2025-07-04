import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import cv2
import base64
import openai
import torch
from PIL import Image
import numpy as np
import random 
import torch.nn.functional as F
from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from visualizer import get_local
get_local.activate()

from custom_infer.STVG.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array, print_gpu
from custom_infer.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval

from custom_infer.STVG.V_Tuning.spatial_tuning import spatial_llm_logit, spatial_llm_att
# from custom_infer.STVG.T_Tuning.temporal_tuning import temporal_llm_logit, temporal_llm_att

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

import copy

def llava_embedd_tuning(cfg_pretrained, tokenizer, model, image_processor, video, questions, grid_size, targets, args, vid_i, logger):

    """
    基本思路：
    Epoch 1 :
    Step 1: 用 blur_track 来获得 spatial_localization
    Step 2 (optional): 基于 spatial_localization 去 blur track_proposal 无关的frames
    Step 3: 通过 model.generate()函数获得 attention 在各帧的分布
    ==========================
    More epoches:
    Step 4: 根据 temporal_localization 重新获得 spatial_localization
    ....
    """
    contra_cap = "please describe this video"   # 这里 caption 的内容会影响结果
    
    
    # =========================== step 1.2 Spatial tuning ======= > spatial localization
    llm_reasoning = True
    if llm_reasoning:
        # ===================================== step 1.1. 预先准备全部 blur_track 的 video_embedding
        with torch.no_grad():
            # =========================== w/o blur track
            if model.dtype == torch.bfloat16:
                video_proc = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(torch.bfloat16).cuda()
            else:
                video_proc = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
            video_batch_wo_blur = [video_proc]
            
            prompt_, input_ids_, keywords_, stopping_criteria_, attention_masks_, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids_, 
                position_ids=None, 
                past_key_values=None,
                labels=None,
                attention_mask=attention_masks_,
                images=video_batch_wo_blur, 
                modalities="video",
                )
            
            F_n = video_batch_wo_blur[0].size(0)
            if cfg_pretrained.mm_newline_position == "one_token":
                v_n = grid_size*grid_size * F_n + 1
                # v_n = v_n-1
            elif cfg_pretrained.mm_newline_position == "no_token":
                v_n = grid_size*grid_size * F_n + 0
            else:
                breakpoint()
            img_token_idx = int(torch.where(input_ids_ == -200)[1])
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
            
        
        tokens_format = [img_token_idx, img_token_idx+v_n, v_n]
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        spatial_prompt_history = spatial_prompt
    
        tmp_questions = [f"Is there {questions.strip('.').strip()} in this video?", "please describe this video"]         
        print(tmp_questions)
        llava_att_pred_track_id, llava_rela_att = spatial_llm_att(
                                                model,
                                                tmp_questions.copy(),
                                                pre_img_token_embedds,
                                                img_token_embedds,
                                                None,
                                                spatial_prompt,
                                                spatial_prompt_history,
                                                grid_size,
                                                video_batch_wo_blur,
                                                tokens_format, 
                                                targets,
                                                stopping_criteria_, tokenizer, args, cfg_pretrained
                                            )
        spatial_pred_track_id = llava_att_pred_track_id.item()

        contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=True)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')
        
        # 保存 llava_rela_att 到指定文件夹
        npy_dict = {}
        npy_dict = {'S_pred': spatial_pred_track_id, 'T_LLaVa_att': llava_rela_att.cpu().to(torch.float32).numpy(), 'vid_i':vid_i, 'gt_viou':gt_viou}
        name_f = targets['vid_name']
        np.save(f'{args.output_dir}/_llava_att_npy/{name_f}.npy', npy_dict)
        print('Save LLaVA-Att done!')
    else:
        
        # with open("custom_infer/STVG/output/focus_attr_pred_track_id.json", "r", encoding="utf-8") as f:
        #     oracle_track_ids = json.load(f)
        
        # track_id = oracle_track_ids[str(vid_i)]
        
        name_f = targets['vid_name']
        npy_f = f'{args.output_dir[:-7]}/_llava_att_npy/{name_f}.npy'
        llava_dict = np.load(npy_f, allow_pickle=True).item()
        spatial_pred_track_id = llava_dict['S_pred']
        llava_rela_att = torch.from_numpy(llava_dict['T_LLaVa_att']).cuda()
        
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        contrastive_content = f"_{questions[0:120]}_"
        bbox_pred_tmp, temporal_pred_tmp, _ = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=False)
        
        print('Evaluate and Visualize the llava_att on spatial prediction done!')
    # breakpoint()
    return bbox_pred_tmp, temporal_pred_tmp
