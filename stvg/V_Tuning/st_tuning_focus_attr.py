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

# from visualizer import get_local
# get_local.activate()

from stvg.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
# from stvg.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336
# from stvg.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
# from stvg.utils import Q_to_prompt, get_Q_att, ttt_eval

from stvg.V_Tuning.spatial_tuning import spatial_llm_logit, spatial_llm_att
# from custom_infer.STVG.T_Tuning.temporal_tuning import temporal_llm_logit, temporal_llm_att

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

import copy

def llava_embedd_tuning(cfg_pretrained, tokenizer, model, image_processor, video, questions, grid_size, targets, args, vid_i, logger):

    """
    TODO
    """
    contra_cap = "please describe this video" 
        
    # ===================================== step 1.1. 预先准备全部 track 的 video_embedding
    with torch.no_grad():
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
            # v_n = v_n - 1 
        elif cfg_pretrained.mm_newline_position == "no_token":
            v_n = grid_size*grid_size * F_n + 0
        else:
            breakpoint()
        
        img_token_idx = int(torch.where(input_ids_ == -200)[1])
        img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
        pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
        
        # ============================ reverse track
        # video_proc_ = torch.flip(video_proc, dims=[0])
        # video_batch_ = [video_proc_]
        # (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
        #     input_ids=input_ids_, 
        #     position_ids=None, 
        #     past_key_values=None,
        #     labels=None,
        #     attention_mask=attention_masks_,
        #     images=video_batch_, 
        #     modalities="video",
        #     )
        # img_token_embedds_w_reverse = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
        # # llava_ov
        # img_token_embedds_w_reverse_ = torch.flip(img_token_embedds[0][:-1].reshape(F_n, grid_size*grid_size, -1), dims=[0])
        # img_token_embedds_w_reverse = torch.cat([img_token_embedds_w_reverse_.reshape(1, F_n*grid_size*grid_size, -1), img_token_embedds[:, -1:]], dim=1)
        # llava_next_video
        # img_token_embedds_w_reverse = torch.flip(img_token_embedds.reshape(F_n, grid_size*grid_size, -1), dims=[0])
        # img_token_embedds_w_reverse = img_token_embedds_w_reverse.reshape(1, F_n*grid_size*grid_size, -1)
        # img_token_embedds_w_reverse = None
        
    # =========================== step 1.2 Spatial tuning ======= > spatial localization
    tokens_format = [img_token_idx, img_token_idx+v_n, v_n]                             
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds)) # vp_tuning
    spatial_prompt_history = spatial_prompt
    
    
    if args.ttt_epoch > 0:
        tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_tuning')
        logger.info(f'focus_attribute_tuning:  {tmp_questions}')
        spatial_prompt, spatial_prompt_history, spatial_tuning_loss_list = spatial_llm_logit(
                                                model, 
                                                tmp_questions.copy(), 
                                                pre_img_token_embedds, 
                                                img_token_embedds,
                                                grid_size, 
                                                video_batch_wo_blur, 
                                                tokens_format, 
                                                args.ttt_epoch,
                                                stopping_criteria_, tokenizer, args
                                                )
    # tmp_questions = [spatial_question[0], "please describe this video"]         
    # tmp_questions = [spatial_question[0], spatial_question[1]]  
    tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_infer')
    logger.info(f'focus_attribute_infer:  {tmp_questions}')
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
    
    contrastive_content  = f'track_{spatial_pred_track_id}_{tmp_questions[0][0:120]}_'
    spatial_pred = {'Space_pred': spatial_pred_track_id}
    bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=True)
    print('Evaluate and Visualize the llava_att on spatial prediction done!')
    
    # 保存 llava_rela_att 到指定文件夹
    # npy_dict = {}
    # npy_dict = {
    #     'S_pred': spatial_pred_track_id, 
    #     'T_LLaVa_att': llava_rela_att.cpu().to(torch.float32).numpy(), 
    #     'vid_i': vid_i, 
    #     'GT_viou': gt_viou,
    #     # 'spatial_tuning_loss':spatial_tuning_loss_list
    #     # 'attr_for_track_score_pre': max_att_each_proposal_pre_tuning.cpu().numpy(),
    #     # 'attr_pred_id_pre':  llava_att_pred_track_id_pre_tuning.item(),
    #     # 'attr_for_track_score_post':max_att_each_proposal.cpu().numpy(),
    #     # 'attr_pred_id_post':  spatial_pred_track_id,
    #     # 'oracle_id': track_id
    #     }
    
    # name_f = targets['vid_name']
    # np.save(f'{args.output_dir}/_llava_att_npy/{name_f}.npy', npy_dict)
    

        
    return bbox_pred_tmp, temporal_pred_tmp




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
            tmp_questions = attr_Q
            tmp_questions.append("please describe this video")
            # Method 2.   attr_Q - action_Q
            # tmp_questions = [ref_Q]
            # tmp_questions.append("please describe this video")
        return tmp_questions
    else:
        breakpoint()