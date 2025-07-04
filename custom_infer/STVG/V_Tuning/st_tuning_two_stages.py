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

from custom_infer.STVG.T_Tuning.spatial_tuning import spatial_llm_logit, spatial_llm_att
from custom_infer.STVG.T_Tuning.temporal_tuning import temporal_llm_logit, temporal_llm_att
from custom_infer.STVG.T_Tuning.utils import prepare_instructions

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
    # contra_cap = "Is there the woman in black clothes in this video?"
    
    #questions
    
    # ===================================== step 1.1. 预先准备全部 blur_track 的 video_embedding
    inputs_blur_track_embeds = []
    with torch.no_grad():
        # =========================== w/o blur track
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
        v_n = grid_size*grid_size * F_n
        img_token_idx = int(torch.where(input_ids_ == -200)[1])
        img_token_embedds_wo_blur = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
        pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
            
        # ============================ blur track
        for track_k in range(len(targets['proposal_bboxes'])):
            video_blur_proposal, blur_images = blur_proposal_video_array(video, targets, blur_propsal_id=track_k)
            video_proc = image_processor.preprocess(video_blur_proposal, return_tensors="pt")["pixel_values"].half().cuda()
            video_blur_batch = [video_proc]
            
            prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids_C, 
                position_ids=None, 
                past_key_values=None,
                labels=None,
                attention_mask=attention_masks_C,
                images=video_blur_batch, 
                modalities="video",
                )
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            inputs_blur_track_embeds.append(img_token_embedds)
    
    tokens_format = [img_token_idx, img_token_idx+v_n, v_n]
    
    # =========================== step 1.2 Spatial tuning ======= > spatial localization
    for spatial_t_epoch in [args.ttt_epoch,]:
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        spatial_prompt_history = spatial_prompt

        spatial_question = copy.deepcopy(questions[:])
        # ============================ stage 1
        tmp_questions = prepare_instructions(questions, targets, mode='tuning_spatial_loc')
        # tmp_questions = [spatial_question[1], spatial_question[2]] # ['Is there The man goes to the car in this video?', 'Is there the man standing by the stairs goes to the car in this video?']
        logger.info(f'tuning_spatial_loc:  {tmp_questions}')
        spatial_prompt_, spatial_prompt_history_ = spatial_llm_logit(
                                                model, 
                                                tmp_questions.copy(), 
                                                pre_img_token_embedds, 
                                                img_token_embedds_wo_blur, 
                                                inputs_blur_track_embeds,
                                                spatial_prompt, spatial_prompt_history, 
                                                grid_size, 
                                                video_batch_wo_blur, 
                                                tokens_format, 
                                                spatial_t_epoch,
                                                stopping_criteria_, tokenizer, args,
                                                args.s_lr
                                                )
        tmp_questions = prepare_instructions(questions, targets, mode='infer_spatial_loc')
        # tmp_questions = [spatial_question[0], "please describe this video"]
        # tmp_questions = [spatial_question[0], spatial_question[1]] # ['Is there the man standing by the stairs in this video?', 'Is there The man goes to the car in this video?']
        logger.info(f'infer_spatial_loc:  {tmp_questions}')
        llava_att_pred_track_id, _ = spatial_llm_att(
                                                model,
                                                tmp_questions.copy(),
                                                pre_img_token_embedds,
                                                img_token_embedds_wo_blur,
                                                inputs_blur_track_embeds,
                                                spatial_prompt_,
                                                spatial_prompt_history_,
                                                grid_size,
                                                video_batch_wo_blur,
                                                tokens_format, 
                                                targets,
                                                stopping_criteria_, tokenizer, args
                                            )
        spatial_pred_track_id = llava_att_pred_track_id.item()
        # breakpoint()
        
        # ============================ stage 2
        temporal_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        temporal_prompt_history = temporal_prompt
        tmp_questions = prepare_instructions(questions, targets, mode='tuning_temporal_loc')
        logger.info(f'tuning_temporal_loc:  {tmp_questions}')
        temporal_prompt_, temporal_prompt_history_ = spatial_llm_logit(
                                                model, 
                                                tmp_questions.copy(), 
                                                pre_img_token_embedds, 
                                                img_token_embedds_wo_blur, 
                                                inputs_blur_track_embeds,
                                                temporal_prompt, temporal_prompt_history, 
                                                grid_size, 
                                                video_batch_wo_blur, 
                                                tokens_format, 
                                                spatial_t_epoch,
                                                stopping_criteria_, tokenizer, args,
                                                args.t_lr,
                                                )
        tmp_questions = prepare_instructions(questions, targets, mode='infer_temporal_loc')
        logger.info(f'infer_temporal_loc:  {tmp_questions}')
        # tmp_questions = [spatial_question[1], "please describe this video"]      
        # tmp_questions = [spatial_question[1], spatial_question[0]]
        _, llava_rela_att = spatial_llm_att(
                                            model,
                                            tmp_questions.copy(),
                                            pre_img_token_embedds,
                                            img_token_embedds_wo_blur,
                                            inputs_blur_track_embeds,
                                            temporal_prompt_,
                                            temporal_prompt_history_,
                                            grid_size,
                                            video_batch_wo_blur,
                                            tokens_format, 
                                            targets,
                                            stopping_criteria_, tokenizer, args
                                            )

        contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args)
        logger.info('Evaluate and Visualize the llava_att on spatial prediction done!')
    # breakpoint()
    return bbox_pred_tmp, temporal_pred_tmp

    # =========================== step 2.1 blur video_frames ======= > temporal localization
    # spatial_pre_action = targets['proposal_bboxes'][spatial_pred_track_id].sum(axis=-1) > 0.0
    # spatial_pre_action_inerval = [np.where(spatial_pre_action==1)[0][0], np.where(spatial_pre_action==1)[0][-1] + 1]

    # track_mask = torch.Tensor(targets['proposal_mask'][spatial_pred_track_id])
    # prompt_mask = F.interpolate(track_mask.unsqueeze(0), size=(grid_size, grid_size), mode='bilinear', align_corners=False).squeeze(0) > 0.1
    # prompt_mask = prompt_mask.reshape(1, -1, 1).to(img_token_embedds.device).squeeze().squeeze()
    prompt_mask  = None
    with torch.no_grad():
        # video_blur_temporal_array, video_blur_temporal_images = blur_temp_video_array(video, targets, blur_interval=spatial_pre_action_inerval)
        # targets['pil_images'] = video_blur_temporal_images
        # video_blur_proposal, blur_images = blur_proposal_video_array(video, targets, blur_propsal_id=spatial_pred_track_id)
        # targets['pil_images'] = blur_images

        # 全部 blur ， 注意 blur_interval 设置
        video_blur, video_blur_temporal_images = blur_temp_video_array(video, targets, blur_interval=[-1, -1])
        video_proc = image_processor.preprocess(video_blur, return_tensors="pt")["pixel_values"].half().cuda()
        prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
        (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids_C,
            position_ids=None, 
            past_key_values=None,
            labels=None,
            attention_mask=attention_masks_C, 
            images=[video_proc], 
            modalities="video",
            )
        inputs_blur_embedds=inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
        
    # =========================== step 2.2 Temporal tuning
    print('========================= Temporal Tuning Start !')
    # breakpoint()
    ttt_epochs = [2,] #[0, 2, 4, 6]
    for epoch in ttt_epochs:
        temporal_questions = copy.deepcopy(questions[:])                # 
        temporal_questions.append("please describe this video")
        print(temporal_questions)
    
        temporal_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        temporal_prompt_history = temporal_prompt
        tokens_format = [img_token_idx, img_token_idx+v_n, v_n]
        temporal_prompt, temporal_prompt_history = temporal_llm_logit(
                                                model, 
                                                temporal_questions, 
                                                pre_img_token_embedds,
                                                img_token_embedds_wo_blur,
                                                inputs_blur_embedds, 
                                                temporal_prompt, 
                                                temporal_prompt_history, 
                                                grid_size,
                                                [video_proc],
                                                tokens_format,
                                                targets,
                                                stopping_criteria_, tokenizer, epoch, prompt_mask, args
                                            )
        llava_att_pred_track_id, llava_rela_att = temporal_llm_att(
                                                model, 
                                                temporal_questions, 
                                                pre_img_token_embedds,
                                                img_token_embedds_wo_blur,
                                                inputs_blur_embedds, 
                                                temporal_prompt, 
                                                temporal_prompt_history, 
                                                grid_size,
                                                [video_proc],
                                                tokens_format,
                                                targets,
                                                stopping_criteria_, tokenizer, epoch, prompt_mask, args
                                            )
        temporal_pred_track_id = spatial_pred_track_id#.item()
        q_to_img_rela_att = llava_rela_att

        # ============== 当前 ttt_epoch : 可视化当前 ttt_epoch 的结果
        contrastive_content  = f'track_{temporal_pred_track_id}_{temporal_questions[-2][10:120]}_'
        T_pred = {'Space_pred': temporal_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, q_to_img_rela_att, vid_i, preprocess, contrastive_content, T_pred, args)
    print('========================= Temporal Tuning end !')
    # for head_n in range(0, 32, 4):
    #     contrastive_content  = f'Head_{head_n}_track_{spatial_pred_track_id}_{temporal_question_list[0][10:120]}_'
    #     blur_pred = {'Space_pred': spatial_pred_track_id}
    #     bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, q_to_img_rela_att[0][head_n, :].mean(dim=0), vid_i, preprocess, contrastive_content, blur_pred, args)
    # for layer_n in range(0, 32, 4):
    #     contrastive_content  = f'Layer_{layer_n}_track_{spatial_pred_track_id}_{temporal_question_list[0][10:120]}_'
    #     blur_pred = {'Space_pred': spatial_pred_track_id}
    #     bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, q_to_img_rela_att[0][:, layer_n].mean(dim=0), vid_i, preprocess, contrastive_content, blur_pred, args)
    # breakpoint()
    return bbox_pred_tmp, temporal_pred_tmp

