import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu
import torch.nn.functional as F
from transformers import AutoConfig

import cv2
import base64
import openai
import torch
from PIL import Image
import numpy as np
import random 

from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

# from visualizer import get_local
# get_local.activate()

from stvg.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
# from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, \
#     blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array, print_gpu
# from stvg.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
from stvg.util import Q_to_prompt, get_Q_att, ttt_eval



import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)


"""
v1: 2025-02-28
将每个 whole_Q 拆分成 attr_Q 和 action_Q 两个部分，没有拆对的话，则都用 "please describe this video" 代替

max( LLM_[whole_Q] - LLM_[attr_Q] + LLM_[whole_Q] - LLM_[action_Q])

"""

def spatial_llm_logit(
    model,
    spatial_questions,
    pre_img_token_embedds, 
    img_token_embedds,
    grid_size,
    tokens_format,
    epochs,
    stopping_criteria_Q,
    tokenizer,
    args,
    st_lr,
    ret_logit=False,
    **kwargs
    ):
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(spatial_prompt))
    spatial_prompt_history = spatial_prompt
    spatial_tuning_loss_list = []
    
    img_token_start, img_token_end, tokens_N = tokens_format
    for ep in range(epochs):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        track_score = []
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
            
            inputs_blur_embed_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_blur_embed_q.shape[1])).to(inputs_blur_embed_q.device)
            
            new_inputs_blur_embed_q = inputs_blur_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_blur_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            
            # with torch.no_grad():
            outputs = model.generate(
                inputs=None, 
                images=None, 
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_blur_embed_q,
                position_ids=position_ids_q,
                modalities="video", 
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=10, 
                top_p=0.1,
                num_beams=1,
                use_cache=True, 
                stopping_criteria=[stopping_criteria_Q],
                output_scores=True,
                return_dict_in_generate=True,
            )
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # scores = outputs.scores[0].detach().cpu().numpy()
            yes_score, no_score = outputs.scores[0][0][args.yes_id], outputs.scores[0][0][args.no_id]

            get_local.clear()
            torch.cuda.empty_cache()
            # print_gpu()
            track_score.append(yes_score-no_score)
            # track_score.append(outputs.scores[0][0][args.yes_id]/(outputs.scores[0][0].sum()+0.0001) * 10000)
            
        track_score = torch.stack(track_score).squeeze()   # Q_n
        
        
        spatial_tuning_loss = 0.0
        for q_id in range(len(spatial_questions[1:])):
            spatial_tuning_loss += (1.0 - (track_score[q_id+1] - track_score[0]))
        print(spatial_tuning_loss, ' _st_lr=', args.s_lr)
        spatial_tuning_loss_list.append(spatial_tuning_loss.detach().cpu().numpy())
        spatial_prompt_history = spatial_prompt
        grad_cond = torch.autograd.grad(spatial_tuning_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        spatial_prompt = spatial_prompt - args.s_lr * grad_cond
        
        get_local.clear()
        torch.cuda.empty_cache()
    if ret_logit:
        return spatial_prompt, spatial_prompt_history, track_score
    else:
        return spatial_prompt, spatial_prompt_history, spatial_tuning_loss_list



def spatial_action_att_tuning(
    model,
    spatial_questions,
    pre_img_token_embedds, img_token_embedds, gt_track_mask,
    spatial_prompt,
    spatial_prompt_history,
    grid_size,
    video_batch_wo_blur,
    tokens_format,
    epochs,
    stopping_criteria_Q,
    tokenizer,
    args,
    st_lr,
    cfg_pretrained
    ):
    """
    blur gt_track 之外的其他帧， 最大化其他帧的 logit 与 gt_track 所在 frames 的差
    """
    img_token_start, img_token_end, tokens_N = tokens_format
    
    for ep in range(epochs):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            # breakpoint()
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])

            inputs_embed_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embed_q.shape[1])).to(inputs_embed_q.device)
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            # with torch.no_grad():
            outputs = model.generate(
                inputs=None, 
                images=None, 
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_embed_q,
                position_ids=position_ids_q,
                modalities="video", 
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=5, 
                top_p=0.1,
                num_beams=1,
                use_cache=True, 
                stopping_criteria=[stopping_criteria_Q],
                output_scores=True,
                return_dict_in_generate=True,
            )
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # print('>>> LLM output : ', output)
            # tokenizer.convert_ids_to_tokens(input_ids_Q[0][-7:])
            Q_to_img_att = get_Q_att(video_batch_wo_blur, input_ids_Q, grid_size, args, cfg_pretrained)   # 32 x 32  x words_N x f_n x 8*8
            # breakpoint()
            Q_to_img_att = Q_to_img_att[:, :, args.token_idx].mean(dim=0).mean(dim=0)
            Q_to_img_att_list.append(Q_to_img_att)
            # breakpoint()
        
            get_local.clear()
            torch.cuda.empty_cache()
        
        rela_att = Q_to_img_att_list[1] / (Q_to_img_att_list[0] + 0.0001)
        # rela_att = torch.flip(rela_att, dims=[0])
        f_n, token_N = rela_att.shape
        
        action_loss = 1. - (rela_att.max(dim=-1)[0] * gt_track_mask.to(rela_att.device)).sum() / (rela_att.max(dim=-1)[0].sum() + 0.001)
        
        print(action_loss, ' _st_lr=', st_lr)
        spatial_prompt_history = spatial_prompt
        grad_cond = torch.autograd.grad(action_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        if torch.isnan(grad_cond).any():
            pass
        else:
            spatial_prompt = spatial_prompt - st_lr * grad_cond
        # print(grad_cond.max())
        get_local.clear()
        torch.cuda.empty_cache()
    
        # print_gpu()
    return spatial_prompt, spatial_prompt_history



def spatial_llm_att(
    model,
    spatial_questions,
    pre_img_token_embedds, 
    inputs_embeds,
    grid_size,
    tokens_format,
    targets,
    stopping_criteria_Q,
    tokenizer,
    args,
    cfg_pretrained
    ):
    
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(inputs_embeds))
    spatial_prompt_history = spatial_prompt
    # spatial_questions.append('please describe this video')
    img_token_start, img_token_end, tokens_N = tokens_format
    
    
    for ep in range(1):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])

            inputs_embed_q = torch.cat([pre_img_token_embedds, inputs_embeds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embed_q.shape[1])).to(inputs_embed_q.device)
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            # new_inputs_embed_q[:, -6:, :] += spatial_prompt_cur
            with torch.no_grad():
                outputs = model.generate(
                    inputs=None, 
                    images=None, 
                    attention_mask=attention_mask_q, 
                    inputs_embeds=new_inputs_embed_q,
                    position_ids=position_ids_q,
                    modalities="video", 
                    do_sample=False, 
                    temperature=0.0, 
                    max_new_tokens=5, 
                    top_p=0.1,
                    num_beams=1,
                    use_cache=True, 
                    stopping_criteria=[stopping_criteria_Q],
                    output_scores=True,
                    return_dict_in_generate=True,
                    output_attentions=True
                )
            
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # print('>>> LLM output : ', output)
            # scores = outputs.scores[0].detach().cpu().numpy()
            
            Q_to_img_att = get_Q_att(args.opts[1], outputs['attentions'], input_ids_Q, grid_size, args, cfg_pretrained)   # 32 x 32  x words_N x f_n x 8*8

            Q_to_img_att = Q_to_img_att[:, :, args.token_idx][-26:-6].mean(dim=0).mean(dim=0)
            Q_to_img_att_list.append(Q_to_img_att)
            # breakpoint()
        
            # get_local.clear()
            # torch.cuda.empty_cache()
        
        rela_att = Q_to_img_att_list[0] / (Q_to_img_att_list[1] + 0.0001)
        # rela_att = (Q_to_img_att_list[0] / (Q_to_img_att_list[1] + 0.0001)).mean(dim=0)
        # rela_att = torch.flip(rela_att, dims=[0])
        f_n, token_N = rela_att.shape

        # max_att_proposal_id = box2mask_for_track_id(targets['proposal_bboxes'].copy(), rela_att, targets['ori_size'], grid_size)
        proposal_m = F.interpolate(torch.Tensor(targets['proposal_mask'].copy()), size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        proposal_m = proposal_m.reshape(-1, f_n, token_N)
        proposal_m = proposal_m.to(rela_att.device)
        att_each_proposal = rela_att.unsqueeze(dim=0) * proposal_m
        # ============== 计算每个track的最大激活值, 查找哪个 track 具有最大的激活值
        max_att_each_proposal, _ = att_each_proposal.reshape(-1, f_n*token_N).max(dim=-1)
        max_att_proposal_id = torch.argmax(max_att_each_proposal)
        
        # ============== 平均每个track的最大激活值，来比较谁更可能是 target
        # each_track_f_max_v, _ = att_each_proposal.max(dim=-1)           # O_n x F_n 
        # each_track_f_max_v_mean = each_track_f_max_v.sum(dim=-1) / ((proposal_m.sum(dim=-1)>0.0).sum(dim=-1) + 0.0001)
        # mean_att_proposal_id = torch.argmax(each_track_f_max_v_mean)
        # breakpoint()
        return max_att_proposal_id, rela_att


    
