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
from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, \
    blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array, print_gpu
from custom_infer.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval


import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)


def temporal_llm_logit(
    model,
    temporal_questions,
    pre_img_token_embedds,
    inputs_embedds,
    inputs_blur_embedds,
    temporal_prompt,
    temporal_prompt_history,
    grid_size,
    video_batch,
    tokens_format,
    targets,
    stopping_criteria_Q,
    tokenizer,
    epochs,
    prompt_mask,
    args
    ):

    img_token_start, img_token_end, tokens_N = tokens_format
    manual_post_img_token_embedds_l = []
    
    for ep in range(epochs):
        Temp_prompt_cur = args.beta * temporal_prompt + (1 - args.beta) * temporal_prompt_history  # EMA
        
        T_segment_score = []
        Q_to_img_att_list = []
        pos_score = []
        for q_id, Qe in enumerate(temporal_questions[:-1]):
            
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
            manual_post_img_token_embedds_l.append(manual_post_img_token_embedds)
            
            inputs_embed_q = torch.cat([pre_img_token_embedds, inputs_embedds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embed_q.shape[1])).to(inputs_embed_q.device)
            
            new_inputs_embed_q = inputs_embed_q
            if prompt_mask is None:
                new_inputs_embed_q[:, img_token_start:img_token_end, :] += Temp_prompt_cur
            else:
                new_inputs_embed_q[:, img_token_start:img_token_end, :][:, prompt_mask] += Temp_prompt_cur[:, prompt_mask]
            outputs = model.generate(
                inputs=None, 
                images=None, 
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_embed_q,
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
            scores = outputs.scores[0]#.detach().cpu().numpy()
            # print(f'response ====> {output}')
            # print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
            # tmp_loss = scores[0][3582] - scores[0][1217]
            # grad_cond = torch.autograd.grad(tmp_loss.requires_grad_(True), [temporal_prompt], retain_graph=True,)[0]
            
            yes_logit = scores[0][3582]
            no_logit = scores[0][1217]

            pos_score.append(yes_logit-no_logit)
            get_local.clear()
            torch.cuda.empty_cache()
        
        # breakpoint()
        temporal_tuning_loss  = 1. - (pos_score[-1] - pos_score[0]) # light action
        # temporal_tuning_loss  = 1. - (pos_score[-1] - pos_score[1]) # light attribute
        # temporal_tuning_loss = torch.stack(temporal_tuning_loss).sum() / (len(temporal_tuning_loss) + 0.0001)
        print(temporal_tuning_loss, ' _temporal_lr=', args.lr)
        temporal_prompt_history = temporal_prompt
        grad_cond = torch.autograd.grad(temporal_tuning_loss.requires_grad_(True), [temporal_prompt], retain_graph=True,)[0]
        temporal_prompt = temporal_prompt - args.lr * grad_cond
        # print(temporal_prompt_history.sum()-temporal_prompt.sum())
        # print(np.round(alpha_.reshape(-1).detach().cpu().numpy(), 1))
        # print(np.vectorize(lambda x: f"{x:.2f}")(alpha_.reshape(-1).detach().cpu().numpy()))
        # print("grad_cond: ", grad_cond.max())
        get_local.clear()
        torch.cuda.empty_cache()
    
    return temporal_prompt, temporal_prompt_history



def temporal_llm_att(
    model,
    temporal_questions,
    pre_img_token_embedds,
    inputs_embedds,
    inputs_blur_embedds,
    temporal_prompt,
    temporal_prompt_history,
    grid_size,
    video_batch,
    tokens_format,
    targets,
    stopping_criteria_Q,
    tokenizer,
    epochs,
    prompt_mask,
    args
    ):

    img_token_start, img_token_end, tokens_N = tokens_format
    # manual_post_img_token_embedds_l = []
    
    temporal_prompt = temporal_prompt.detach()
    temporal_prompt_history = temporal_prompt_history.detach()
    Temp_prompt_cur = args.beta * temporal_prompt + (1 - args.beta) * temporal_prompt_history  # EMA
    Q_to_img_att_list = []
    # ['Is there the man standing by the stairs goes to the car in this video?', 'please describe this video']
    for q_id, Qe in enumerate(temporal_questions[-2:]):
        
        prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
        img_token_idx = int(torch.where(input_ids_Q == -200)[1])
        manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
            
        inputs_embed_q = torch.cat([pre_img_token_embedds, inputs_embedds, manual_post_img_token_embedds], dim=1)
        position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embed_q.shape[1])).to(inputs_embed_q.device)
        
        new_inputs_embed_q = inputs_embed_q
        if prompt_mask is None:
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += Temp_prompt_cur
        else:
            new_inputs_embed_q[:, img_token_start:img_token_end, :][:, prompt_mask] += Temp_prompt_cur[:, prompt_mask]
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
            # print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
            # yes_score, no_score = outputs.scores[0][0][3582], outputs.scores[0][0][1217]

            Q_to_img_att = get_Q_att(video_batch, input_ids_Q, grid_size)   # 32 x 32  x words_N x f_n x 8*8
            Q_to_img_att_list.append(Q_to_img_att[:, :, -5].mean(dim=0).mean(dim=0))
            
            get_local.clear()
            torch.cuda.empty_cache()
    
    # rela_att = (Q_to_img_att_list[0] * prompt_mask.reshape(20, 36)) / (Q_to_img_att_list[1] * prompt_mask.reshape(20, 36) + 0.0001)
    rela_att = Q_to_img_att_list[0]  / (Q_to_img_att_list[1] + 0.0001)
    f_n, token_N = rela_att.shape
    # breakpoint()
    
    proposal_m = F.interpolate(torch.Tensor(targets['proposal_mask'].copy()), size=(grid_size, grid_size), mode='bilinear', align_corners=False)
    proposal_m = proposal_m.reshape(-1, f_n, token_N)
    proposal_m = proposal_m.to(rela_att.device)
    att_each_proposal = rela_att.unsqueeze(dim=0) * proposal_m
    # ============== 计算每个track的最大激活值, 查找哪个 track 具有最大的激活值
    max_att_each_proposal, _ = att_each_proposal.reshape(-1, f_n*token_N).max(dim=-1)
    max_att_proposal_id = torch.argmax(max_att_each_proposal)
        
    return max_att_proposal_id, rela_att