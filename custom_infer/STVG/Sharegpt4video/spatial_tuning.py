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

import sys
sys.path.append("/home/yzq/mnt/code/MLLM/LLaVA-NeXT/custom_infer/STVG/Sharegpt4video")
from ShareGPT4Video.llava.mm_utils import (get_model_name_from_path, process_images, tokenizer_image_token)
from ShareGPT4Video.llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from ShareGPT4Video.llava.conversation import conv_templates

from visualizer import get_local
get_local.activate()

from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval


import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

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

def spatial_llm_att(
    model,
    spatial_questions,
    pre_img_token_embedds, 
    inputs_embeds,
    img_token_embedds_w_reverse,
    spatial_prompt,
    spatial_prompt_history,
    grid_size,
    video_batch,
    tokens_format,
    targets,
    stopping_criteria_Q,
    tokenizer,
    args,
    cfg_pretrained,
    ret_score=False
    ):
    
    generate_confg = {
        'do_sample': False, 'temperature': 1.0, 'top_p': 0.9, 'num_beams': 1, 'max_new_tokens': 20, 'pad_token_id': 128256, 
        'use_cache': True, 'output_attentions': True, 'return_dict_in_generate': True, 'output_logits':True}
    img_token_start, img_token_end, tokens_N = tokens_format
    
    scores = []
    for ep in range(1):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            Qe_prompt = prepare_prompt(Qe, "llava_llama_3")
            input_ids_Q = tokenizer_image_token(Qe_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_Q = input_ids_Q.to(model.device)
            img_token_idx = int(torch.where(input_ids_Q == -200)[0])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[img_token_idx+1:]).unsqueeze(dim=0)
            
            inputs_embed_q = torch.cat([pre_img_token_embedds, inputs_embeds, manual_post_img_token_embedds], dim=1)
            # position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embed_q.shape[1])).to(inputs_embed_q.device)
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            with torch.no_grad():
                outputs = model.generate(
                    inputs=None, 
                    images=None, 
                    attention_mask=None, 
                    inputs_embeds=new_inputs_embed_q,
                    position_ids=None,
                    **generate_confg
                )
            
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # print('>>> LLM output : ', output, 'logit_yes_no_score: ', outputs.scores[0][0][args.yes_id], outputs.scores[0][0][args.no_id])
            scores.append([outputs.logits[0][0][args.yes_id].cpu().numpy(), outputs.logits[0][0][args.no_id].cpu().numpy()])
            # tokenizer.convert_ids_to_tokens(input_ids_Q[0][-7:])
            # Q_to_img_att = get_Q_att(video_batch, input_ids_Q, grid_size, args, cfg_pretrained)   # 32 x 32  x words_N x f_n x 8*8
            
            attentions = outputs['attentions'][0]
            Q_to_img_att = torch.stack(attentions).squeeze()[:, :, -6:, img_token_start:img_token_start+tokens_N].cpu()
            # # 因为 shareGPT是把 4x4 的图片 Grid压缩到一个 1x1的图片，因此需要上采样的4x4的图片map
            # Q_to_img_att = Q_to_img_att.reshape(32, 32, -1, 24, 24).reshape(32, 32, -1, 24, 4, 6).reshape(32, 32, -1, 4, 6, 4, 6)
            # Q_to_img_att = Q_to_img_att.reshape(32, 32, -1, 16, 36)
            Q_to_img_att = Q_to_img_att[:, :, -6:]
            Q_to_img_att_list.append(Q_to_img_att.mean(dim=0).mean(dim=0))
            
        
            get_local.clear()
            torch.cuda.empty_cache()
        
        rela_att = Q_to_img_att_list[0] / (Q_to_img_att_list[1] + 0.0001)
        rela_att = rela_att.reshape(6, 24, 24)
        
        rela_att = rela_att.unfold(1, 6, 6).unfold(2, 6, 6)
        rela_att = rela_att.contiguous().view(6, -1, 6, 6)
        rela_att = rela_att.reshape(6, 16, 36)

        all_tokens_rela_att = rela_att
        # rela_att_max_v = rela_att.reshape(5, -1).max(dim=-1)[0]
        # indices = torch.argsort(rela_att_max_v, descending=True)
        rela_att = rela_att[args.token_idx]
        # rela_att = rela_att.mean(dim=0)
        # rela_att = rela_att[-1]
        # rela_att = torch.flip(rela_att, dims=[0])
        f_n, token_N = rela_att.shape
        
        
        proposal_m = F.interpolate(torch.Tensor(targets['proposal_mask'].copy()), size=(6, 6), mode='bilinear', align_corners=False)
        proposal_m = proposal_m.reshape(-1, f_n, token_N)
        proposal_m = proposal_m.to(rela_att.device)
        att_each_proposal = rela_att.unsqueeze(dim=0) * proposal_m
        # ============== 计算每个track的最大激活值, 查找哪个 track 具有最大的激活值
        max_att_each_proposal, _ = att_each_proposal.reshape(-1, f_n*token_N).max(dim=-1)
        max_att_proposal_id = torch.argmax(max_att_each_proposal)
        # max_att_each_proposal_L.append(max_att_each_proposal)
        
        # ============== 平均每个track的最大激活值，来比较谁更可能是 target
        # each_track_f_max_v, _ = att_each_proposal.max(dim=-1)           # O_n x F_n 
        # each_track_f_max_v_mean = each_track_f_max_v.sum(dim=-1) / ((proposal_m.sum(dim=-1)>0.0).sum(dim=-1) + 0.0001)
        # mean_att_proposal_id = torch.argmax(each_track_f_max_v_mean)
        
        # max_att_proposal_id = torch.argmax(torch.stack(max_att_each_proposal_L).mean(dim=0))
        
        if ret_score:
            return max_att_proposal_id, rela_att, scores[0][0]-scores[0][1]
        return max_att_proposal_id, rela_att, all_tokens_rela_att
    
    
    

def spatial_llm_logit(
    model,
    spatial_questions,
    pre_img_token_embedds, img_token_embedds_wo_blur, inputs_blur_track_embeds,
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
    ret_logit=False,
    **kwargs
    ):
    
    generate_confg = {
        'do_sample': False, 'temperature': 1.0, 'top_p': 0.9, 'num_beams': 1, 'max_new_tokens': 20, 'pad_token_id': 128256, 
        'use_cache': True, 'output_attentions': True, 'return_dict_in_generate': True, 'output_logits':True}
    
    # spatial_prompt = torch.nn.Parameter(torch.zeros_like(spatial_prompt))
    # spatial_prompt_history = spatial_prompt
    spatial_tuning_loss_list = []
    assert tokenizer.convert_tokens_to_ids('yes') == args.yes_id
    
    img_token_start, img_token_end, tokens_N = tokens_format
    for ep in range(epochs):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        track_score = []
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            Qe_prompt = prepare_prompt(Qe, "llava_llama_3")
            input_ids_Q = tokenizer_image_token(Qe_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            input_ids_Q = input_ids_Q.to(model.device)
            img_token_idx = int(torch.where(input_ids_Q == -200)[0])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[img_token_idx+1:]).unsqueeze(dim=0)
            
            # for k in range(len(inputs_blur_track_embeds)):
            inputs_blur_embed_q = torch.cat([pre_img_token_embedds, img_token_embedds_wo_blur, manual_post_img_token_embedds], dim=1)
            # position_ids_q, attention_mask_q = None, torch.ones((1, inputs_blur_embed_q.shape[1])).to(inputs_blur_embed_q.device)
            
            new_inputs_blur_embed_q = inputs_blur_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            
            new_inputs_blur_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            # with torch.no_grad():
            outputs = model.generate(
                inputs=None, 
                images=None, 
                attention_mask=None, 
                inputs_embeds=new_inputs_blur_embed_q,
                position_ids=None,
                **generate_confg
            )
            
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # scores = outputs.scores[0].detach().cpu().numpy()
            yes_score, no_score = outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id]
            get_local.clear()
            torch.cuda.empty_cache()
            # print_gpu()
            track_score.append(yes_score-no_score)
        
        track_score = torch.stack(track_score).squeeze()   # Q_n
        
        
        spatial_tuning_loss = 0.0
        for q_id in range(len(spatial_questions[1:])):
            spatial_tuning_loss += (1.0 - (track_score[q_id+1] - track_score[0]))
        print(spatial_tuning_loss, ' _st_lr=', st_lr)
        spatial_tuning_loss_list.append(spatial_tuning_loss.detach().cpu().numpy())
        spatial_prompt_history = spatial_prompt
        grad_cond = torch.autograd.grad(spatial_tuning_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        spatial_prompt = spatial_prompt - st_lr * grad_cond
        
        get_local.clear()
        torch.cuda.empty_cache()
    if ret_logit:
        return spatial_prompt, spatial_prompt_history, track_score
    else:
        return spatial_prompt, spatial_prompt_history, spatial_tuning_loss_list


