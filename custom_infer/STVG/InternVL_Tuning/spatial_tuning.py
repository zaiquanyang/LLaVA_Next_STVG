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

from custom_infer.STVG.InternVL_Tuning.conversation import get_conv_template
from custom_infer.STVG.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
# from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array, print_gpu
# from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval
from custom_infer.vis import print_gpu



def prepare_internVL_embedds(model, video, input_ids, video_embedds=None):
    
    visual_token_id = 151667
    # if video_embedds is None:
    #     video_embedds = model.extract_feature(video)
    # else:
        
    input_embeds = model.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == visual_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = video_embedds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)
            
    img_token_mask = selected
    return input_embeds, img_token_mask


def prepare_input_ids(model, tokenizer, question, generation_config, num_patches_list):
    
    IMG_START_TOKEN = "<img>"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_END_TOKEN = "</img>"
    
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    # video_prefix = ''.join([f'<image>\n' for i in range(len(num_patches_list))])
    question = video_prefix + question
    
    template = get_conv_template("internvl2_5")
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    # yes_token_id = tokenizer.convert_tokens_to_ids('yes')
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    
    # print(query)
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * int(num_patches) + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
    
    
    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    generation_config['eos_token_id'] = eos_token_id
    
    return input_ids, attention_mask, generation_config


def spatial_llm_att(
    model,
    spatial_questions,
    pre_img_token_embedds, 
    video_embeds,
    video_embeds_w_reverse,
    spatial_prompt,
    spatial_prompt_history,
    grid_size,
    video_batch,
    tokens_format,
    targets,
    generation_config,
    num_patches_list,
    tokenizer,
    args,
    processor,
    ret_score=False
    ):
    
    # img_token_start, img_token_end, tokens_N = tokens_format
    
    scores = []
    for ep in range(1):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            input_ids, attention_mask, generation_config  = prepare_input_ids(model, tokenizer, Qe, generation_config, num_patches_list)
            inputs_embed_q, img_token_mask = prepare_internVL_embedds(model, video_batch, input_ids, video_embedds=video_embeds)
            attention_mask_q = attention_mask.to(inputs_embed_q.device)
            # print(tokenizer.convert_ids_to_tokens(input_ids[0][-20:]))
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            
            new_inputs_embed_q[0][img_token_mask] += spatial_prompt_cur
            with torch.no_grad():
                outputs = model.language_model.generate(
                            inputs_embeds=new_inputs_embed_q, 
                            attention_mask=attention_mask_q,
                            output_logits=True,
                            **generation_config
                        )
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            print('>>> LLM output : ', output, 'logit_yes_no_score: ', outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id])
           
            ori_attention_maps = torch.stack(outputs['attentions'][0]).squeeze().cpu()
            # tokenizer.convert_ids_to_tokens(input_ids[0][-6:])
            
            question_to_img_att = ori_attention_maps[:, :, -6:, img_token_mask.cpu()]
            Q_to_img_att_list.append(question_to_img_att.mean(dim=0).mean(dim=0))

            torch.cuda.empty_cache()

        video_grid_thw_ = [video_batch.shape[0], 32, 32]
        rela_att = Q_to_img_att_list[0] / (Q_to_img_att_list[1] + 0.0001)
        all_tokens_rela_att = rela_att
        
        # rela_att_max_v = rela_att.reshape(6, -1).max(dim=-1)[0]
        # indices = torch.argsort(rela_att_max_v, descending=True)
        rela_att =  rela_att[args.token_idx]
        
        rela_att = rela_att.reshape(video_grid_thw_[0], -1)
        f_n, token_N = rela_att.shape
        
        proposal_m = F.interpolate(torch.Tensor(targets['proposal_mask'].copy()), size=(int(video_grid_thw_[1]//2), int(video_grid_thw_[2]//2)), mode='bilinear', align_corners=False)
        proposal_m = proposal_m.reshape(-1, f_n, token_N)
        proposal_m = proposal_m.to(rela_att.device)
        att_each_proposal = rela_att.unsqueeze(dim=0) * proposal_m
        
        # ============== 计算每个track的最大激活值, 查找哪个 track 具有最大的激活值
        max_att_each_proposal, _ = att_each_proposal.reshape(-1, f_n*token_N).max(dim=-1)
        max_att_proposal_id = torch.argmax(max_att_each_proposal)

    return max_att_proposal_id, rela_att, all_tokens_rela_att




def spatial_llm_logit(
        model,
        spatial_questions,
        pre_video_token_embedds, 
        video_embeds, 
        video_embeds_w_reverse,
        spatial_prompt,
        spatial_prompt_history,
        grid_size,
        video_batch,
        generation_config,
        num_patches_list,
        epochs,
        tokenizer, 
        args, 
        processor,
        st_lr,
        ret_logit=False,
        **kwargs
    ):
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(spatial_prompt))
    spatial_prompt_history = spatial_prompt
        
    # tokenizer.convert_tokens_to_ids('yes')
    # img_token_start, img_token_end, tokens_N = tokens_format
    for ep in range(epochs):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        track_score = []
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            input_ids, attention_mask, generation_config  = prepare_input_ids(model, tokenizer, Qe, generation_config, num_patches_list)
            inputs_embed_q, img_token_mask = prepare_internVL_embedds(model, video_batch, input_ids, video_embedds=video_embeds)
            attention_mask_q = attention_mask.to(inputs_embed_q.device)
            # print(tokenizer.convert_ids_to_tokens(input_ids[0][-20:]))
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_embed_q[0][img_token_mask] += spatial_prompt_cur.squeeze()
            print(f'Ep={ep}, Q_id={q_id}')
            print_gpu()
            outputs = model.language_model.generate(
                        inputs_embeds=new_inputs_embed_q, 
                        attention_mask=attention_mask_q,
                        output_logits=True,
                        **generation_config
                    )
            
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # scores = outputs.scores[0].detach().cpu().numpy()
            yes_score, no_score = outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id]

            # get_local.clear()
            torch.cuda.empty_cache()
            print_gpu()
            track_score.append(yes_score-no_score)
        
        track_score = torch.stack(track_score).squeeze()   # Q_n
        
        spatial_tuning_loss = 0.0
        for q_id in range(len(spatial_questions[1:])):
            spatial_tuning_loss += (1.0 - (track_score[q_id+1] - track_score[0]))
        print(spatial_tuning_loss, ' _st_lr=', st_lr)
        spatial_prompt_history = spatial_prompt
        grad_cond = torch.autograd.grad(spatial_tuning_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        spatial_prompt = spatial_prompt - st_lr * grad_cond
        
        # get_local.clear()
        torch.cuda.empty_cache()
        
    if ret_logit:
        return spatial_prompt, spatial_prompt_history, track_score
    else:
        return spatial_prompt, spatial_prompt_history



