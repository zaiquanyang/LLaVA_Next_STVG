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

from visualizer import get_local
get_local.activate()

from custom_infer.STVG.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array, print_gpu
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval



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
    stopping_criteria_Q,
    tokenizer,
    args,
    processor,
    ret_score=False
    ):
    
    # spatial_questions.append('please describe this video')
    img_token_start, img_token_end, tokens_N = tokens_format
    
    scores = []
    for ep in range(1):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            cur_inputs = prepare_inputs(processor, tokenizer, Qe, video_batch)
            input_ids, pixel_values_videos, video_grid_thw = cur_inputs['input_ids'], cur_inputs['pixel_values_videos'], cur_inputs['video_grid_thw']
            video_embeds, pre_video_embedds, post_video_embedds, attention_mask, _ = prepare_qwen_embedds(model, cur_inputs, args.visual_token_id, video_embeds)
            inputs_embed_q = torch.cat([pre_video_embedds, video_embeds, post_video_embedds], dim=1)
            attention_mask_q = attention_mask.to(inputs_embed_q.device)
            # print(tokenizer.convert_ids_to_tokens(input_ids[0][-20:]))
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            with torch.no_grad():
                outputs = model.generate(
                            inputs_embeds=new_inputs_embed_q, 
                            attention_mask=attention_mask_q, 
                            max_new_tokens=10,
                            use_cache=True,
                            # output_scores=True,
                            output_logits=True,
                            return_dict_in_generate=True,
                        )
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            print('>>> LLM output : ', output, 'logit_yes_no_score: ', outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id])
            
            # scores.append([outputs.scores[0][0][args.yes_id].cpu().numpy(), outputs.scores[0][0][args.no_id].cpu().numpy()])
            # tokenizer.convert_ids_to_tokens(input_ids_Q[0][-7:])
            # breakpoint()
            ori_attention_maps = get_local.cache['Qwen2VLSdpaAttention.forward'] # sdpa_attention_forward, Qwen2VLSdpaAttention.forward
            get_local.clear()
            torch.cuda.empty_cache()
            
            question_attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] > 1]
            question_to_img_att = torch.cat([att.cpu() for att in question_attention_maps], 0)
            question_to_img_att = question_to_img_att[:, :, -6:, img_token_start:(img_token_start + tokens_N)]
            Q_to_img_att_list.append(question_to_img_att.mean(dim=0).mean(dim=0))
            
        video_grid_thw_ = video_grid_thw[0].cpu().numpy()
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
        tokens_format,
        epochs,
        tokenizer, args, processor,
        st_lr,
        ret_logit=False,
        **kwargs
    ):
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(spatial_prompt))
    spatial_prompt_history = spatial_prompt
        
    # tokenizer.convert_tokens_to_ids('yes')
    img_token_start, img_token_end, tokens_N = tokens_format
    for ep in range(epochs):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        track_score = []
        Q_to_img_att_list = []
        for q_id, Qe in enumerate(spatial_questions[:]):
            
            cur_inputs = prepare_inputs(processor, tokenizer, Qe, video_batch)
            input_ids, pixel_values_videos, video_grid_thw = cur_inputs['input_ids'], cur_inputs['pixel_values_videos'], cur_inputs['video_grid_thw']
            video_embeds, pre_video_embedds, post_video_embedds, attention_mask, _ = prepare_qwen_embedds(model, cur_inputs, args.visual_token_id, video_embeds)
            inputs_embed_q = torch.cat([pre_video_embedds, video_embeds, post_video_embedds], dim=1)
            
            position_ids_q, attention_mask_q = None, attention_mask.to(inputs_embed_q.device)
            
            new_inputs_embed_q = inputs_embed_q
            # 1 x v_n x 4096,   1 x v_n x 1
            new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
            # with torch.no_grad():
            outputs = model.generate(
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_embed_q,
                max_new_tokens=10, 
                top_p=0.1,
                num_beams=1,
                use_cache=True, 
                # output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
            generate_ids = outputs.sequences    # requires_grad
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # scores = outputs.scores[0].detach().cpu().numpy()
            yes_score, no_score = outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id]

            get_local.clear()
            torch.cuda.empty_cache()
            # breakpoint()
            track_score.append(yes_score-no_score)
        
        track_score = torch.stack(track_score).squeeze()   # Q_n
        
        spatial_tuning_loss = 0.0
        for q_id in range(len(spatial_questions[1:])):
            spatial_tuning_loss += (1.0 - (track_score[q_id+1] - track_score[0]))
        print(spatial_tuning_loss, ' _st_lr=', st_lr)
        spatial_prompt_history = spatial_prompt
        grad_cond = torch.autograd.grad(spatial_tuning_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        spatial_prompt = spatial_prompt - st_lr * grad_cond
        
        get_local.clear()
        torch.cuda.empty_cache()
        
    if ret_logit:
        return spatial_prompt, spatial_prompt_history, track_score
    else:
        return spatial_prompt, spatial_prompt_history




def prepare_qwen_embedds(model, inputs, visual_token_id, video_embeds=None):
    input_ids, pixel_values_videos, video_grid_thw = inputs['input_ids'], inputs['pixel_values_videos'], inputs['video_grid_thw']
    
    inputs_embeds = model.model.embed_tokens(input_ids)
    
    pixel_values_videos = pixel_values_videos.type(model.visual.get_dtype())
    with torch.no_grad():       # 注意，不加 no_grad 会爆显存
        if video_embeds is None:
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        else:
            video_embeds = video_embeds.squeeze()
    n_video_tokens = (input_ids == visual_token_id).sum().item()
    n_video_features = video_embeds.shape[0]
    if n_video_tokens != n_video_features:
        raise ValueError(
            f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
        )
    video_mask = (
        (input_ids == visual_token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    visual_start_pos = input_ids[0].tolist().index(visual_token_id)
    visual_end_pos = visual_start_pos + n_video_features
    
    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    
    attention_mask = inputs["attention_mask"].to(inputs_embeds.device)

    pre_video_embedds = inputs_embeds[:, :visual_start_pos, :]
    post_video_embedds = inputs_embeds[:, visual_end_pos:, : ]
    video_embeds = inputs_embeds[:, visual_start_pos:visual_end_pos, :]
    
    tokens_format = [visual_start_pos, visual_end_pos, n_video_features]
    
    return video_embeds, pre_video_embedds, post_video_embedds, attention_mask, tokens_format


def prepare_inputs(processor, tokenizer, question, video):
    vid_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": [
                        "placeholder",
                        "placeholder",
                    ],
                    "fps": 1.0,
                },
                {"type": "text", "text": f"{question}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        vid_messages, tokenize=False, add_generation_prompt=True
    )
    # print(tokenizer.tokenize(text))
    # breakpoint()
    inputs = processor(
        text=[text],
        images=None,
        videos=(torch.Tensor(video).permute(0, 3, 1, 2), ),
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][-30:])
    
    return inputs
