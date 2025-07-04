import json
import os

import torch
from PIL import Image
import numpy as np
import random 
import torch.nn.functional as F

from visualizer import get_local
get_local.activate()

# from custom_infer.utils import boxes_to_masks
from custom_infer.STVG.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336
from custom_infer.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval

from custom_infer.STVG.Qwen_Tuning.tuning_func import spatial_llm_logit, spatial_llm_att

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

import copy




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

def llava_embedd_tuning(cfg_pretrained, tokenizer, model, processor, video, questions, grid_size, targets, args, vid_i, logger):
    
    frames_N, H, W, _ = video.shape
    patch_h, patch_w = H // 28, W // 28
    
    video_inputs = np.repeat(video, repeats=2, axis=0)        # 因为 Qwen2_VL会对输入的视频按时间步长=2进行下采样，所以我们需要先进行预处理
    
    inputs = prepare_inputs(processor, tokenizer, questions, video_inputs)
    input_ids, pixel_values_videos, video_grid_thw = inputs['input_ids'], inputs['pixel_values_videos'], inputs['video_grid_thw']
    
    llm_reasoning = True
    if llm_reasoning:
        # =============================== prepare input_embedds for Qwen2_VL
        visual_token_id = 151656
        video_embeds, pre_video_embedds, post_video_embedds, attention_mask, tokens_format = prepare_qwen_embedds(model, inputs, visual_token_id)
        
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(video_embeds))
        spatial_prompt_history = spatial_prompt
        
        if args.ttt_epoch > 0:
            tmp_questions = prepare_instructions(questions, targets, mode='focus_actions_tuning')
            logger.info(f'focus_actions_tuning:  {tmp_questions}')
            spatial_prompt, spatial_prompt_history = spatial_llm_logit(
                                                    model, 
                                                    tmp_questions.copy(), 
                                                    pre_video_embedds, 
                                                    video_embeds, 
                                                    None,
                                                    spatial_prompt, spatial_prompt_history, 
                                                    grid_size, 
                                                    video_inputs, 
                                                    tokens_format, 
                                                    args.ttt_epoch,
                                                    tokenizer, args, processor,
                                                    args.s_lr
                                                    )
        
        tmp_questions = prepare_instructions(questions, targets, mode='focus_actions_infer')
        logger.info(f'focus_actions_infer:  {tmp_questions}')
        llava_att_pred_track_id, llava_rela_att, all_tokens_rela_att = spatial_llm_att(
                    model,
                    tmp_questions.copy(),
                    pre_video_embedds,
                    video_embeds,
                    None,
                    spatial_prompt,
                    spatial_prompt_history,
                    None,
                    video_inputs,
                    tokens_format, 
                    targets,
                    None, 
                    tokenizer, args, processor
                )
        spatial_pred_track_id = llava_att_pred_track_id.item()
    
        contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=True)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')

        npy_dict = {}
        npy_dict = {
            'S_pred': spatial_pred_track_id, 
            'T_LLaVa_att': llava_rela_att.cpu().to(torch.float32).numpy(), 
            # 'vid_i': vid_i, 
            # 'GT_viou': gt_viou, 
            # 'attr_for_track_score_pre': max_att_each_proposal_pre_tuning.cpu().numpy(),
            # 'attr_pred_id_pre':  llava_att_pred_track_id_pre_tuning.item(),
            # 'attr_for_track_score_post':max_att_each_proposal.cpu().numpy(),
            # 'attr_pred_id_post':  spatial_pred_track_id,
            # 'oracle_id': track_id
            }
        
        name_f = targets['vid_name']
        np.save(f'{args.output_dir}/_llava_att_npy/{name_f}.npy', npy_dict)

    else:
        pass
    
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
            attr_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]     #["please describe this video"]
            decompose_fail = True
        else:
            attr_Q = [f"Is there {sent.strip('.').strip()} in this video?"]

    for sent in gpt_4_proc_action:
        if "please describe this video" in sent:
            action_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]   #["please describe this video"]
            decompose_fail = True
        else:
            action_Q = [f"Is there {sent.strip('.').strip()} in this video?"]
    # assert decompose_fail == False, print('please setting decompose_fail right !')
    
    # sub_action_Q = []
    # for sent in gpt_4_proc_sub_actions:
    #     sub_action_Q.append(f"Is there {sent.strip('.').strip()} in this video?")
            
    tmp_questions = []
    
    if mode == "focus_actions_tuning":
        if "please describe this video" in action_Q:          # 当前这个解耦无效-fail
            tmp_questions = ["please describe this video", ref_Q]
        else:
            # tmp_questions = [attr_Q[0], ref_Q]                                                 # ref_Q - attr_Q
            tmp_questions = ["please describe this video"]                        # action_Q - " "
            tmp_questions.extend(action_Q)
        return tmp_questions
    elif mode == "focus_actions_infer":
        if "please describe this video" in action_Q:          # 当前这个解耦无效-fail
            tmp_questions = [ref_Q, "please describe this video"]
        else:
            # Method 1. 
            # tmp_questions = action_Q
            # tmp_questions.append("please describe this video")
            # Method 2.
            # tmp_questions = [action_Q[0], attr_Q[0]]
            # Method 3.
            tmp_questions = [ref_Q, "please describe this video"]
        return tmp_questions

    elif mode == "focus_attribute_tuning":
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