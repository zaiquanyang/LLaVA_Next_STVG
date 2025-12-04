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
# from stvg.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, blur_proposal_video_array, blur_video_frames
# from stvg.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss
from stvg.util import Q_to_prompt, get_Q_att, ttt_eval
from stvg.V_Tuning.spatial_tuning import spatial_llm_logit, spatial_llm_att

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

import copy

def llava_embedd_tuning(cfg_pretrained, tokenizer, model, image_processor, video, questions, grid_size, targets, args, vid_i, logger):

    """
    为了有效验证 hightlight_action 对时序定位的作用，这里我们选择oracl_track_id来规避空间定位的影响
    """
    contra_cap = "please describe this video"   # 这里 caption 的内容会影响结果
    # with open("custom_infer/STVG/output/oracle_track_id.json", "r", encoding="utf-8") as f:
    #     track_ids = json.load(f)
    
    # track_id = track_ids[str(vid_i)]
    
    # gt_track_all_frames = []
    
    llm_reasoning = True
    if llm_reasoning:
        # ===================================== step 1.1. 预先准备全部 的 video_embedding
        with torch.no_grad():
            # ======================================================= w/o blur track
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
                # v_n = v_n -1
            elif cfg_pretrained.mm_newline_position == "no_token":
                v_n = grid_size*grid_size * F_n + 0
            else:
                breakpoint()
            img_token_idx = int(torch.where(input_ids_ == -200)[1])
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]

            # ======================================================== w blur track
            # track_mask = targets['proposal_bboxes'][track_id].sum(axis=-1)>0
            # blur_frames_ids = ~(targets['proposal_bboxes'][track_id].sum(axis=-1)>0)
            # if blur_frames_ids.sum() == 0:
            #     blur_frames_ids = np.ones_like(blur_frames_ids)
                
            # video_blur_proposal, blur_images = blur_video_frames(video, targets, blur_frames_ids=blur_frames_ids)
            # # targets['pil_images'] = blur_images
            
            # if model.dtype == torch.bfloat16:
            #     video_proc = image_processor.preprocess(video_blur_proposal, return_tensors="pt")["pixel_values"].to(torch.bfloat16).cuda()
            # else:
            #     video_proc = image_processor.preprocess(video_blur_proposal, return_tensors="pt")["pixel_values"].half().cuda()
            # video_blur_batch = [video_proc]
            
            # prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            # (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
            #     input_ids=input_ids_C, 
            #     position_ids=None, 
            #     past_key_values=None,
            #     labels=None,
            #     attention_mask=attention_masks_C,
            #     images=video_blur_batch, 
            #     modalities="video",
            #     )
            # img_token_embedds_w_blur = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            
            
        # =========================== step 1.2 Spatial tuning ======= > spatial localization
        tokens_format = [img_token_idx, img_token_idx+v_n, v_n]
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
        spatial_prompt_history = spatial_prompt

        # ============================ stage 1
        if args.ttt_epoch > 0:
            tmp_questions = prepare_instructions(questions, targets, mode='focus_actions_tuning')
            logger.info(f'tuning_action_loc:  {tmp_questions}')
            spatial_prompt, spatial_prompt_history, _ = spatial_llm_logit(
                                                    model, 
                                                    tmp_questions.copy(), 
                                                    pre_img_token_embedds, 
                                                    img_token_embedds, 
                                                    None,
                                                    spatial_prompt, spatial_prompt_history, 
                                                    grid_size, 
                                                    video_batch_wo_blur, 
                                                    tokens_format, 
                                                    args.ttt_epoch,
                                                    stopping_criteria_, tokenizer, args,
                                                    args.t_lr
                                                    )
            # tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_tuning')
            # logger.info(f'tuning_attr_loc:  {tmp_questions}')
            # spatial_prompt, spatial_prompt_history = unify_llm_att(
            #                                         model, 
            #                                         tmp_questions.copy(), 
            #                                         pre_img_token_embedds, 
            #                                         img_token_embedds, 
            #                                         track_mask,
            #                                         spatial_prompt, spatial_prompt_history, 
            #                                         grid_size, 
            #                                         video_batch_wo_blur, 
            #                                         tokens_format, 
            #                                         args.ttt_epoch,
            #                                         stopping_criteria_, 
            #                                         tokenizer, 
            #                                         args,
            #                                         args.s_lr,
            #                                         targets,
            #                                         cfg_pretrained
            #                                         )
        
        # tmp_questions = [spatial_question[1], "please describe this video"]         
        tmp_questions = prepare_instructions(questions, targets, mode='focus_actions_infer')
        logger.info(f'infer_action_loc:  {tmp_questions}')
        assert len(tmp_questions) == 2, print('inferring must need two Questions !')
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
        # spatial_pred_track_id = track_id
        contrastive_content  = f'track_{spatial_pred_track_id}_{tmp_questions[0][0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=False)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')

        # 保存 llava_rela_att 到指定文件夹
        npy_dict = {}
        npy_dict = {'S_pred': spatial_pred_track_id, 'T_LLaVa_att': llava_rela_att.cpu().to(torch.float32).numpy(), 'vid_i': vid_i, 'GT_viou': gt_viou}
        name_f = targets['vid_name']
        np.save(f'{args.output_dir}/_llava_att_npy/{name_f}.npy', npy_dict)
        
    else:
        with open("custom_infer/STVG/output/hc_0319_decom_yes_1k/llava_next_video_7b_f_20_pool_4/focus_attr_pred_track_id.json", "r", encoding="utf-8") as f:
            track_ids = json.load(f)
        # breakpoint()
        track_id = track_ids[str(vid_i)]
        spatial_pred = {'Space_pred': track_id}
        
        name_f = targets['vid_name']
        npy_f = f'{args.output_dir[:-7]}/_llava_att_npy/{name_f}.npy'
    
        llava_dict = np.load(npy_f, allow_pickle=True).item()
        llava_rela_att = torch.from_numpy(llava_dict['T_LLaVa_att']).cuda()
        # spatial_pred_track_id = llava_dict['S_pred']
        # spatial_pred = {'Space_pred': spatial_pred_track_id}
        
        # spatial_pred = {}
        contrastive_content  = f'_{questions[0:120]}_'
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=False)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')
        
        # llava_dict['vid_i'] = vid_i
        # llava_dict['GT_Action'] = targets['actioness'].numpy()
        # np.save(npy_f, llava_dict)

        # breakpoint()
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
            tmp_questions = action_Q
            tmp_questions.append("please describe this video")
            # Method 2.
            # tmp_questions = [action_Q[0], attr_Q[0]]
            # Method 3.
            # tmp_questions = [ref_Q, "please describe this video"]
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