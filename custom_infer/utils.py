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

from llava.mm_utils import process_anyres_image,tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle

from visualizer import get_local
get_local.activate()

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

from custom_infer.STVG.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
from custom_infer.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, \
    blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array
from custom_infer.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss

def load_video(video_path,args):
    if args.for_get_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), fps//2)]
    frame_time = [i/fps for i in frame_idx]
    
    if len(frame_idx) > args.for_get_frames_num or args.force_sample:
        sample_fps = args.for_get_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    pil_images = [Image.fromarray(frame) for frame in spare_frames]
    # import pdb;pdb.set_trace()

    return spare_frames[:], frame_time, video_time, pil_images[:]

def load_img_list(vid_path, frames_N):

    all_frames_list = os.listdir(vid_path)
    if len(all_frames_list) <= frames_N:
        sample_frames_list = all_frames_list
    else:
        sample_frames_list = random.choices(all_frames_list, k=frames_N)
    sample_frames_list.sort(key=lambda p: int(os.path.splitext(p)[0]))
    
    # 读取已经抽完帧的图片数据
    pil_img_list = []
    arr_img_list = []
    for img_f in sample_frames_list:
        img_ = Image.open(os.path.join(vid_path, img_f)).convert('RGB')
        pil_img_list.append(img_)
        arr_img_list.append(np.array(img_))

    arr_images = np.stack(arr_img_list)
    return arr_images, None, None, pil_img_list


def load_video_base64(path):
    video = cv2.VideoCapture(path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    return base64Frames

def Q_to_prompt(question, args, model, tokenizer):

    qs = question
    assert args.add_time_instruction == False
    # if args.add_time_instruction:
    #     time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    #     qs = f'{time_instruciton}\n{qs}'
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    if tokenizer.pad_token_id is None:
        if "qwen" in tokenizer.name_or_path.lower():
            print("Setting pad token to bos token for qwen model.")
            tokenizer.pad_token_id = 151643
            
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    
    return prompt, input_ids, keywords, stopping_criteria, attention_masks, tokenizer


# def get_Q_att(video, input_ids, grid_size):
    
#     ori_attention_maps = get_local.cache['sdpa_attention_forward'] #get_local.cache['LlamaSdpaAttention.forward'], sdpa_attention_forward
#     get_local.clear()
#     torch.cuda.empty_cache()
    
#     F_n = video[0].size(0)
#     prompt_n = input_ids.size(-1)
    
#     v_n = grid_size*grid_size * F_n
#     llm_in_tokens_N = v_n +  prompt_n - 1 
#     img_token_idx = int(torch.where(input_ids == -200)[1])
#     # ========================================== question_attention_maps
#     question_attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] == llm_in_tokens_N]
#     question_to_img_att_ = torch.cat([att.to('cuda') for att in question_attention_maps], 0)            # Layers_N x head_N x llm_in_tokens_N x llm_in_tokens_N
    
#     question_to_img_att = question_to_img_att_.mean(0).mean(0)                                                  # llm_in_tokens_N x llm_in_tokens_N
#     question_to_img_att = question_to_img_att[(img_token_idx + v_n - 1):, img_token_idx:(img_token_idx + v_n)] #  Q_L x Patch_n
    
#     Q_to_img_att = question_to_img_att.reshape(-1, F_n, grid_size*grid_size)
    
#     return Q_to_img_att



def get_Q_att(video, input_ids, grid_size, args=None, cfg_pretrained=None):
    
    ori_attention_maps = get_local.cache['sdpa_attention_forward'] #get_local.cache['LlamaSdpaAttention.forward'], sdpa_attention_forward
    get_local.clear()
    torch.cuda.empty_cache()
    # print("ori_attention_maps: ", len(ori_attention_maps))
    F_n = video[0].size(0)
    prompt_n = input_ids.size(-1)
    
    if cfg_pretrained.mm_newline_position == "one_token":
        v_n = grid_size*grid_size * F_n + 1
        # v_n = v_n -1
    elif cfg_pretrained.mm_newline_position == "no_token":
        v_n = grid_size*grid_size * F_n + 0
    else:
        breakpoint()
    llm_in_tokens_N = v_n +  prompt_n - 1 
    img_token_idx = int(torch.where(input_ids == -200)[1])
    
    # ========================================== question_attention_maps
    question_attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] == llm_in_tokens_N]
    question_to_img_att_ = torch.cat([att.to('cuda') for att in question_attention_maps], 0)            # Layers_N x head_N x llm_in_tokens_N x llm_in_tokens_N
    
    # question_to_img_att = question_to_img_att_.mean(0).mean(0)                                                  # llm_in_tokens_N x llm_in_tokens_N
    question_to_img_att = question_to_img_att_[:, :, (img_token_idx + v_n - 1):, img_token_idx:(img_token_idx + v_n)] #  Q_L x Patch_n
    
    
    if cfg_pretrained.mm_newline_position == "one_token":
        question_to_img_att = question_to_img_att[:, :, :, :-1]
        # 因为之前已经取消了 last_one_token的输入，所以这里没必要再调整
        # question_to_img_att = question_to_img_att[:, :, :, :]
    elif cfg_pretrained.mm_newline_position == "no_token":
        question_to_img_att = question_to_img_att[:, :, :, :]
    else:
        breakpoint()
        
    Q_to_img_att = question_to_img_att.reshape(args.layers_N, args.heads_N, -1, F_n, grid_size*grid_size)
    
    return Q_to_img_att


def ttt_eval(targets, Q_to_img_att_list, vid_i, preprocess, contrastive_content, blur_pred, args, save=True):
    
    output_dir = args.output_dir
    
    proposal_mask=targets['proposal_mask'].copy()
    proposal_bboxes = targets['proposal_bboxes'].copy()
    actioness = targets['actioness'].clone()
    boxs, image_size, mode = targets['boxs'].bbox, targets['boxs'].size, targets['boxs'].mode
    pil_images = targets['pil_images']
    ori_size = targets['ori_size']
    vid_name = targets['vid_name']
    
    # =============================== step-1 . Generate spatial-temporal attention prediction
    rela_att = Q_to_img_att_list.detach() if Q_to_img_att_list is not None else None
    if rela_att is not None:
        bbox_pred_tmp, temporal_pred_tmp, pred_track_ID, pred_actionness, v_iou, gt_viou = evaluation_llava_att(
            llava_pred=Q_to_img_att_list.detach(), 
            actioness=actioness, 
            gt_boxs=boxs.clone(), # 注意这里要 clone，因为evaluation_single会修改传入的gt_boxs
            proposal_mask=proposal_mask, 
            proposal_bboxes=proposal_bboxes,
            vid_i=vid_i, 
            targets=targets,
            blur_pred=blur_pred,
            args=args
        )
    else:
        bbox_pred_tmp, temporal_pred_tmp, pred_track_ID, pred_actionness, v_iou = evaluation_blur_pred(
            llava_pred=rela_att, 
            actioness=actioness, 
            gt_boxs=boxs.clone(), # 注意这里要 clone，因为evaluation_single会修改传入的gt_boxs
            proposal_mask=proposal_mask, 
            proposal_bboxes=proposal_bboxes,
            vid_i=vid_i, 
            targets=targets,
            blur_pred=blur_pred
        )
    if save:
        # ===============================  step-2 .可视化预测结果
        st_iou = int(np.round(v_iou, 2)*100)
        gt_viou = int(np.round(gt_viou, 2)*100)
        proposal_bbox_w_action = proposal_bboxes[pred_track_ID].copy()#[pred_actionness==1]
        proposal_bbox_w_action = original_resize_336(proposal_bbox_w_action, ori_size, args)
        proposal_bbox_w_action[pred_actionness==0] = [0,0,0,0]

        vid_rela = show_video_relevance(
                pil_images[:], 
                preprocess, 
                att_maps=rela_att[:, :], 
                norm_mode='video',
                actioness=actioness, 
                bboxes=(boxs * (224/336)).numpy().astype(int),
                pred_bboxes=(proposal_bbox_w_action * (224/336)).astype(int),
                )
        vid_rela.save(f'{output_dir}/{vid_i}_{gt_viou}_{st_iou}_{contrastive_content[:120]}.png')
        return bbox_pred_tmp, temporal_pred_tmp, gt_viou
    else:
        gt_viou = int(np.round(gt_viou, 2)*100)
        return bbox_pred_tmp, temporal_pred_tmp, gt_viou

  
def llava_embedd_infer(cfg_pretrained, tokenizer, model, image_processor, video, questions, grid_size, targets, args, vid_i):

    # ============== blur proposal region 
    proposal_num = len(targets['proposal_bboxes'])
    t_id = min(proposal_num-1, 2)
    # video_blur_proposal, blur_images = blur_proposal_video_array(video, targets, blur_propsal_id=t_id)
    # targets['pil_images'] = blur_images
    # ============== blur gt region 
    # video_blur_gt = blur_gt_video_array(video, targets)
    
    video_proc = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda()
    video_batch = [video_proc]
    contra_cap = "please describe this video"#"please describe this video"   # 这里 caption 的内容会影响结果
    print(questions)
    # breakpoint()
    # contra_cap = "Is there the man raises his finger and points at the side, then points at the top and walks up behind in this video ?"
    print('contra_cap: ', contra_cap)
    # questions_list = questions
    questions_list = [
        "Is there woman puts down knife and fork and touches man's hand in this video ?", \
        "Is there woman in this video ?", \
        "Is there person puts down knife and fork and touches man's hand in this video ?",
        ]
    # ===================================== 预先当前 inputs_embeds_q
    with torch.no_grad():
        prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
        
        (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids_C, 
            position_ids=None, 
            attention_mask=attention_masks_C, 
            past_key_values=None,
            labels=None, 
            images=video_batch, 
            modalities="video",  
            image_sizes=None
            )
    F_n = video_batch[0].size(0)
    # prompt_n = input_ids_C.size(-1)
    v_n = grid_size*grid_size * F_n
    
    img_token_idx = int(torch.where(input_ids_C == -200)[1])
    img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
    pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
    
    Q_to_img_att_list = []
    ttt_epochs = 1
    visual_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
    vprompt_history = visual_prompt
    
    for ep in range(ttt_epochs):
        vprompt_cur = args.beta * visual_prompt + (1 - args.beta) * vprompt_history  # EMA
        
        # ===================================== 提取当前 ttt_epoch 的 contrastive_attention
        prompt_, input_ids_C, keywords_, stopping_criteria_C, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
        assert img_token_idx==int(torch.where(input_ids_C == -200)[1])
        
        with torch.no_grad():
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_C[:, img_token_idx+1:])
            inputs_embeds_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embeds_q.shape[1])).to(inputs_embeds_q.device)

        
            new_inputs_embeds_c = inputs_embeds_q
            new_inputs_embeds_c[:, img_token_idx:img_token_idx + v_n, :] += vprompt_cur
            
            output_ids = model.generate(
                inputs=input_ids_C, 
                images=video_batch, 
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_embeds_c,
                position_ids=position_ids_q,
                modalities="video", 
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=40, 
                top_p=0.1,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria_C],
                output_scores=True,
                return_dict_in_generate=True
            )
            C_to_img_att = get_Q_att(video_batch, input_ids_C, grid_size)
        
        
            for q_id, Qe in enumerate(questions_list[:]):
                # ===================================== load prompt
                prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
                assert img_token_idx==int(torch.where(input_ids_Q == -200)[1])
                manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
                inputs_embeds_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
                position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embeds_q.shape[1])).to(inputs_embeds_q.device)
                
                new_inputs_embeds_q = inputs_embeds_q
                new_inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :] += vprompt_cur
                outputs = model.generate(
                    inputs=input_ids_Q, 
                    images=video_batch, 
                    attention_mask=attention_mask_q, 
                    inputs_embeds=new_inputs_embeds_q,
                    position_ids=position_ids_q,
                    modalities="video", 
                    do_sample=False, 
                    temperature=0.0, 
                    max_new_tokens=40, 
                    top_p=0.1,
                    num_beams=1,
                    use_cache=True, 
                    stopping_criteria=[stopping_criteria_Q],
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                Q_to_img_att = get_Q_att(video_batch, input_ids_Q, grid_size)
                Q_to_img_att_list.append(Q_to_img_att)
                
                rela = Q_to_img_att[-5] / (C_to_img_att[-5] + 0.0001)
                # rela = C_to_img_att[-5] / (Q_to_img_att[-5] + 0.0001)
                
                generate_ids = outputs.sequences    # requires_grad
                scores = outputs.scores[0].detach().cpu().numpy()
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
                print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
                
                # ============== 当前 ttt_epoch : 可视化当前 ttt_epoch 的结果
                contrastive_content  = f'yes={scores[0][3582]:.2f}_no={scores[0][1217]:.2f}_({Qe[:100]})_({contra_cap})_'
                bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, rela, vid_i, preprocess, contrastive_content, args)
            
        # ============== 当前 ttt_epoch : compute loss L1
        # ttt_loss = args.alpha * phrase_consistency_l(Q_to_img_att_list, C_to_img_att, track_proposal=targets['proposal_mask'].copy(), grid_size=grid_size)
        # ttt_loss = outputs.scores[0][0][3582] - outputs.scores[0][0][1217]
        # vprompt_history = vprompt_cur
        # grad_cond = torch.autograd.grad(ttt_loss.requires_grad_(True), [visual_prompt], retain_graph=True, allow_unused=False)[0]
        # visual_prompt = visual_prompt - args.lr * grad_cond
        
        # tokenizer.convert_tokens_to_ids('yes') # 3582
        # tokenizer.batch_decode([3582], skip_special_tokens=True)[0].strip()
        get_local.clear()
        torch.cuda.empty_cache()
        
    breakpoint()
    return bbox_pred_tmp, temporal_pred_tmp



def llava_embedd_tuning(cfg_pretrained, tokenizer, model, image_processor, video, questions, grid_size, targets, args, vid_i):

    contra_cap = "please describe this video"   # 这里 caption 的内容会影响结果
    # print(questions)
    # breakpoint()
    # print('contra_cap: ', contra_cap)
    questions_list = questions
    temporal_question_list = questions #["Is there the man kisses the woman and turns away in this video"]
    spatial_question_list = questions #["Is there the black-clothed man in this video?"]
    # questions_list = [
    #     # "Is there the man in the gray suit turns his head or not in this video ?", \
    #     "Is the man in the gray suit or not in this video ?", \
    #     "Is the person a man or not in this video ?",
    #     ]
    
    # # ===================================== step 1. 预先准备全部 blur_track 的 video_embedding
    inputs_blur_track_embeds = []
    inputs_blur_temporal_embedds = []
    with torch.no_grad():
        # ============================ blur track
        for track_k in range(min(len(targets['proposal_bboxes']), 2)):
            video_blur_proposal, blur_images = blur_proposal_video_array(video, targets, blur_propsal_id=track_k)
            video_proc = image_processor.preprocess(video_blur_proposal, return_tensors="pt")["pixel_values"].half().cuda()
            video_batch = [video_proc]
            
            prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids_C, 
                position_ids=None, 
                attention_mask=attention_masks_C, 
                past_key_values=None,
                labels=None, 
                images=video_batch, 
                modalities="video",  
                image_sizes=None
                )
            F_n = video_batch[0].size(0)
            v_n = grid_size*grid_size * F_n
            img_token_idx = int(torch.where(input_ids_C == -200)[1])
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
            inputs_blur_track_embeds.append(img_token_embedds)
        
        # ============================ blur temporal segment
        bins = np.linspace(0, F_n, F_n//5 + 1).astype(int)  # 生成分段的边界点
        
        for bin_id in range(len(bins)-1):
            video_blur_temporal, blur_images = blur_temp_video_array(video, targets, blur_interval=[bins[bin_id], bins[bin_id+1]])
            video_proc = image_processor.preprocess(video_blur_temporal, return_tensors="pt")["pixel_values"].half().cuda()
            video_batch = [video_proc]
            
            prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids_C, 
                position_ids=None, 
                attention_mask=attention_masks_C, 
                past_key_values=None,
                labels=None, 
                images=video_batch, 
                modalities="video",  
                image_sizes=None
                )
            F_n = video_batch[0].size(0)
            v_n = grid_size*grid_size * F_n
            img_token_idx = int(torch.where(input_ids_C == -200)[1])
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            pre_img_token_embedds = inputs_embeds_q[:, :img_token_idx, :]
            inputs_blur_temporal_embedds.append(img_token_embedds)

    
    # ===================================== step 2. 每一次 ttt 推理 需要 generate N_q x N_track 次
    # =========================== step 2.1 Temproal tuning
    temporal_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
    temporal_prompt_history = temporal_prompt
    tokens_format = [img_token_idx, img_token_idx+v_n, v_n]
    temporal_logit_pred = temporal_llm_tuning(
                model, temporal_question_list, 
                pre_img_token_embedds, inputs_blur_temporal_embedds, 
                temporal_prompt, temporal_prompt_history, 
                grid_size, 
                tokens_format, 
                bins, stopping_criteria_, tokenizer, args
            )
    temporal_pred_bin_id = temporal_logit_pred.item()
    
    # =========================== step 2.2 Spatial tuning
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(img_token_embedds))
    spatial_prompt_history = spatial_prompt
    spatial_logit_pred = spatial_llm_tuning(
                        model,
                        spatial_question_list,
                        pre_img_token_embedds, inputs_blur_track_embeds,
                        spatial_prompt,
                        spatial_prompt_history,
                        grid_size,
                        tokens_format,
                        bins,
                        stopping_criteria_,
                        tokenizer,
                        args
                    )
    spatial_pred_track_id = spatial_logit_pred.item()
    
    blur_pred = {}
    blur_pred['Time_pred'] = [bins[temporal_pred_bin_id], bins[temporal_pred_bin_id+1]]
    blur_pred['Space_pred'] = spatial_pred_track_id
    
    # ===================================== step 3.  ttt 更新完毕，重新推理 
    # 同时 blur temporal 和 spatial
    
    for mode in ['']: #['s', 't', 'st', '']:
        blur_mode = mode
        blur_video_st, blur_st_images = blur_st_video_array(video, targets, blur_propsal_id=spatial_pred_track_id, \
            blur_interval=[bins[temporal_pred_bin_id], bins[temporal_pred_bin_id+1]], mode=blur_mode)
        # targets['pil_images'] = blur_st_images
        contra_cap = "please describe this video"   # 这里 caption 的内容会影响结果
        # ===================================== 提取当前 ttt_epoch 的 contrastive_attention
        with torch.no_grad():
            video_proc = image_processor.preprocess(blur_video_st, return_tensors="pt")["pixel_values"].half().cuda()
            video_batch = [video_proc]
            
            prompt_, input_ids_C, keywords_, stopping_criteria_, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            (inputs_, position_ids_q, attention_mask_q, _, inputs_embeds_q, _) = model.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids_C, 
                position_ids=None, 
                attention_mask=attention_masks_C, 
                past_key_values=None,
                labels=None, 
                images=video_batch, 
                modalities="video",  
                image_sizes=None
                )
            img_token_embedds = inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :]
            
            prompt_, input_ids_C, keywords_, stopping_criteria_C, attention_masks_C, tokenizer_ = Q_to_prompt(contra_cap, args, model, tokenizer)
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_C[:, img_token_idx+1:])
            inputs_embeds_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
            position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embeds_q.shape[1])).to(inputs_embeds_q.device)
        
            new_inputs_embeds_c = inputs_embeds_q
            # new_inputs_embeds_c[:, img_token_idx:img_token_idx + v_n, :] += visual_prompt#[logit_pred]
            output_ids = model.generate(
                inputs=input_ids_C, 
                images=None, 
                attention_mask=attention_mask_q, 
                inputs_embeds=new_inputs_embeds_c,
                position_ids=position_ids_q,
                modalities="video", 
                do_sample=False, 
                temperature=0.0, 
                max_new_tokens=40, 
                top_p=0.1,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria_C],
                output_scores=True,
                return_dict_in_generate=True
            )
            C_to_img_att = get_Q_att(video_batch, input_ids_C, grid_size)
        
            for q_id, Qe in enumerate(questions_list[:]):
                # ===================================== load prompt
                prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
                manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
                inputs_embeds_q = torch.cat([pre_img_token_embedds, img_token_embedds, manual_post_img_token_embedds], dim=1)
                position_ids_q, attention_mask_q = None, torch.ones((1, inputs_embeds_q.shape[1])).to(inputs_embeds_q.device)
                
                new_inputs_embeds_q = inputs_embeds_q
                # new_inputs_embeds_q[:, img_token_idx:img_token_idx + v_n, :] += visual_prompt#[logit_pred]
                outputs = model.generate(
                    inputs=input_ids_Q, 
                    images=None, 
                    attention_mask=attention_mask_q, 
                    inputs_embeds=new_inputs_embeds_q,
                    position_ids=position_ids_q,
                    modalities="video", 
                    do_sample=False, 
                    temperature=0.0, 
                    max_new_tokens=40, 
                    top_p=0.1,
                    num_beams=1,
                    use_cache=True, 
                    stopping_criteria=[stopping_criteria_Q],
                    output_scores=True,
                    return_dict_in_generate=True,
                    # output_attentions=True,
                )
                Q_to_img_att = get_Q_att(video_batch, input_ids_Q, grid_size)
                # Q_to_img_att_list.append(Q_to_img_att)
                rela = Q_to_img_att[-5] / (C_to_img_att[-5] + 0.0001)
                # rela = C_to_img_att[-5] / (Q_to_img_att[-5] + 0.0001)
                
                generate_ids = outputs.sequences    # requires_grad
                scores = outputs.scores[0].detach().cpu().numpy()
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
                # print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
                
                # ============== 当前 ttt_epoch : 可视化当前 ttt_epoch 的结果
                contrastive_content  = f'blur_{blur_mode}_yes={scores[0][3582]:.2f}_no={scores[0][1217]:.2f}_{temporal_question_list[0][10:120]}_{spatial_question_list[0][0:0]}_'
                bbox_pred_tmp, temporal_pred_tmp = ttt_eval(targets, rela, vid_i, preprocess, contrastive_content, blur_pred, args)
    
    # bbox_pred_tmp, temporal_pred_tmp, max_track_iou = ttt_eval(targets, None, vid_i, preprocess, "", blur_pred, args)
    # breakpoint()
    return bbox_pred_tmp, temporal_pred_tmp

def temporal_llm_tuning(
    model,
    temporal_questions,
    pre_img_token_embedds, inputs_blur_temporal_embedds,
    temporal_prompt,
    temporal_prompt_history,
    grid_size,
    tokens_format,
    bins,
    stopping_criteria_Q,
    tokenizer,
    args
    ):
    
    token_bins = [(bins[bk]*grid_size**2, bins[bk+1]*grid_size**2) for bk in range(len(bins)-1)]
    img_token_start, img_token_end, tokens_N = tokens_format
    zero_mask = torch.zeros((1, tokens_N))
    token_masks = []
    for bin in token_bins:
        zero_mask_copy = zero_mask.clone()
        zero_mask_copy[img_token_start+bin[0] : img_token_start+bin[1]] = 1.
        token_masks.append(zero_mask_copy.unsqueeze(dim=-1))
    
    for ep in range(1):
        Temp_prompt_cur = args.beta * temporal_prompt + (1 - args.beta) * temporal_prompt_history  # EMA
        
        T_segment_score = []
        for q_id, Qe in enumerate(temporal_questions[:]):
            T_segment_score_Q = []
            
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
            
            for k in range(len(inputs_blur_temporal_embedds)):
                inputs_blur_embed_q = torch.cat([pre_img_token_embedds, inputs_blur_temporal_embedds[k], manual_post_img_token_embedds], dim=1)
                position_ids_q, attention_mask_q = None, torch.ones((1, inputs_blur_embed_q.shape[1])).to(inputs_blur_embed_q.device)
                
                new_inputs_blur_embed_q = inputs_blur_embed_q
                # 1 x v_n x 4096,   1 x v_n x 1
                new_inputs_blur_embed_q[:, img_token_start:img_token_end, :] += (Temp_prompt_cur* token_masks[k].to(inputs_blur_embed_q.device))
                outputs = model.generate(
                    inputs=None, 
                    images=None, 
                    attention_mask=attention_mask_q, 
                    inputs_embeds=new_inputs_blur_embed_q,
                    position_ids=position_ids_q,
                    modalities="video", 
                    do_sample=False, 
                    temperature=0.0, 
                    max_new_tokens=40, 
                    top_p=0.1,
                    num_beams=1,
                    use_cache=True, 
                    stopping_criteria=[stopping_criteria_Q],
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                generate_ids = outputs.sequences    # requires_grad
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
                scores = outputs.scores[0].detach().cpu().numpy()
                # print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
                yes_score, no_score = outputs.scores[0][0][3582], outputs.scores[0][0][1217]
                T_segment_score_Q.append(torch.stack([yes_score, no_score]))

                get_local.clear()
                torch.cuda.empty_cache()
                
            T_segment_score.append(torch.stack(T_segment_score_Q))
        
        T_segment_score = torch.stack(T_segment_score)   # Q_n x  Segment_N x 2
        # breakpoint()
        temporal_loss, temporal_logit_pred = Q_temporal_loss(T_segment_score, grid_size=grid_size)
        temporal_loss = temporal_loss *  args.alpha
        temporal_logit_pred = temporal_logit_pred.cpu()
        args.lr = 0.0
        
        temporal_prompt_history = temporal_prompt
        # grad_cond = torch.autograd.grad(temporal_loss.requires_grad_(True), [temporal_prompt], retain_graph=True, allow_unused=False)[0]
        # temporal_prompt = temporal_prompt - args.lr * grad_cond
        return temporal_logit_pred
    
    

def spatial_llm_tuning(
    model,
    temporal_questions,
    pre_img_token_embedds, inputs_blur_track_embeds,
    spatial_prompt,
    spatial_prompt_history,
    grid_size,
    tokens_format,
    bins,
    stopping_criteria_Q,
    tokenizer,
    args
    ):
    
    # token_bins = [(bins[bk]*grid_size**2, bins[bk+1]*grid_size**2) for bk in range(len(bins)-1)]
    img_token_start, img_token_end, tokens_N = tokens_format
    # zero_mask = torch.zeros((1, tokens_N))
    # token_masks = []
    # for bin in token_bins:
    #     zero_mask_copy = zero_mask.clone()
    #     zero_mask_copy[img_token_start+bin[0] : img_token_start+bin[1]] = 1.
    #     token_masks.append(zero_mask_copy.unsqueeze(dim=-1))
    
    for ep in range(1):
        spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
        track_score = []
        for q_id, Qe in enumerate(temporal_questions[:]):
            track_score_Q = []
            
            prompt, input_ids_Q, keywords, stopping_criteria_Q, attention_masks_Q, tokenizer = Q_to_prompt(Qe, args, model, tokenizer)
            img_token_idx = int(torch.where(input_ids_Q == -200)[1])
            manual_post_img_token_embedds = model.get_model().embed_tokens(input_ids_Q[:, img_token_idx+1:])
            
            for k in range(len(inputs_blur_track_embeds)):
                inputs_blur_embed_q = torch.cat([pre_img_token_embedds, inputs_blur_track_embeds[k], manual_post_img_token_embedds], dim=1)
                position_ids_q, attention_mask_q = None, torch.ones((1, inputs_blur_embed_q.shape[1])).to(inputs_blur_embed_q.device)
                
                new_inputs_blur_embed_q = inputs_blur_embed_q
                # 1 x v_n x 4096,   1 x v_n x 1
                new_inputs_blur_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
                outputs = model.generate(
                    inputs=None, 
                    images=None, 
                    attention_mask=attention_mask_q, 
                    inputs_embeds=new_inputs_blur_embed_q,
                    position_ids=position_ids_q,
                    modalities="video", 
                    do_sample=False, 
                    temperature=0.0, 
                    max_new_tokens=40, 
                    top_p=0.1,
                    num_beams=1,
                    use_cache=True, 
                    stopping_criteria=[stopping_criteria_Q],
                    output_scores=True,
                    return_dict_in_generate=True,
                )
                generate_ids = outputs.sequences    # requires_grad
                output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
                scores = outputs.scores[0].detach().cpu().numpy()
                # print(f'yes_logit = {scores[0][3582]:.2f}, no_logit = {scores[0][1217]:.2f}, response ====> {output}')
                yes_score, no_score = outputs.scores[0][0][3582], outputs.scores[0][0][1217]
                track_score_Q.append(torch.stack([yes_score, no_score]))
            
                get_local.clear()
                torch.cuda.empty_cache()
        
            track_score.append(torch.stack(track_score_Q))
        
        track_score = torch.stack(track_score)   # Q_n x  Segment_N x 2
        
        spatial_loss, spatial_logit_pred = Q_spatial_loss(track_score, grid_size=grid_size)
        spatial_loss = spatial_loss *  args.alpha
        sptial_logit_pred = spatial_logit_pred.cpu()
        args.lr = 0.0
        
        temporal_prompt_history = spatial_prompt
        # grad_cond = torch.autograd.grad(temporal_loss.requires_grad_(True), [spatial_prompt], retain_graph=True, allow_unused=False)[0]
        # temporal_prompt = temporal_prompt - args.lr * grad_cond
        return sptial_logit_pred
    