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

# from visualizer import get_local
# get_local.activate()

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

from stvg.datasets.evaluation.hcstvg_eval import evaluation_llava_att, evaluation_blur_pred
# from stvg.vis import show_image_relevance, show_video_relevance, norm_video_att, original_resize_336, \
#     blur_proposal_video_array, blur_gt_video_array, blur_temp_video_array, blur_st_video_array
# from stvg.loss import Q_spatial_loss, contrastive_logit_adjust, Q_temporal_loss

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




def get_Q_att(video_fn, ori_attention_maps, input_ids, grid_size, args=None, cfg_pretrained=None):
    
    # ori_attention_maps = get_local.cache['sdpa_attention_forward'] #get_local.cache['LlamaSdpaAttention.forward'], sdpa_attention_forward
    # get_local.clear()
    # torch.cuda.empty_cache()
    ori_attention_maps = [torch.stack(ori_attention_map).squeeze() for ori_attention_map in ori_attention_maps]
    
    video_fn = int(video_fn)
    prompt_n = input_ids.size(-1)
    
    if cfg_pretrained.mm_newline_position == "one_token":
        v_n = grid_size*grid_size * video_fn + 1
        # v_n = v_n -1
    elif cfg_pretrained.mm_newline_position == "no_token":
        v_n = grid_size*grid_size * video_fn + 0
    else:
        breakpoint()
    llm_in_tokens_N = v_n +  prompt_n - 1 
    img_token_idx = int(torch.where(input_ids == -200)[1])
    
    # ========================================== question_attention_maps
    question_attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] == llm_in_tokens_N]
    question_to_img_att_ = torch.cat([att.to('cuda') for att in question_attention_maps], 0)            # Layers_N x head_N x llm_in_tokens_N x llm_in_tokens_N

    question_to_img_att = question_to_img_att_[:, :, (img_token_idx + v_n - 1):, img_token_idx:(img_token_idx + v_n)] #  Q_L x Patch_n
    
    
    if cfg_pretrained.mm_newline_position == "one_token":
        question_to_img_att = question_to_img_att[:, :, :, :-1]
        # 因为之前已经取消了 last_one_token的输入，所以这里没必要再调整
        # question_to_img_att = question_to_img_att[:, :, :, :]
    elif cfg_pretrained.mm_newline_position == "no_token":
        question_to_img_att = question_to_img_att[:, :, :, :]
    else:
        breakpoint()
        
    Q_to_img_att = question_to_img_att.reshape(args.layers_N, args.heads_N, -1, video_fn, grid_size*grid_size)
    
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
    rela_att = Q_to_img_att_list.detach()
    
    bbox_pred_tmp, temporal_pred_tmp, pred_track_ID, pred_actionness, v_iou, gt_viou = evaluation_llava_att(
        llava_pred=Q_to_img_att_list.detach(), 
        actioness=actioness, 
        gt_boxs=boxs.clone(),                   # 注意这里要 clone，因为evaluation_single会修改传入的gt_boxs
        proposal_mask=proposal_mask, 
        proposal_bboxes=proposal_bboxes,
        vid_i=vid_i, 
        targets=targets,
        blur_pred=blur_pred,
        args=args
    )

    save = False
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
                bboxes=(boxs * (224/args.img_size)).numpy().astype(int),
                pred_bboxes=(proposal_bbox_w_action * (224/args.img_size)).astype(int),
                )
        vid_rela.save(f'{output_dir}/{vid_i}_{gt_viou}_{st_iou}_{contrastive_content[:120]}.png')
        return bbox_pred_tmp, temporal_pred_tmp, gt_viou
    else:
        gt_viou = int(np.round(gt_viou, 2)*100)
        return bbox_pred_tmp, temporal_pred_tmp, gt_viou

