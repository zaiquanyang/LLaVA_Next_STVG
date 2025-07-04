import json
import os

import torch
from PIL import Image
import numpy as np
import random 
import torch.nn.functional as F

from custom_infer.STVG.InternVL_Tuning.conversation import get_conv_template
from custom_infer.utils import Q_to_prompt, get_Q_att, ttt_eval
from custom_infer.STVG.InternVL_Tuning.spatial_tuning import spatial_llm_att, spatial_llm_logit


import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def prepare_internVL_embedds(model, video, input_ids, visual_token_id, video_embedds=None):
    
    transform = build_transform(input_size=448)
    video = [transform(tile) for tile in video]
    video = torch.stack(video).to(torch.bfloat16).cuda()
    
    if not video_embedds:
        with torch.no_grad():
            video_embedds = model.extract_feature(video)
    with torch.no_grad():
        input_embeds = model.language_model.get_input_embeddings()(input_ids)
        
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == visual_token_id)
    assert selected.sum() != 0
    input_embeds[selected] = video_embedds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)
            
    
    return video_embedds


def prepare_input_ids(model, tokenizer, question, generation_config, num_patches_list, args):
    
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
    assert tokenizer.convert_tokens_to_ids('yes') == args.yes_id, print("the <yes> tokens id is not setting right ?")
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    
    # print(query)
    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * int(num_patches) + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
    
    # breakpoint()
    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)
    generation_config['eos_token_id'] = eos_token_id
    
    return input_ids, attention_mask, generation_config

def llava_embedd_tuning(cfg_pretrained, tokenizer, model, processor, video, questions, grid_size, targets, args, vid_i, logger):

    video_tensor = torch.Tensor(video).to(torch.bfloat16).cuda().permute(0, 3, 1, 2)
    frames_N, _, H, W = video_tensor.shape
    video_pil = targets['pil_images']
    
    num_patches_list = np.ones(frames_N).tolist()
    
    generation_config = dict(max_new_tokens=5, do_sample=True, output_attentions=True, return_dict_in_generate=True)
    input_ids, attention_mask, generation_config = prepare_input_ids(model, tokenizer, 'please describe this video', generation_config, num_patches_list, args)
    
    # # =============================== prepare input_embedds for InternVL
    visual_token_id = 151667#92546
    video_embeds = prepare_internVL_embedds(model, video_pil, input_ids, visual_token_id)
    f_n, patch_hw, c_dim = video_embeds.shape
    
    
    llm_reasoning = True
    if llm_reasoning:
        # =============================== prepare input_embedds for InternVL
        spatial_prompt = torch.nn.Parameter(torch.zeros_like(video_embeds.reshape(1, -1, c_dim)))
        spatial_prompt_history = spatial_prompt
        
        if args.ttt_epoch > 0:
            tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_tuning')
            logger.info(f'focus_attribute_tuning:  {tmp_questions}')
            torch.cuda.empty_cache()
            spatial_prompt, spatial_prompt_history = spatial_llm_logit(
                                                    model, 
                                                    tmp_questions.copy(), 
                                                    None, 
                                                    video_embeds, 
                                                    None,
                                                    spatial_prompt, 
                                                    spatial_prompt_history, 
                                                    None, 
                                                    video, 
                                                    generation_config,
                                                    num_patches_list,
                                                    args.ttt_epoch,
                                                    tokenizer, args, processor,
                                                    args.s_lr
                                                    )
        
        tmp_questions = prepare_instructions(questions, targets, mode='focus_attribute_infer')
        logger.info(f'focus_attribute_infer:  {tmp_questions}')
        
    
        llava_att_pred_track_id, llava_rela_att, all_tokens_rela_att = spatial_llm_att(
                    model,
                    tmp_questions.copy(),
                    None,
                    video_embeds,
                    None,
                    spatial_prompt,
                    spatial_prompt_history,
                    None,
                    video,
                    None, 
                    targets,
                    generation_config,
                    num_patches_list,
                    tokenizer, args, processor
                )
        spatial_pred_track_id = llava_att_pred_track_id.item()
        torch.cuda.empty_cache()
    
        contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
        spatial_pred = {'Space_pred': spatial_pred_track_id}
        bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=True)
        print('Evaluate and Visualize the llava_att on spatial prediction done!')
    
    else:
        pass
    return bbox_pred_tmp, temporal_pred_tmp, None



def prepare_instructions(ref_sent, targets, mode='tuning_spatial_loc'):
    
    ref_Q = f"Is there {ref_sent.strip('.').strip()} in this video?"
    gpt_4_proc_attr = targets['gpt_4_proc_attr']            # 默认解耦出一个
    gpt_4_proc_action = targets['gpt_4_proc_action']        # 默认解耦出一个，当 考虑sub-actions的时候考虑多个
    # gpt_4_proc_sub_actions = targets['gpt_4_proc_sub_actions']
    
    attr_Q, action_Q = ["please describe this video"], ["please describe this video"]
    decompose_fail = False
    for sent in gpt_4_proc_attr:
        if "please describe this video" in sent:
            attr_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]#["please describe this video"]
            decompose_fail = True
        else:
            attr_Q = [f"Is there {sent.strip('.').strip()} in this video?"]

    for sent in gpt_4_proc_action:
        if "please describe this video" in sent:
            action_Q = [f"Is there {ref_sent.strip('.').strip()} in this video?"]#["please describe this video"]
            decompose_fail = True
        else:
            action_Q = [f"Is there {sent.strip('.').strip()} in this video?"]
    # assert decompose_fail == False, print('please setting decompose_fail right !')
    
            
    tmp_questions = []
    
    if mode == "focus_attribute_tuning":
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
            tmp_questions = attr_Q
            tmp_questions.append("please describe this video")
            # Method 2.   attr_Q - action_Q
            # tmp_questions = [ref_Q]
            # tmp_questions.append("please describe this video")
        return tmp_questions
    else:
        breakpoint()