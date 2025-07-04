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


# def spatial_llm_att(
#     model,
#     spatial_questions,
#     pre_img_token_embedds, 
#     video_embeds,
#     img_token_embedds_w_reverse,
#     spatial_prompt,
#     spatial_prompt_history,
#     grid_size,
#     video_batch,
#     tokens_format,
#     targets,
#     stopping_criteria_Q,
#     tokenizer,
#     args,
#     processor,
#     ret_score=False
#     ):
    
#     # spatial_questions.append('please describe this video')
#     img_token_start, img_token_end, tokens_N = tokens_format
    
#     scores = []
#     for ep in range(1):
#         spatial_prompt_cur = args.beta * spatial_prompt + (1 - args.beta) * spatial_prompt_history  # EMA
        
#         Q_to_img_att_list = []
#         for q_id, Qe in enumerate(spatial_questions[:]):
            
#             cur_inputs = prepare_inputs(processor, tokenizer, Qe, video_batch)
#             input_ids, pixel_values_videos, video_grid_thw = cur_inputs['input_ids'], cur_inputs['pixel_values_videos'], cur_inputs['video_grid_thw']
#             video_embeds, pre_video_embedds, post_video_embedds, attention_mask, _ = prepare_qwen_embedds(model, cur_inputs, args.visual_token_id, video_embeds)
#             inputs_embed_q = torch.cat([pre_video_embedds, video_embeds, post_video_embedds], dim=1)
#             attention_mask_q = attention_mask.to(inputs_embed_q.device)
#             print(tokenizer.convert_ids_to_tokens(input_ids[0][-20:]))
            
#             new_inputs_embed_q = inputs_embed_q
#             # 1 x v_n x 4096,   1 x v_n x 1
#             new_inputs_embed_q[:, img_token_start:img_token_end, :] += spatial_prompt_cur
#             with torch.no_grad():
#                 outputs = model.generate(
#                             inputs_embeds=new_inputs_embed_q, 
#                             attention_mask=attention_mask_q, 
#                             max_new_tokens=10,
#                             use_cache=True,
#                             # output_scores=True,
#                             output_logits=True,
#                             return_dict_in_generate=True,
#                         )
#             generate_ids = outputs.sequences    # requires_grad
#             output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
#             print('>>> LLM output : ', output, 'logit_yes_no_score: ', outputs.logits[0][0][args.yes_id], outputs.logits[0][0][args.no_id])
#             # breakpoint()
#             # scores.append([outputs.scores[0][0][args.yes_id].cpu().numpy(), outputs.scores[0][0][args.no_id].cpu().numpy()])
#             # tokenizer.convert_ids_to_tokens(input_ids_Q[0][-7:])
#             # breakpoint()
#             ori_attention_maps = get_local.cache['Qwen2VLSdpaAttention.forward'] # sdpa_attention_forward, Qwen2VLSdpaAttention.forward
#             get_local.clear()
#             torch.cuda.empty_cache()
            
#             question_attention_maps = [att for i, att in enumerate(ori_attention_maps) if att.shape[-2] > 1]
#             question_to_img_att = torch.cat([att.cpu() for att in question_attention_maps], 0)
#             question_to_img_att = question_to_img_att[:, :, -6:, img_token_start:(img_token_start + tokens_N)]
#             Q_to_img_att_list.append(question_to_img_att.mean(dim=0).mean(dim=0))
        
#         video_grid_thw_ = video_grid_thw[0].cpu().numpy()
#         rela_att = Q_to_img_att_list[0] / (Q_to_img_att_list[1] + 0.0001)
        
#         all_tokens_rela_att = rela_att

#         # rela_att_max_v = rela_att.reshape(6, -1).max(dim=-1)[0]
#         # indices = torch.argsort(rela_att_max_v, descending=True)
#         rela_att =  rela_att[args.token_idx]
        
#         rela_att = rela_att.reshape(video_grid_thw_[0], -1)
#         f_n, token_N = rela_att.shape
        
#         proposal_m = F.interpolate(torch.Tensor(targets['proposal_mask'].copy()), size=(int(video_grid_thw_[1]//2), int(video_grid_thw_[2]//2)), mode='bilinear', align_corners=False)
#         proposal_m = proposal_m.reshape(-1, f_n, token_N)
#         proposal_m = proposal_m.to(rela_att.device)
#         att_each_proposal = rela_att.unsqueeze(dim=0) * proposal_m
#         # breakpoint()
#         # ============== 计算每个track的最大激活值, 查找哪个 track 具有最大的激活值
#         max_att_each_proposal, _ = att_each_proposal.reshape(-1, f_n*token_N).max(dim=-1)
#         max_att_proposal_id = torch.argmax(max_att_each_proposal)

#     return max_att_proposal_id, rela_att, all_tokens_rela_att




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

    # with open("custom_infer/STVG/output/oracle_track_id.json", "r", encoding="utf-8") as f:
    #     oracle_track_ids = json.load(f)
    # # breakpoint()
    # track_id = oracle_track_ids[str(vid_i)]
        
    frames_N, H, W, _ = video.shape
    patch_h, patch_w = H // 28, W // 28
    
    video_inputs = np.repeat(video, repeats=2, axis=0)        # 因为 Qwen2_VL会对输入的视频按时间步长=2进行下采样，所以我们需要先进行预处理
    
    inputs = prepare_inputs(processor, tokenizer, questions, video_inputs)
    input_ids, pixel_values_videos, video_grid_thw = inputs['input_ids'], inputs['pixel_values_videos'], inputs['video_grid_thw']
    
    # # =============================== prepare input_embedds for Qwen2_VL
    visual_token_id = 151656
    video_embeds, pre_video_embedds, post_video_embedds, attention_mask, tokens_format = prepare_qwen_embedds(model, inputs, visual_token_id)
    
    spatial_prompt = torch.nn.Parameter(torch.zeros_like(video_embeds))
    spatial_prompt_history = spatial_prompt
        
    inputs_embeds = torch.cat([pre_video_embedds, video_embeds, post_video_embedds], dim=1)
    tmp_questions = [f"Is there {questions.strip('.').strip()} in this video?", 'please describe this video']
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

    
    # original_h, original_w = targets['proposal_mask'].shape[-2:]
    # oracle_track = targets['proposal_bboxes'][track_id].copy()
    # oracle_track[:, 0] = oracle_track[:, 0] * args.img_size / original_w
    # oracle_track[:, 2] = oracle_track[:, 2] * args.img_size / original_w
    # oracle_track[:, 1] = oracle_track[:, 1] * args.img_size / original_h
    # oracle_track[:, 3] = oracle_track[:, 3] * args.img_size / original_h
    
    # oracle_mask = boxes_to_masks(oracle_track, (args.img_size, args.img_size))
    # oracle_fg_mask = F.interpolate(torch.Tensor(oracle_mask).reshape(1, frames_N, args.img_size, args.img_size), size=(patch_h, patch_w), mode='bilinear', align_corners=False)[0] > 0
    # oracle_bg_mask = ~oracle_fg_mask
    # llava_rela_att_336 = llava_rela_att.reshape(frames_N, patch_h, patch_w).to(torch.float32)
    
    # # all_tokens_rela_att
    # fg_max = (oracle_fg_mask.unsqueeze(dim=0)*all_tokens_rela_att.cpu().to(torch.float32).reshape(6, frames_N, patch_h, patch_w)).reshape(6, -1).max(axis=-1)[0]
    # bg_max = (oracle_bg_mask.unsqueeze(dim=0)*all_tokens_rela_att.cpu().to(torch.float32).reshape(6, frames_N, patch_h, patch_w)).reshape(6, -1).max(axis=-1)[0] + 0.0001
    # # count_tokens_N[np.argmax(fg_max / bg_max) -  args.tokens_N ] += 1
    # # breakpoint()
    # hit_token_id = (np.argmax(fg_max / bg_max) -  6).item()
    
    # # max_hit_in_fg = torch.max((llava_rela_att_336 * oracle_fg_mask).reshape(frames_N, -1), dim=-1)[0]           #
    # # max_hit_in_bg = torch.max((llava_rela_att_336 * oracle_bg_mask).reshape(frames_N, -1), dim=-1)[0]
    # # max_v_in_oracle, max_frame = torch.max(max_hit_in_fg, dim=0)
    # # max_v_in_bg = max_hit_in_bg[max_frame]
    
    
    contrastive_content  = f'track_{spatial_pred_track_id}_{questions[0:120]}_'
    spatial_pred = {'Space_pred': spatial_pred_track_id}
    bbox_pred_tmp, temporal_pred_tmp, gt_viou = ttt_eval(targets, llava_rela_att, vid_i, preprocess, contrastive_content, spatial_pred, args, save=False)
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
    # =========================================== easy mode 
    # with torch.no_grad():
    #     output = model(
    #         **inputs,
    #         output_attentions=True,
    #         )

    # attentions = output['attentions']
    # attentions = torch.stack(attentions).squeeze()
    
    # pos = inputs['input_ids'][0].tolist().index(visual_token_id)
    # img_tokens_num = (inputs['input_ids'][0]==visual_token_id).sum()
    # question_to_img_att  = attentions[:, :, :, pos:pos+img_tokens_num]
    
    
    return bbox_pred_tmp, temporal_pred_tmp#max_v_in_oracle/(max_v_in_bg+0.0001)
    # return question_to_img_att, video_inputs[0], patch_h, patch_w
