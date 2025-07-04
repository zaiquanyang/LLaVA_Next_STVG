import os
import numpy as np

dir = "/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/custom_infer/STVG/output/hc_0319_decom_yes_1k/llava_next_video_7b_f_20_pool_4/focus_attr_2_lr_8.0_0.0_attrQ_cap_attrQ_cap_record_proposal_score"

def vis_llava_att_dist(dir):
    llava_att_dir = os.path.join(dir, '_llava_att_npy')
    # llava_att_dist = np.zeros(20)
    print(len(os.listdir(llava_att_dir)))
    pre_tuning_proposal_score = []
    post_tuning_proposal_score = []
    
    oracle_id = []
    
    for file_name in os.listdir(llava_att_dir):
        att_attention_f = os.path.join(llava_att_dir, file_name)
        
        llava_dict = np.load(att_attention_f, allow_pickle=True).item()
        
        oracle_id.append(llava_dict['oracle_id'])
        pre_tuning_proposal_score.append(llava_dict['attr_for_track_score_pre'])
        post_tuning_proposal_score.append(llava_dict['attr_for_track_score_post'])
        
        # reverse_loss_list.append(reverse_loss)
    return oracle_id, pre_tuning_proposal_score, post_tuning_proposal_score


oracle_id, pre_tuning_proposal_score, post_tuning_proposal_score = vis_llava_att_dist(dir)

pre_hit_right = 0
post_hit_right = 0

for ii in range(len(oracle_id)):
    # breakpoint()
    if np.argmax(pre_tuning_proposal_score[ii]) == oracle_id[ii]:
        pre_hit_right += 1
    
    contrstive_score = post_tuning_proposal_score[ii] - pre_tuning_proposal_score[ii]
    if contrstive_score.max() > 100.0:
        proposal_score = contrstive_score
    else:
        proposal_score = post_tuning_proposal_score[ii]
    if np.argmax(proposal_score) == oracle_id[ii]:
        post_hit_right += 1
print(pre_hit_right, post_hit_right, len(oracle_id))