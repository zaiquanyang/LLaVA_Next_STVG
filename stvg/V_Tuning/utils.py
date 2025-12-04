import os
import torch

def Strainght_through_Gumbel(logits):
    
    # sample = torch.rand_like(logits, requires_grad=True)
    argmax = torch.zeros_like(logits)
    argmax_one_hot = argmax.scatter(-1, torch.argmax(logits, dim=-1).unsqueeze(dim=-1), 1)
    
    C = argmax_one_hot - torch.softmax(logits, dim=-1)
    C = C.detach()
    
    SG_logits = C + torch.softmax(logits, dim=-1)
    
    return SG_logits


def prepare_instructions(ref_sent, targets, mode='tuning_spatial_loc'):
    
    ref_Q = f"Is there {ref_sent.strip('.').strip()} in this video?"
    gpt_4_proc_attr = targets['gpt_4_proc_attr']            # 默认解耦出一个
    gpt_4_proc_action = targets['gpt_4_proc_action']        # 默认解耦出一个，当 考虑sub-actions的时候考虑多个
    attr_Q, action_Q = "please describe this video", "please describe this video"
    decompose_fail = False
    for sent in gpt_4_proc_attr:
        if "please describe this video" in sent:
            attr_Q = "please describe this video"
            decompose_fail = True
        else:
            attr_Q = f"Is there {sent.strip('.').strip()} in this video?"

    for sent in gpt_4_proc_action:
        if "please describe this video" in sent:
            action_Q = "please describe this video"
            decompose_fail = True
        else:
            action_Q = f"Is there {sent.strip('.').strip()} in this video?"
    assert decompose_fail == False, print('please setting decompose_fail right !')
    # breakpoint()
    tmp_questions = []
    
    
    if mode == 'whole_tuning':
        tmp_questions = ["please describe this video", ref_Q]
        return tmp_questions
    elif mode == 'whole_infer':
        tmp_questions = [ref_Q, "please describe this video"]
        return tmp_questions
    
    elif mode == "tuning_spatial_loc":
        if "please describe this video" in attr_Q:          # 当前这个解耦无效-fail
            tmp_questions = ["please describe this video", ref_Q]
        else:
            tmp_questions = [action_Q, ref_Q]                                                   # ref_Q - action_Q
            # tmp_questions = ["please describe this video", attr_Q]                              # attr_Q - ""
        return tmp_questions
    elif mode == "infer_spatial_loc":
        if "please describe this video" in attr_Q:          # 当前这个解耦无效-fail
            tmp_questions = [ref_Q, "please describe this video"]
        else:
            # Method 1.     attr_Q - " "
            # tmp_questions = [attr_Q, "please describe this video"]
            # Method 2.   attr_Q - action_Q
            tmp_questions = [attr_Q, action_Q]
        return tmp_questions
     
    elif mode == "tuning_temporal_loc":
        if "please describe this video" in action_Q:          # 当前这个解耦无效-fail
            tmp_questions = ["please describe this video", ref_Q]
        else:
            tmp_questions = [attr_Q, ref_Q]                                                 # ref_Q - attr_Q
            # tmp_questions = ["please describe this video", action_Q]                        # action_Q - " "
        return tmp_questions
    elif mode == "infer_temporal_loc":
        if "please describe this video" in action_Q:          # 当前这个解耦无效-fail
            tmp_questions = [ref_Q, "please describe this video"]
        else:
            # Method 1. 
            tmp_questions = [action_Q, "please describe this video"]
            # Method 2.
            # tmp_questions = [action_Q, attr_Q]
            # Method 3.
            # tmp_questions = [ref_Q, "please describe this video"]
        return tmp_questions