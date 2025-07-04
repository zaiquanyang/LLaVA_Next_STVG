import matplotlib.pyplot as plt
import numpy as np
import os

def vis_llava_att_dist(dir):
    llava_att_dir = os.path.join(dir, '_llava_att_npy')
    llava_att_dist = np.zeros(20)
    gt_att_dist = np.zeros(20)

    print(len(os.listdir(llava_att_dir)))
    for file_name in os.listdir(llava_att_dir):
        att_attention_f = os.path.join(llava_att_dir, file_name)
        
        llava_dict = np.load(att_attention_f, allow_pickle=True).item()
        att_attention = llava_dict['T_LLaVa_att']
        att_attention_max = np.max(att_attention, axis=-1)#.reshape(1, -1)
        att_attention_max = att_attention_max / att_attention_max.max()
        llava_att_dist += att_attention_max

        if 'GT_Action' in  list(llava_dict.keys()):
            gt_att_dist += llava_dict['GT_Action']

    llava_att_dist = llava_att_dist / llava_att_dist.sum()
    gt_att_dist = gt_att_dist / (gt_att_dist.sum() + 0.001)
    
    return llava_att_dist, gt_att_dist


# wo_tuning_at_ac_at_ac = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0307_ignore_decom_fail/focus_attr_reverse_vid_ep_0_lr_0.0_0.0_at-ac_at-ac')
# wo_tuning_at_cap_at_cap = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0307_ignore_decom_fail/focus_attr_reverse_vid_ep_0_lr_0.0_0.0_at-cap_at-cap')
act_wo_tuning, _ = vis_llava_att_dist('custom_infer/STVG/output/hc_0317_decom_yes_1k/llava_OV_f_20_pool_5/focus_act_-3_0_lr_0.0_0.0_(actionQ_cap)')
act_w_tuning, gt_att_dist = vis_llava_att_dist('custom_infer/STVG/output/hc_0317_decom_yes_1k/llava_OV_f_20_pool_5/focus_act_-3_2_lr_0.0_8.0_(actionQ_cap)')

# wo_tuning_at_ac_at_ac = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0308_decom_yes_2k/focus_attr_reverse_vid_ep_0_lr_0.0_0.0_at-ac_at-ac')
# wo_tuning_at_none_at_none = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0308_decom_yes_2k/focus_attr_reverse_vid_ep_0_lr_0.0_0.0_at-none_at-none')


# print(wo_tuning_ref_cap_ref_cap)

x = np.arange(20)  # x轴对应的索引或数据

# 创建折线图
plt.plot(x, act_wo_tuning*100, label='act_wo_tuning', marker='o', linestyle='-', color='b')  # 折线1
plt.plot(x, act_w_tuning*100, label='act_w_tuning', marker='s', linestyle='--', color='r')  # 折线2
plt.plot(x, gt_att_dist*100, label='gt_att_dist', marker='s', linestyle='--', color='y')  # 折线2
# plt.plot(x, wo_tuning_ref_cap_ref_cap*100, label='wo_tuning_ref_cap_ref_cap', marker='s', linestyle='--', color='g')  # 折线2
# plt.plot(x, wo_tuning_ref_none_ref_none*100, label='wo_tuning_ref_none_ref_none', marker='s', linestyle='--', color='b')  # 折线2
# plt.plot(x, focus_act_ac_cap_ac_cap*100, label='focus_act_ac_cap_ac_cap', marker='s', linestyle='--', color='r')  # 折线2

# 添加标题和标签
plt.title("frames hit dist")
plt.xlabel("frames_idx")
plt.ylabel("hit_ratio")

# 添加图例
plt.legend()

# 显示网格（可选）
# plt.grid(True)
plt.savefig("/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/custom_infer/llava_dist.png", dpi=300, bbox_inches="tight", transparent=False)
# 显示图形
# plt.show()