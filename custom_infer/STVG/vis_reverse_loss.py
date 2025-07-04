import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


def obtain_logit_pred_track():
    llava_att_dir = os.path.join('/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/custom_infer/STVG/output/blur_pred_focus_whole_divide_logit', '_llava_att_npy')
    GT_viou_dict = {}
    for npy_f in os.listdir(llava_att_dir):
        att_attention_f = os.path.join(llava_att_dir, npy_f)
        
        llava_dict = np.load(att_attention_f, allow_pickle=True).item()
        GT_viou = llava_dict['GT_viou']
        GT_viou_dict[npy_f.split('.')[0]] = GT_viou
    
    return GT_viou_dict
        
def vis_llava_att_dist(dir):
    llava_att_dir = os.path.join(dir, '_llava_att_npy')
    llava_att_dist = np.zeros(20)
    print(len(os.listdir(llava_att_dir)))
    GT_viou_dict = {}
    reverse_loss_list = []
    for file_name in os.listdir(llava_att_dir):
        att_attention_f = os.path.join(llava_att_dir, file_name)
        
        llava_dict = np.load(att_attention_f, allow_pickle=True).item()
        GT_viou = llava_dict['GT_viou']
        reverse_loss = llava_dict['reverse_loss']
        
        GT_viou_dict[file_name.split('.')[0]] = GT_viou
        reverse_loss_list.append(reverse_loss)
    # breakpoint()
    # sorted_indices = sorted(range(len(reverse_loss_list)), key=lambda i: reverse_loss_list[i])
    # GT_viou_list = [GT_viou_list[idx] for idx in sorted_indices]
    # reverse_loss_list = [reverse_loss_list[idx] for idx in sorted_indices]

    return GT_viou_dict, reverse_loss_list


blur_logit_pred = obtain_logit_pred_track()


logit_viou_dict, reverse_loss_list = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0308_decom_yes_2k/focus_attr_ep_2_lr_8.0_8.0_(attrQ_none)_(attrQ_none)_alpha_1.0_0.0')
reverse_viou_dict, reverse_loss_list_ = vis_llava_att_dist('custom_infer/STVG/output/hc_f_20_pool_4_0308_decom_yes_2k/focus_attr_ep_2_lr_8.0_8.0_(attrQ_none)_(attrQ_none)_alpha_0.0_1.0')
sorted_indices = sorted(range(len(reverse_loss_list)), key=lambda i: reverse_loss_list[i])


all_keys = list(logit_viou_dict.keys())

logit_viou_list = []
for key_id in sorted_indices:
    logit_viou_list.append(logit_viou_dict[all_keys[key_id]])


refine_logit_viou_list = []
for kk, key_id in enumerate(sorted_indices):
    if kk > 600:
        refine_logit_viou_list.append(blur_logit_pred[all_keys[key_id]])
    else:
        refine_logit_viou_list.append(logit_viou_dict[all_keys[key_id]])

# breakpoint()
# reverse_viou_list = []
# for key_id in sorted_indices:
#     try:
#         if all_keys[key_id] in list(reverse_viou_dict.keys()):
#             reverse_viou_list.append(reverse_viou_dict[all_keys[key_id]])
#     except:
#         breakpoint()
# logit_viou_list = [logit_viou_list[idx] for idx in sorted_indices]
# reverse_viou_list = [reverse_viou_list[idx] for idx in sorted_indices]

# breakpoint()
logit_viou_sum = []
reverse_viou_sum = []

print(len(logit_viou_list), len(refine_logit_viou_list))
for idx in range(20):
    logit_viou_sum.append(sum(logit_viou_list[idx*50:(idx+1)*50]) / 50)
    reverse_viou_sum.append(sum(refine_logit_viou_list[idx*50:(idx+1)*50]) / 50)

plt.plot(logit_viou_sum, marker='o', color='b', label='logit_viou_sum')
plt.plot(reverse_viou_sum, marker='o', color='r', label='reverse_viou_sum')

print(sum(logit_viou_list) / len(logit_viou_list))
print(sum(refine_logit_viou_list) / len(refine_logit_viou_list))
# breakpoint()
# 添加标签和标题
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('reverse_tuning_dist')
plt.legend()

plt.savefig("/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/custom_infer/reverse_logit.png", dpi=300, bbox_inches="tight", transparent=False)

# 显示图形
# plt.show()