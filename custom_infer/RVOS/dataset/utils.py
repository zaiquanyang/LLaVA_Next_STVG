import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import pycocotools.mask as mask_utils 
from torchvision.transforms import InterpolationMode

import json
import pickle

def load_rle_mask(rle_path, size=None):
    # rle_path = os.path.join("/home/yzq/data/coco/refer/sam_rle/filter", rle_path)
    
    with open(rle_path, "rb") as file:  # "rb" 表示以二进制读取模式打开文件
        rle_data = pickle.load(file)
    
    def rle2mask(rle_dict):
        mask = mask_utils.decode(rle_dict) 
        return mask

    def mask2rle(mask):
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) 
        
        return rle  
    
    rle_tracks, track_rank = rle_data['track_dict'], rle_data['track_rank']
    track_keys = list(rle_tracks.keys())
    H, W = rle2mask(rle_tracks[track_keys[0]][0]).shape
    track_masks = torch.zeros((len(track_keys), len(rle_tracks[track_keys[0]]), H, W))       # O_n x F_n x H x W
    
    if size is None:
        pass
    else:
        track_masks = TF.resize(track_masks, size, interpolation=InterpolationMode.NEAREST)#[0]
    
    for o_n, o_k in enumerate(track_keys):
        track_rle = rle_tracks[o_k]
        for f_n, f_obj_rle in enumerate(track_rle):
            try:
                frame_obj_mask = rle2mask(f_obj_rle)
            except:
                breakpoint()
            if size is None:            # 不缩放
                track_masks[o_n][f_n] = torch.Tensor(frame_obj_mask)
            else:                       # 缩放 SAM_mask
                track_masks[o_n][f_n] = TF.resize(torch.Tensor(frame_obj_mask).unsqueeze(dim=0), size, interpolation=InterpolationMode.NEAREST).squeeze()
    
    return track_masks, track_rank