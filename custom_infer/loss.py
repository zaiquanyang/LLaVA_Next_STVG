import torch
import math
from PIL import Image, ImageDraw, ImageFont
import logging
import os
import torchvision.transforms as T
import json
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.patches import Rectangle
import torch.nn.functional as F


def phrase_consistency_l(Q_to_img_att_list, C_to_img_att, track_proposal, grid_size):
    
    
    pre_q_map = [(C_to_img_att[-5]/(att[-5] + 0.0001)).reshape(-1, grid_size, grid_size) for att in Q_to_img_att_list]      # [f_n x 6 x 6, f_n x 6 x 6, f_n x 6 x 6],
    track_proposal = torch.Tensor(track_proposal).to(C_to_img_att.device)
    track_proposal = F.interpolate(track_proposal, size=(grid_size, grid_size), mode='bilinear', align_corners=False)       # o_n x f_n x  6 x 6
    
    o_n_score = torch.zeros(len(track_proposal)).to(C_to_img_att.device)
    
    for q_map in pre_q_map:
        
        o_socre_all_f = F.adaptive_max_pool2d((q_map.unsqueeze(dim=0) * track_proposal), output_size=(1, 1)).squeeze().squeeze()  # o_n x f_n 
        o_socre = o_socre_all_f.max(dim=-1)[0]
        o_n_score += o_socre

    o_n_score_entropy = o_n_score.max() / o_n_score.sum()
    
    return 1. - o_n_score_entropy


def contrastive_logit_adjust(track_score_for_Q, grid_size=None):
    
    track_rela_score_for_Q = []
    for track_k in range(len(track_score_for_Q)):
        track_k_score_for_Q = []
        for q_i in range(len(track_score_for_Q[track_k])):
            pos_score = track_score_for_Q[track_k][q_i][0] / (track_score_for_Q[track_k][q_i][1] + 0.0001)
            track_k_score_for_Q.append(pos_score)
        track_rela_score_for_Q.append(torch.stack(track_k_score_for_Q))

    track_rela_score_for_Q = torch.stack(track_rela_score_for_Q)
    
    track_rela_score_sum = track_rela_score_for_Q.sum(dim=-1)
    loss = 1. - track_rela_score_sum.max() / (track_rela_score_sum.sum() + 0.0001)
    logit_pred_id = torch.argmax(track_rela_score_for_Q.sum(dim=-1))
    return loss, logit_pred_id

def Q_temporal_loss(T_segment_score, grid_size=None):
    # T_segment_score: Q_n x  Segment_N x 2
    
    T_segment_rela_score = T_segment_score[:, :, 0] / (T_segment_score[:, :, 1] + 0.0001)
    Q_pred_bin = torch.argmax(T_segment_rela_score, dim=-1)
    loss = 0.0
    
    return loss, Q_pred_bin

def Q_spatial_loss(track_score, grid_size=None):
    
    # track_score: Q_n x  Track_N
    
    track_rela_score = track_score[:, :, 0] / (track_score[:, :, 1] + 0.0001)
    Q_pred_bin = torch.argmax(track_rela_score, dim=-1)
    loss = 0.0
    
    return loss, Q_pred_bin