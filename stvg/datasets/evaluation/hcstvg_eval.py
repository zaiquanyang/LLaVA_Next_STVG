import os
from typing import Dict, List

import numpy as np
from utils.comm import is_main_process, all_gather

import torch
from functools import reduce
from utils.box_utils import np_box_iou
import json

def save_json(path, data):
    with open(path, "w") as f:
        return json.dump(data, f)

class HCSTVGiouEvaluator:
    def __init__(
        self,
        vidstg_path: str,
        subset: str = "test",
        iou_thresholds: list = None,
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        """
        assert subset in ["train", "test", "val"], f"Wrong HCSTVG subset {subset}"
    
        gt_data = []
        cache_dir = os.path.join(vidstg_path, 'data_cache')
        dataset_cache = os.path.join(cache_dir, f'hcstvg-{subset}-anno.cache')
        gt_data = torch.load(dataset_cache) 

        self.vid2steds = {}  # map video_id to [start, end] of the GT tube
        self.vid2box = {}  # map video to bbox
        self.vid2names = {}
        self.vid2sents = {}
        
        for data_item in gt_data:
            video_id = data_item['item_id']
            temp_gt = data_item['gt_temp_bound']
            self.vid2names[video_id] = data_item['vid']
            self.vid2sents[video_id] = data_item['description']
            box_dict = data_item['bboxs']
            self.vid2box[video_id]={key : [box_dict[key]] for key in box_dict}
            self.vid2steds[video_id] = temp_gt

        self.iou_thresholds = iou_thresholds

    def evaluate(self, predictions: List[Dict], video_predictions: List[Dict]):
        vid_metrics = {}
        
        for video_id, video_pred in video_predictions.items():
            if video_id in vid_metrics:
                print(f"Warning, multiple predictions found for video {video_id}")
                continue
            
            gt_sted = self.vid2steds[video_id]
            pred_sted = video_pred["sted"]

            # compute temporal iou
            max_start = max(gt_sted[0], pred_sted[0])
            min_end = min(gt_sted[1], pred_sted[1])
            min_start = min(gt_sted[0], pred_sted[0])
            max_end = max(gt_sted[1], pred_sted[1])
            if min_end <= max_start:
                tiou = 0
            else:
                intersection = min_end - max_start
                gt_span = gt_sted[1] - gt_sted[0]
                pred_span = pred_sted[1] - pred_sted[0]
                union = gt_span + pred_span - intersection
                tiou = intersection / union

            # compute viou and gt_viou
            vid_metrics[video_id] = {
                "gt_sted": gt_sted,
                "pred_sted": pred_sted,
                "tiou": tiou,
                "img_metrics": {},
            }

            union_predgt = set([
                frame_id for frame_id in range(min_start, max_end+1)
            ])
            inter_predgt = set(
                [frame_id for frame_id in range(max_start, min_end+1)]
            )
            
            viou = 0
            gt_viou = 0
            prediction = predictions[video_id]

            for fid in self.vid2box[video_id].keys():  # iterate on all frames of the annotated moment to update GT metrics
                if fid not in prediction:
                    # raise RuntimeError(f"No prediction for frame {fid}")
                    # print(self.vid2box[video_id].keys(), fid)
                    continue
                pred_boxes = prediction[fid]
                gt_boxes = self.vid2box[video_id][fid]
                
                iou = np_box_iou(np.array(pred_boxes).reshape(1, 4), np.array(gt_boxes))[0][0]
                if fid in inter_predgt:
                    viou += iou
                gt_viou += iou
            # breakpoint()
            viou = viou / max(len(union_predgt), 1)
            vid_metrics[video_id]["viou"] = viou
            recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if viou > thresh:
                    recalls[thresh] += 1
            vid_metrics[video_id].update(
                {
                    f"viou@{thresh}": recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

            # compute gt_viou@R
            gt_viou = gt_viou / max(len(self.vid2box[video_id]), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            vid_metrics[video_id].update(
                {
                    f"gt_viou@{thresh}": gt_recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

        return vid_metrics, self.vid2names, self.vid2sents


class HCSTVGEvaluator(object):
    def __init__(
        self,
        logger,
        vidstg_path,
        subset,
        iou_thresholds,
        save_pred=False,
        save_dir=None
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param save_pred: whether to save predictions in the output of summarize
        """
        self.evaluator = HCSTVGiouEvaluator(
            vidstg_path,
            subset=subset,
            iou_thresholds=iou_thresholds,
        )
        self.predictions = {}
        self.video_predictions = {}
        self.video_cross_attn = {}
        self.results = None
        self.iou_thresholds = iou_thresholds
        self.save_pred = save_pred
        self.save_dir = save_dir
        self.logger = logger
        
        self.tsa_weights = {}
        self.text_weights = {}
        self.spatial_weights = {}
        self.pred_sted = {}

    def accumulate(self):
        pass

    def update(self, predictions):
        self.predictions.update(predictions)

    def update_cross_attn(self, cross_weights):
        self.video_cross_attn.update(cross_weights)

    def video_update(self, video_predictions):
        self.video_predictions.update(video_predictions)

    def synchronize_between_processes(self):
        all_predictions = all_gather(self.predictions)
        self.predictions = reduce(lambda a, b: a.update(b) or a, all_predictions, {})
        all_video_predictions = all_gather(self.video_predictions)
        self.video_predictions = reduce(lambda a, b: a.update(b) or a, all_video_predictions, {})

    def summarize(self):
        if is_main_process():
            self.logger.info("#######  Start Calculating the metrics  ########")
            self.results, vid2names, vid2sents = self.evaluator.evaluate(
                self.predictions, self.video_predictions
            )
            
            metrics = {"gt_viou": 0}
            metrics.update({"tiou": 0, "viou": 0})
            for thresh in self.iou_thresholds:  # init metrics
                metrics[f"viou@{thresh}"] = 0
                metrics[f"gt_viou@{thresh}"] = 0
            counter = 0
            result_str = ''
            result_str += '\n' + '=' * 100 + '\n'
            for x in self.results.values():  # sum results
                metrics["tiou"] += x["tiou"]
                metrics["viou"] += x["viou"]
                metrics["gt_viou"] += x["gt_viou"]
                for thresh in self.iou_thresholds: 
                    metrics[f"viou@{thresh}"] += x[f"viou@{thresh}"]
                    metrics[f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
                counter += 1
                
            for key in metrics:  # average results
                metrics[key] = metrics[key] / counter
                result_str += f"{key}: {metrics[key]:.4f}" + '\n'

            result_str += '=' * 100 + '\n'
            self.logger.info(result_str)
            
            out = {f"{name}": metrics[name] for name in metrics}
            
            if self.save_pred:
                out["predictions"] = self.predictions
                out["gt"] = self.evaluator.vid2box
                out["video_predictions"] = self.video_predictions
                out["vid_metrics"] = self.results
                out['vid2names'] = vid2names
                out['vid2sents'] = vid2sents
                res_path = os.path.join(self.save_dir, 'test_results.json')
                save_json(res_path, out)

            return out

        return None




import torch.nn.functional as F

def evaluation_llava_att(llava_pred, actioness, gt_boxs, proposal_mask, proposal_bboxes, vid_i, targets, blur_pred, args):
    
    llava_pred = llava_pred.cpu()
    f_n, hw = llava_pred.shape
    h, w = int(np.sqrt(hw)), int(np.sqrt(hw))
    llava_pred = llava_pred.reshape(-1, f_n, h, w)
    proposal_mask, proposal_bboxes = torch.Tensor(proposal_mask), torch.Tensor(proposal_bboxes)
    original_h, original_w = proposal_mask.shape[-2:]
    
    gt_boxs[:, 0] = gt_boxs[:, 0] * original_w/336
    gt_boxs[:, 2] = gt_boxs[:, 2] * original_w/336
    gt_boxs[:, 1] = gt_boxs[:, 1] * original_h/336
    gt_boxs[:, 3] = gt_boxs[:, 3] * original_h/336
    
    
    proposal_mask_w_action = proposal_mask# * actioness.reshape(1, f_n, 1, 1)                            # O_n x F_n x original_h x original_w
    llava_pred_up = F.interpolate(llava_pred.cuda(), size=(original_h, original_w), mode='bilinear', align_corners=False)    # 1 x F_n x original_h x original_w
    proposal_llava_pred = proposal_mask_w_action * llava_pred_up.cpu()                                        # O_n x F_n x original_h x original_w
    proposal_llava_pred_max = F.adaptive_max_pool2d(proposal_llava_pred, output_size=(1, 1)).squeeze().squeeze()        #  O_n x F_n
    proposal_max_v_, proposal_max_frame_ID = torch.max(proposal_llava_pred_max, dim=1)
    proposal_max_v, proposal_max_track_ID = torch.max(proposal_max_v_, dim=0)
    
    # # # ======================= 选择在 gt_action 时间段内， spation与gt最接近的 track作为预测
    # max_track_iou = -1
    # max_track_ID = -1
    # for track_k, track_bboxes in enumerate(proposal_bboxes):
        
    #     track_bboxes_w_action = track_bboxes[actioness==1]              # 取出 action==True 的帧
    #     # 计算 track_bboxes_w_action 和 gt_boxs 的 IoU
    #     track_iou = np_box_iou(np.array(track_bboxes_w_action), np.array(gt_boxs))  # 12 x 12
    #     track_iou = np.diagonal(track_iou)
    #     if track_iou.mean() > max_track_iou:
    #         max_track_iou = track_iou.mean()
    #         max_track_ID = track_k
    # # print('{} / {}'.format(max_track_ID, max_track_iou))
    # blur_pred['Space_pred'] = max_track_ID

    pred_track_ID =  blur_pred['Space_pred'] if 'Space_pred' in blur_pred.keys() else proposal_max_track_ID
    
    track_llava_att = proposal_llava_pred_max[pred_track_ID]            # 只考虑 pred_track 内部的 att_map 来进行时序预测
    # track_llava_att = proposal_llava_pred_max.max(dim=0)[0]             # 考虑全部 proposal_track 内部的 att_map 来进行时序预测
    track_llava_att_norm = track_llava_att / (track_llava_att.max() + 0.0001)
    # print(track_llava_att_norm)
    # 1. 设定一个 固定的 threshold 来选择定位的帧，并以第一帧作为起点，最后一帧作为终点
    # pred_action_mask = (track_llava_att_norm > 0.5).numpy()
    # pred_track_start = np.argmax(pred_action_mask == 1)
    # pred_track_end = len(pred_action_mask) - 1 - np.argmax(pred_action_mask[::-1] == 1)
    
    # 2. 选择最大激活帧，然后以该帧为中心向两边传播，直到综合得分小于某一阈值
    
    # temporal_pred_tmp[vid_i]['sted'] = [frame_ids[pred_track_start], frame_ids[pred_track_end]]
    # pred_track_mask[pred_track_start:pred_track_end] = 1.0
    
    if 'Time_pred' in blur_pred.keys():
        t_start_fid, t_end_fid = blur_pred['Time_pred']    
        pred_action_mask = np.zeros_like(actioness)
        pred_action_mask[t_start_fid:t_end_fid] = 1.0
    else:
        center_frame_id = proposal_max_frame_ID[pred_track_ID].item()
        all_frames = track_llava_att_norm.shape[0]
        # left_i_max = center_frame_id
        start_score = track_llava_att_norm[center_frame_id]
        pred_action_mask = np.zeros_like(track_llava_att_norm)
        pred_action_mask[center_frame_id] = 1.0
        left_i, right_j = 1, 1
        
        for join_i in range(all_frames-1):
            if (center_frame_id-left_i)>=0:
                left_socre = track_llava_att_norm[center_frame_id-left_i]
            else:
                left_socre = -100

            if (center_frame_id+right_j)<all_frames:
                right_socre = track_llava_att_norm[center_frame_id+right_j]
            else:
                right_socre = -100
            
            if pred_action_mask.sum() > args.frames_thre:
                break
            else:
                if (left_socre > right_socre):
                    pred_action_mask[center_frame_id-left_i] = 1.0
                    start_score += left_socre
                    left_i += 1
                elif (left_socre < right_socre):
                    pred_action_mask[center_frame_id+right_j] = 1.0
                    start_score += right_socre
                    right_j += 1
                else:
                    print(left_socre, right_socre)
                    break
            
            # if (left_socre > right_socre) and (left_socre+start_score)/(pred_action_mask.sum()+1) > args.temporal_thre:
            #     pred_action_mask[center_frame_id-left_i] = 1.0
            #     start_score += left_socre
            #     left_i += 1
            # elif (left_socre < right_socre) and (right_socre + start_score)/(pred_action_mask.sum()+1) > args.temporal_thre:
            #     pred_action_mask[center_frame_id+right_j] = 1.0
            #     start_score += right_socre
            #     right_j += 1
            # else:
            #     break

    if pred_action_mask.sum() > args.frames_thre+1:
        print('not matched with {} frames !'.format(args.frames_thre))
        breakpoint()
        
    # if actioness[center_frame_id]:         # hit 成功，即认为temporal完全正确
    #     pred_action_mask = np.array(actioness)
    # else:                                   # hit 失败，即认为temporal完全失败
    #     pred_action_mask = pred_action_mask * 0.0
    #     pred_action_mask[0:2] = 1.0
    
    # 基于tuble_track对 pred_action_mask 进行调整?
    
    gt_track_start = np.argmax(actioness == 1)
    pred_track_start = np.argmax(pred_action_mask == 1)
    pred_track_end = len(pred_action_mask) - 1 - np.argmax(pred_action_mask[::-1] == 1)

    pred_track_bboxes = proposal_bboxes[pred_track_ID]
    
    union_predgt = (actioness + pred_action_mask) > 0
    v_iou = 0
    gt_viou = 0
    for ii, (gt_i, pred_i) in enumerate(zip(actioness, pred_action_mask)):
        
        if (gt_i*pred_i) > 0.:
            # gt_i-gt_track_start
            try:
                track_iou = np_box_iou(np.array(pred_track_bboxes[ii]).reshape(1, 4), np.array(gt_boxs[ii-gt_track_start]).reshape(1, 4))
                v_iou += track_iou
            except:
                breakpoint()
        if gt_i > 0:
            try:
                track_iou = np_box_iou(np.array(pred_track_bboxes[ii]).reshape(1, 4), np.array(gt_boxs[ii-gt_track_start]).reshape(1, 4))
                gt_viou += track_iou
            except:
                breakpoint()

    v_iou = v_iou / max(union_predgt.sum(), 1)
    gt_viou = gt_viou / max(actioness.sum(), 1)
    
    # # ======================= 选择在 gt_action 时间段内， spation与gt最接近的 track作为预测
    # max_track_iou = -1
    # max_track_ID = -1
    # for track_k, track_bboxes in enumerate(proposal_bboxes):
        
    #     track_bboxes_w_action = track_bboxes[actioness==1]              # 取出 action==True 的帧
    #     # 计算 track_bboxes_w_action 和 gt_boxs 的 IoU
    #     track_iou = np_box_iou(np.array(track_bboxes_w_action), np.array(gt_boxs))  # 12 x 12
    #     track_iou = np.diagonal(track_iou)
    #     if track_iou.mean() > max_track_iou:
    #         max_track_iou = track_iou.mean()
    #         max_track_ID = track_k
    # # print('{} / {}'.format(max_track_ID, max_track_iou))

    bbox_pred_tmp, temporal_pred_tmp = {}, {}
    bbox_pred_tmp[vid_i] = {}
    temporal_pred_tmp[vid_i] = {}
    frame_ids = targets['frame_ids']

    for f_k, frame_id in enumerate(targets['full_frames_ids']):
        bbox_pred_tmp[vid_i][frame_id] = targets['full_proposal_bboxes'][pred_track_ID][f_k]
    temporal_pred_tmp[vid_i]['sted'] = [frame_ids[pred_track_start], frame_ids[pred_track_end]]
    temporal_pred_tmp[vid_i]['qtype'] = targets['qtype']
    return bbox_pred_tmp, temporal_pred_tmp, pred_track_ID, pred_action_mask, v_iou, gt_viou#, proposal_llava_pred_max[pred_track_ID]



def evaluation_blur_pred(llava_pred, actioness, gt_boxs, proposal_mask, proposal_bboxes, vid_i, targets, blur_pred):
    
    original_h, original_w = proposal_mask.shape[-2:]
    gt_boxs[:, 0] = gt_boxs[:, 0] * original_w/336
    gt_boxs[:, 2] = gt_boxs[:, 2] * original_w/336
    gt_boxs[:, 1] = gt_boxs[:, 1] * original_h/336
    gt_boxs[:, 3] = gt_boxs[:, 3] * original_h/336
    
    t_start_fid, t_end_fid = blur_pred['Time_pred']
    pred_track_ID = blur_pred['Space_pred']
    
    pred_action_mask = np.zeros_like(actioness)
    pred_action_mask[t_start_fid:t_end_fid] = 1.0

    gt_track_start = np.argmax(actioness == 1)
    pred_track_start = np.argmax(pred_action_mask == 1)
    assert pred_track_start == t_start_fid
    
    pred_track_end = len(pred_action_mask) - 1 - np.argmax(pred_action_mask[::-1] == 1)
    pred_track_bboxes = proposal_bboxes[pred_track_ID]
    
    union_predgt = (actioness + pred_action_mask) > 0
    v_iou = 0
    for ii, (gt_i, pred_i) in enumerate(zip(actioness, pred_action_mask)):
        
        if (gt_i*pred_i) > 0.:
            # gt_i-gt_track_start
            try:
                track_iou = np_box_iou(np.array(pred_track_bboxes[ii]).reshape(1, 4), np.array(gt_boxs[ii-gt_track_start]).reshape(1, 4))
                v_iou += track_iou
            except:
                breakpoint()
    
    v_iou = v_iou / max(union_predgt.sum(), 1)
    # breakpoint()

    bbox_pred_tmp, temporal_pred_tmp = {}, {}
    bbox_pred_tmp[vid_i] = {}
    temporal_pred_tmp[vid_i] = {}
    frame_ids = targets['frame_ids']

    for f_k, frame_id in enumerate(targets['full_frames_ids']):
        bbox_pred_tmp[vid_i][frame_id] = targets['full_proposal_bboxes'][pred_track_ID][f_k]
    temporal_pred_tmp[vid_i]['sted'] = [frame_ids[pred_track_start], frame_ids[pred_track_end]]
    
    # # ======================= 选择在 gt_action 时间段内， spation与gt最接近的 track作为预测
    # max_track_iou = -1
    # max_track_ID = -1
    
    # original_h, original_w = proposal_mask.shape[-2:]
    # gt_boxs[:, 0] = gt_boxs[:, 0] * original_w/336
    # gt_boxs[:, 2] = gt_boxs[:, 2] * original_w/336
    # gt_boxs[:, 1] = gt_boxs[:, 1] * original_h/336
    # gt_boxs[:, 3] = gt_boxs[:, 3] * original_h/336
    
    # for track_k, track_bboxes in enumerate(proposal_bboxes):
        
    #     track_bboxes_w_action = track_bboxes[actioness==1]              # 取出 action==True 的帧
    #     # 计算 track_bboxes_w_action 和 gt_boxs 的 IoU
    #     track_iou = np_box_iou(np.array(track_bboxes_w_action), np.array(gt_boxs))  # 12 x 12
    #     track_iou = np.diagonal(track_iou)
    #     if track_iou.mean() > max_track_iou:
    #         max_track_iou = track_iou.mean()
    #         max_track_ID = track_k
    # # print('{} / {}'.format(max_track_ID, max_track_iou))
    
    return bbox_pred_tmp, temporal_pred_tmp, pred_track_ID, pred_action_mask, v_iou#, proposal_llava_pred_max[pred_track_ID]