import os
import json
from copy import deepcopy
import torch
import random
import pycocotools.mask as mask_utils 

from tqdm import tqdm
import torch.utils.data as data
import numpy as np
from PIL import Image
import ffmpeg
from decord import VideoReader, cpu

from torchvision.transforms import ToTensor, ToPILImage, Resize
from utils.bounding_box import BoxList
from .data_utils import make_hcstvg_input_clip


import cv2
import torch
import numpy as np

class HCSTVGDataset(data.Dataset):

    def __init__(self, cfg, split, transforms=None) -> None:
        super(HCSTVGDataset,self).__init__()
        assert split in ['train', 'test', 'val']
        self.cfg = cfg.clone()
        self.split = split
        self.transforms = transforms

        self.data_dir = cfg.DATA_DIR
        # self.anno_dir = os.path.join(self.data_dir,'annos/hcstvg_v2' if 'hc-stvg2' in self.data_dir else 'annos/hcstvg_v1')
        self.anno_dir = os.path.join(self.data_dir,'annos/hcstvg')
        self.sent_file = os.path.join(self.anno_dir, f'{split}_proc.json')  # split
        self.epsilon = 1e-10
        # breakpoint()
        self.all_gt_data = self.load_data()
        self.clean_miss()
        self.vocab = None
        
        if cfg.DATA_TRUNK is not None:
            self.all_gt_data = self.all_gt_data[:cfg.DATA_TRUNK]
    
    def clean_miss(self):
        miss_name = '10__Gvp-cj3bmIY.mp4'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
        miss_name = '1_aMYcLyh9OhU.mkv'
        for item in self.all_gt_data:
            if item['vid'] == miss_name:
                self.all_gt_data.remove(item)
                break
        
    def get_video_info(self,index):
        video_info = {}
        data_item = self.all_gt_data[index]
        video_info['height'] = data_item['height']
        video_info['width'] = data_item['width']
        return video_info

    def load_frames(self, data_item, load_video=True):
        video_name = data_item['vid']
        frame_ids = data_item['frame_ids']
        frame_ids = [id_ * 10 for id_ in frame_ids]
        patience = 20
        max_rate = 1.4
        video_time = -1
        fps=-1
        if load_video:
            video_path = os.path.join(self.data_dir,'v2_video' if 'hc-stvg2' in self.data_dir else 'v1_video', video_name)
            h, w = data_item['height'], data_item['width']
            succ_flag = False
            for _ in range(patience):
                try:
                    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
                    total_frame_num = len(vr)
                    
                    video_time = total_frame_num / vr.get_avg_fps()
                    fps = round(vr.get_avg_fps())
                    # print('{} has {} frames: time/fps: {}/{}'.format(video_name, len(vr), video_time, fps))
                    
                    succ_flag = True
                    if succ_flag:
                        break
                except Exception:
                    # print(video_name)
                    aa = 0
            # print('{} has {} frames: time/fps: {}/{}'.format(video_name, len(vr), video_time, fps))
            if video_time < 19:
                print('there is something wrong with this video {}, because it only have {}-s'.format(video_name, video_time))
                succ_flag = False
            if not succ_flag:
                print("video load wrong", video_name)
                frames = np.ones((20, self.cfg.INPUT.RESOLUTION, int(self.cfg.INPUT.RESOLUTION*max_rate), 3), dtype=np.uint8)
                # raise RuntimeError("Load Video Error")
            else:
                frames = vr.get_batch(frame_ids).asnumpy()
            
            frames = np.array([
                np.array(Image.fromarray(frame).resize((self.cfg.INPUT.RESOLUTION, self.cfg.INPUT.RESOLUTION), Image.BICUBIC)) for frame in frames
                ])
            pil_images = [Image.fromarray(frame) for frame in frames]    
            frames = np.stack(frames)

        else:
            raise NotImplementedError("Not Implement load from frames")

        return frames, pil_images, video_time, fps

    def __getitem__(self, index: int):
        """
        Usage:
            In training, sample a random clip from video
            In testing, chunk the video to a set of clips
        """
        video_data = deepcopy(self.all_gt_data[index]) 
        class_catgory = video_data['object']
        # print('In the all_gt_data, this frame_count is {}.'.format(video_data['frame_count']))
        data_item = make_hcstvg_input_clip(self.cfg, self.split, video_data)
        
        frames, pil_images, video_time, fps = self.load_frames(data_item)   # T * C * H * W

        # load the sampled gt bounding box
        frame_ids = data_item['frame_ids']
        temp_gt = data_item['gt_temp_bound']
        action_idx = np.where(data_item['actioness'])[0]
        start_idx, end_idx = action_idx[0], action_idx[-1]
        bbox_idx = [frame_ids[idx] - temp_gt[0] for idx in range(start_idx, end_idx + 1)]
        bboxs = torch.from_numpy(data_item['bboxs'][bbox_idx]).reshape(-1, 4)
        assert bboxs.shape[0] == len(action_idx)
        
        o_w, o_h = data_item['width'], data_item['height']
        w, h = frames.shape[1], frames.shape[2]        # 因为在load_frames的时候resize了图片，因此这里需要更新data_item，来应对box的处理
        # resize bbox
        bboxs[:, 0] = bboxs[:, 0] * (w/o_w)
        bboxs[:, 2] = bboxs[:, 2] * (w/o_w)
        bboxs[:, 1] = bboxs[:, 1] * (h/o_h)
        bboxs[:, 3] = bboxs[:, 3] * (h/o_h)
        bboxs = BoxList(bboxs, (w, h), 'xyxy')
        
        sentence = data_item['description']
        sentence = sentence.lower()
        input_dict = {'frames': frames, 'boxs': bboxs, 'text': sentence, 'actioness' : data_item['actioness']}

        # if self.transforms is not None:
        #     input_dict = self.transforms(input_dict)
        
        targets = {
            'vid': data_item['vid'],
            'vid_name': data_item['vid'].split('.')[0],
            'item_id' : data_item['item_id'],
            'frame_ids' : data_item['frame_ids'],
            'actioness' : torch.from_numpy(data_item['actioness']) ,
            'start_heatmap' : torch.from_numpy(data_item['start_heatmap']),
            'end_heatmap' : torch.from_numpy(data_item['end_heatmap']),
            'boxs' : input_dict['boxs'],
            'qtype' : 'None',
            'img_size' : input_dict['frames'].shape[1:3],
            'ori_size' : (o_h, o_w),
            'class_catgory':class_catgory,
            'proposal_mask':data_item['proposal_mask'],
            'proposal_bboxes':data_item['proposal_bboxes'],
            'full_proposal_bboxes':data_item['full_proposal_bboxes'],
            'full_frames_ids': video_data['frame_ids'],
            'gpt_4_proc_attr':video_data['gpt_4_proc_attr'],
            'gpt_4_proc_action':video_data['gpt_4_proc_action'],
            # 'gpt_4_proc_sub_actions':video_data['gpt_4_proc_sub_actions'],
            'pil_images':pil_images,
        }

        return input_dict['frames'], sentence, targets, pil_images, video_time, fps

    def __len__(self) -> int:
        return len(self.all_gt_data)

    def load_data(self):
        """
        Prepare the Input Data Cache and the evaluation data groundtruth
        """
        cache_dir = os.path.join(self.data_dir,'data_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
         # Used for Model Input
        dataset_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-input.cache')
        # Used For Evaluateion
        gt_anno_cache = os.path.join(cache_dir, f'hcstvg-{self.split}-anno.cache')
        
        # Used For GDINO_SAM2
        GDINO_Proposal_Path = f"{self.data_dir}/dino_sam2_proposals"
        if os.path.exists(dataset_cache):
            data = torch.load(dataset_cache)
            return data

        # Used for GPT_4_Contrastive_Prompt
        decompose_attr_action = f"{self.data_dir}/annos/hcstvg/gpt4_decompose_attr_action.json"
        with open(decompose_attr_action, 'r') as fr:
            decompose_attr_action_data = json.load(fr)
        
        # decompose_sub_actions = "/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/custom_infer/STVG/data/hc-stvg2/annos/hcstvg_v2/gpt4_decompose_actions.json"
        # with open(decompose_sub_actions, 'r') as fr:
        #     decompose_sub_actions_data = json.load(fr)
            
        gt_data, gt_anno = [], []
        vstg_anno = self.preprocess(self.sent_file)
        
        for anno_id in tqdm(vstg_anno):  
            gt_file = vstg_anno[anno_id]
            frame_nums = gt_file['frame_count']
            video_name = gt_file['vid']
            # breakpoint()    
            # 剔除 119_E-6ruyZFfZs.mkv
            if video_name in ['119_E-6ruyZFfZs.mkv', '331_phrYEKv0rmw.mp4']:    # 331_phrYEKv0rmw.mp4 是因为 GDINO没提取出box
                continue
            start_fid = 0
            end_fid = frame_nums - 1
            temp_gt_begin = max(0, gt_file['tube_start_frame'])
            temp_gt_end = min(gt_file['tube_end_frame'], end_fid)

            assert len(gt_file['target_bboxs']) == temp_gt_end - temp_gt_begin + 1
            
            frame_ids = []
            for frame_id in range(start_fid, end_fid+1):
                frame_ids.append(frame_id)
            assert len(frame_ids) == frame_nums, print('frame_nums != len(frame_ids) !') 
            actioness = np.array([int(fid <= temp_gt_end and fid >= temp_gt_begin) for fid in frame_ids]) 
            
            # prepare the temporal heatmap
            action_idx = np.where(actioness)[0]
            start_idx, end_idx = action_idx[0], action_idx[-1]
            
            start_heatmap = np.ones(actioness.shape) * self.epsilon
            pesudo_prob = (1 - (start_heatmap.shape[0] - 3) * self.epsilon - 0.5) / 2
            
            start_heatmap[start_idx] = 0.5
            if start_idx > 0:
                start_heatmap[start_idx-1] = pesudo_prob
            if start_idx < actioness.shape[0] - 1:
                start_heatmap[start_idx+1] = pesudo_prob

            end_heatmap = np.ones(actioness.shape) * self.epsilon
            end_heatmap[end_idx] = 0.5
            if end_idx > 0:
                end_heatmap[end_idx-1] = pesudo_prob
            if end_idx < actioness.shape[0] - 1:
                end_heatmap[end_idx+1] = pesudo_prob

            bbox_array = []
            for idx in range(len(gt_file['target_bboxs'])):
                bbox = gt_file['target_bboxs'][idx]
                x1, y1, w, h = bbox
                bbox_array.append(np.array([x1,y1,min(x1+w, gt_file['width']), min(y1+h, gt_file['height'])]))
                # try:
                #     assert (x1 <= gt_file['width']) and (x1 + w <= gt_file['width']), print(video_name, bbox)
                #     assert (y1 <= gt_file['height']) and (y1 + h <= gt_file['height']), print(video_name, bbox)
                # except:
                #     breakpoint()
            bbox_array = np.array(bbox_array)
            assert bbox_array.shape[0] == temp_gt_end - temp_gt_begin + 1
            
            gt_bbox_dict = {fid : bbox_array[fid - temp_gt_begin].tolist() for fid in range(temp_gt_begin, temp_gt_end + 1)}
            
            gt_item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'bboxs' : gt_bbox_dict,
                'description' : gt_file['sentence'],
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'frame_count' : gt_file['frame_count']
            }
            
            item = {
                'item_id' : gt_file['id'],
                'vid' : video_name,
                'frame_ids' : frame_ids,
                'width' : gt_file['width'],
                'height' : gt_file['height'],
                'start_heatmap': start_heatmap,
                'end_heatmap': end_heatmap,
                'actioness': actioness,
                'bboxs' : bbox_array,
                'gt_temp_bound' : [temp_gt_begin, temp_gt_end],
                'description' : gt_file['sentence'],
                'object' : 'person',
                'frame_count' : gt_file['frame_count']
            }
            # add GDNIO_Proposals
            with open(os.path.join(GDINO_Proposal_Path, video_name.split('.')[0]+'.json'), "r") as file:
                gdino_data = json.load(file)
            gdino_frames = gdino_data['sample_frames']
            gdino_tracks_list = gdino_data['dino_sam2_proposals']
            
            packed_bboxes, packed_masks = self.pack_tracks(gdino_tracks_list)
            
            item['proposal_frames'] = gdino_frames
            item['proposal_bboxes'] = packed_bboxes         # F_n x 4
            item['proposal_masks'] = packed_masks           # F_n x h x w
            
            # add contrastive_prompt
            try:
                # gpt_4_proc_subject =  contrastive_prompt_data[video_name]['decompose_subject']
                gpt_4_proc_attr = decompose_attr_action_data[video_name]['decompose_attribute']
                gpt_4_proc_action = decompose_attr_action_data[video_name]['decompose_action']
                # gpt_4_proc_sub_actions = decompose_sub_actions_data[video_name]['decompose_sub_actions']
            except:
                breakpoint()
            
            # item['gpt_4_proc_subject'] = gpt_4_proc_subject
            item['gpt_4_proc_attr'] = gpt_4_proc_attr
            item['gpt_4_proc_action'] = gpt_4_proc_action
            # item['gpt_4_proc_sub_actions'] = gpt_4_proc_sub_actions

            gt_data.append(item)
            gt_anno.append(gt_item)
        
        # random.shuffle(gt_data)
        torch.save(gt_data, dataset_cache)
        torch.save(gt_anno, gt_anno_cache)
        return gt_data

    def preprocess(self,anno_file):
        """
        preoprocess from the original annotation
        """
        pair_cnt = 0
        print(f"Prepare {self.split} Data")
        
        with open(anno_file, 'r') as fr:
            hcstvg_anno = json.load(fr)
        
        
        proc_hcstvg_anno = {}
        for vid in tqdm(hcstvg_anno):
            # breakpoint()
            anno = hcstvg_anno[vid]
            data_pairs = {}
            data_pairs['vid'] = vid
            data_pairs['width'] = anno['img_size'][1]#anno['width']
            data_pairs['height'] = anno['img_size'][0]#anno['height']
            data_pairs['frame_count'] = anno['img_num']
            data_pairs['tube_start_frame'] = anno['st_frame']# - 1
            data_pairs['tube_end_frame'] = data_pairs['tube_start_frame'] + len(anno['bbox']) - 1
            data_pairs['tube_start_time'] = anno['st_time']
            data_pairs['tube_end_time'] = anno['ed_time']
            data_pairs['id'] = pair_cnt
            data_pairs['sentence'] = anno['English']
            data_pairs['target_bboxs'] = anno['bbox']
            proc_hcstvg_anno[pair_cnt] = data_pairs
            pair_cnt += 1
            # if pair_cnt > 1000:
            #     break

        print(f'{self.split} pair number : {pair_cnt}')
        return proc_hcstvg_anno

    def pack_tracks(self, tracks_list):

        packed_bboxes = []
        packed_masks = []
        for track in tracks_list:
            
            track_id, inst_bboxes, inst_rles, class_name, box_format, img_width, img_height = track.items()
            rle_2_mask_list = []
            for rle_ in inst_rles[1]:
                # mask_ = mask_utils.decode(rle_)
                rle_2_mask_list.append(rle_)
            
            # packed_mask_ = np.stack(rle_2_mask_list)
            packed_bbox_ = np.stack(inst_bboxes[1]).squeeze(axis=1)
        
            packed_bboxes.append(packed_bbox_)
            packed_masks.append(rle_2_mask_list)

        return packed_bboxes, packed_masks