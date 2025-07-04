from inspect import getcallargs
import os 
import sys
from tkinter.tix import Tree
from traceback import print_tb
# from dataset.RefTR_Dataset import denorm 
import torch.utils.data as data
import torch 
import numpy as np 
from PIL import Image 
import cv2 
import transformers
from dataset.vid_refer import Vid_REFER
import CLIP.clip as clip
# import clip
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
import pdb 
import imageio
import random 
import pycocotools.mask as mask_utils 
import json
import torchvision.transforms as transforms
from dataset.utils import load_rle_mask

# from spacy.lang.en import English
# import spacy
# nlp = spacy.load("en_core_web_sm")

def pil_resize(img, size, order):
    if size[0] == img.shape[0] and size[1] == img.shape[1]:
        return img

    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST

    return np.asarray(Image.fromarray(img).resize(size[::-1], resample))

def pil_rescale(img, scale, order):
    height, width = img.shape[:2]
    target_size = (int(np.round(height*scale)), int(np.round(width*scale)))
    return pil_resize(img, target_size, order)

def rle2mask(rle_dict):

    mask = mask_utils.decode(rle_dict) 
    return mask

def mask2rle(mask):

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))) 
    
    return rle    

class Vid_ReferDataset(data.Dataset):
    def __init__(self,
                 refer_data_root='/home/yzq/mnt/code/RIS/segmentation/MeViS/RVOS_Data',
                 dataset='mevis',
                 splitBy='unc',
                 bert_tokenizer='clip',
                 image_transforms=None,
                 max_tokens=20, 
                 num_frames=5,
                 split='train',
                 eval_mode=True,
                 efficient_sample_N=40,
                 size=448,
                 scales=False,
                 negative_samples=0,
                 positive_samples=1,
                 pseudo_path=None) -> None:
        """
        parameters:
            args: argparse obj
            image_transforms: transforms apply to image and mask
            max_tokens: determined the max length of token 
            split: ['train','val','testA','testB']
            eval_mode: whether in training or evaluating 
        """

        self.clip = ('clip' in bert_tokenizer)
        self.negative_samples = negative_samples
        self.positive_samples = positive_samples 
        self.classes=[]
        self.image_transforms=image_transforms
        self.split=split
        self.refer=Vid_REFER(refer_data_root, dataset)
        self.scales = scales 
        self.size = size 
        self.pseudo_path = pseudo_path
        self.dataset = dataset
        print('\nPreparing dataset .....')
        print(dataset, split)
        print(refer_data_root, dataset) 
        print(f'pseudo_path = {pseudo_path}')

        self.max_tokens=max_tokens
        self.num_frames = num_frames
        self.efficient_sample_N = efficient_sample_N
        ref_ids=self.refer.getRefIds(split=self.split)
        # breakpoint()
        img_ids=self.refer.getImgIds(ref_ids)
        
        self.refer_data_root = refer_data_root
        self.img_folder = os.path.join(refer_data_root, dataset, 'JPEGImages')
        mask_json = os.path.join(refer_data_root, dataset, split + '/mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            self.mask_dict = json.load(fp)
            
        # change dict to list
        all_imgs=self.refer.Imgs
        self.imgs=list(all_imgs[i] for i in img_ids)
        
        self.ref_ids=ref_ids
        self.tokenizer = clip.tokenize 
        
        self.eval_mode = eval_mode

        self.input_ids=[]
        self.word_masks=[]
        self.all_sentences = []
        # get negative samples, 
        self.refid2index = {}

        for index, r in enumerate(self.ref_ids):
            self.refid2index[r] = index 

            # for each image
            ref = self.refer.Refs[r]
            # List[Tensor] Tensor shape [1,len]
            sentences_for_ref = []
            attentions_for_ref = []
            sentence_raw_for_re = []

            # for each sentence
            for i,(el,sent_id) in enumerate(zip(ref['sentences'],ref['sent_ids'])):
                sentence_raw = el['sent']
            
                word_id = self.tokenizer(sentence_raw).squeeze(0)[:self.max_tokens]
       
                word_id = np.array(word_id)
                word_mask = np.array(word_id>0,dtype=int)

                sentences_for_ref.append(torch.tensor(word_id).unsqueeze(0))
                attentions_for_ref.append(torch.tensor(word_mask).unsqueeze(0))
                sentence_raw_for_re.append(sentence_raw)

            self.input_ids.append(sentences_for_ref)
            self.word_masks.append(attentions_for_ref)
            self.all_sentences.append(sentence_raw_for_re)
        print('Dataset prepared!')

    def __len__(self):
        return len(self.ref_ids)
    
    # def efficient_sample_frames(self, vid_id, efficient_sample_N):
        
        vid_full_path = os.path.join(self.refer_data_root, self.dataset, 'JPEGImages', vid_id)
        sample_path = os.path.join(self.refer_data_root, self.dataset, f'JPEGImages_Sample_{efficient_sample_N}', vid_id)
        if os.path.exists(sample_path):
            # print('sample {}  frames from {} frames'.format(len(os.listdir(sample_path)), len(os.listdir(vid_full_path))))
            return sample_path
        
        # 将 vid_full_path 中帧，部分拷贝到一个 tmp_video_dir 下，然后用 tmp_video_dir 提取 mask
        frames = sorted(os.listdir(vid_full_path))
        step = 1 if len(frames) < efficient_sample_N else (len(frames) // efficient_sample_N + 1)
        
        # 遍历并采样
        if os.path.exists(sample_path):
            shutil.rmtree(sample_path)
        os.makedirs(sample_path, exist_ok=True)
            
        for idx, frame in enumerate(frames):
            # 只选择符合采样间隔的帧
            if idx % step == 0:
                source_path = os.path.join(vid_full_path, frame)
                target_path = os.path.join(sample_path, frame)
                shutil.copy(source_path, target_path)  # 复制文件
                # print(f"Copied: {source_path} -> {target_path}")
        # print('sample {}  frames from {} frames'.format(len(os.listdir(sample_path)), len(os.listdir(vid_full_path))))

        return sample_path
        
    def __getitem__(self, index):
        this_ref_id = self.ref_ids[index]
        this_vid_id = self.refer.getImgIds(this_ref_id)
        this_vid = self.refer.Imgs[this_vid_id[0]]
        vid_id = this_vid_id[0]
        vid_path = os.path.join(self.refer.IMAGE_DIR, this_vid['file_name'])

        ref = self.refer.loadRefs(this_ref_id)[0]
        
        # ===================================================== prepare sentences
        ann_id = ref['ann_id']
        if isinstance(ann_id, int):
            ann_id = [ann_id]
        else:
            breakpoint()
        
        # ===================================================== prepare sam_tracks
        sam_tracks, track_rank = load_rle_mask(rle_path=os.path.join(self.refer_data_root, self.dataset, 'JPEGImages_Sample_50_rle_320_Top_10', f'{vid_id}_{ann_id[0]}_track.pkl'), size=(320, 320)) # O_n x F_n x H x W
        
        # ================  记得按照 frame_id 排序 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # sample_path = self.efficient_sample_frames(this_vid['file_name'], self.efficient_sample_N)
        sample_path = os.path.join(self.refer_data_root, self.dataset, f'JPEGImages_Sample_{self.efficient_sample_N}', vid_id)
        all_sample_frames_list = os.listdir(sample_path)
        all_sample_frames_list.sort(key=lambda p: int(os.path.splitext(p)[0]))
        
        # ================  如果是 Training model, 需要采样每个 video 固定的 frame 数量 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if not self.eval_mode:
            # training的话需要对每个 video 采样一样数量的 帧
            # gt_mask_track = self.mask_dict[str(ann_id[0])]
            # all_valid_frames = [f_id for f_id in all_sample_frames_list if gt_mask_track[int(f_id.split('.')[0])] is not None]
            # if len(all_valid_frames) == 0:
            #     all_valid_frames = all_sample_frames_list[:10]
            # sample_frames_list =  random.choices(all_valid_frames, k=self.num_frames)
            sample_frames_list =  random.choices(all_sample_frames_list, k=self.num_frames)
            sample_frames_list.sort(key=lambda p: int(os.path.splitext(p)[0]))
        else:
            sample_frames_list = all_sample_frames_list
        
        # load video and frame masks
        imgs, labels, boxes, masks, valid = [], [], [], [], []
        sam_masks = []
        for sample_f in sample_frames_list:
            frame_indx = int(sample_f.split('.')[0])
            
            img_path = os.path.join(sample_path, sample_f)
            img = Image.open(img_path).convert('RGB')

            # h, w = img.shape
            mask = np.zeros(img.size[::-1], dtype = np.float32)
            
            for x in ann_id:
                frm_anno = self.mask_dict[str(x)][frame_indx]
                if frm_anno is not None:
                    mask += mask_utils.decode(frm_anno)
                    
            if (mask > 0).any():
                # y1, y2, x1, x2 = self.bounding_box(mask)
                # box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                valid.append(1)
            else:  # some frame didn't contain the instance
                # box = torch.tensor([0, 0, 0, 0]).to(torch.float)
                valid.append(0)        
            
            label = torch.tensor(ref['category_id'])
            # append
            imgs.append(img)
            labels.append(label)
            mask = torch.from_numpy(mask)
            masks.append(mask)
            
            sam_masks.append(sam_tracks[:, all_sample_frames_list.index(sample_f)].unsqueeze(dim=1))
        
        sam_masks = torch.cat(sam_masks,dim=1)      # O_n x f_n x h x w
        # transform
        w, h = img.size
        labels = torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
        
        target = {
            # 'frames_idx': torch.tensor(sample_indx),  # [T,]
            'labels': labels,  # [T,]
            # 'boxes': boxes,  # [T, 4], xyxy
            'masks': masks,  # [T, H, W]
            'valid': torch.tensor(valid),  # [T,]
            # 'caption': exp,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'size': torch.as_tensor([int(h), int(w)]),
            'anno_id': ann_id
        }
        # breakpoint()
        if self.image_transforms is not None:
            imgs, target = self.image_transforms(imgs, target)
            imgs = torch.stack(imgs, dim=0)  # [T, 3, H, W]
        
        # # FIXME: handle "valid", since some box may be removed due to random crop
        # if torch.any(target['valid'] == 1):  # at leatst one instance
        #     instance_check = True
        # else:
        #     idx = random.randint(0, self.__len__() - 1)
                
        # ===================================================== prepare sentences
        if self.eval_mode:
            pseudo_gt = None 
            embedding=[]
            att=[]
            sentences = [] 

            for s in range(len(self.input_ids[index])):
                e=self.input_ids[index][s]
                a=self.word_masks[index][s]
                sent = self.all_sentences[index][s]

                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))
                sentences.append(sent)
            # all sentence
            word_ids = torch.cat(embedding, dim=-1)
            word_masks = torch.cat(att, dim=-1)
        else: 
            # for training, random select one sentence 
            choice_sent = np.random.choice(len(self.input_ids[index]))
            word_ids = self.input_ids[index][choice_sent]
            word_masks = self.word_masks[index][choice_sent]
            sentences = self.all_sentences[index][choice_sent]

            pseudo_gt = None 
            
            if self.negative_samples > 0:
                ###########
                img2ref = self.refer.imgToRefs[this_img_id[0]]
                cat2ref = self.refer.catToRefs[ref['category_id']]
                neg_index = []
                for item in img2ref:  
                    t_ref_id = item['ref_id']
                    t_category_id = item['category_id']
                    try:
                        if t_ref_id != this_ref_id:  # and this_category_id == t_category_id
                            neg_index.append(self.refid2index[t_ref_id])
                    except: ### for refcocog google, its refindex is not match
                        break 
                        import pdb
                        pdb.set_trace() 
                ###########
                # while len(neg_index) < self.negative_samples:
                #     same_cat_ref = random.sample(cat2ref, 1)[0]
                #     try:
                #         if same_cat_ref['ref_id'] in self.ref_ids:
                #             neg_index.append(self.refid2index[same_cat_ref['ref_id']])
                #     except:
                #         breakpoint()
                
                # neg_index.extend([self.refid2index[same_cat_ref['ref_id']] for same_cat_ref in random.sample(cat2ref, self.negative_samples) ])
                if len(neg_index) > 0:
                    neg_sents = []
                    neg_word_ids = []
                    ## random select negtive samples from same random index 
                    # n_index = neg_index[np.random.choice(len(neg_index))]
                    while len(neg_sents) < self.negative_samples:
                        ## different random index 
                        n_index = neg_index[np.random.choice(len(neg_index))]
                        choice_sent = np.random.choice(len(self.input_ids[n_index]))
                        neg_word_ids.append(self.input_ids[n_index][choice_sent])
                        neg_sents.append(self.all_sentences[n_index][choice_sent])
                    neg_word_ids = torch.cat(neg_word_ids, dim=0)
                else:
                    # random index, then randomly select one sentence 
                    neg_sents = []
                    neg_word_ids = []
                    while len(neg_sents) < self.negative_samples:
                        n_index = np.random.choice(len(self.input_ids))
                        choice_sent = np.random.choice(len(self.input_ids[n_index]))
                        tmp_sent = self.all_sentences[n_index][choice_sent]
                        if tmp_sent != sentences:
                            neg_sents.append(tmp_sent)
                            neg_word_ids.append(self.input_ids[n_index][choice_sent])
                    neg_word_ids = torch.cat(neg_word_ids, dim=0)

        # breakpoint()
        samples = {
            "img": imgs,
            "word_ids": word_ids,
            "word_masks": word_masks,
            "sam_masks": sam_masks,
            "sam_rank": torch.Tensor(track_rank),
            # "vid_id": this_vid,
        }
        if self.negative_samples > 0:
            samples['neg_sents'] = neg_sents
            samples['neg_word_ids'] = neg_word_ids
        targets = {
            # "target": target.unsqueeze(0),
            "vid_path": sample_path,
            # "sentences": sentences,
            # "boxes": bbox,
            # "orig_size": np.array([h, w]),
            # "img_path_full": img_path_full,
            ##############################
            "sentences": sentences,
            # 'frames_idx': target['frames_idx'],     # [T,]
            'labels': target['labels'],             # [T,]
            # 'boxes': boxes,  # [T, 4], xyxy
            'masks': target['masks'],               # [T, H, W]
            'valid': target['valid'],               # [T,]
            'orig_size': target['orig_size'],
            'size': target['size'],
            'vid_id': ref['image_id'],
            'anno_id': ann_id
        }
        if pseudo_gt is not None:
            targets['pseudo_gt'] = pseudo_gt
            
        return samples, targets



def vis_map(image, se_map, save_dir, vid_name, vid_frames, sents):
    
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    image = inv_normalize(image)
    image = transforms.functional.resize(image, (320, 320))

    # save raw image
    raw_img = Image.fromarray(np.uint8(image.permute(1, 2, 0) * 255))#.save(os.path.join(save_dir, vid_name))

    se_pred = se_map
    se_pred = cv2.applyColorMap(np.uint8(se_pred * 255), cv2.COLORMAP_JET)[:, :, ::-1]
    blend_pred_map = cv2.addWeighted(np.uint8(image.permute(1, 2, 0) * 255), 0.3, se_pred, 0.7, 0)

    if isinstance(sents, list):
        sent = sents[0]
    else:
        sent = sents
    
    save_vid_dir = os.path.join(save_dir, '{}'.format(vid_name))
    if os.path.exists(save_vid_dir):
        pass
    else:
        os.makedirs(save_vid_dir, exist_ok=True)
    # Image.fromarray(blend_pred_map).save(os.path.join(save_vid_dir, '{}.png'.format(sent)))
    # print('save {} done !'.format(os.path.join(save_vid_dir, '{}.png'.format(sent))))
    
    return raw_img, Image.fromarray(blend_pred_map), sent

def inv_normalize_img(torch_img_list):

    arr_img_list = []

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for img in torch_img_list:
        
        img = inv_normalize(img)
        img = transforms.functional.resize(img, (320, 320))

        # save raw image
        img = Image.fromarray(np.uint8(img.permute(1, 2, 0) * 255))#.save(os.path.join(save_dir, vid_name))

        arr_img_list.append(np.array(img))

    return arr_img_list



def blend_img_rgba(pred, image_arr_list, blend=True):
    image_blend_list = []
    for pred_mask, img in zip(pred, image_arr_list):
        # sam_2_pred = pred_track
        # breakpoint()
        pil_img = Image.fromarray(img)
        # pred_mask 必须为 HxW 或者 HxWx3
        pred_mask = pred_mask.squeeze()
        # pred = cv2.applyColorMap(np.uint8(pred_mask.squeeze() * 255), cv2.COLORMAP_JET)[:, :, ::-1]
        # pred_blend = cv2.addWeighted(np.uint8(img.permute(1, 2, 0) * 255), 0.3, pred, 0.7, 0)

        # RGBA 格式可视化，  # alpha  = 0.6
        color = np.array([0.6, 0.0, 0.2, 0.6])

        # 创建 RGBA mask，其中 alpha 通道为透明度
        h, w = pred_mask.shape
        mask_image = np.zeros((h, w, 4), dtype=np.float32)  # (H, W, 4) 四通道
        mask_image[..., :] = pred_mask[..., None] * color  # 填充 RGB_alpha 颜色

        # 转换为 uint8 并创建一个 PIL.Image 对象
        mask_image = (mask_image * 255).astype(np.uint8)
        mask_image = Image.fromarray(mask_image, mode="RGBA")

        # 将背景图像转换为 RGBA 格式
        pil_img = pil_img.convert("RGBA")
        # 使用 alpha_composite 方法叠加 mask 到原图
        if blend:
            pred_blend = Image.alpha_composite(pil_img, mask_image)
        else:
            pred_blend = pil_img

        image_blend_list.append(np.array(pred_blend))

    return image_blend_list


import shutil
import math

def save_masks(mask_list, vid_id, save_dir):
    os.makedirs(save_dir,exist_ok=True)
    # breakpoint()
    
    row_size, column_size = math.ceil(len(mask_list) / 10), 10
    column_gap, row_gap = 10, 10
    if len(mask_list[0].shape) == 3:
        if mask_list[0].shape[-1] == 4:
            # RGBA
            # 计算拼接图片的尺寸
            total_width = (mask_list[0].shape[1] + column_gap) * column_size
            max_height = (mask_list[0].shape[0] + row_gap) * row_size
            # 创建大图片
            final_image = Image.new('RGBA', (total_width, max_height))
            for i, img in enumerate(mask_list):
                i_col = i % column_size
                i_row = (i-i_col) // column_size
                img = Image.fromarray(img)
                final_image.paste(img, (i_col * (img.width+column_gap), i_row*(img.height + row_gap)))
            # breakpoint()
            final_image.save(os.path.join(save_dir, f"{vid_id}.png"))
        else:
            # RGB
            # 计算拼接图片的尺寸
            total_width = mask_list[0].shape[1] * column_size
            max_height = mask_list[0].shape[0] * row_size
            # 创建大图片
            final_image = Image.new('RGB', (total_width, max_height))
            for i, img in enumerate(mask_list):
                i_col = i % column_size
                i_row = (i-i_col) // column_size
                img = Image.fromarray(img)
                final_image.paste(img, (i_col * img.width, i_row*img.height))
            # breakpoint()
            final_image.save(os.path.join(save_dir, f"{vid_id}.png"))
    else:
        # 计算拼接图片的尺寸
        total_width = mask_list[0].shape[1] * len(mask_list)
        max_height = mask_list[0].shape[0]
        # 创建大图片
        final_image = Image.new('RGB', (total_width, max_height))
        for i, img in enumerate(mask_list):
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            final_image.paste(img, (i * img.width, 0))
        final_image.save(os.path.join(save_dir, f"mask_{vid_id}.png"))      
    
if __name__ == '__main__':
    from dataset.transform import get_img_transform, get_vid_transform
    import numpy as np 
    import json 
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    ref_vid_train = Vid_ReferDataset(
        dataset='mevis', 
        split='train', 
        eval_mode=True, 
        efficient_sample_N=50,
        image_transforms=get_vid_transform(train_mode=True, size=320), 
        num_frames=1
        )
    
    # ref_vid_val = Vid_ReferDataset(
    #     dataset='mevis', 
    #     split='valid_u', 
    #     eval_mode=True, 
    #     efficient_sample_N=20,
    #     image_transforms=get_vid_transform(train_mode=True, size=320), 
    #     num_frames=1
    #     )
    

    All_Frames = 0
    Not_All_Frames = 0
    Frame_Ratio_list = []
    for idx, vid_data in enumerate(ref_vid_train):
        
        samples, targets = vid_data
        transformed_images = samples['img']             # bs x 3 x r_h x r_w
        transformed_masks = targets['masks']
        
        if (transformed_masks.sum(dim=-1).sum(dim=-1) > 1).sum() < len(transformed_masks):
            Not_All_Frames += 1
        else:
            All_Frames += 1

        # if targets['vid_id'] == "0fbea9d79c31": # 2774fb6a02ec, 540068688f95, aba4e44c5964, 0403db3d05dc
        #     print(targets['sentences'])
        # else:
        #     print(idx)
        #     continue
        
        if idx // 10 != 2:
            continue
        # Frame_Ratio_list.append(np.round(
        #     (transformed_masks.sum(dim=-1).sum(dim=-1) > 1).sum() / len(transformed_masks), 3))

        # if idx % 100 == 0:
        #     print(All_Frames, Not_All_Frames)

        #     # 绘制直方图
        #     plt.hist(Frame_Ratio_list, bins=10, range=(0, 1), color='skyblue', edgecolor='black', alpha=0.7)

        #     # 添加标题和标签
        #     plt.title('Distribution of Values')
        #     plt.xlabel('Value Range')
        #     plt.ylabel('Frequency')

        #     # 保存图像到本地
        #     plt.savefig('value_distribution.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
        #     plt.close()
        # visulization
        paste_raw_T = []
        paste_blend_T = []
        
        text_ = targets['sentences'][0]
        pil_img_list = inv_normalize_img(transformed_images)
        image_blend_list = blend_img_rgba(np.array(transformed_masks), pil_img_list, blend=True)
        
        save_masks(image_blend_list, targets['vid_id']+f'_anno_{text_}', 'tmp_save')
        
        image_blend_list = blend_img_rgba(np.array(transformed_masks), pil_img_list, blend=False)
        save_masks(image_blend_list, targets['vid_id']+f'_img_{text_}', 'tmp_save')
        
        breakpoint()
    # train_loader=DataLoader(refcoco_train,
    #                         batch_size=12,
    #                         num_workers=2,
    #                         pin_memory=True,
    #                         sampler=None)
    # for idx,(img, target, bbox, word_ids, word_mask, _, raw_sentences) in enumerate(train_loader):
    #     print(idx, img.shape)

    #     if idx > 10: break 
    
    
