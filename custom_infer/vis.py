from PIL import Image, ImageDraw
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

def show_image_relevance(image_relevance, image, preprocess, norm_all_attention=False, att_hw=(24,24), bbox=None, tokens=None, prompt=""):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * 0.8 + np.float32(img) * 0.8
        cam = cam / np.max(cam)
        return cam

    # plt.clf() 
    # plt.plot()
    # fig = plt.gcf()
    words_N, patch_N = image_relevance.shape
    # fig, axs = plt.subplots(1, 1)
    
    image = preprocess(image)
    
    image = image.permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    origin_img = np.uint8(255 * image)                      # RGB
    # empty_image = np.zeros_like(origin_img[0])
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)          # BGR

    dim = int(image_relevance[0].numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, words_N, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(words_N, 224, 224).cuda().data.cpu().numpy()

    vis_list = []
    vis_list.append(origin_img)
    
    att_map_max, att_map_min = image_relevance.max(), image_relevance.min()
    
    for att_map in image_relevance:
        if norm_all_attention:              # 
            att_map = (att_map - att_map_min) / (att_map_max - att_map_min)
        else:
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
        vis = show_cam_on_image(image, att_map)
        vis = np.uint8(255 * vis)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_BGR2RGB)
        vis_list.append(vis)

    combined_img = np.hstack((origin_img, vis))
    
    ################################################## PIL ###################################################
    coumn_size = 8
    row_size = math.ceil(len(vis_list) / coumn_size)
    img_h, img_w, _ = vis_list[0].shape
    text_h, text_w = 30, 10
    
    # 创建一个空的画布，大小为 I 行 J 列的图像
    summary_h = 200
    canvas_height = (img_h + text_h) * row_size + summary_h                                # 额外加入一些空间用于文本
    canvas_width = (img_w + text_w) * coumn_size - text_w
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))   # 白色背景
    draw = ImageDraw.Draw(canvas)
    # print('为 {} 创建一个 {} x {} 的画布!'.format(len(vis_list), coumn_size, row_size))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 16) 
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 24) 

    # 将每张图片放到画布上
    for index, img in enumerate(vis_list):
        row = index // coumn_size
        col = index % coumn_size
        if row < row_size:                              # 确保不超出行数
            # 将 OpenCV 图像转换为 Pillow 图像
            img_pil = Image.fromarray(img)
            tmp_draw = ImageDraw.Draw(img_pil)
            # breakpoint()
            if bbox is not None:
                for id_, box in enumerate(bbox):
                    if id_ == 0:
                        tmp_draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline="green", width=4)
                    else:
                        tmp_draw.rectangle([box[0], box[1], box[0]+box[2], box[1]+box[3]], outline="red", width=2)
            # 计算图片放置位置
            x = col * (img_h + text_w)
            y = row * (img_h + text_h)  # 留出空间用于文本

            # 绘制图像
            canvas.paste(img_pil, (x, y))

            # 绘制文本
            if tokens is not None:
                text = f"Token: {tokens[index]}"  # 你可以根据需要修改文本
                draw.text((x + 50, y + img_h + 5), text, fill=(0, 0, 0), font=font)
    
    # # 绘制文本, 防止文本过长，假设每行只写20个单词
    # text_1, text_2 = prompt

    # draw.text((0 + 50, y + img_h + 55*1), text_1, fill=(0, 0, 0), font=font_big)
    # draw.text((0 + 50, y + img_h + 55*2), text_2, fill=(0, 0, 0), font=font_big)
    # canvas.show()
    # plt.imshow(vis_combined)
    # plt.imshow(image)
    # plt.axis('off')
        
    # breakpoint()
    return canvas



def show_video_relevance(clip, preprocess=None, att_maps=None, norm_mode='video', bboxes=None, pred_bboxes=None, actioness=None):
    
    def show_cam_on_image(img, mask, box=None, pred_box=None):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap * 0.8 + np.float32(img) * 0.8
        cam = cam / np.max(cam)
        
        if box is not None:
            x1, y1, x2, y2 = box
            # 将 cam 转为 uint8 格式以支持绘图
            cam_uint8 = np.uint8(255 * cam)
            # 绘制矩形框（红色，线条粗细为 2）
            # breakpoint()
            cam_with_box = cv2.rectangle(cam_uint8, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cam = np.float32(cam_with_box) / 255.0

        if pred_box is not None:
            x1, y1, x2, y2 = pred_box
            # 将 cam 转为 uint8 格式以支持绘图
            cam_uint8 = np.uint8(255 * cam)
            # 绘制矩形框（红色，线条粗细为 2）
            # breakpoint()
            cam_with_box = cv2.rectangle(cam_uint8, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cam = np.float32(cam_with_box) / 255.0
        
        cam_with_box = cam

        return cam_with_box

    # assert clip[0].size[0] == 336, print('the size of image for visulization is not 336?')
    image_list = [preprocess(img_)  for img_ in clip]
    
    # image_sequence = preprocess(clip)['pixel_values_images']
    images = np.stack(image_list)           # frames_N, 3, 224, 224
    
    images = np.transpose(images, (0, 2, 3, 1))
    images_min = images.min(axis=(1, 2), keepdims=True)  # 计算每张图的最小值
    images_max = images.max(axis=(1, 2), keepdims=True)

    images = (images - images_min) / (images_max - images_min)
    
    origin_img = np.uint8(255 * images)                         # RGB
    empty_image = np.zeros_like(origin_img[0])
    image_bgr_list =[ cv2.cvtColor(rgb_bgr, cv2.COLOR_RGB2BGR) for rgb_bgr in images]          # BGR

    frames_N, img_h, img_w, _ = origin_img.shape
    text_h, text_w = 20, 20

    # 创建一个空的画布，大小为 I 行 J 列的图像
    coumn_size = 10
    row_size = math.ceil(2*frames_N / coumn_size)
    
    summary_h = 200
    canvas_height = (img_h + text_h) * row_size + summary_h                                # 额外加入一些空间用于文本
    canvas_width = (img_w + text_w) * coumn_size - text_w
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))   # 白色背景
    draw = ImageDraw.Draw(canvas)

    # print('为 {} 创建一个 {} x {} 的画布!'.format(len(vis_list), coumn_size, row_size))
    # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 16) 
    # font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 24) 

    orignal_img_list = [origin_img[i] for i in range(origin_img.shape[0])]
    vis_list = []
    vis_list.extend(orignal_img_list)
    padding_count = (10-len(vis_list)%10) if len(vis_list)%10 != 0 else 0
    vis_list.extend([empty_image] * padding_count)
    
    # frames_N, HW = att_maps.shape
    # dim = int(HW ** 0.5)
    # att_maps = att_maps.reshape(1, frames_N, dim, dim)
    # att_maps = torch.nn.functional.interpolate(att_maps, size=224, mode='bilinear')
    # att_maps = att_maps.reshape(frames_N, 224, 224).cuda().data.cpu().numpy()
    
    # bboxes[:] = torch.Tensor(bboxes[:] * (224/336)).to(torch.int64)
    # pred_bboxes[:] = torch.Tensor(pred_bboxes[:] * (224/336)).to(torch.int64)

    action_ids = list(np.where(actioness == 1)[0])
    sorted(action_ids)
    
    att_maps = norm_video_att(att_maps, mode='None')
    for kk, att_map in enumerate(att_maps):
        # breakpoint()
        if norm_mode=='video':              # 
            att_maps_min, att_maps_max = att_maps.min(), att_maps.max()  # for video-level normalization
            att_map = (att_map - att_maps_min) / (att_maps_max - att_maps_min)
        elif norm_mode=='image':
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
        elif norm_mode=='image_video':
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())
            att_maps_min, att_maps_max = att_maps.min(), att_maps.max()
            att_map = (att_map - att_maps_min) / (att_maps_max - att_maps_min)
        else:
            att_map = att_map
            
        if kk in action_ids:
            att_vis = show_cam_on_image(image_bgr_list[kk], att_map, box=bboxes[action_ids.index(kk)], pred_box=pred_bboxes[kk])
        else:
            att_vis = show_cam_on_image(image_bgr_list[kk], att_map, box=None, pred_box=pred_bboxes[kk])
        att_vis = np.uint8(255 * att_vis)
        att_vis = cv2.cvtColor(np.array(att_vis), cv2.COLOR_BGR2RGB)
        vis_list.append(att_vis)
    
    # 将每张图片放到画布上
    for index, img in enumerate(vis_list):
        row = index // coumn_size
        col = index % coumn_size

        # 将 OpenCV 图像转换为 Pillow 图像
        img_pil = Image.fromarray(img)
        
        # 计算图片放置位置
        x = col * (img_h + text_h)
        y = row * (img_w + text_w)  # 留出空间用于文本

        # 绘制图像
        canvas.paste(img_pil, (x, y))

    return canvas


def norm_video_att(video_att, mode='video'):
    
    att_maps = video_att
    frames_N, HW = att_maps.shape
    dim = int(HW ** 0.5)
    att_maps = att_maps.reshape(1, frames_N, dim, dim)
    att_maps = torch.nn.functional.interpolate(att_maps, size=224, mode='bilinear')
    att_maps = att_maps.reshape(frames_N, 224, 224).cuda().to(torch.float32).data.cpu().numpy()

    if mode=='video':
        att_maps_min, att_maps_max = att_maps.min(), att_maps.max()  # for video-level normalization
        att_maps = (att_maps - att_maps_min) / (att_maps_max - att_maps_min)
    elif mode=='image':
        for kk, att_map in enumerate(att_maps):
            att_maps[kk] = (att_map - att_map.min()) / (att_map.max() - att_map.min())
    else:
        pass
    
    return att_maps
    
def original_resize_336(proposal_bbox_w_action, ori_size, args):

    proposal_bbox_w_action[:, 0] = proposal_bbox_w_action[:, 0] * args.img_size / ori_size[1]
    proposal_bbox_w_action[:, 2] = proposal_bbox_w_action[:, 2] * args.img_size / ori_size[1]
    proposal_bbox_w_action[:, 1] = proposal_bbox_w_action[:, 1] * args.img_size / ori_size[0]
    proposal_bbox_w_action[:, 3] = proposal_bbox_w_action[:, 3] * args.img_size / ori_size[0]

    return proposal_bbox_w_action

def blur_gt_video_array(video, targets, blur_gt=True):
    
    action_ids = list(np.where(targets['actioness'] == 1)[0])
    start_action_id = action_ids[0]
    boxs, image_size, mode = targets['boxs'].bbox, targets['boxs'].size, targets['boxs'].mode
    
    video_copy = video.copy()
    for frame_idx, (img_arr, action) in enumerate(zip(video_copy, targets['actioness'])):
        img_pil = Image.fromarray(img_arr).convert('RGBA')
        # 对整个图片进行高斯模糊
        blurred_image = img_pil.filter(ImageFilter.GaussianBlur(radius=10))  # 调整模糊半径
        if frame_idx in action_ids:
            box_ = boxs[frame_idx-start_action_id]
            x1, y1, x2, y2 = box_.numpy().astype(int)
            # breakpoint()
            # 转换为 numpy 数组
            img_np = np.array(img_pil)
            blurred_image_np = np.array(blurred_image)
            # print('blurred_image_np : ', blurred_image_np.shape)
            # 保留矩形框内的原始内容
            # 将模糊图片的矩形区域替换为原始图片的对应区域
            
            blurred_image_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
            final_image = Image.fromarray(blurred_image_np[..., :3])
            # print('final_image:, ', np.array(final_image).shape)
            # 保存或显示结果
            # final_image.save(f"blur_demo/blur_image_{frame_idx}.png")
            
        else:
            blurred_image_np = np.array(blurred_image)
            final_image = Image.fromarray(blurred_image_np[..., :3])
            # print('final_image:, ', np.array(final_image).shape)
            # 保存或显示结果
            # final_image.save(f"blur_demo/blur_image_{frame_idx}.png")
        video_copy[frame_idx] = np.array(final_image)
    return video



def blur_proposal_video_array(video, targets, blur_propsal_id=0):
    
    ori_size = targets['ori_size']
    proposal_bboxes = targets['proposal_bboxes'][blur_propsal_id]#, targets['boxs'].size, targets['boxs'].mode
    proposal_bbox_336 = original_resize_336(proposal_bboxes.copy(), ori_size)
    
    video_copy = video.copy()
    blur_images = []
    for frame_idx, img_arr in enumerate(video_copy):
        img_pil = Image.fromarray(img_arr).convert('RGBA')
        # 对整个图片进行高斯模糊
        blurred_image = img_pil.filter(ImageFilter.GaussianBlur(radius=100))  # 调整模糊半径
        
        box_ = proposal_bbox_336[frame_idx]
        x1, y1, x2, y2 = box_.astype(int)
        # breakpoint()
        # 转换为 numpy 数组
        img_np = np.array(img_pil)
        blurred_image_np = np.array(blurred_image)
        # print('blurred_image_np : ', blurred_image_np.shape)
        # 保留矩形框内的原始内容
        # 将模糊图片的矩形区域替换为原始图片的对应区域
        
        blurred_image_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
        final_image = Image.fromarray(blurred_image_np[..., :3])
        # print('final_image:, ', np.array(final_image).shape)
        # 保存或显示结果
        # final_image.save(f"blur_demo/blur_image_{frame_idx}.png")
            
        # else:
        #     blurred_image_np = np.array(blurred_image)
        #     final_image = Image.fromarray(blurred_image_np[..., :3])
        #     # print('final_image:, ', np.array(final_image).shape)
        #     # 保存或显示结果
        #     # final_image.save(f"blur_demo/blur_image_{frame_idx}.png")
        video_copy[frame_idx] = np.array(final_image)
        blur_images.append(final_image)
    return video_copy, blur_images



def blur_video_frames(video, targets, blur_frames_ids=0):
    
    ori_size = targets['ori_size']
    # proposal_bboxes = targets['proposal_bboxes'][blur_propsal_id]#, targets['boxs'].size, targets['boxs'].mode
    # proposal_bbox_336 = original_resize_336(proposal_bboxes.copy(), ori_size)
    
    video_copy = video.copy()
    blur_images = []
    for frame_idx, img_arr in enumerate(video_copy):
        img_pil = Image.fromarray(img_arr).convert('RGBA')
        if blur_frames_ids[frame_idx]:
            # 对整个图片进行高斯模糊
            blurred_image = img_pil.filter(ImageFilter.GaussianBlur(radius=100))  # 调整模糊半径
            blurred_image_np = np.array(blurred_image)
            final_image = Image.fromarray(blurred_image_np[..., :3])
            
        else:
            img_np = np.array(img_pil)
            final_image = Image.fromarray(img_np[..., :3])
        # breakpoint()
        # print('final_image:, ', np.array(final_image).shape)
        video_copy[frame_idx] = np.array(final_image)
        blur_images.append(final_image)
    return video_copy, blur_images

def blur_temp_video_array(video, targets, blur_interval=0):
    
    start_frame, end_frame = blur_interval
    ori_size = targets['ori_size']
    # proposal_bboxes = targets['proposal_bboxes'][blur_propsal_id]#, targets['boxs'].size, targets['boxs'].mode
    # proposal_bbox_336 = original_resize_336(proposal_bboxes.copy(), ori_size)
    
    video_copy = video.copy()
    blur_images = []
    for frame_idx, img_arr in enumerate(video_copy):
        img_pil = Image.fromarray(img_arr).convert('RGBA')
        
        if (frame_idx >= start_frame ) and (frame_idx < end_frame):
            # 如果是要保留的图像帧，则不执行高斯模糊
            blurred_image_np = np.array(img_pil)
        else:
            # 如果不是要保留的图像帧，则对整个图片进行高斯模糊
            blurred_image = img_pil.filter(ImageFilter.GaussianBlur(radius=100))  # 调整模糊半径
            blurred_image_np = np.array(blurred_image)
            
        final_image = Image.fromarray(blurred_image_np[..., :3])
        video_copy[frame_idx] = np.array(final_image)
        blur_images.append(final_image)
    return video_copy, blur_images

def blur_st_video_array(video, targets, blur_propsal_id=None, blur_interval=None, mode='st'):
    
    start_frame, end_frame = blur_interval
    
    ori_size = targets['ori_size']
    proposal_bboxes = targets['proposal_bboxes'][blur_propsal_id]#, targets['boxs'].size, targets['boxs'].mode
    proposal_bbox_336 = original_resize_336(proposal_bboxes.copy(), ori_size)
    
    video_copy = video.copy()
    blur_images = []
    for frame_idx, img_arr in enumerate(video_copy):
        img_pil = Image.fromarray(img_arr).convert('RGBA')
        # 对整个图片进行高斯模糊
        blurred_image = img_pil.filter(ImageFilter.GaussianBlur(radius=10))  # 调整模糊半径
        
        box_ = proposal_bbox_336[frame_idx]
        x1, y1, x2, y2 = box_.astype(int)
        
        # 转换为 numpy 数组
        img_np = np.array(img_pil)
        blurred_image_np = np.array(blurred_image)
        if 'st' == mode:
            # 仅保留指定时刻片段内的 box 
            if (frame_idx >= start_frame ) and (frame_idx < end_frame):
                # 保留矩形框内的原始内容
                # 将模糊图片的矩形区域替换为原始图片的对应区域
                blurred_image_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
                final_image = Image.fromarray(blurred_image_np[..., :3])
            else:
                final_image = Image.fromarray(blurred_image_np[..., :3])
        elif 's' == mode:
            # 仅保留 box 
            blurred_image_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
            final_image = Image.fromarray(blurred_image_np[..., :3])
        elif 't' == mode:  
            # 仅保留指定时刻片段
            if (frame_idx >= start_frame ) and (frame_idx < end_frame):
                blurred_image_np[:, :] = img_np[:, :]
                final_image = Image.fromarray(blurred_image_np[..., :3])
            else:
                final_image = Image.fromarray(blurred_image_np[..., :3])
        else:
            # 不进行 blur
            blurred_image_np[:, :] = img_np[:, :]
            final_image = Image.fromarray(blurred_image_np[..., :3])
        
        video_copy[frame_idx] = np.array(final_image)
        blur_images.append(final_image)
        
    return video_copy, blur_images


import pynvml

def print_gpu(logger=None):
    # 初始化 NVML
    pynvml.nvmlInit()
    # 获取显卡数量
    device_count = pynvml.nvmlDeviceGetCount()

    print(f"Number of GPUs: {device_count}\n")

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取第 i 个 GPU 的句柄
        name = pynvml.nvmlDeviceGetName(handle)  # 获取 GPU 名称
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 获取显存信息

        total_memory = memory_info.total / 1024**2  # 总显存，单位 MB
        used_memory = memory_info.used / 1024**2   # 已用显存，单位 MB
        free_memory = memory_info.free / 1024**2   # 空闲显存，单位 MB
        if logger is None:
            print(f"GPU {i}: {name}")
            print(f"  Total Memory: {total_memory:.2f} MB")
            print(f"  Used Memory: {used_memory/total_memory*100:.2f} %")
            # logger.info(f"  Free Memory: {free_memory/total_memory*100:.2f} %\n")
        else:
            # 移除 `.decode('utf-8')`
            logger.info(f"GPU {i}: {name}")
            logger.info(f"  Total Memory: {total_memory:.2f} MB")
            logger.info(f"  Used Memory: {used_memory/total_memory*100:.2f} %")
            # logger.info(f"  Free Memory: {free_memory/total_memory*100:.2f} %\n")

    # 关闭 NVML
    pynvml.nvmlShutdown()



def plot_stvg_metrics(plot_metrics_dict, output_dir):
    
    
    import matplotlib.pyplot as plt

    # 示例字典，每个键对应一个列表
    data = plot_metrics_dict

    # 确定子图的行数和列数
    num_plots = len(data)  # 总子图数量
    cols = 3               # 每行显示的子图数量
    rows = (num_plots + cols - 1) // cols  # 行数，向上取整

    # 创建子图
    # fig, axes = plt.subplots(rows, cols, figsize=(10, 6), constrained_layout=True)
    fig, axes = plt.subplots(rows, cols, figsize=(24, 8), constrained_layout=True)
    
    # 将子图轴展平成一维数组，方便迭代
    axes = axes.flatten()

    # 遍历字典中的每个列表，在每个子图中绘制
    for i, (label, values) in enumerate(data.items()):
        
        axes[i].plot(values, marker='o', label=label)  # 绘制折线图
        axes[i].set_title(label, fontsize=10)         # 设置子图标题
        axes[i].grid(True, linestyle="--", alpha=0.6) # 添加网格
        axes[i].legend(fontsize=8)                    # 添加图例

        axes[i].set_ylim(min(values) // 10 * 10, (max(values) // 10 + 1) * 10)
        
        # 在每个数据点上显示对应的数值
        for j, value in enumerate(values):
            axes[i].annotate(
                f"{value}",                      # 显示的数值
                (j, value),                      # 数值的位置 (x, y)
                textcoords="offset points",      # 偏移方式
                xytext=(0, 5),                   # 偏移量 (x偏移, y偏移)
                ha='center', fontsize=8, color="black"  # 文本水平居中
            )
        
    # 如果子图数目少于网格数，隐藏多余的子图
    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    # 添加整体标题
    fig.suptitle("Subplots for Each List in Dictionary", fontsize=16)

    # 显示图形
    # plt.show()
    plt.savefig(os.path.join(output_dir, "_metrics.png"), dpi=300)