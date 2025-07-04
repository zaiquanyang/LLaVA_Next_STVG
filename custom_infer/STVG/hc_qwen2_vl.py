import argparse
import torch
import numpy as np
from visualizer import get_local
get_local.activate()

from custom_infer.STVG.utils.logger import setup_logger
from custom_infer.STVG.utils.comm import synchronize, get_rank
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import clip
_, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

import json
import os
import math
from torch.utils.tensorboard import SummaryWriter
import inspect
from transformers import AutoConfig
import importlib

from custom_infer.STVG.datasets import build_evaluator, build_dataset, build_transforms
from custom_infer.STVG.config import cfg


st_tuning = []
infer_module_name = []
# blur_pred = importlib.import_module("custom_infer.STVG.V_Tuning.st_tuning_blur_pred")
base = importlib.import_module("custom_infer.STVG.Qwen_Tuning.st_tuning_base")
focus_attr = importlib.import_module("custom_infer.STVG.Qwen_Tuning.st_tuning_focus_attr")
# focus_attr_w_reverse = importlib.import_module("custom_infer.STVG.V_Tuning.st_tuning_focus_attr_w_reverse")
focus_act = importlib.import_module("custom_infer.STVG.Qwen_Tuning.st_tuning_focus_act")
# focus_whole = importlib.import_module("custom_infer.STVG.V_Tuning.st_tuning_focus_whole")
# st_tuning.append(blur_pred)
st_tuning.append(base)
st_tuning.append(focus_attr)
# st_tuning.append(focus_attr_w_reverse)
st_tuning.append(focus_act)
# st_tuning.append(focus_whole)
# infer_module_name.append(blur_pred.__name__.split('.')[-1][10:])
infer_module_name.append(base.__name__.split('.')[-1][10:])
infer_module_name.append(focus_attr.__name__.split('.')[-1][10:])
# infer_module_name.append(focus_attr_w_reverse.__name__.split('.')[-1][10:])
infer_module_name.append(focus_act.__name__.split('.')[-1][10:])
# infer_module_name.append(focus_whole.__name__.split('.')[-1][10:])



print(infer_module_name)



def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default="")
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default="")
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default="")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mm_resampler_type", type=str, default="spatial_pool")
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="average")
    parser.add_argument("--image_aspect_ratio", type=str, default="anyres")
    parser.add_argument("--image_grid_pinpoints", type=str, default="[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--for_get_frames_num", type=int, default=4)
    parser.add_argument("--load_8bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--load_4bit",  type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--prompt", type=str, default=None) 
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--mm_newline_position", type=str, default="no_token")
    parser.add_argument("--force_sample", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--add_time_instruction", type=str, default=False)

    parser.add_argument("--eval_freq", type=int, default=30)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # Learning rate and other optimization hyperparameters
    parser.add_argument('--cfg_file', type=str, default="cfg_file", help='cfg_file')
    parser.add_argument('--model_name', type=str, default="llava", help='mode_name')
    parser.add_argument('--dataset_name', type=str, default="stvg", help='mode_name')
    parser.add_argument('--exp_comment', type=str, default="llava", help='exp_comment')
    parser.add_argument('--split', type=str, default="val", help='val')
    
    parser.add_argument('--img_size', type=int, default=336, help='img_size')

    parser.add_argument('--yes_id', type=int, default=-1, help='yes_id')
    parser.add_argument('--no_id', type=int, default=-1, help='no_id')
    parser.add_argument('--s_lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--t_lr', type=float, default=1, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.5, help='Beta parameter')
    parser.add_argument('--action_alpha', type=float, default=1., help='Alpha parameter')
    parser.add_argument('--sont_alpha', type=float, default=1., help='Alpha parameter')
    parser.add_argument('--attribute_alpha', type=float, default=1., help='Alpha parameter')
    parser.add_argument('--T', type=int, default=5, help='T parameter')
    parser.add_argument('--ttt_epoch', type=int, default=2, help='ttt_epochs')
    parser.add_argument('--layers_N', type=int, default=28, help='llm_layers_N')
    
    parser.add_argument('--token_idx', type=int, default=-3, help='token_idx')
    parser.add_argument('--visual_token_id', type=int, default=151656, help='visual_token_id')

    parser.add_argument('--llm_infer_mode', type=str, default="", help='llm_infer_mode')
    parser.add_argument('--temporal_thre', type=float, default=0.5, help='temporal_thre')
    parser.add_argument('--frames_thre', type=int, default=10, help='temporal_thre')

    return parser.parse_args()


def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # ================================ Initialize the model
    
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct", 
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="sdpa",
    #     device_map="auto",
    #     cache_dir="/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/model_zoo",
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/model_zoo/Qwen2-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    
    # MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # default processer
    image_processor = AutoProcessor.from_pretrained("/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/model_zoo/Qwen2-VL-7B-Instruct", )
    tokenizer = AutoTokenizer.from_pretrained("/home/zaiquyang2/scratch/mllm/LLaVA-NeXT/model_zoo/Qwen2-VL-7B-Instruct", )
    


    # ========================================== STVG_dataset
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    vid_dataset = build_dataset(cfg, split=args.split, transforms=None)
    
    # ========================================== Exp_Setting
    grid_size = math.ceil(27 / args.mm_spatial_pool_stride)
    model_name = str(args.model_name).lower()
    check_case_id = 167
    exp_name = f"{args.dataset_name}/{model_name}_f_{cfg.INPUT.SAMPLE_FRAMES}/{args.llm_infer_mode}_ep_{args.ttt_epoch}_lr_{args.s_lr}_{args.t_lr}{args.exp_comment}"
    # exp_name = f"hc_f_{cfg.INPUT.SAMPLE_FRAMES}_pool_{args.mm_spatial_pool_stride}_0305_ignore_decom_fail/focus_act_ep_{args.ttt_epoch}_lr_{args.s_lr}_{args.t_lr}_(ac_cap)_(ac_cap)"
    output_dir = os.path.join("custom_infer/STVG/output/", exp_name)
    

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, '_llava_att_npy'), exist_ok=True)
    args.output_dir = output_dir
    writer = SummaryWriter(f'{output_dir}/_thre_{args.frames_thre}_f_in_attr_track')
    logger = setup_logger("Video Grounding", output_dir, get_rank(), filename='_log.txt')
    evaluator = build_evaluator(cfg, logger, mode=args.split)  # mode = ['val','test']
    
    # with open("custom_infer/STVG/data/hc-stvg2/annos/better_track.json", "r", encoding="utf-8") as f:
    #     vid_names_w_better_track = json.load(f)
    
    att_ratio_L = []
    count_tokens_N = np.zeros(6)
    for vid_i in range(len(vid_dataset)):
        try:
            vid_data = vid_dataset[vid_i]
        except Exception as e:
            print(f"Error occurred at sample {vid_i}: {e}")
            import traceback
            traceback.print_exc()  # 打印详细的错误堆栈信息
            continue
        
        video_data, sentence, targets, pil_images, video_time, fps = vid_data
        ref_sent = sentence
        vid_path = targets['vid']

        # ===================================== 1. 只考虑 top1k 的 case
        # if vid_i > 1000:
        #     break
        
        # if vid_i != check_case_id: # check case 33, 49, 78
        #     continue
        # ===================================== 2. 只考虑 track 的比较好的 case
        # if vid_path.split('.')[0] not in vid_names_w_better_track:
        #     continue
        
        print(vid_i, '/', len(vid_dataset), vid_path)
        
        # ===================================== LLava-TTT
        # gpt_4_proc_attr = targets['gpt_4_proc_attr']
        # gpt_4_proc_action = targets['gpt_4_proc_action']
        # gpt_4_proc_attr_q, gpt_4_proc_action_q = [], []
        # ===================================== 3. 只考虑可以 decompose 出 attribute 和 action 俩部分的case
        # decompose_fail = False
        # for sent in gpt_4_proc_attr:
        #     if "please describe this video" in sent:
        #         gpt_4_proc_attr_q.append(sent)
        #         decompose_fail = True
        #     else:
        #         gpt_4_proc_attr_q.append(f"Is there {sent.strip('.').strip()} in this video?")

        # for sent in gpt_4_proc_action:
        #     if "please describe this video" in sent:
        #         gpt_4_proc_action_q.append(sent)
        #         decompose_fail = True
        #     else:
        #         gpt_4_proc_action_q.append(f"Is there {sent.strip('.').strip()} in this video?")
        # if decompose_fail:
        #     continue
        
        # q_0 = [f"Is there {ref_sent.strip('.').strip()} in this video?"]
        # Q_list = []
        # Q_list.extend(gpt_4_proc_attr_q)
        # Q_list.extend(gpt_4_proc_action_q)
        # Q_list.extend(q_0)
        # ['Is there the man standing by the stairs in this video?', 'Is there The man goes to the car in this video?', 'Is there the man standing by the stairs goes to the car in this video?']
        try:
            st_tuning_module = st_tuning[infer_module_name.index(args.llm_infer_mode)]
            # # breakpoint()
            bbox_pred_tmp, temporal_pred_tmp = st_tuning_module.llava_embedd_tuning(
                                                            tokenizer=tokenizer,
                                                            cfg_pretrained=None, 
                                                            model=model, 
                                                            processor=image_processor,
                                                            video=video_data, 
                                                            questions=ref_sent, 
                                                            grid_size=grid_size, 
                                                            targets=targets, 
                                                            args=args, 
                                                            vid_i=vid_i,
                                                            logger=logger
                                                        )
            # att_ratio_L.append(att_ratio.item())
            # count_tokens_N[hit_token_id] += 1
        except:
            import traceback
            traceback.print_exc()
            continue
        # if max_track_iou > 0.55:# and 'then' in ref_sent:
        #     high_quality_video_names.append(vid_path.split('.')[0])
        
        evaluator.update(bbox_pred_tmp)
        evaluator.video_update(temporal_pred_tmp)
        temporal_duration = list(temporal_pred_tmp[vid_i]['sted'])
        
        writer.add_scalar('zs_check/temporal_duration', temporal_duration[1]-temporal_duration[0], vid_i)
        if vid_i % args.eval_freq  == 0:
            res = evaluator.summarize()
            logger.info('{} ====== > {}'.format(max(vid_i-args.eval_freq, 0), vid_i))
            #
            for key in res.keys():
                # breakpoint()
                writer.add_scalar(f'zs_eval/{key}', round(res[key]*100, 1), vid_i)
            
            # writer.add_scalar(f'zs_check/att_ratio_mean', round(sum(att_ratio_L) / (len(att_ratio_L) + 0.001), 1), vid_i)
            # print(np.array(count_tokens_N) / sum(count_tokens_N))
    # np.save(f"{output_dir}/_attention_ratio_token_{args.token_idx}.npy", att_ratio_L)





if __name__ == "__main__":
    args = parse_args()
    run_inference(args)