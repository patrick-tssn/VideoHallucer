import torch

from llavavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llavavid.conversation import conv_templates, SeparatorStyle
from llavavid.model.builder import load_pretrained_model
from llavavid.utils import disable_torch_init
from llavavid.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

import json
import os
import math
from tqdm import tqdm
from decord import VideoReader, cpu

from transformers import AutoConfig

import time

import numpy as np


from base import ViLLMBaseModel


class Args:
    def __init__(self):
        self.video_path = None  # Required argument
        self.output_dir = None  # Required argument
        self.output_name = None  # Required argument
        self.model_path = "facebook/opt-350m"
        self.model_base = None
        self.conv_mode = None
        self.chunk_idx = 0
        self.mm_resampler_type = "spatial_pool"
        self.mm_spatial_pool_stride = 4
        self.mm_spatial_pool_out_channels = 1024
        self.mm_spatial_pool_mode = "average"
        self.image_aspect_ratio = "anyres"
        self.image_grid_pinpoints = "[(224, 448), (224, 672), (224, 896), (448, 448), (448, 224), (672, 224), (896, 224)]"
        self.mm_patch_merge_type = "spatial_unpad"
        self.overwrite = True
        self.for_get_frames_num = 4
        self.load_8bit = False

    def parse_bool(self, value):
        return str(value).lower() == 'true'

    def set_args(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key in ['overwrite', 'load_8bit']:
                    setattr(self, key, self.parse_bool(value))
                else:
                    setattr(self, key, value)
            else:
                raise AttributeError(f"Argument '{key}' not found in Args class.")

class LLaVANeXT(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        self.device = model_args["device"]

        args = Args()
        if '7B' in model_args["model_path"]:
            conv_mode = "vicuna_v1"
        elif "34B" in model_args["model_path"]:
            conv_mode = "mistral_direct"
        args.set_args(
            model_path=model_args["model_path"],
            conv_mode=conv_mode,
            for_get_frames_num=32,
            mm_spatial_pool_stride=2
        )

        # Initialize the model
        model_name = get_model_name_from_path(args.model_path)
        # Set model configuration parameters if they exist
        if args.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = args.mm_resampler_type
            overwrite_config["mm_spatial_pool_stride"] = args.mm_spatial_pool_stride
            overwrite_config["mm_spatial_pool_out_channels"] = args.mm_spatial_pool_out_channels
            overwrite_config["mm_spatial_pool_mode"] = args.mm_spatial_pool_mode
            overwrite_config["patchify_video_feature"] = False

            cfg_pretrained = AutoConfig.from_pretrained(args.model_path)

            
            if "224" in cfg_pretrained.mm_vision_tower:
                # suppose the length of text tokens is around 1000, from bo's report
                least_token_number = args.for_get_frames_num*(16//args.mm_spatial_pool_stride)**2 + 1000
            else:
                least_token_number = args.for_get_frames_num*(24//args.mm_spatial_pool_stride)**2 + 1000

            scaling_factor = math.ceil(least_token_number/4096)
            # import pdb;pdb.set_trace()

            if scaling_factor >= 2:
                if "mistral" not in cfg_pretrained._name_or_path.lower() and "7b" in cfg_pretrained._name_or_path.lower():
                    print(float(scaling_factor))
                    overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
                overwrite_config["max_sequence_length"] = 4096 * scaling_factor
                overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_8bit=args.load_8bit, overwrite_config=overwrite_config)
        else:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

        self.args = args
        
        
    def generate(self, instruction, video_path):
        
        if os.path.exists(video_path):
            video = load_video(video_path, self.args)
            video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].half().cuda(self.device)

        qs = instruction
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda(self.device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        cur_prompt = instruction
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            # import pdb;pdb.set_trace()
            start_time = time.time()
            output_ids = self.model.generate(inputs=input_ids, images=video, attention_mask=attention_masks, modalities="video", do_sample=True, temperature=0.2, max_new_tokens=1024, use_cache=True, stopping_criteria=[stopping_criteria])
            end_time = time.time()
            # print(f"Time taken for inference: {end_time - start_time} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outptus[:-len(stop_str)]
        outputs = outputs.strip()

        return outputs


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    # fps = round(vr.get_avg_fps())
    # frame_idx = [i for i in range(0, len(vr), fps)]
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

