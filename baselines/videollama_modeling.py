import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import * 


from base import ViLLMBaseModel


class Args:
    def __init__(self):
        self.cfg_path = 'video_llama/video_llama_eval_withaudio.yaml'
        self.gpu_id = 0
        self.model_type = 'llama_v2'
        self.options = None  # Assuming None is the default when no options are provided



class VideoLLaMA(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        args = Args()
        args.gpu_id = model_args['device']
        model_path = model_args['model_path']
        cfg = Config(args)

        if '7B' in model_path:
            cfg.model_cfg.llama_model = os.path.join(model_path, "llama-2-7b-chat-hf")
            cfg.model_cfg.imagebind_ckpt_path = model_path
            cfg.model_cfg.ckpt = os.path.join(model_path, "VL_LLaMA_2_7B_Finetuned.pth")
            cfg.model_cfg.ckpt_2 = os.path.join(model_path, "AL_LLaMA_2_7B_Finetuned.pth")
        elif '13B' in model_path:
            cfg.model_cfg.llama_model = os.path.join(model_path, "llama-2-13b-chat-hf")
            cfg.model_cfg.imagebind_ckpt_path = model_path
            cfg.model_cfg.ckpt = os.path.join(model_path, "VL_LLaMA_2_13B_Finetuned.pth")
            cfg.model_cfg.ckpt_2 = os.path.join(model_path, "AL_LLaMA_2_13B_Finetuned.pth")
        
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        
        
    def generate(self, instruction, video_path):
        
        # assert len(videos) == 1
        # video_path = videos[0]
        # instruction = instruction[0] if type(instruction)==list else instruction
        
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = ""
        img_list = []
        llm_message = self.chat.upload_video_without_audio(video_path, chat_state, img_list)
        self.chat.ask(instruction, chat_state)

        num_beams = 1
        temperature = 0.2
        with torch.inference_mode():
            llm_message = self.chat.answer(
                conv=chat_state,
                img_list=img_list,
                num_beams=num_beams,
                temperature=temperature,
                max_new_tokens=300,
                max_length=2000
            )[0]

        
        outputs = llm_message.strip()
        # print(outputs)
        return outputs