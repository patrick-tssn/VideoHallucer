import os

import torch
from peft import get_peft_model, LoraConfig, TaskType

# videochat
from pllava.utils.easydict import EasyDict
from pllava.tasks.eval.model_utils import load_pllava
from pllava.tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)
from pllava.tasks.eval.demo import pllava_theme


SYSTEM="""You are Pllava, a large vision-language assistant. 
You are able to understand the video content that the user provides, and assist the user with a variety of tasks using natural language.
Follow the instructions carefully and explain your answers in detail based on the provided video.
"""
INIT_CONVERSATION: Conversation = conv_plain_v1.copy()

from base import ViLLMBaseModel

class PLLaVA(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        # init model
        self.model, self.processor = load_pllava(
            model_args['model_path'],
            16,
            use_lora=True,
            weight_dir=model_args['model_path'],
            lora_alpha=4,
            use_multi_gpus=None
        )
        device = model_args['device']
        self.model = self.model.to(f'cuda:{device}')
        self.chat = ChatPllava(self.model, self.processor)
        self.model_args = model_args
        
    def generate(self, instruction, video_path):

        num_beams = 1
        temperature = 0.2
        max_new_tokens = 200
        if '34B' in self.model_args["model_path"]: max_new_tokens = 5 # FIXME

        chat_state = INIT_CONVERSATION.copy()
        img_list = []
        llm_message, img_list, chat_state = self.chat.upload_video(video_path, chat_state, img_list, None, )

        chat_state = self.chat.ask(instruction, chat_state, SYSTEM)
        llm_message, llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=max_new_tokens, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "")

        outputs = llm_message.strip()
        return outputs
