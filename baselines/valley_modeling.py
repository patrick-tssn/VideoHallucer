import argparse
from transformers import AutoTokenizer
from valley.model.valley_model import ValleyLlamaForCausalLM
import torch
from enum import Enum

from valley.util.config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_FRAME_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)

from base import ViLLMBaseModel

class Valley(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        device = torch.device("cuda:"+str(model_args["device"]) if torch.cuda.is_available() else "cpu")
        model_path = model_args["model_path"]
        self.model = ValleyLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        init_vision_token(self.model, self.tokenizer)

        self.model = self.model.to(device)
        self.model.eval()

        
    def generate(self, instruction, video_path):

        # input the query
        query = f"{DEFAULT_VIDEO_TOKEN} {instruction}"
        # input the system prompt
        system_prompt = "You are Valley, a large language and vision assistant trained by ByteDance. You are able to understand the visual content or video that the user provides, and assist the user with a variety of tasks using natural language. Follow the instructions carefully and explain your answers in detail."

        # we support openai format input
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hi!"},
            {"role": "assistent", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": query},
        ]

        gen_kwargs = dict(
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
        )


        response = self.model.completion(self.tokenizer, video_path, message, gen_kwargs, self.device)
        # print(response)
        outputs = response[0].strip()
        
        # print(outputs)
        return outputs

def init_vision_token(model, tokenizer):
    vision_config = model.get_model().vision_tower.config
    (
        vision_config.im_start_token,
        vision_config.im_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    (
        vision_config.vi_start_token,
        vision_config.vi_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
    vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(
        DEFAULT_VIDEO_FRAME_TOKEN
    )
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
