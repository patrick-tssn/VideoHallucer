import os
import random

import torch


from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init


from base import ViLLMBaseModel


class Args:
    def __init__(self):
        self.model_type = 'av' # a, v, av
        self.options = None  # Assuming None is the default when no options are provided

class VideoLLaMA2(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        args = Args()
        
        self.device = "cuda:"+str(model_args['device'])
        model_path = model_args['model_path']
        
        self.model, self.processor, self.tokenizer = model_init(model_path, device=self.device)
        
    @torch.no_grad()
    def generate(self, instruction, video_path):
        preprocess = self.processor["video"]
        audio_video_tensor = preprocess(video_path, va=True)
        outputs = mm_infer(
            audio_video_tensor,
            instruction,
            model=self.model,
            tokenizer=self.tokenizer,
            modal="video",
            do_sample=False
        )
        
        outputs = outputs.strip()
        # print(outputs)
        return outputs