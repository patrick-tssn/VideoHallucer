import os
import torch
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

from videolavit.models import build_model

from base import ViLLMBaseModel

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class VideoLaVIT(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        
        model_path = os.path.join(model_args["model_path"], "language_model_sft")
        device_id = model_args["device"]
        model_dtype = "bf16"
        # model_dtype = "fp16"

        max_video_clips = 16
        torch.cuda.set_device(device_id)
        self.device = torch.device("cuda")

        # For Multi-Modal Understanding
        self.runner = build_model(model_path=model_path, model_dtype=model_dtype, understanding=True, 
                device_id=device_id, use_xformers=True, max_video_clips=max_video_clips,)


    def generate(self, instruction, video_path):

        
        output = self.runner({"video": video_path, "text_input": instruction}, length_penalty=1, \
            use_nucleus_sampling=False, num_beams=1, max_length=512, temperature=1.0)[0]
        outputs = output.strip()
        
        # print(outputs)
        return outputs