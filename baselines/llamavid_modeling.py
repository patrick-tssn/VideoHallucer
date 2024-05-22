import os
import torch

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from base import ViLLMBaseModel

class Args:
    def __init__(self):
        self.model_path = "facebook/opt-350m"
        self.model_base = None
        self.image_file = None
        self.num_gpus = 1
        self.conv_mode = None
        self.temperature = 0.5  # set to 0.2 for image
        self.top_p = 0.7
        self.max_new_tokens = 512
        self.load_8bit = False
        self.load_4bit = False
        self.debug = False

class LLaMAVID(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        self.args = Args()
        if '7B' in model_args["model_path"]:
            self.args.model_path = os.path.join(model_args["model_path"], "llama-vid-7b-full-224-video-fps-1")
        elif '13B' in model_args["model_path"]:
                self.args.model_path = os.path.join(model_args["model_path"], "llama-vid-13b-full-224-video-fps-1")
        device = "cuda:" + str(model_args["device"])
        
        # Model
        disable_torch_init()

        self.model_name = get_model_name_from_path(self.args.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.args.model_path, self.args.model_base, self.model_name, self.args.load_8bit, self.args.load_4bit, device=device)

        
        
    def generate(self, instruction, video_path):

        if 'llama-2' in self.model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower() or "vid" in self.model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, self.args.conv_mode, self.args.conv_mode))
        else:
            self.args.conv_mode = conv_mode

        conv = conv_templates[self.args.conv_mode].copy()
        if "mpt" in self.model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles


        if video_path is not None:
            if '.mp4' in video_path:
                image = load_video(video_path)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
                image_tensor = [image_tensor]
            else:
                image = load_image(video_path)
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        else:
            image_tensor = None

        
        self.model.update_prompt([[instruction]])

        if video_path is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                instruction = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + instruction
            else:
                instruction = DEFAULT_IMAGE_TOKEN + '\n' + instruction
            conv.append_message(conv.roles[0], instruction)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                max_new_tokens=self.args.max_new_tokens,
                # streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith("</s>"):
            outputs = outputs.split("</s>")[0]
        # print(outputs)
        return outputs


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def load_video(video_path, fps=1):
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames