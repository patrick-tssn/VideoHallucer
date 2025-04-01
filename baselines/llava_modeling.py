import os, shutil, cv2
import requests
from PIL import Image
from io import BytesIO

import torch
from transformers import TextStreamer

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path



from base import ViLLMBaseModel

class LLaVA(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_path = model_args['model_path']
        self.device = model_args['device']
        cache_dir = 'cache_dir'

        load_4bit, load_8bit = False, False
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(self.model_path, None, model_name, load_8bit, load_4bit, device=self.device, cache_dir=cache_dir)
        self.conv_mode = "llava_v1"
        
    def generate(self, instruction, video_path):
        
        
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles
        image_path = extract_frame_from_video(video_path)
        image = load_image(image_path)
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        if image is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                instruction = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + instruction
            else:
                instruction = DEFAULT_IMAGE_TOKEN + '\n' + instruction
            image = None
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512,
                # streamer=streamer,
                use_cache=True,)

        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return outputs

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image



def create_frame_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

def extract_frame_from_video(video_file_path, FPS=1):
    # print(f"Extracting {video_file_path} at 1 frame per second. This might take a bit...")
    # print(video_file_path)
    FRAME_EXTRACTION_DIRECTORY = "./cache_dir/llava"
    FRAME_PREFIX = "llava_"
    create_frame_output_dir(FRAME_EXTRACTION_DIRECTORY)
    vidcap = cv2.VideoCapture(video_file_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps  # Time interval between frames (in seconds)
    output_file_prefix = os.path.basename(video_file_path).replace('.', '_')
    frame_count = 0
    count = 0
    while vidcap.isOpened():
        success, frame = vidcap.read()
        if not success: # End of video
            break
        if int(count / fps) == frame_count: # Extract a frame every second
            min = frame_count // 60
            sec = frame_count % 60
            time_string = f"{min:02d}:{sec:02d}"
            image_name = f"{output_file_prefix}{FRAME_PREFIX}{time_string}.jpg"
            output_filename = os.path.join(FRAME_EXTRACTION_DIRECTORY, image_name)
            cv2.imwrite(output_filename, frame)
            frame_count += 1
        count += 1
    vidcap.release() # Release the capture object\n",
    files = os.listdir(FRAME_EXTRACTION_DIRECTORY)
    files = sorted(files)
    selected_file = files[len(files)//2]
    file_path = os.path.join(FRAME_EXTRACTION_DIRECTORY, selected_file)
    return file_path
    # print(f"Completed video frame extraction!\n\nExtracted: {frame_count} frames")

