import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from base import ViLLMBaseModel

class VideoLLaVA(ViLLMBaseModel):
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
        self.tokenizer, self.model, processor, _ = load_pretrained_model(self.model_path, None, model_name, load_8bit, load_4bit, device=self.device, cache_dir=cache_dir)
        self.video_processor = processor['video']
        self.conv_mode = "llava_v1"
        
    def generate(self, instruction, video_path):
        
        # assert len(videos) == 1
        # video_path = videos[0]
        # instruction = instruction[0] if type(instruction)==list else instruction
        
        conv = conv_templates[self.conv_mode].copy()
        roles = conv.roles
        video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video.to(self.model.device, dtype=torch.float16) for video in video_tensor]
        else:
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        # print(f"{roles[1]}: {instruction}")
        instruction = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + instruction
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        # print(outputs)
        return outputs