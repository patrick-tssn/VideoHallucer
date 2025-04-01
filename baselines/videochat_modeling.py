import os

import torch
from peft import get_peft_model, LoraConfig, TaskType

# videochat
from video_chat2.utils.config import Config
from video_chat2.utils.easydict import EasyDict
from video_chat2.models.videochat2_it import VideoChat2_it
from video_chat2.conversation import Chat

from base import ViLLMBaseModel

class VideoChat(ViLLMBaseModel):
    def __init__(self, model_args):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        config_file = "video_chat2/configs/config.json"
        cfg = Config.from_file(config_file)
        cfg.model.vision_encoder.num_frames = 4
        cfg.model.vit_blip_model_path = os.path.join(model_args["model_path"], "umt_l16_qformer.pth")
        cfg.model.llama_model_path = os.path.join(model_args["model_path"], "vicuna-7b-v0")
        cfg.model.videochat2_model_path = os.path.join(model_args["model_path"], "videochat2_7b_stage2.pth")
        cfg.device = model_args["device"]
        # cfg.model.videochat2_model_path = ""
        # cfg.model.debug = True
        model = VideoChat2_it(config=cfg.model)
        model = model.to(torch.device(cfg.device))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        state_dict = torch.load(os.path.join(model_args["model_path"], "videochat2_7b_stage3.pth"), "cpu")
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model = model.eval()
        self.chat = Chat(model)
        
    def generate(self, instruction, video_path):
        
        chat_state = EasyDict({
            "system": "",
            "roles": ("Human", "Assistant"),
            "messages": [],
            "sep": "###"
        })
        img_list = []
        num_segments = 8
        num_beams = 1
        temperature = 1.0
        llm_message, img_list, chat_state = self.chat.upload_video(video_path, chat_state, img_list, num_segments)
        chat_state = self.chat.ask(instruction, chat_state)

        llm_message,llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "") # handle <s>
        outputs = llm_message.strip()
        return outputs
