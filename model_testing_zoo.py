import os, sys
import json
import argparse
sys.path.append(os.getcwd())

video_path = "../assets/test_video.mp4"

instruction = "Is there a dog in the video? Response with 'yes' or 'no'."
# instruction = "Describe the Video: "

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="", 
                    choices=["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video", "ShareGPT4Video",
                             "Gemini-1.5-pro", "GPT4O",
                             "LLaVA", "GPT4V", 
                             "Video-LLaMA-2-13B", "LLaMA-VID-13B", 
                             "PLLaVA-13B", "PLLaVA-34B", "LLaVA-NeXT-Video-34B",
                             "VideoLLaMA2"])
args = parser.parse_args()
TESTING_MODEL=args.model_name


def load_model(TESTING_MODEL):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Valley2":
        from valley_modeling import Valley
        ckpt_path = f"{CKPT_DIR}/Valley2-7b"
        model = Valley({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Video-LLaMA-2-13B":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-13B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoChat2":
        from videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling import VideoLLaVA
        ckpt_path = f"{CKPT_DIR}/VideoLLaVA/Video-LLaVA-7B"
        model = VideoLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaMA-VID-13B":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-13B"
        model = LLaMAVID({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "VideoLaVIT":
        from videolavit_modeling import VideoLaVIT
        ckpt_path = f"{CKPT_DIR}/Video-LaVIT-v1"
        model = VideoLaVIT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video/checkpoints"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA-13B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-13b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "PLLaVA-34B":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-34b"
        model = PLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "LLaVA-NeXT-Video-34B":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-34B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "ShareGPT4Video":
        from sharegpt4video_modeling import ShareGPT4Video
        ckpt_path = f"{CKPT_DIR}/ShareGPT4Video/sharegpt4video-8b"
        model = ShareGPT4Video({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Gemini-1.5-pro":
        from gemini_modeling import Gemini
        model = Gemini({"model_path": None, "device": 0})
    elif TESTING_MODEL == "LLaVA":
        from llava_modeling import LLaVA
        ckpt_path = f"{CKPT_DIR}/LLaVA/llava-v1.5-7b"
        model = LLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "GPT4V":
        from gpt4v_modeling import GPT4V
        model = GPT4V({"model_path": None, "device": 0})
    elif TESTING_MODEL == "GPT4O":
        from gpt4o_modeling import GPT4O
        model = GPT4O({"model_path": None, "device": 0})
    elif TESTING_MODEL == "VideoLLaMA2":
        from videollama2_modeling import VideoLLaMA2
        model = VideoLLaMA2({"model_path": f"{CKPT_DIR}/VideoLLaMA2.1-7B-AV", "device": 0})

    return model

model = load_model(TESTING_MODEL)


pred = model.generate(
    instruction=instruction,
    video_path=video_path,
)
print('-'*20)
print(f'Instruction:\t{instruction}')
print(f'Answer:\t{pred}')
print('-'*20)
