import os, sys
import json
import argparse
sys.path.append(os.getcwd())

video_path = "../assets/test_video.mp4"

instruction = "Is there a man in the video? response with 'yes' or 'no'."

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str,
                    default="", 
                    choices=["Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "Gemini-1.5-pro"])
args = parser.parse_args()
TESTING_MODEL=args.model_name


def load_model(TESTING_MODEL):
    if TESTING_MODEL == "Video-LLaMA-2":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": 0})
    if TESTING_MODEL == "VideoChat2":
        from videochat_modeling1 import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": 0})
    if TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling import VideoLLaVA
        ckpt_path = f'{CKPT_DIR}/Video-LLaVA-7B'
        model = VideoLLaVA({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "Gemini-1.5-pro":
        from gemini_modeling import Gemini
        model = Gemini({"model_path": None, "device": 0})
    
    return model

model = load_model(TESTING_MODEL)


pred = model.generate(
    instruction=instruction,
    video_path=video_path,
)
print(f'Instruction:\t{instruction}')
print(f'Answer:\t{pred}')
print('-'*20)
