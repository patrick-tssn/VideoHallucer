import argparse
import json
import sys
import os
import numpy as np
import random
import uuid
from collections import defaultdict
from typing import Callable
from tqdm import tqdm

from evaluation_exp_utils import evaluate

sys.path.append(os.getcwd())

configs = json.load(open("./config.json"))

DATA_DIR = configs['DATA_DIR']
CKPT_DIR = configs['CKPT_DIR']





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
    elif TESTING_MODEL == "VideoLaVIT":
        from videolavit_modeling import VideoLaVIT
        ckpt_path = f"{CKPT_DIR}/Video-LaVIT-v1"
        model = VideoLaVIT({"model_path": ckpt_path, "device": 0})
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video/checkpoints"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": 0})
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

    return model


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="", 
                        choices=["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "Gemini-1.5-pro", "LLaVA", "GPT4V"])

    parser.add_argument(
        "--output_dir_path", type=str, default="results",
    )
    parser.add_argument("--device", type=int, default=0)

    # Per-dataset evaluation flags
    parser.add_argument(
        "--eval_obj_rel",
        action="store_true",
        default=False,
        help="Whether to evaluate on object&relation hallucination",
    )
    parser.add_argument(
        "--eval_temporal",
        action="store_true",
        default=False,
        help="Whether to evaluate on temporal hallucination.",
    )
    parser.add_argument(
        "--eval_semantic",
        action="store_true",
        default=False,
        help="Whether to evaluate on other semantic detail hallucination.",
    )
    parser.add_argument(
        "--eval_fact",
        action="store_true",
        default=True,
        help="Whether to evaluate on fact hallucination.",
    )
    parser.add_argument(
        "--eval_nonfact",
        action="store_true",
        default=False,
        help="Whether to evaluate on fact hallucination.",
    )
    parser.add_argument(
        "--detect_fact",
        action="store_true",
        default=False,
        help="Whether to detect factual and nonfactula knowledge.",
    )

    ## Object-Relation Dataset
    parser.add_argument(
        "--obj_rel_path",
        type=str,
        default="object_relation/object_relation.json",
    )
    parser.add_argument(
        "--obj_rel_video_dir_path",
        type=str,
        default="object_relation/videos",
    )
    ## Temporal Dataset
    parser.add_argument(
        "--temporal_path",
        type=str,
        default="temporal/temporal.json",
    )
    parser.add_argument(
        "--temporal_video_dir_path",
        type=str,
        default="temporal/videos",
    )
    ## Other Semantic Detail Dataset
    parser.add_argument(
        "--semantic_path",
        type=str,
        default="semantic_detail/semantic_detail.json",
    )
    parser.add_argument(
        "--semantic_video_dir_path",
        type=str,
        default="semantic_detail/videos",
    )
    ## External Fact Dataset
    parser.add_argument(
        "--fact_path",
        type=str,
        default="external_factual/external_factual.json",
    )
    parser.add_argument(
        "--fact_video_dir_path",
        type=str,
        default="external_factual/videos",
    )
    ## External Non-Fact Dataset
    parser.add_argument(
        "--nonfact_path",
        type=str,
        default="external_nonfactual/external_nonfactual.json",
    )
    parser.add_argument(
        "--nonfact_video_dir_path",
        type=str,
        default="external_nonfactual/videos",
    )
    ## Fact-Nonfact Detect Dataset
    parser.add_argument(
        "--factdet_path",
        type=str,
        default="fact_detect/fact_detect.json",
    )
    parser.add_argument(
        "--factdet_video_dir_path",
        type=str,
        default="fact_detect/videos",
    )
    args = parser.parse_args()
    
    model = load_model(args.model_name)
    # model = None
    final_result = {}

    if args.eval_obj_rel:
        obj_rel_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.obj_rel_path),
            qa_type='obj_rel',
            video_dir_path=os.path.join(DATA_DIR, args.obj_rel_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["obj_rel"] = obj_rel_scores

    if args.eval_temporal:
        temporal_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.temporal_path),
            qa_type='temporal',
            video_dir_path=os.path.join(DATA_DIR, args.temporal_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["temporal"] = temporal_scores

    if args.eval_semantic:
        semantic_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.semantic_path),
            qa_type='semantic',
            video_dir_path=os.path.join(DATA_DIR, args.semantic_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["semantic"] = semantic_scores
    
    if args.eval_fact:
        fact_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.fact_path),
            qa_type='fact',
            video_dir_path=os.path.join(DATA_DIR, args.fact_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["fact"] = fact_scores

    if args.eval_nonfact:
        nonfact_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.nonfact_path),
            qa_type='nonfact',
            video_dir_path=os.path.join(DATA_DIR, args.nonfact_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["nonfact"] = nonfact_scores
    
    if args.detect_fact:
        factdet_scores = evaluate(
            model=model,
            model_name=args.model_name,
            qa_path=os.path.join(DATA_DIR, args.factdet_path),
            qa_type='factdet',
            video_dir_path=os.path.join(DATA_DIR, args.factdet_video_dir_path),
            output_dir_path=args.output_dir_path   
        )
        final_result["factdet"] = factdet_scores
    
    res_dct = final_result["fact"]
    ori_acc = res_dct["ori_accuracy"]
    self_acc = res_dct["self_accuracy"]
    gt_acc = res_dct["gt_accuracy"]

    
    
    eval_result_path = os.path.join(args.output_dir_path, f"exp_{args.model_name}_evaluation_results.json")
    with open(eval_result_path, "w") as jp:
        json.dump(final_result, jp, indent=4)
    print("="*20)
    print("Original Accuracy: ", ori_acc)
    print("Self-Explanation Accuracy: ", self_acc)
    print("GT-Explanation Accuracy: ", gt_acc)
    print("="*20)

if __name__ == "__main__":
    main()